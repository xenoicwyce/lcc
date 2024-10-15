# Standard Libraries
import json
import copy

# Third-party Libraries
import numpy as np
from pathlib import Path
from pydantic import BaseModel, Field

# Qiskit and Related Libraries
from qiskit.primitives import BackendSampler
from qiskit_optimization.applications import Maxcut
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.algorithms import MinimumEigenOptimizer, MinimumEigenOptimizationResult
from qiskit_optimization.problems import QuadraticProgram
from qiskit_algorithms import SamplingVQE, QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer import AerSimulator

# Docplex
from docplex.mp.model import Model

from .utils import to_serializable


class Knapsack:
    def __init__(self, model_path):
        self.converter = QuadraticProgramToQubo()
        with open(model_path, 'r', encoding='utf-8') as f:
            self.params = json.load(f)
        
        self.quadratic_program = self.construct_qiskit_quadratic_program()
        self.maxcut_qp = self.convert_qp_to_maxcut()

    @staticmethod
    def construct_cplex_model(params) -> Model:
        """
        Construct a CPLEX model for the Multiple Knapsack problem.
        
        Args:
            params (dict): Parameters for the problem including number of items, number of bins, 
                        weights, capacities, and values.
                        
        Returns:
            Model: CPLEX model object.
        """
        mdl = Model("Multiple Knapsack Model")

        x = mdl.binary_var_matrix(params["num_items"], params["num_bins"], name="x")

        # Each item can be in at most one bin
        for i in range(params["num_items"]):
            mdl.add_constraint(mdl.sum(x[i, j] for j in range(params["num_bins"])) <= 1)

        # The total weight in each bin cannot exceed its capacity
        for j in range(params["num_bins"]):
            mdl.add_constraint(
                mdl.sum(x[i, j] * params["weights"][i] for i in range(params["num_items"]))
                <= params["capacities"][j]
            )

        # Objective function
        total_value = mdl.sum(
            x[i, j] * params["values"][i]
            for i in range(params["num_items"])
            for j in range(params["num_bins"])
        )
        mdl.maximize(total_value)

        return mdl
    
    def construct_qiskit_quadratic_program(self) -> QuadraticProgram:
        """
        Convert a CPLEX model to a QuadraticProgram for Qiskit optimization.
        
        Args:
            params (dict): Parameters for the problem.
                        
        Returns:
            QuadraticProgram: QuadraticProgram object.
        """
        mdl = self.construct_cplex_model(self.params)
        qp = from_docplex_mp(mdl)
        return qp
    
    def convert_qp_to_maxcut(self) -> QuadraticProgram:
        qubo = self.converter.convert(self.quadratic_program)

        linear = qubo.objective.linear.to_array()
        quadratic = qubo.objective.quadratic.to_array()
        weighted_max_cut_qubo = QUBO(quadratic, linear)
        weighted_max_cut_qubo.linear_to_sqaure()

        max_cut_graph = weighted_max_cut_qubo.to_maxcut()
        max_cut = Maxcut(max_cut_graph)
        return max_cut.to_quadratic_program()
    
    @staticmethod
    def maxcut_to_qubo_solution(cut_solution) -> list[int]:
        """
        Convert the solution of the weighted max cut to the original QUBO solution.
        
        Parameters:
        cut_solution (list): A binary list representing the cut solution.
        
        Returns:
        list: The solution to the original QUBO problem.
        """
        n = len(cut_solution) - 1  # Subtract 1 because cut_solution includes the extra 0th node
        qubo_solution = []

        for i in range(1, n + 1):
            if cut_solution[i] != cut_solution[0]:
                qubo_solution.append(0)
            else:
                qubo_solution.append(1)
    
        return qubo_solution
    
    def solve_exact(self) -> int:
        solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
        result = solver.solve(self.quadratic_program)
        exact_cost = self.quadratic_program.objective.evaluate(result.x)

        return exact_cost
    
    def solve_vqe(
        self,
        ansatz,
        to_maxcut=False,
        optimizer=None, 
    ) -> MinimumEigenOptimizationResult:        
        backend = AerSimulator(
            method='statevector',
            statevector_parallel_threshold=10,
        )
        sampler = BackendSampler(backend=backend)
        optimizer = COBYLA() if optimizer is None else optimizer
        ansatz_c = copy.deepcopy(ansatz)

        svqe = SamplingVQE(sampler, ansatz_c, optimizer)
        solver = MinimumEigenOptimizer(svqe)
        
        if to_maxcut:
            result = solver.solve(self.maxcut_qp)
        else:
            result = solver.solve(self.quadratic_program)

        return result
    
    def solve_qaoa(
        self,
        reps=1,
        to_maxcut=False,
        optimizer=None,
    ) -> MinimumEigenOptimizationResult:
        backend = AerSimulator(
            method='statevector',
            statevector_parallel_threshold=10,
        )
        sampler = BackendSampler(backend=backend)
        optimizer = COBYLA() if optimizer is None else optimizer
        qaoa = QAOA(sampler, optimizer, reps=reps)
        solver = MinimumEigenOptimizer(qaoa)

        if to_maxcut:
            result = solver.solve(self.maxcut_qp)
        else:
            result = solver.solve(self.quadratic_program)
        
        return result
    

class KnapsackResult(BaseModel):
    ins_name: str
    num_vars: int
    num_qubo_vars: int
    num_qubits: int
    exact_cost: int = 0
    qubo_solution: list[list[int]] = Field(default_factory=list)
    qp_solution: list[list[int]] = Field(default_factory=list)
    maxcut_cost: list[float] = Field(default_factory=list)
    qp_cost: list[float] = Field(default_factory=list)
    feasibility: list[bool] = Field(default_factory=list)
    success: list[bool] = Field(default_factory=list)
    almost_success: list[bool] = Field(default_factory=list)
    feasibility_rate: float = 0.
    success_rate: float = 0.
    almost_success_rate: float = 0.

    def post_process(self) -> None:
        self.success = [
            (self.qp_cost[i] == self.exact_cost) and self.feasibility[i] for i in range(len(self.feasibility))
        ]
        self.almost_success = [
            (0.9*self.exact_cost <= self.qp_cost[i] <= self.exact_cost) and self.feasibility[i] for i in range(len(self.feasibility))
        ]
        self.feasibility_rate = np.sum(self.feasibility) / len(self.feasibility)
        self.success_rate = np.sum(self.success) / len(self.success)
        self.almost_success_rate = np.sum(self.almost_success) / len(self.almost_success)


    def dump(self, file_path) -> None:
        file_path = Path(file_path)
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)

        with file_path.open('w') as f:
            json.dump(self.model_dump(), f, default=to_serializable, indent=4)


class QUBO:
    def __init__(self, quad: np.array, linear: np.array) -> None:
        self.num_vars = self._check_shape(quad, linear)

        self.quad = quad.copy()
        self.linear = linear.copy()

    def to_maxcut(self) -> np.array:
        """
        Create a graph where its MAXCUT problem is equivalent to the original QUBO 
        Assume that the QUBO is in the following form, i.e. the linear terms are on the diagonal:
            Q = sum_{i=1}^{n} sum_{j=1}^{n} q_{i,j} * x_i * x_j
        """
        graph = np.zeros((self.num_vars + 1, self.num_vars + 1))
        
        # node 0 to all other nodes
        for i in range(1, self.num_vars + 1):
            graph[0, i] = np.sum(self.quad[i-1, :]) + np.sum(self.quad[:, i-1])
            graph[i, 0] = graph[0, i]
        
        # all other nodes to each other
        for i in range(1, self.num_vars + 1):
            for j in range(i + 1, self.num_vars + 1):
                graph[i, j] = self.quad[i-1, j-1] + self.quad[j-1, i-1]
                graph[j, i] = graph[i, j]

        return graph

    def linear_to_square(self) -> None:
        """
        Convert linear terms (c_i * x_i) to square terms (c_ii * x_i^2)
        """
        for i in range(self.num_vars):
            self.quad[i, i] += self.linear[i]
        self.linear = np.zeros(self.num_vars)
    
    def square_to_linear(self) -> None:
        """
        Convert square terms (c_ii * x_i^2) to linear terms (c_i * x_i)
        """
        for i in range(self.num_vars):
            self.linear[i] += self.quad[i, i]
        np.fill_diagonal(self.quad, 0)

    def _check_shape(self, quad: np.array, linear: np.array) -> None:
        """
        Check if the shape of the linear and quadratic terms match each other
        Return the number of variables in the QUBO
        """
        quad_shape, linear_shape = quad.shape, linear.shape
        
        #TODO: Add check for data type 
        quad_dtype, linear_dtype = quad.dtype, linear.dtype

        if len(quad_shape) != 2:
            raise ValueError("The quadratic terms are not 2D")
        if quad_shape[0] != quad_shape[1]:
            raise ValueError("The quadratic terms are not square")
        if len(linear_shape) != 1:
            raise ValueError("The linear terms are not 1D")
        if quad_shape[0] != linear_shape[0]:
            raise ValueError("The shape of the linear and quadratic terms do not match")
        
        return quad_shape[0]
