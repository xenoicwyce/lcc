"""
This python file contains the implementation of the linear programming model, and
all the classical functions used in the pauli correlation encoding method

"""

import json # required for reading the json file
import time # required for timing the execution of the code
import numpy as np # required for array manipulation
import pandas as pd # required for data manipulation
from docplex.mp.model import Model # required for creating the model
from qiskit_optimization import QuadraticProgram # required for creating the quadratic program
from qiskit_optimization.translators import from_docplex_mp # required for converting the model to a quadratic program
# from cplex.exceptions import CplexError # required for handling the cplex errors
import networkx as nx # required for creating the graph
import matplotlib.pyplot as plt # required for plotting the graph
import itertools

"""
Function to make an LP model for the given data
"""
def construct_cplex_model(params) -> Model:
    mdl = Model("RBDO Model")

    upper_bounds = params["demands"] * params["num_suppliers"]
    order_amount = mdl.integer_var_matrix(
        params["num_suppliers"], params["num_parts"], lb=0, ub=upper_bounds, name="y"
    )

    # Objective function
    total_cost = mdl.sum(
        order_amount[(i, j)] * params["costs"][i][j]
        for i in range(params["num_suppliers"])
        for j in range(params["num_parts"])
    )
    mdl.minimize(total_cost)

    # Constraints
    for j in range(params["num_parts"]):
        # Demand
        mdl.add_constraint(
            mdl.sum(order_amount[(k, j)] for k in range(params["num_suppliers"]))
            == params["demands"][j]
        )
        # Risk tolerance
        mdl.add_constraint(
            mdl.sum(
                params["risk_scores"][k] * order_amount[(k, j)]
                for k in range(params["num_suppliers"])
            )
            <= params["risk_tolerance_per_part"][j] * params["demands"][j]
        )

    return mdl


def construct_qiskit_quadratic_program(params) -> QuadraticProgram:
    mdl = construct_cplex_model(params)
    qp = from_docplex_mp(mdl)
    return qp


def interpret(sol, params):
    """
    Interpret the solution of the problem.
    
    Args:
        sol (dict): Solution of the problem.
        params (dict): Parameters for the problem.
                       
    Returns:
        list: Interpretation of the solution.
    """
    result = []
    for i in range(params["num_items"]):
        for j in range(params["num_bins"]):
            if sol[f"x_{i}_{j}"] == 1:
                result.append(j)
                break
        else:
            result.append(-1)
    return result


def interpret_dict(sol, params):
    """
    Interpret the solution of the problem.
    
    Args:
        sol (dict): Solution of the problem.
        params (dict): Parameters for the problem.
                       
    Returns:
        dict: Interpretation of the solution.
    """
    result = {}
    for i in range(params["num_items"]):
        for j in range(params["num_bins"]):
            if sol[f"x_{i}_{j}"] == 1:
                if j not in result:
                    result[j] = [i]
                else:
                    result[j].append(i)
    return result


def solve_lp_from_json(json_file):
    """
    Solve the LP problem defined in a JSON file.
    
    Args:
        json_file (str): Path to the JSON file containing the problem definition.
                       
    Returns:
        tuple: A tuple containing the status, solution, and objective value.
    """
    try:
        # Load the LP model from JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            params = json.load(f)

        # Construct the CPLEX model
        mdl = construct_cplex_model(params)

        # Create a CPLEX problem instance
        prob = mdl.get_cplex()

        # Solve the problem
        prob.solve()

        # Get solution status
        status = prob.solution.get_status_string()

        # Get solution values
        solution = prob.solution.get_values()

        # Get objective value
        obj_value = prob.solution.get_objective_value()

        return status, solution, obj_value

    except CplexError as exc:
        print(exc)
        return None, None, None

def create_graph_from_weight_matrix(w):
    G = nx.Graph()
    n = len(w)

    # Add nodes
    for i in range(n):
        G.add_node(i)

    # Add edges with weights, ignoring zero-weight edges
    for i in range(n):
        for j in range(i + 1, n):
            if w[i, j] != 0:
                G.add_edge(i, j, weight=w[i, j])

    return G

def draw_graph(G, colors, pos):
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)

def print_graph_info(G):
    """
    Print the number of nodes and edges in the graph.

    Args:
        G (networkx.Graph): The graph to analyze.
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")


def brute_force_maxcut(G, W):
    """
    Perform brute-force search to find the maximum cut in a graph and plot the final graph.

    Args:
        G (networkx.Graph): The input graph.
        W (numpy.ndarray): The weight matrix representing edge weights.

    Returns:
        tuple: A tuple containing the best solution (list of 0s and 1s) and its corresponding cost.
    """
    num_nodes = G.number_of_nodes()
    best_cost_brute = 0
    best_solution_brute = None

    # Generate all possible binary configurations
    binary_configs = list(itertools.product([0, 1], repeat=num_nodes))

    # Iterate over all configurations
    for x in binary_configs:
        cost = 0
        # Calculate the cost for the current configuration
        for i in range(num_nodes):
            for j in range(num_nodes):
                cost += W[i, j] * x[i] * (1 - x[j])

        # Update best cost and solution if a better solution is found
        if best_cost_brute < cost:
            best_cost_brute = cost
            best_solution_brute = x

    # Color nodes based on the best solution
    colors = ["r" if best_solution_brute[i] == 0 else "c" for i in range(num_nodes)]

    # Plot the graph with colored nodes
    pos = nx.spring_layout(G)
    draw_graph(G, colors, pos)

    return best_solution_brute, best_cost_brute


def max_cut_to_qubo_solution(cut_solution):
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

    def linear_to_sqaure(self) -> None:
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

def cut_to_sol(cut) -> np.array:
    """
    Convert graph cut to QUBO solution, following:
        sol[i-1] = cut[0] XNOR cut[i]
    """
    assert len(cut.shape) == 1
    sol = np.zeros(cut.shape[0] - 1)
    for i in range(1, cut.shape[0]):
        sol[i-1] = (cut[0] == cut[i])
    return sol