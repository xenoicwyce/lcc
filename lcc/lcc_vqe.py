import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import (
    BackendSamplerV2 as BackendSampler,
    BackendEstimatorV2 as BackendEstimator,
)
from qiskit.quantum_info import SparsePauliOp, Pauli

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import GurobiOptimizer
from qiskit_algorithms.optimizers import COBYLA, OptimizerResult
from qiskit_aer import AerSimulator

TWO_PI = 2 * np.pi


class FullVQE:
    def __init__(
        self,
        quadratic_program: QuadraticProgram,
        reps: int = 1,
        shots: int = None,
        sampler=None,
        estimator=None,
        optimizer=None,        
    ) -> None:
        self.qp = quadratic_program
        self.converter = QuadraticProgramToQubo()
        self.qubo = self.converter.convert(self.qp)

        hamiltonian, offset = self.qubo.to_ising()
        self.hamiltonian: SparsePauliOp = hamiltonian
        self.offset: float = offset

        self.num_qubits = hamiltonian.num_qubits
        self.full_ansatz = self.generate_full_ansatz(hamiltonian.num_qubits, reps)

        backend_sv = AerSimulator(method='statevector')
        backend_mps = AerSimulator(method='matrix_product_state')
        self.sampler = BackendSampler(backend=backend_mps) if sampler is None else sampler
        self.estimator = BackendEstimator(backend=backend_sv) if estimator is None else estimator
        self.optimizer = COBYLA() if optimizer is None else optimizer
        self.shots = shots

        self.optimal_params = None
        self.optimal_solution = None
    
    @staticmethod
    def generate_full_ansatz(num_qubits: int, reps: int) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        theta = ParameterVector('θ', (reps + 1) * num_qubits)
        
        for k in range(num_qubits):
            qc.ry(theta[k], k)

        for r in range(1, reps + 1):
            for k in range(num_qubits):
                qc.cz(k, (k + 1) % num_qubits)

            for k in range(num_qubits):
                qc.ry(theta[r * num_qubits + k], k)

        return qc
    
    def generate_random_params(self, scale=TWO_PI):
        return np.random.rand(self.full_ansatz.num_parameters) * scale
    
    def generate_pubs(
        self, 
        params: list[float] | np.ndarray,
    ) -> list[tuple[QuantumCircuit, SparsePauliOp, np.ndarray]]:
        return [(self.full_ansatz, self.hamiltonian, params)]
    
    def compute_energy(self, params: list[float] | np.ndarray) -> float:
        """
        Computes the sum of expectation of the ZZ terms.
        """
        pubs = self.generate_pubs(params)
        results = self.estimator.run(pubs).result()
        evs = [result.data.evs for result in results]
        return sum(evs)
    
    def _sample_optimal_circuit(self) -> dict[str, int]:
        """
        Run the full circuit with sampler to get the solution. 
        """
        if self.optimal_params is None:
            raise ValueError('Problem not yet solved. Run LCCVQE.solve() to solve the problem.')
        
        ansatz = self.full_ansatz.copy()
        ansatz.measure_all()
        result = self.sampler.run([(ansatz, self.optimal_params)], shots=self.shots).result()[0]
        return result.data.meas.get_counts()

    def _sample_most_likely(self) -> list[int]:
        counts = self._sample_optimal_circuit()
        highest_count = max(counts.values())

        for bit_string, count in counts.items():
            if count == highest_count:
                return list(map(int, bit_string[::-1])) # flip the bit-string due to qiskit ordering
            
    def get_qp_solution(self) -> list[float]:
        qubo_solution = self._sample_most_likely()
        return self.converter.interpret(qubo_solution)

    def solve(
        self, 
        initial_point: list[float] | np.ndarray = None,
        run_sampler: bool = False,
    ) -> OptimizerResult:
        """
        Calls the Scipy minimize function and returns the OptimizerResult object.
        """
        if initial_point is None:
            initial_point = self.generate_random_params()
        else:
            assert np.asarray(initial_point).shape[0] == (2 * self.num_qubits), 'Parameter length does not match.'
        
        def obj_func(params):
            return self.compute_energy(params)

        result = self.optimizer.minimize(obj_func, initial_point)
        self.optimal_params = result.x

        if run_sampler:
            self.optimal_solution = self.get_qp_solution()

        return result

    def solve_gurobi(self) -> float:
        result = GurobiOptimizer().solve(self.qp)
        return result.fval


class LCCVQE(FullVQE):
    """
    LCC for VQE ansatz with circular entanglement.
    * Only works for one-local (Z) or two-local (ZZ) operations.
    * Currently only consider the TwoLocal ansatz with RY rotation, CZ entanglement, and reps=1.
    """
    def __init__(
        self,
        quadratic_program: QuadraticProgram,
        shots: int = None,
        sampler=None,
        estimator=None,
        optimizer=None,        
    ) -> None:
        super().__init__(
            quadratic_program,
            reps=1,
            shots=shots,
            sampler=sampler,
            estimator=estimator,
            optimizer=optimizer,
        )

    @staticmethod
    def get_pauli_indices(pauli: Pauli) -> list[int]:
        return np.argwhere(pauli.z).reshape(-1).tolist()
    
    @staticmethod
    def generate_local_ansatz(num_qubits: int) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        theta = ParameterVector('θ', 2 * num_qubits)

        # First layer RY
        for k in range(num_qubits):
            qc.ry(theta[k], k)

        # Entangling CNOT
        for k in range(num_qubits - 1):
            qc.cz(k, k + 1)

        # Second layer RY
        for k in range(num_qubits):
            qc.ry(theta[num_qubits + k], k)

        return qc

    def _distance(self, i, j):
        return min(abs(i - j), abs(i + self.num_qubits - j), abs(j - i + self.num_qubits))

    def generate_pubs(
        self, 
        params: list[float] | np.ndarray,
    ) -> list[tuple[QuantumCircuit, SparsePauliOp, np.ndarray]]:
        pubs = []

        for observable in self.hamiltonian:
            pauli_indices = self.get_pauli_indices(observable.paulis[0])
            if len(pauli_indices) == 1:
                # one-local, 3-qubit
                qc = self.generate_local_ansatz(3)
                local_ob = SparsePauliOp('IZI', observable.coeffs[0])
                i = pauli_indices[0]
                first_layer = np.array([(i - 1), i, (i + 1)]) % self.num_qubits

            elif len(pauli_indices) == 2:
                # two-local
                i, j = pauli_indices
                if self._distance(i, j) == 1:
                    # 4-qubit
                    qc = self.generate_local_ansatz(4)
                    local_ob = SparsePauliOp('IZZI', observable.coeffs[0])
                    first_layer = np.array([i - 1, i, j, j + 1]) % self.num_qubits

                elif self._distance(i, j) == 2:
                    # 5-qubit
                    qc = self.generate_local_ansatz(5)
                    local_ob = SparsePauliOp('IZIZI', observable.coeffs[0])

                    if i + 2 == j:
                        first_layer = np.array([i - 1, i, i + 1, j, j + 1]) % self.num_qubits
                    else:
                        first_layer = np.array([j - 1, j, i - 1, i, i + 1]) % self.num_qubits

                elif self._distance(i, j) > 2:
                    # 6-qubit
                    qc = self.generate_local_ansatz(6)
                    local_ob = SparsePauliOp('IZIIZI', observable.coeffs[0])
                    first_layer = np.array([i - 1, i, i + 1, j - 1, j, j + 1]) % self.num_qubits
            else:
                raise ValueError(f'Pauli indices must be of 1 or 2 length. Got {len(pauli_indices)} instead.')
            
            second_layer = first_layer + self.num_qubits
            param_indices = np.hstack([first_layer, second_layer]).tolist()

            # construct pub
            params = np.asarray(params)
            pubs.append((qc, local_ob, params[param_indices]))
            
        return pubs