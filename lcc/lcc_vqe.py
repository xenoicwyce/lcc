import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import (
    BackendSamplerV2 as BackendSampler,
    BackendEstimatorV2 as BackendEstimator,
    SamplerResult,
)
from qiskit.quantum_info import SparsePauliOp, Pauli

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import GurobiOptimizer
from qiskit_algorithms.optimizers import COBYLA, OptimizerResult
from qiskit_aer import AerSimulator

TWO_PI = 2 * np.pi


class LCCVQE:
    """
    LCC for VQE ansatz with circular entanglement.
    * Only works for one-local (Z) or two-local (ZZ) operations.
    * Currently only consider the TwoLocal ansatz with RY rotation, CZ entanglement, and reps=1.
    """
    def __init__(
        self,
        quadratic_program: QuadraticProgram,
        sampler=None,
        estimator=None,
        optimizer=None,
    ) -> None:
        self.qp = quadratic_program
        converter = QuadraticProgramToQubo()
        qubo = converter.convert(quadratic_program)
        self.qubo = qubo

        hamiltonian, offset = qubo.to_ising()
        self.hamiltonian: SparsePauliOp = hamiltonian
        self.offset: float = offset

        self.num_qubits = hamiltonian.num_qubits

        backend_sv = AerSimulator(method='statevector')
        backend_mps = AerSimulator(method='matrix_product_state')
        self.sampler = BackendSampler(backend=backend_mps) if sampler is None else sampler
        self.estimator = BackendEstimator(backend=backend_sv) if estimator is None else estimator
        self.optimizer = COBYLA() if optimizer is None else optimizer

        self.optimal_params = None
    
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
    
    @staticmethod
    def full_twolocal_ansatz(num_qubits: int) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        theta = ParameterVector('θ', 2 * num_qubits)
        
        for k in range(num_qubits):
            qc.ry(theta[k], k)

        for k in range(num_qubits):
            qc.cz(k, (k + 1) % num_qubits)

        for k in range(num_qubits):
            qc.ry(theta[num_qubits + k], k)

        return qc
    
    def generate_random_params(self, scale=TWO_PI):
        return np.random.rand(2 * self.num_qubits) * scale
    
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
    
    def compute_energy(self, params: list[float] | np.ndarray) -> float:
        """
        Computes the sum of expectation of the ZZ terms.
        """
        pubs = self.generate_pubs(params)
        results = self.estimator.run(pubs).result()
        evs = [result.data.evs for result in results]
        return sum(evs)
    
    def sample_most_likely(self, params: list[float] | np.ndarray) -> list[int]:
        """
        Run the full circuit with sampler to get the solution.
        """
        ansatz = self.full_twolocal_ansatz(self.num_qubits)        
        result = self.sampler.run([(ansatz, params)]).result()[0]
        counts = result.data.meas.get_counts()

        highest_count = max(counts.values())
        for bit_string, count in counts.items():
            if count == highest_count:
                return list(map(int, bit_string))

    def solve(self, initial_point: list[float] | np.ndarray = None) -> OptimizerResult:
        """
        Calls the Scipy minimize function and returns the OptimizerResult object.
        """
        if initial_point is None:
            initial_point = np.random.rand(2 * self.num_qubits) * 2 * np.pi
        else:
            assert np.asarray(initial_point).shape[0] == (2 * self.num_qubits), 'Parameter length does not match.'
        
        def obj_func(params):
            return self.compute_energy(params)

        result = self.optimizer.minimize(obj_func, initial_point)
        self.optimal_params = result.x

        return result

    def solve_gurobi(self) -> float:
        result = GurobiOptimizer().solve(self.qp)
        return result.fval
