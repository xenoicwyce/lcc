
import numpy as np

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp, Pauli

from qiskit_algorithms.optimizers import OptimizerResult, Optimizer

from qiskit_aer.backends.aerbackend import AerBackend
from qiskit_aer import StatevectorSimulator

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

from .lcc_functions import make_lcc_circuit


class LCCVQEgeneral:

    def __init__(self, quadratic_program: QuadraticProgram, qc: QuantumCircuit,  shots: int = None, simulator=None, optimizer=None) -> None:
        self.qp = quadratic_program
        self.qc = qc

        self.num_qubits = qc.num_qubits

        converter = QuadraticProgramToQubo()
        qubo = converter.convert(self.qp)

        hamiltonian, offset = qubo.to_ising()
        self.hamiltonian: SparsePauliOp = hamiltonian.simplify()
        self.offset: float = offset

        self.shots = shots
        self.simulator: AerBackend = simulator
        self.optimizer: Optimizer = optimizer

        self.lcc_pubs = self.create_lcc_pubs()

        self.params_indices_dict = {key: value for value, key in enumerate(qc.parameters)}

        self.optimal_params = None

    @staticmethod
    def get_pauli_indices(pauli: Pauli) -> list[int]:
        return np.argwhere(pauli.z).reshape(-1).tolist()

    def create_lcc_pubs(self) -> list[tuple[QuantumCircuit, float]]:

        lcc_pubs = []
        for obs in self.hamiltonian:
            qc_tmp = self.qc.copy()
            indices = self.get_pauli_indices(obs.paulis[0])

            qc_tmp.add_register(ClassicalRegister(len(indices)))
            qc_tmp.measure(indices, [i for i in range(len(indices))])

            if isinstance(self.simulator, StatevectorSimulator):
                qc_tmp, lcc_obs_str = make_lcc_circuit(qc_tmp, measure=False)

            else:
                ValueError(f"{self.simulator} is not yet supported.")

            local_sp = SparsePauliOp(lcc_obs_str, obs.coeffs[0])
            local_hamiltonian_diag = np.diag(local_sp.to_matrix())

            lcc_pubs.append((qc_tmp, local_hamiltonian_diag))

        return lcc_pubs

    def compute_energy(self, points: list[float] | np.ndarray) -> float:

        total_energy = 0.0

        for lcc_qc, local_hamiltonian_diag in self.lcc_pubs:

            lcc_points_dict = {}
            for param in lcc_qc.parameters:
                index = self.params_indices_dict[param]
                lcc_points_dict[param] = points[index]

            lcc_qc.assign_parameters(lcc_points_dict, inplace=True)

            if isinstance(self.simulator, StatevectorSimulator):
                job = self.simulator.run(lcc_qc)
                result = job.result()
                state = result.get_statevector().data

                if self.shots is None:
                    quasi_dist = np.abs(state)**2
                else:
                    quasi_dist = np.random.multinomial(self.shots, np.abs(state)**2)

            else:
                raise ValueError(f"{self.simulator} is not yet supported.")

            local_energy = np.inner(quasi_dist, local_hamiltonian_diag)
            total_energy += local_energy

        # energy must be a real number.
        total_energy = total_energy.real
        return total_energy

    def solve(self, initial_point: list[float] | np.ndarray = None) -> OptimizerResult:

        if initial_point is None:
            # [-pi, pi]
            initial_point = (2 * np.random.random(self.qc.num_parameters) - 1) * np.pi
        else:
            assert np.asarray(initial_point).shape[0] == self.qc.num_parameters, "Parameter length does not match."

        def obj_func(points):
            return self.compute_energy(points)

        result = self.optimizer.minimize(obj_func, initial_point)
        self.optimal_params = result.x

        return result
