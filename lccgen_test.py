import networkx as nx
import numpy as np

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp, Pauli

from qiskit_algorithms.optimizers import OptimizerResult, L_BFGS_B, Optimizer

from qiskit_aer.backends.aerbackend import AerBackend
from qiskit_aer import StatevectorSimulator

from qiskit_optimization.applications import Maxcut
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

from lcc.lcc_functions import make_lcc_circuit

from qiskit.quantum_info import Statevector

from lcc.lcc_vqe_general import LCCVQEgeneral

if __name__ == "__main__":
    n = 6
    G = nx.random_regular_graph(3, n, seed=0)
    max_cut = Maxcut(G)
    qp = max_cut.to_quadratic_program()

    rotation_blocks = ["ry"]
    entanglement_blocks = ["cz"]
    entanglement = "circular"
    reps = 2
    qc = TwoLocal(n, rotation_blocks, entanglement_blocks, entanglement, reps).decompose()

    simulator = StatevectorSimulator()
    optimizer = L_BFGS_B()
    lcc_vqe_gen = LCCVQEgeneral(qp, qc, simulator=simulator, optimizer=optimizer)

    points = (2 * np.random.random(len(qc.parameters)) - 1) * np.pi

    result = lcc_vqe_gen.solve()
    print(result)
    expectation = -(result.fun + lcc_vqe_gen.offset)
    print(expectation)

    state = Statevector(qc.assign_parameters(result.x, inplace=False))
    energy_full = state.expectation_value(lcc_vqe_gen.hamiltonian)
    expectation_value_full = -(energy_full + lcc_vqe_gen.offset)
    print(expectation_value_full)
