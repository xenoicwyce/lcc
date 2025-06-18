import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit import Qubit
from qiskit.dagcircuit import DAGCircuit

from qiskit.converters import circuit_to_dagdependency, dag_to_circuit, circuit_to_dag

from qiskit.transpiler.passes import RemoveBarriers
from qiskit import transpile

from qiskit_aer import StatevectorSimulator

import rustworkx as rx


def make_lcc_graph(graph: rx.PyDiGraph, start_nodes: list) -> rx.PyDiGraph:

    # 辺の向きを逆にして、measure nodeからBFSを行うことで、LCCのグラフを作る。
    graph.reverse()

    new_graph = rx.PyDiGraph()  # 結果のグラフ
    queue = [(node, None) for node in start_nodes]  # (現在のノード, 親ノード)
    node_map = {}  # {元のノードID: 新グラフのノードID}
    added_edges = set()  # 追加済みエッジ (from, to) の集合

    # 初期ノードを追加
    for node in start_nodes:
        node_map[node] = new_graph.add_node(node)

    while queue:
        node, parent = queue.pop(0)

        # まだ追加されていないノードを追加
        if node not in node_map:
            node_map[node] = new_graph.add_node(node)

        # 親が存在する場合、エッジを追加（重複を避ける）
        # Note: 初期ノードには親は存在しない
        if parent is not None:
            edge = (node_map[parent], node_map[node])
            if edge not in added_edges:
                new_graph.add_edge(*edge, None)
                added_edges.add(edge)

        # 隣接ノードをキューに追加
        for neighbor in graph.neighbors(node):
            queue.append((neighbor, node))

    # 出来上がったグラフは辺の向きが逆になっているので、戻す。
    new_graph.reverse()
    return new_graph


def count_gates(qc: QuantumCircuit) -> dict:
    gate_count = {qubit: 0 for qubit in qc.qubits}
    for gate in qc.data:
        for qubit in gate.qubits:
            gate_count[qubit] += 1
    return gate_count


def removed_idle_wires(qc: QuantumCircuit) -> list:
    gate_count = count_gates(qc)
    removed_wires = []
    for qubit, count in gate_count.items():
        if count == 0:
            removed_wires.append(qubit._index)

    return removed_wires


def make_lcc_circuit(qc: QuantumCircuit, measure: bool = True) -> tuple[QuantumCircuit, str]:

    qc = RemoveBarriers()(qc)
    dagqc = circuit_to_dagdependency(qc)

    dag_graph = dagqc._multi_graph

    measurement_qubits = []
    measure_node_list = []
    for node in dag_graph.nodes():
        if node.op.name == "measure":
            measure_node_list.append(node)
            measurement_qubits.append(node.qargs[0]._index)

    if len(measurement_qubits) == 0:
        raise ValueError("input qc must have measurement operation.")

    # measurement gatesからBFSをすることでLCC回路のDAGのグラフを構築する。
    # measurement gatesに対応するnodesを見つける。
    node_index_map = {dag_graph[index]: index for index in dag_graph.node_indexes()}
    start_nodes = [node_index_map[i] for i in measure_node_list]

    # make lcc graph
    lcc_graph = make_lcc_graph(dag_graph, start_nodes)
    lcc_nodes = sorted(list(lcc_graph.nodes()))

    # make LCC DAG circuit
    # https://github.com/Qiskit/qiskit/blob/stable/1.4/qiskit/converters/dagdependency_to_dag.py#L18-L54
    lccdagcircuit = DAGCircuit()
    # lccdagcircuit.name = dagqc.name
    # lccdagcircuit.metadata = dagqc.metadata
    lccdagcircuit.add_qubits(dagqc.qubits)
    lccdagcircuit.add_clbits(dagqc.clbits)

    for register in dagqc.qregs.values():
        lccdagcircuit.add_qreg(register)

    for register in dagqc.cregs.values():
        lccdagcircuit.add_creg(register)

    for node in lcc_nodes:
        inst = dag_graph[node].op.copy()
        lccdagcircuit.apply_operation_back(inst, dag_graph[node].qargs, dag_graph[node].cargs)

    lcc_qc = dag_to_circuit(lccdagcircuit)

    # modify measurement gates and clbits to reduce clbits.

    lcc_qc.remove_final_measurements(inplace=True)
    if measure:
        lcc_qc.add_bits(ClassicalRegister(len(measurement_qubits)))  # 上書き
        lcc_qc.measure(measurement_qubits, [i for i in range(len(measurement_qubits))])

    # remove idle qubits
    lcc_qc, idle_wires = remove_idle_qubits(lcc_qc)

    new_measureemnt_position = update_measurement_positions(qc.num_qubits, measurement_qubits, idle_wires)
    lcc_obs_tmp = ["I"] * lcc_qc.num_qubits
    for idx in new_measureemnt_position:
        lcc_obs_tmp[idx] = "Z"

    lcc_obs_str = "".join(lcc_obs_tmp)[::-1]

    return lcc_qc, lcc_obs_str


def remove_idle_qubits(qc: QuantumCircuit) -> tuple[QuantumCircuit, list]:

    dag = circuit_to_dag(qc)

    idle_wires = list(dag.idle_wires())

    idle_qubit_wires = []
    idle_clbit_wires = []

    for wire in idle_wires:
        if isinstance(wire, Qubit):
            idle_qubit_wires.append(wire)
        else:
            idle_clbit_wires.append(wire)

    dag.remove_qubits(*idle_qubit_wires)
    dag.remove_clbits(*idle_clbit_wires)

    return dag_to_circuit(dag), [qubit._index for qubit in idle_qubit_wires]


def update_measurement_positions(num_qubits: int, measure_qubits: list, deleted_qubits: list) -> list:
    # 新しい量子ビットリストを作成
    remaining_qubits = [i for i in range(num_qubits) if i not in deleted_qubits]

    # 新しい測定ゲートの位置を計算
    new_measure_qubits = []
    for mq in measure_qubits:
        # 削除された量子ビットのインデックスを調整
        adjusted_position = remaining_qubits.index(mq)
        new_measure_qubits.append(adjusted_position)

    return new_measure_qubits


if __name__ == "__main__":

    num_qubits = 10

    qreg = QuantumRegister(num_qubits)
    creg = ClassicalRegister(num_qubits)
    qc = QuantumCircuit(qreg, creg)

    theta1_1 = ParameterVector("theta1_1", num_qubits)
    theta1_2 = ParameterVector("theta1_2", num_qubits)

    for i in range(num_qubits):
        qc.ry(theta1_1[i], i)

    for i in range(num_qubits):
        qc.cz(i, (i+1) % num_qubits)

    for i in range(num_qubits):
        qc.ry(theta1_2[i], i)

    measure_qubits = [1, 9]
    measure_clbits = [1, 9]
    qc.measure(measure_qubits, measure_clbits)

    print(qc)

    lcc_qc, lcc_obs_str = make_lcc_circuit(qc)

    print(lcc_qc)
    print(lcc_obs_str)

    params = np.random.random(lcc_qc.num_parameters)

    sim = StatevectorSimulator()

    lcc_qc.assign_parameters(params, inplace=True)
    result = sim.run(lcc_qc).result()
    print(result)

    # state = result.get_statevector().data
    # for index, data in enumerate(state):
    #     print(index, data)
