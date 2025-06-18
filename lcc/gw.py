import cvxpy as cvx
import networkx as nx
import numpy as np

# https://github.com/rvg77/max-cut/blob/master/src/models/maxcut.py


def cut_cost(x, L):
    return 0.25 * x @ L @ x


def GoemansWilliamson(G: nx.Graph, num_trials: int = 30):
    n = G.number_of_nodes()
    L = nx.laplacian_matrix(G, nodelist=sorted(G.nodes))

    # SDP solution
    X = cvx.Variable((n, n), PSD=True)
    obj = 0.25 * cvx.trace(L.toarray() @ X)
    constr = [cvx.diag(X) == 1]
    problem = cvx.Problem(cvx.Maximize(obj), constraints=constr)
    problem.solve(solver=cvx.SCS)

    # GW algorithm
    u, s, _ = np.linalg.svd(X.value)
    U = u * np.sqrt(s)

    sol_list = []
    for i in range(num_trials):
        r = np.random.randn(n)
        r = r / np.linalg.norm(r)
        cut = np.sign(r @ U.T)
        cost = cut_cost(cut, L)

        cut = (np.array(cut) + 1) / 2

        bitstr = "".join(list(map(str, map(int, cut))))
        sol_list.append([bitstr, float(cost)])

    sol_list.sort(key=lambda x: -x[1])

    return sol_list
