import numpy as np
import torch
import gurobipy as gp
from gurobipy import GRB

from src.problems.vertex_cover import VertexCoverProblem


# -----------------------------------------------------------
#  RECONSTRUCT A, b FROM THE PYG GRAPH (same as before)
# -----------------------------------------------------------
def reconstruct_A_b_from_data(data):
    n_vars = data.n_vars.item()
    n_constraints = data.n_constraints.item()
    A = torch.zeros(n_constraints, n_vars, dtype=torch.float32)
    b = data.constr_rhs.clone().float()

    edge_index = data.edge_index
    edge_attr = data.edge_attr.view(-1)

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    coeffs = edge_attr.tolist()

    # Only pick constraint -> variable edges
    for u, v, c in zip(src, dst, coeffs):
        if u >= n_vars and v < n_vars:  # constraint -> var
            row_idx = u - n_vars
            col_idx = v
            A[row_idx, col_idx] = c

    return A, b

def build_true_scenario_milp(instance, scenario_idx):
    """
    Build the exact MILP for ONE scenario in the SAME canonical form
    used by the graph encoding:

        Variables:
            z = [x_0..x_{N-1}, y_0..y_{N-1}, t]

        Constraints in A z <= b form:
          - Cover (for each edge (u,v)):
                -x_u - y_u - x_v - y_v <= -1
          - Compatibility (for each j):
                x_j + y_j <= 1
          - Budget:
                sum_j sce[i,j] * y_j - t <= 0

    Return:
        A_true: np.ndarray [n_constraints, n_vars]
        b_true: np.ndarray [n_constraints]
    """
    A_adj = instance.data["adjacency"]
    det = instance.data["det"]
    sce = instance.data["sce"]

    # Normalize shapes
    if A_adj.ndim == 3:
        A_adj = A_adj[0]
    if det.ndim > 1:
        det = det[0]
    if sce.ndim > 2:
        sce = sce[0]

    N = A_adj.shape[0]
    sce_i = sce[scenario_idx]

    # ---------- Build Gurobi model ----------
    m = gp.Model()
    m.Params.OutputFlag = 0

    # Variable ordering must match graph encoding:
    #   [x_0..x_{N-1}, y_0..y_{N-1}, t]
    x = m.addVars(N, vtype=GRB.BINARY, name="x")
    y = m.addVars(N, vtype=GRB.BINARY, name="y")
    t = m.addVar(lb=0.0, name="t")

    # --- Cover constraints in <= form ---
    #   x_u + y_u + x_v + y_v >= 1
    #   <=> -x_u - y_u - x_v - y_v <= -1
    cover_constrs = []
    for u in range(N):
        for v in range(u + 1, N):
            if A_adj[u, v] != 0:
                expr = -x[u] - y[u] - x[v] - y[v]
                c = m.addConstr(expr <= -1.0, name=f"cover_{u}_{v}")
                cover_constrs.append(c)

    # --- Compatibility constraints (already <= form) ---
    #   x_j + y_j <= 1
    compat_constrs = []
    for j in range(N):
        expr = x[j] + y[j]
        c = m.addConstr(expr <= 1.0, name=f"compat_{j}")
        compat_constrs.append(c)

    # --- Budget constraint in <= form ---
    #   sum_j sce_i[j] * y_j <= t
    #   <=> sum_j sce_i[j] * y_j - t <= 0
    budget_expr = gp.quicksum(float(sce_i[j]) * y[j] for j in range(N)) - t
    budget_constr = m.addConstr(budget_expr <= 0.0, name="budget")

    # Dummy objective (not used in A,b comparison)
    m.setObjective(gp.quicksum(det[j] * x[j] for j in range(N)) + t, GRB.MINIMIZE)

    m.update()

    # ---------- Extract A_true and b_true ----------
    # Order of constraints:
    #  [all cover_constrs, all compat_constrs, budget_constr]
    all_constrs = cover_constrs + compat_constrs + [budget_constr]

    # Gurobi's A is full constraint matrix for ALL constraints in model
    A_mat = m.getA().toarray().astype(float)  # [num_rows, num_vars]
    b_all = np.array([c.RHS for c in m.getConstrs()], dtype=float)

    # We want rows corresponding to our constraints in the order we added them.
    # getConstrs() returns constraints in the order added, so indices line up.
    # Therefore, we can just slice the first len(all_constrs) rows.
    num_rows_needed = len(all_constrs)
    A_true = A_mat[:num_rows_needed, :]
    b_true = b_all[:num_rows_needed]

    # Sanity: number of variables should be 2N + 1 (x, y, t)
    assert A_true.shape[1] == 2 * N + 1, (
        f"Expected 2N+1 vars, got {A_true.shape[1]} columns"
    )

    return A_true, b_true


# -----------------------------------------------------------
#  FULL SANITY CHECK WITH ASSERTS
# -----------------------------------------------------------
def test_vc_graph_rows_match_gurobi_milp(n_nodes=5, n_scenarios=3, p_edge=5.0):
    problem = VertexCoverProblem(
        n_nodes=n_nodes,
        n_scenarios=n_scenarios,
        p_edge=p_edge,
    )

    instance = problem.sample()
    data_list = problem.to_pyg_data(instance)

    # Pick scenario 0
    scenario_idx = 0
    data_graph = data_list[scenario_idx]

    # A_graph, b_graph: from GNN bipartite MILP encoding
    A_graph, b_graph = reconstruct_A_b_from_data(data_graph)
    A_graph = A_graph.numpy()
    b_graph = b_graph.numpy()

    # A_true, b_true: from actual MILP using Gurobi
    A_true, b_true = build_true_scenario_milp(instance, scenario_idx)

    # Rows in the true MILP: cover + compat + budget.
    # The encoded graph matches exactly those rows in SAME ORDER.

    # ---- ASSERT 1: Shape consistency ----
    assert A_graph.shape == A_true.shape, f"Shape mismatch: graph {A_graph.shape}, true {A_true.shape}"
    assert b_graph.shape == b_true.shape, f"RHS shape mismatch: {b_graph.shape} vs {b_true.shape}"

    # ---- ASSERT 2: Numerical equality ----
    # Allow small tolerance since Gurobi may print -0 or tiny eps.
    if not np.allclose(A_graph, A_true, atol=1e-6):
        diff = A_graph - A_true
        raise AssertionError(f"A mismatch! Max diff = {np.max(np.abs(diff))}")

    if not np.allclose(b_graph, b_true, atol=1e-6):
        diff = b_graph - b_true
        raise AssertionError(f"b mismatch! Max diff = {np.max(np.abs(diff))}")

    print("SUCCESS: Graph MILP encoding EXACTLY matches Gurobi MILP rows.")
    print(f"A shape: {A_graph.shape}, b shape: {b_graph.shape}")


if __name__ == "__main__":
    test_vc_graph_rows_match_gurobi_milp()
