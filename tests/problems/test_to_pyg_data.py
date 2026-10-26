import numpy as np
import torch
from src.core.types import Instance
from src.problems.vertex_cover import VertexCoverProblem


# --- Helpers to create debug instances ---
def make_vc_debug_instance():
    N, S = 3, 2
    # Simple line graph: 0-1-2
    adjacency = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=int)

    # Det Costs (for x)
    det = np.array([10, 20, 30], dtype=int)

    # Scenario Costs (for y)
    sce = np.array([
        [1, 2, 3],  # S0
        [3, 2, 1],  # S1
    ], dtype=int)

    return Instance({
        "n_nodes": np.array([N]),
        "n_scenarios": np.array([S]),
        "n_min_cost": np.array([1]),
        "n_max_cost": np.array([30]),
        "p_edge": np.array([0.0]),
        "det": det,
        "sce": sce,
        "adjacency": adjacency,
    })


def _assert_bipartite_graph(data) -> None:
    n_vars = int(data.n_vars.item())
    src, dst = data.edge_index
    is_var = src < n_vars
    is_con = dst >= n_vars
    is_var_rev = dst < n_vars
    is_con_rev = src >= n_vars
    assert torch.all((is_var & is_con) | (is_var_rev & is_con_rev))


def _edges_to_constraint(edge_index: torch.Tensor, edge_attr: torch.Tensor, dst: int) -> dict:
    if edge_index.numel() == 0:
        return {}
    mask = edge_index[1] == int(dst)
    srcs = edge_index[0][mask].tolist()
    coeffs = edge_attr[mask, 0].tolist()
    return dict(zip(srcs, coeffs))


def test_feature_shapes_and_types():
    """Verify x has 9 dims and edge_attr has 1 dim."""
    problem = VertexCoverProblem(n_nodes=3, n_scenarios=2)
    inst = make_vc_debug_instance()

    # Test Scenario Graph
    data_list = problem._build_pyg_data(inst)
    d0 = data_list[0]

    assert d0.x.shape[1] == 9, f"Expected 9 node features, got {d0.x.shape[1]}"
    assert d0.edge_attr.shape[1] == 1, f"Expected 1 edge feature, got {d0.edge_attr.shape[1]}"

    # Test Deterministic Graph
    d_det = problem.to_pyg_data_det(inst)
    assert d_det.x.shape[1] == 9
    assert d_det.edge_attr.shape[1] == 1


def test_vertex_cover_graphs_are_bipartite():
    """
    Motivation:
    - Ensures vertex-cover MILP graphs only connect variables to constraints.
    - Guards against index mixups in the graph encoding.
    """
    problem = VertexCoverProblem(n_nodes=3, n_scenarios=2)
    inst = make_vc_debug_instance()

    data_s0 = problem._build_pyg_data(inst)[0]
    data_det = problem.to_pyg_data_det(inst)

    _assert_bipartite_graph(data_s0)
    _assert_bipartite_graph(data_det)


def test_vertex_cover_graph_coefficients_and_rhs():
    """
    Motivation:
    - Verifies cover/compat/budget constraint coefficients for a tiny instance.
    - Catches index/sign errors in the MILP encoding.
    """
    N = 2
    S = 1
    problem = VertexCoverProblem(n_nodes=N, n_scenarios=S)

    adjacency = np.array([[0, 1], [1, 0]], dtype=np.int64)
    det = np.array([1.0, 2.0], dtype=np.float32)
    sce = np.array([[3.0, 4.0]], dtype=np.float32)

    inst = Instance({
        "n_nodes": np.array([N]),
        "n_scenarios": np.array([S]),
        "det": det,
        "sce": sce,
        "adjacency": adjacency,
    })

    data = problem._build_pyg_data(inst)[0]

    y_start = N
    t_idx = 2 * N
    n_vars = 2 * N + 1

    constr_start = n_vars
    idx_cover = constr_start
    idx_compat0 = constr_start + 1
    idx_compat1 = constr_start + 2
    idx_budget = constr_start + 3

    cover_edges = _edges_to_constraint(data.edge_index, data.edge_attr, idx_cover)
    assert cover_edges[0] == -1.0
    assert cover_edges[y_start + 0] == -1.0
    assert cover_edges[1] == -1.0
    assert cover_edges[y_start + 1] == -1.0
    assert data.x[idx_cover, 6].item() == -1.0

    compat0_edges = _edges_to_constraint(data.edge_index, data.edge_attr, idx_compat0)
    assert compat0_edges[0] == 1.0
    assert compat0_edges[y_start + 0] == 1.0
    assert data.x[idx_compat0, 6].item() == 1.0

    compat1_edges = _edges_to_constraint(data.edge_index, data.edge_attr, idx_compat1)
    assert compat1_edges[1] == 1.0
    assert compat1_edges[y_start + 1] == 1.0
    assert data.x[idx_compat1, 6].item() == 1.0

    budget_edges = _edges_to_constraint(data.edge_index, data.edge_attr, idx_budget)
    assert budget_edges[t_idx] == -1.0
    assert budget_edges[y_start + 0] == 3.0
    assert budget_edges[y_start + 1] == 4.0
    assert data.x[idx_budget, 6].item() == 0.0


def test_edge_feature_values():
    """Verify edge features contain signed coeffs."""
    problem = VertexCoverProblem(n_nodes=3, n_scenarios=2)
    inst = make_vc_debug_instance()
    d0 = problem._build_pyg_data(inst)[0]

    # Extract edges for Cover Constraint 0 (Edge 0-1)
    # x0 + y0 + x1 + y1 >= 1  ->  -1 coeffs
    # We look for an edge with coeff -1

    found_neg = False
    found_pos = False

    for i in range(d0.edge_attr.shape[0]):
        val = d0.edge_attr[i, 0].item()
        if val == -1.0: found_neg = True
        if val == 1.0: found_pos = True

    assert found_neg, "Should have -1 edges (Cover)"
    assert found_pos, "Should have +1 edges (Compat)"


def test_node_feature_semantics_scenario():
    """Check specific 2RO node features (Hint, Obj, Roles)."""
    problem = VertexCoverProblem(n_nodes=3, n_scenarios=2)
    inst = make_vc_debug_instance()
    d0 = problem._build_pyg_data(inst)[0]  # Scenario 0: y costs [1, 2, 3]
    d1 = problem._build_pyg_data(inst)[1]  # Scenario 1: y costs [3, 2, 1]

    N = 3
    y_start = N

    # 1. Check Cost Hints (Index 5)
    # For S0, y nodes should have [1, 2, 3]
    y_hints_s0 = d0.x[y_start:y_start + N, 5].tolist()
    assert y_hints_s0 == [1.0, 2.0, 3.0]

    # For S1, y nodes should have [3, 2, 1]
    y_hints_s1 = d1.x[y_start:y_start + N, 5].tolist()
    assert y_hints_s1 == [3.0, 2.0, 1.0]

    # x nodes should have 0 hint
    assert d0.x[0:N, 5].sum() == 0.0

    # 2. Check Obj Coeffs (Index 4)
    # x nodes should have det costs [10, 20, 30]
    assert d0.x[0:N, 4].tolist() == [10.0, 20.0, 30.0]

    # 3. Check Roles (One-Hot)
    # x0: [0=0(var), 1=1(x), 2=0(y), 3=0(t)]
    assert d0.x[0, 0] == 0.0
    assert d0.x[0, 1] == 1.0

    # y0: [0=0(var), 1=0(x), 2=1(y), 3=0(t)]
    assert d0.x[y_start, 2] == 1.0


def test_cosine_similarity():
    """Verify Index 7 (Cos Sim) is calculated and non-zero."""
    problem = VertexCoverProblem(n_nodes=3, n_scenarios=2)
    inst = make_vc_debug_instance()
    d0 = problem._build_pyg_data(inst)[0]

    # Constraints start after vars (2N+1 = 7)
    constr_start = 7

    # Check that SOME constraints have non-zero cosine sim
    # (Since det costs are > 0, compat constraints x+y<=1 must have alignment)
    cos_sims = d0.x[constr_start:, 7]
    assert (cos_sims != 0).any(), "Cosine similarity should not be all zero"
    assert (cos_sims <= 1.001).all() and (cos_sims >= -1.001).all()


def test_det_graph_has_no_hints():
    """Verify Deterministic graph has 0 for cost hints."""
    problem = VertexCoverProblem(n_nodes=3, n_scenarios=2)
    inst = make_vc_debug_instance()
    d_det = problem.to_pyg_data_det(inst)

    N = 3
    y_start = N

    # y nodes in Det graph must have 0 hint (we don't know scenario yet)
    y_hints = d_det.x[y_start:y_start + N, 5]
    assert y_hints.sum() == 0.0

    # Det graph should NOT have budget constraint
    # Scen graph has M+N+1 constraints
    # Det graph has M+N constraints
    M = 2  # edges
    expected_cons = M + N
    assert d_det.n_constraints.item() == expected_cons


import numpy as np
import torch
from src.problems.vertex_cover import VertexCoverProblem
from src.core.types import Instance


def test_to_pyg_data_det_matches_shared_part():
    # -----------------------------
    # 1. Setup Problem
    # -----------------------------
    N = 3
    S = 2
    problem = VertexCoverProblem(n_nodes=N, n_scenarios=S)

    # -----------------------------
    # 2. Instance Data
    # -----------------------------
    # Adjacency: 0-1 (One edge)
    A = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ], dtype=np.float32)

    det = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    sce = np.array([
        [10.0, 20.0, 30.0],
        [11.0, 21.0, 31.0],
    ], dtype=np.float32)

    instance = Instance(data={
        "adjacency": torch.tensor(A),
        "det": torch.tensor(det),
        "sce": torch.tensor(sce),
        "n_nodes": torch.tensor([N]),
        "n_scenarios": torch.tensor([S]),
    })

    # -----------------------------
    # 3. Build Graphs
    # -----------------------------
    data_s0 = problem._build_pyg_data(instance)[0]
    data_det = problem.to_pyg_data_det(instance)

    # -----------------------------
    # 4. Dimension Sanity Checks
    # -----------------------------
    # Det Vars: x(N) + y(N) + t(1) = 2N + 1.
    # (Implementation keeps t to match Scenario graph structure)
    assert int(data_det.n_vars.item()) == 2 * N + 1

    # Det Constraints: M(1 edge) + N(compat) = 4. (No Budget constraint)
    expected_det_cons = 1 + N
    assert int(data_det.n_constraints.item()) == expected_det_cons

    # Scenario graph has 1 extra constraint (Budget)
    assert int(data_s0.n_constraints.item()) == expected_det_cons + 1

    # -----------------------------
    # 5. Extract Shared Subgraph from Scenario Graph
    # -----------------------------
    # We filter edges from data_s0 that are connected to the Budget Constraint.
    # Since data_det keeps 't' (as an isolated node), the indexing for variables
    # and shared constraints is IDENTICAL between Det and Scenario graphs.

    n_vars = int(data_s0.n_vars.item())
    budget_constr_idx = n_vars + expected_det_cons  # The last constraint in S0

    ei = data_s0.edge_index
    ea = data_s0.edge_attr

    filtered_pairs = []
    for e in range(ei.size(1)):
        u, v = int(ei[0, e]), int(ei[1, e])

        # Exclude only edges connected to the budget constraint
        # (This automatically drops the t-to-budget edges, leaving t isolated like in data_det)
        if u != budget_constr_idx and v != budget_constr_idx:
            filtered_pairs.append((u, v, float(ea[e, 0].item())))

    filtered_pairs.sort()

    # -----------------------------
    # 6. Compare with Deterministic Graph
    # -----------------------------
    det_pairs = _edge_multiset_1d(data_det.edge_index, data_det.edge_attr)

    assert filtered_pairs == det_pairs, \
        "Deterministic MILP graph mismatch! It must match the scenario backbone (excluding budget)."

    # -----------------------------
    # 7. RHS Sanity
    # -----------------------------
    rhs = data_det.constr_rhs.tolist()
    assert rhs[0] == -1.0  # cover
    assert all(r == 1.0 for r in rhs[1:])  # compat


def test_epigraph_variable_t_features():
    """
    Verify the unique properties of the Epigraph variable 't'.
    It sits at index 2N.
    Expected:
      - Role is_t (idx 3) = 1.0
      - Obj Coeff (idx 4) = 1.0 (It captures the second stage cost)
      - Cost Hint (idx 5) = 0.0 (It doesn't have a cost itself)
    """
    problem = VertexCoverProblem(n_nodes=3, n_scenarios=2)
    inst = make_vc_debug_instance()
    d0 = problem._build_pyg_data(inst)[0]

    N = 3
    t_idx = 2 * N

    t_feat = d0.x[t_idx]

    assert t_feat[0].item() == 0.0, "t is a variable"
    assert t_feat[1].item() == 0.0, "t is not x"
    assert t_feat[2].item() == 0.0, "t is not y"
    assert t_feat[3].item() == 1.0, "t must have is_t=1"

    assert t_feat[4].item() == 1.0, "t must have obj_coeff=1"
    assert t_feat[5].item() == 0.0, "t should not have a scenario cost hint (y has it)"


def test_constraint_node_rhs_values():
    """
    Verify RHS values (Index 6) for all constraint types.
    Structure: [M Cover] -> [N Compat] -> [1 Budget]
    Expected:
      - Cover: -1.0
      - Compat: 1.0
      - Budget: 0.0
    """
    problem = VertexCoverProblem(n_nodes=3, n_scenarios=2)
    inst = make_vc_debug_instance()
    d0 = problem._build_pyg_data(inst)[0]

    N = 3
    M = 2  # Edges 0-1, 1-2
    n_vars = 2 * N + 1

    constr_start = n_vars

    # 1. Cover Constraints (First M)
    cover_rhs = d0.x[constr_start: constr_start + M, 6]
    assert torch.all(cover_rhs == -1.0), "Cover constraints must have RHS -1"

    # 2. Compatibility Constraints (Next N)
    compat_rhs = d0.x[constr_start + M: constr_start + M + N, 6]
    assert torch.all(compat_rhs == 1.0), "Compat constraints must have RHS 1"

    # 3. Budget Constraint (Last 1)
    budget_rhs = d0.x[constr_start + M + N, 6]
    assert budget_rhs.item() == 0.0, "Budget constraint must have RHS 0"


def test_budget_constraint_topology():
    """
    Verify the edges connected to the Budget Constraint.
    It should be connected to:
      - y_j nodes with weight = c^{(i)}_j
      - t node with weight = -1.0
    """
    problem = VertexCoverProblem(n_nodes=3, n_scenarios=2)
    inst = make_vc_debug_instance()
    # Scenario 0 costs: [1, 2, 3]
    d0 = problem._build_pyg_data(inst)[0]

    N = 3
    n_vars = 2 * N + 1
    M = 2
    budget_idx = n_vars + M + N  # Last node
    t_idx = 2 * N
    y_start = N

    # Find all edges connected to budget_idx
    # Edge index is [2, E]
    mask = (d0.edge_index[1] == budget_idx)  # Incoming edges (Variable -> Constraint)
    src_nodes = d0.edge_index[0][mask]
    weights = d0.edge_attr[mask, 0]  # Signed values

    # Check connection from t
    t_mask = (src_nodes == t_idx)
    assert t_mask.sum() == 1, "t must be connected to budget constraint"
    assert weights[t_mask].item() == -1.0, "t coefficient in budget must be -1"

    # Check connection from y_0 (Cost 1 in S0)
    y0_mask = (src_nodes == y_start + 0)
    assert y0_mask.sum() == 1, "y0 must be connected to budget constraint"
    assert weights[y0_mask].item() == 1.0, "y0 coefficient matches scenario cost"

    # Check connection from y_2 (Cost 3 in S0)
    y2_mask = (src_nodes == y_start + 2)
    assert weights[y2_mask].item() == 3.0


def test_zero_cost_edge_filtering():
    """
    Optimization Check: If a scenario cost is 0, we should NOT create an edge
    between y and the budget constraint (sparsity optimization).
    """
    problem = VertexCoverProblem(n_nodes=3, n_scenarios=2)
    inst = make_vc_debug_instance()

    # Force y_1 in Scenario 0 to have cost 0
    # Current inst S0: [1, 2, 3]. Let's hack the instance data or just make a new one.
    # We'll just modify the tensor directly for the test logic if possible,
    # but cleaner to make a specific instance.

    inst.data["sce"][0, 1] = 0.0  # Set y_1 cost to 0

    d0 = problem._build_pyg_data(inst)[0]

    N = 3
    n_vars = 2 * N + 1
    M = 2
    budget_idx = n_vars + M + N
    y_start = N
    y1_idx = y_start + 1

    # Find edges connecting y1 -> Budget
    mask = (d0.edge_index[0] == y1_idx) & (d0.edge_index[1] == budget_idx)

    assert mask.sum() == 0, "Should not create edge for y_1 if cost is 0"

    # Verify y_0 (cost 1) still has edge
    y0_idx = y_start + 0
    mask_0 = (d0.edge_index[0] == y0_idx) & (d0.edge_index[1] == budget_idx)
    assert mask_0.sum() == 1


def test_structural_degree_feature():
    """
    Verify Index 8 (Degree) is populated.
    In our Make Debug Instance: 0-1-2
    Degrees: 0->1, 1->2, 2->1
    """
    problem = VertexCoverProblem(n_nodes=3, n_scenarios=2)
    inst = make_vc_debug_instance()
    d0 = problem._build_pyg_data(inst)[0]

    # Check x nodes (first N)
    degrees = d0.x[0:3, 8]
    assert degrees[0].item() == 1.0
    assert degrees[1].item() == 2.0
    assert degrees[2].item() == 1.0

    # Check y nodes (next N) - should copy the graph degree
    degrees_y = d0.x[3:6, 8]
    assert torch.equal(degrees, degrees_y)

import numpy as np
import torch

from src.core.types import Instance, Labels
from src.problems.selection import SelectionProblem


def make_sel_debug_instance(N: int = 3, S: int = 2, p: int = 2) -> Instance:
    det = np.array([10, 20, 30], dtype=np.float32)
    sce = np.array(
        [
            [1, 2, 3],  # scenario 0
            [3, 2, 1],  # scenario 1
        ],
        dtype=np.float32,
    )
    return Instance(
        {
            "n_nodes": np.array([N]),
            "n_scenarios": np.array([S]),
            "p_select": np.array([p]),
            "det": det,
            "sce": sce,
            # placeholder to satisfy shared batching utilities
            "adjacency": np.zeros((N, N), dtype=np.int32),
        }
    )


def _edge_multiset_1d(edge_index: torch.Tensor, edge_attr: torch.Tensor):
    if edge_index.numel() == 0:
        return []
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    w = edge_attr[:, 0].tolist()
    return sorted(zip(src, dst, w))


def test_selection_edge_attr_is_1d():
    problem = SelectionProblem(n_nodes=3, n_scenarios=2, p_select=2)
    inst = make_sel_debug_instance()

    d0 = problem._build_pyg_data(inst)[0]
    assert d0.edge_attr.ndim == 2
    assert d0.edge_attr.shape[1] == 1

    d_det = problem.to_pyg_data_det(inst)
    assert d_det.edge_attr.ndim == 2
    assert d_det.edge_attr.shape[1] == 1


def test_selection_scenario_hint_costs_and_det_has_no_hints():
    problem = SelectionProblem(n_nodes=3, n_scenarios=2, p_select=2)
    inst = make_sel_debug_instance()

    d0 = problem._build_pyg_data(inst)[0]  # sce0: [1,2,3]
    d1 = problem._build_pyg_data(inst)[1]  # sce1: [3,2,1]
    d_det = problem.to_pyg_data_det(inst)

    N = 3
    y_start = N

    assert d0.x[y_start:y_start + N, 5].tolist() == [1.0, 2.0, 3.0]
    assert d1.x[y_start:y_start + N, 5].tolist() == [3.0, 2.0, 1.0]
    assert float(d_det.x[y_start:y_start + N, 5].sum().item()) == 0.0


def test_to_pyg_data_det_matches_scenario_backbone_excluding_budget():
    """
    Deterministic graph must match scenario graph edges after removing
    all edges touching the budget constraint node.
    """
    N = 3
    S = 2
    p = 2
    problem = SelectionProblem(n_nodes=N, n_scenarios=S, p_select=p)
    inst = make_sel_debug_instance(N=N, S=S, p=p)

    data_s0 = problem._build_pyg_data(inst)[0]
    data_det = problem.to_pyg_data_det(inst)

    # Scenario indexing:
    # n_vars = 2N+1
    # constraints: [sel] + [disjoint N] + [budget]
    n_vars = 2 * N + 1
    idx_budget = n_vars + (1 + N)  # last constraint in scenario graph

    # Filter edges from scenario graph that touch budget constraint node
    ei = data_s0.edge_index
    ea = data_s0.edge_attr
    filtered_edges = []
    for e in range(ei.size(1)):
        u = int(ei[0, e])
        v = int(ei[1, e])
        if u == idx_budget or v == idx_budget:
            continue
        filtered_edges.append((u, v, float(ea[e, 0].item())))
    filtered_edges.sort()

    det_edges = _edge_multiset_1d(data_det.edge_index, data_det.edge_attr)
    assert filtered_edges == det_edges


def test_selection_collate_fn_builds_batched_graphs_with_1d_edge_attr():
    problem = SelectionProblem(n_nodes=3, n_scenarios=2, p_select=2)

    inst1 = make_sel_debug_instance()
    inst2 = make_sel_debug_instance()
    lab1 = Labels(obj=0.0, feasible=True, x=np.array([1.0, 0.0, 1.0], dtype=np.float32))
    lab2 = Labels(obj=1.0, feasible=False, x=np.array([0.0, 1.0, 1.0], dtype=np.float32))

    out = problem.collate_fn(
        [
            {"instance": inst1, "label": lab1},
            {"instance": inst2, "label": lab2},
        ]
    )

    b_inst = out["instance"]
    assert "scen_graph" in b_inst.data
    assert "det_graph" in b_inst.data

    scen_g = b_inst.data["scen_graph"]
    det_g = b_inst.data["det_graph"]

    assert scen_g["edge_attr"].ndim == 2
    assert scen_g["edge_attr"].shape[1] == 1

    assert det_g["edge_attr"].ndim == 2
    assert det_g["edge_attr"].shape[1] == 1

    # sanity: batch sizes
    assert int(det_g["batch"].max().item()) + 1 == 2  # B=2 instances
    assert int(scen_g["inst_batch"].max().item()) + 1 == 2  # scenario -> instance mapping exists
