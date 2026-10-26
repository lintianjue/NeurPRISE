import math
import itertools
import numpy as np
import pytest
import torch

from src.core.types import Instance
from src.problems.selection import SelectionProblem
from src.training.datamodule import DataModule


def _require_gurobi():
    gp = pytest.importorskip("gurobipy")
    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
    except gp.GurobiError as exc:
        pytest.skip(f"Gurobi unavailable: {exc}")
    return gp


def _make_sel_instance(
    det: list[float],
    sce: list[list[float]],
    p_select: int,
) -> Instance:
    det_np = np.asarray(det, dtype=np.float32)
    sce_np = np.asarray(sce, dtype=np.float32)
    n_items = int(det_np.shape[0])
    n_scenarios = int(sce_np.shape[0])

    return Instance(
        {
            "n_items": np.array([n_items], dtype=np.int64),
            "n_scenarios": np.array([n_scenarios], dtype=np.int64),
            "p_select": np.array([p_select], dtype=np.int64),
            "det": det_np,
            "sce": sce_np,
            # Placeholder adjacency (SelectionProblem.sample() provides it)
            "adjacency": np.zeros((n_items, n_items), dtype=np.int64),
        }
    )


def _find_edges_to_dst(edge_index: torch.Tensor, edge_attr: torch.Tensor, dst: int):
    """Return list of (src, coeff) for edges src -> dst."""
    if edge_index.numel() == 0:
        return []
    mask = edge_index[1] == int(dst)
    srcs = edge_index[0][mask].tolist()
    coeffs = edge_attr[mask, 0].tolist()
    return list(zip(srcs, coeffs))


def _assert_bipartite_graph(data) -> None:
    n_vars = int(data.n_vars.item())
    src, dst = data.edge_index
    is_var = src < n_vars
    is_con = dst >= n_vars
    is_var_rev = dst < n_vars
    is_con_rev = src >= n_vars
    assert torch.all((is_var & is_con) | (is_var_rev & is_con_rev))


def _robust_obj_for_x(det: np.ndarray, sce: np.ndarray, p: int, x01: np.ndarray) -> float:
    """
    Compute the 2RO objective for a fixed first-stage decision x (0/1).
    This is a solver-independent oracle used for a manually verifiable test.

    Given x:
      - Let r = p - sum(x). If r < 0: infeasible (can't "unselect" items).
      - In each scenario i, y^i must pick exactly r items among indices where x_j = 0
        (due to x_j + y^i_j <= 1 and sum(x + y^i) == p),
        minimizing sum(sce[i,j] * y^i_j).
      - Epigraph variable t equals the worst-case (max over scenarios) of that minimal recourse cost.
      - Total objective = sum(det * x) + t.
    """
    x_cnt = int(x01.sum())
    r = int(p - x_cnt)
    if r < 0:
        return float("inf")

    base = float((det * x01).sum())

    worst_recourse = 0.0
    free_idx = np.where(x01 == 0)[0].tolist()

    for i in range(int(sce.shape[0])):
        if r == 0:
            best_recourse = 0.0
        else:
            if len(free_idx) < r:
                return float("inf")
            costs = [float(sce[i, j]) for j in free_idx]
            costs.sort()
            best_recourse = float(sum(costs[:r]))
        worst_recourse = max(worst_recourse, best_recourse)

    return base + float(worst_recourse)


def _bruteforce_best_x(det: np.ndarray, sce: np.ndarray, p: int) -> tuple[float, list[int]]:
    """
    Brute-force optimal x for small N.
    Returns (best_obj, best_x_list_01).

    Motivation:
    - Provides a tiny, completely solver-independent ground truth check.
    - Easy to verify manually: you can enumerate x choices by hand for N=3.
    """
    N = int(det.shape[0])
    best_obj = float("inf")
    best_x = None

    for bits in itertools.product([0, 1], repeat=N):
        x = np.asarray(bits, dtype=np.int64)
        obj = _robust_obj_for_x(det=det, sce=sce, p=p, x01=x)
        if obj < best_obj - 1e-12:
            best_obj = obj
            best_x = x.tolist()

    assert best_x is not None
    return best_obj, best_x


def test_deferring_can_be_better_than_full_first_stage():
    """
    Motivation:
    - In 2RO selection, it is NOT always optimal to fully decide the first stage.
    - Deferring items can reduce worst-case recourse cost when scenario costs vary.
    - This test provides a tiny, solver-free counterexample.
    """
    det = np.asarray([100.0, 100.0, 1.0], dtype=np.float32)
    sce = np.asarray(
        [
            [0.0, 50.0, 50.0],  # scenario 0
            [50.0, 0.0, 50.0],  # scenario 1
        ],
        dtype=np.float32,
    )
    p = 2

    best_obj, best_x = _bruteforce_best_x(det=det, sce=sce, p=p)

    # Best "all first-stage" choice: pick any 2 items in x (no recourse)
    full_stage_objs = []
    for bits in itertools.product([0, 1], repeat=det.shape[0]):
        x = np.asarray(bits, dtype=np.int64)
        if int(x.sum()) == p:
            full_stage_objs.append(_robust_obj_for_x(det=det, sce=sce, p=p, x01=x))
    best_full_stage = min(full_stage_objs)

    assert int(sum(best_x)) < p
    assert best_obj < best_full_stage


def test_to_pyg_data_main_flow_shapes_and_semantics():
    """
    Motivation:
    - Downstream GNN/batching assumes a stable MILP-graph encoding contract:
      correct shapes, consistent indexing, and scenario-dependent features.
    - A regression here may not crash, but will silently poison training/evaluation.

    Checks:
    - Builds S scenario graphs
    - Validates node/edge feature shapes
    - Checks scenario-specific y-hints and constraint RHS semantics
    - Checks budget constraint topology & coefficient signs
    """
    det = [10.0, 20.0, 30.0]
    sce = [
        [1.0, 2.0, 3.0],   # scenario 0
        [3.0, 2.0, 1.0],   # scenario 1
    ]
    p = 2
    inst = _make_sel_instance(det=det, sce=sce, p_select=p)

    problem = SelectionProblem(n_nodes=3, n_scenarios=2, p_select=p)
    data_list = problem.to_pyg_data(inst)
    assert len(data_list) == 2

    d0 = data_list[0]
    d1 = data_list[1]

    # --- Shapes ---
    assert d0.x.shape[1] == 8
    assert d0.edge_attr.shape[1] == 1
    assert d1.x.shape[1] == 8
    assert d1.edge_attr.shape[1] == 1


    N = 3
    n_vars = 2 * N + 1
    n_constr = N + 2
    assert int(d0.n_vars.item()) == n_vars
    assert int(d0.n_constraints.item()) == n_constr

    # Node indexing in SelectionProblem encoding:
    # vars: x[0..N-1], y[N..2N-1], t[2N]


def test_selection_full_and_reduced_from_logged_instance() -> None:
    """
    Motivation:
    - Reproduce a concrete logged instance from eval to validate full vs reduced solve.
    - Ensures the reduced scenario objective matches the recorded value for a single scenario.
    """
    _require_gurobi()
    det = np.asarray(
        [83, 89, 96, 75, 42, 53, 99, 1, 71, 57, 11, 52, 19, 33, 49, 18, 26, 16, 44, 14],
        dtype=np.float32,
    )
    sce = np.asarray(
        [
            [52, 64, 82, 93, 19, 26, 56, 79, 75, 4, 64, 32, 80, 98, 26, 10, 59, 41, 40, 14],
            [1, 49, 87, 59, 17, 86, 95, 50, 8, 97, 10, 51, 78, 37, 71, 23, 4, 20, 81, 88],
            [55, 89, 27, 67, 43, 79, 56, 52, 99, 89, 76, 42, 94, 97, 78, 67, 57, 96, 67, 10],
            [29, 59, 18, 12, 30, 45, 21, 6, 86, 26, 47, 88, 20, 66, 14, 68, 75, 39, 71, 49],
            [6, 28, 90, 48, 77, 28, 7, 65, 83, 92, 67, 17, 4, 22, 73, 78, 28, 13, 28, 1],
            [48, 31, 99, 20, 17, 81, 87, 53, 7, 81, 87, 2, 2, 16, 13, 95, 87, 16, 33, 84],
            [27, 64, 6, 53, 20, 15, 74, 35, 41, 71, 73, 46, 8, 98, 91, 73, 88, 81, 30, 15],
            [59, 50, 75, 64, 57, 24, 87, 43, 91, 6, 26, 41, 1, 31, 22, 22, 82, 47, 94, 77],
            [38, 8, 41, 45, 33, 49, 6, 86, 9, 12, 19, 15, 6, 32, 32, 24, 11, 55, 59, 16],
            [21, 8, 94, 56, 72, 95, 98, 7, 63, 2, 77, 61, 5, 54, 51, 89, 82, 1, 13, 83],
            [69, 63, 98, 27, 32, 33, 96, 68, 74, 29, 37, 61, 44, 63, 41, 43, 13, 90, 11, 7],
            [73, 67, 53, 23, 65, 4, 60, 43, 40, 83, 24, 81, 71, 66, 51, 40, 52, 75, 93, 65],
            [93, 36, 60, 40, 24, 100, 73, 48, 58, 84, 98, 29, 28, 12, 77, 84, 15, 55, 18, 42],
            [44, 26, 100, 62, 84, 51, 79, 8, 13, 28, 8, 99, 35, 47, 44, 52, 6, 70, 47, 100],
            [41, 36, 99, 8, 49, 40, 91, 87, 1, 91, 61, 62, 85, 77, 10, 28, 72, 94, 3, 75],
            [91, 48, 72, 36, 29, 58, 27, 48, 74, 2, 92, 74, 57, 40, 79, 54, 31, 86, 38, 26],
            [28, 83, 67, 31, 62, 47, 45, 42, 23, 77, 49, 60, 51, 32, 2, 57, 67, 98, 89, 78],
            [71, 38, 74, 20, 51, 81, 99, 77, 42, 67, 49, 62, 53, 75, 61, 29, 28, 75, 45, 20],
            [31, 77, 77, 20, 93, 97, 77, 45, 86, 84, 81, 87, 70, 32, 95, 86, 8, 32, 21, 51],
            [5, 34, 15, 32, 98, 88, 83, 40, 59, 11, 45, 54, 54, 70, 43, 75, 24, 43, 7, 83],
            [28, 73, 90, 61, 74, 32, 36, 18, 61, 44, 98, 13, 52, 87, 62, 89, 6, 11, 59, 2],
            [18, 79, 100, 89, 97, 87, 16, 40, 58, 11, 34, 30, 71, 98, 97, 7, 78, 1, 73, 17],
            [10, 18, 100, 15, 21, 10, 94, 3, 71, 83, 66, 54, 47, 44, 1, 24, 19, 77, 18, 70],
            [32, 5, 84, 97, 62, 4, 95, 15, 68, 18, 5, 96, 44, 99, 30, 51, 25, 24, 68, 52],
            [73, 70, 60, 94, 77, 78, 35, 87, 79, 26, 34, 24, 18, 79, 32, 4, 95, 95, 78, 43],
            [32, 66, 44, 34, 36, 80, 6, 93, 12, 92, 23, 1, 20, 60, 68, 1, 26, 19, 2, 27],
            [31, 4, 39, 12, 13, 97, 9, 80, 99, 44, 15, 13, 1, 51, 27, 59, 70, 79, 30, 82],
            [83, 47, 28, 55, 44, 20, 48, 60, 98, 95, 64, 5, 89, 34, 33, 80, 36, 79, 45, 29],
            [38, 48, 43, 86, 94, 25, 77, 84, 6, 80, 64, 13, 54, 58, 36, 22, 18, 92, 52, 88],
            [37, 43, 78, 69, 22, 19, 40, 80, 2, 77, 74, 70, 58, 67, 82, 20, 10, 82, 89, 92],
            [1, 5, 5, 67, 1, 94, 32, 72, 93, 18, 86, 72, 10, 88, 80, 5, 52, 68, 76, 13],
            [22, 71, 92, 66, 24, 18, 25, 7, 78, 29, 66, 71, 89, 4, 27, 77, 56, 26, 86, 21],
            [70, 64, 47, 70, 89, 43, 43, 84, 95, 42, 48, 82, 86, 20, 62, 28, 37, 72, 1, 5],
            [13, 16, 4, 99, 63, 8, 16, 57, 80, 41, 19, 83, 11, 63, 99, 79, 94, 15, 7, 41],
            [25, 97, 93, 30, 93, 39, 33, 37, 35, 28, 34, 49, 80, 84, 69, 58, 69, 19, 68, 66],
            [47, 12, 1, 11, 79, 51, 14, 85, 24, 96, 14, 61, 2, 7, 74, 47, 52, 37, 59, 99],
            [31, 89, 97, 38, 42, 41, 28, 17, 32, 27, 61, 91, 20, 51, 5, 55, 46, 61, 63, 78],
            [39, 83, 100, 77, 66, 66, 80, 60, 4, 2, 54, 89, 23, 70, 69, 86, 91, 29, 25, 46],
            [24, 39, 75, 39, 37, 45, 92, 83, 88, 41, 80, 21, 54, 12, 14, 61, 74, 51, 92, 20],
            [3, 44, 68, 29, 70, 82, 37, 78, 76, 35, 66, 93, 24, 27, 70, 28, 28, 89, 60, 72],
            [63, 6, 40, 62, 98, 28, 57, 45, 65, 13, 44, 83, 86, 9, 100, 77, 21, 6, 52, 67],
            [25, 58, 56, 11, 21, 46, 51, 45, 30, 38, 25, 56, 51, 59, 37, 56, 64, 79, 59, 71],
            [80, 57, 48, 8, 11, 35, 16, 52, 6, 90, 57, 71, 42, 13, 97, 55, 49, 70, 16, 70],
            [72, 86, 18, 58, 73, 30, 7, 43, 75, 28, 65, 23, 89, 81, 60, 78, 8, 48, 70, 5],
            [22, 100, 91, 10, 92, 75, 62, 15, 76, 72, 65, 80, 67, 19, 16, 89, 71, 20, 58, 9],
            [12, 1, 91, 12, 99, 10, 71, 98, 37, 64, 40, 67, 43, 71, 81, 95, 66, 17, 62, 89],
            [30, 86, 5, 87, 29, 86, 78, 91, 70, 45, 82, 2, 40, 76, 59, 69, 58, 44, 5, 94],
            [90, 95, 32, 71, 68, 3, 25, 10, 58, 43, 94, 73, 52, 17, 6, 8, 54, 9, 68, 80],
            [22, 6, 55, 56, 29, 99, 24, 30, 80, 94, 28, 31, 3, 50, 91, 17, 56, 78, 99, 88],
            [11, 71, 43, 76, 90, 41, 53, 96, 76, 46, 38, 82, 81, 100, 59, 6, 63, 97, 77, 99],
        ],
        dtype=np.float32,
    )
    p_select = 10

    full_inst = Instance(
        {
            "n_nodes": np.array([20], dtype=np.int64),
            "n_scenarios": np.array([50], dtype=np.int64),
            "p_select": np.array([p_select], dtype=np.int64),
            "det": det,
            "sce": sce,
            "adjacency": np.zeros((20, 20), dtype=np.int64),
        }
    )
    reduced_sce = sce[2:3]
    reduced_inst = Instance(
        {
            "n_nodes": np.array([20], dtype=np.int64),
            "n_scenarios": np.array([1], dtype=np.int64),
            "p_select": np.array([p_select], dtype=np.int64),
            "det": det,
            "sce": reduced_sce,
            "adjacency": np.zeros((20, 20), dtype=np.int64),
        }
    )

    problem = SelectionProblem(n_nodes=20, n_scenarios=50, p_select=p_select)
    full_labels = problem.solve_exact(full_inst, threads=1, verbose=False)
    reduced_labels = problem.solve_exact(reduced_inst, threads=1, verbose=False)

    assert full_labels.feasible is True
    assert reduced_labels.feasible is True
    assert math.isclose(full_labels.obj, 207.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(reduced_labels.obj, 203.0, rel_tol=0.0, abs_tol=1e-6)


def test_selection_solve_exact_passes_keyword_args() -> None:
    """
    Motivation:
    - Prevent accidental positional-arg misalignment (threads passed as mip_gap).
    - Ensure solve_exact forwards time_limit/threads/verbose by keyword.
    """
    _require_gurobi()
    problem = SelectionProblem(n_nodes=2, n_scenarios=1, p_select=1)
    inst = _make_sel_instance(det=[1.0, 2.0], sce=[[3.0, 4.0]], p_select=1)

    called = {}

    def _fake_solve(self, instance, time_limit=None, mip_gap=None, threads=1, verbose=False):
        called["time_limit"] = time_limit
        called["mip_gap"] = mip_gap
        called["threads"] = threads
        called["verbose"] = verbose
        return {
            "status": 2,
            "has_incumbent": False,
            "obj": None,
            "obj_bound": None,
            "mip_gap": None,
            "x": None,
            "t": None,
        }

    original = SelectionProblem.solve_robust_bigmilp
    SelectionProblem.solve_robust_bigmilp = _fake_solve
    try:
        problem.solve_exact(inst, time_limit=123.0, threads=7, verbose=True)
    finally:
        SelectionProblem.solve_robust_bigmilp = original

    assert called["time_limit"] == 123.0
    assert called["threads"] == 7
    assert called["verbose"] is True
    assert called["mip_gap"] is None


def test_to_pyg_data_zero_cost_edges_are_skipped():
    """
    Motivation:
    - Graph sparsity strongly impacts runtime/memory; the encoding intentionally skips
      y->budget edges when scenario cost is 0.
    - A regression here won't change correctness but can cause large slowdowns.

    Checks:
    - Zero cost y_j does not produce an edge to budget constraint.
    - Non-zero costs still produce edges with correct negative sign.
    """
    det = [1.0, 1.0, 1.0]
    sce = [
        [5.0, 0.0, 7.0],
    ]
    p = 1
    inst = _make_sel_instance(det=det, sce=sce, p_select=p)
    problem = SelectionProblem(n_nodes=3, n_scenarios=1, p_select=p)

    d0 = problem.to_pyg_data(inst)[0]

    N = 3
    n_vars = 2 * N + 1
    y_start = N
    idx_c_budget = n_vars + 1 + N

    budget_in_edges = _find_edges_to_dst(d0.edge_index, d0.edge_attr, idx_c_budget)
    budget_in = dict(budget_in_edges)

    # y_1 cost is 0, so it should not appear
    y1 = y_start + 1
    assert y1 not in budget_in

    # y_0 and y_2 should appear with negative sign
    assert budget_in[y_start + 0] == -5.0
    assert budget_in[y_start + 2] == -7.0


def test_selection_normalize_uses_shared_scale_for_det_and_sce():
    """
    Motivation:
    - Selection normalization should preserve the relative scale between det and sce.
    - Uses a shared min/max across det and sce to avoid reweighting stages.
    """
    det = np.array([1.0, 3.0], dtype=np.float32)
    sce = np.array([[2.0, 4.0], [5.0, 1.0]], dtype=np.float32)
    inst = Instance(
        {
            "n_nodes": np.array([2], dtype=np.int64),
            "n_scenarios": np.array([2], dtype=np.int64),
            "p_select": np.array([1], dtype=np.int64),
            "det": det,
            "sce": sce,
            "adjacency": np.zeros((2, 2), dtype=np.int64),
        }
    )

    inst_norm = DataModule._normalize_instance_shared_pairs(inst)
    det_norm = inst_norm.data["det"].detach().cpu().numpy()
    sce_norm = inst_norm.data["sce"].detach().cpu().numpy()

    assert np.allclose(det_norm, np.array([0.0, 0.5], dtype=np.float32))
    assert np.allclose(
        sce_norm,
        np.array([[0.25, 0.75], [1.0, 0.0]], dtype=np.float32),
    )


def test_to_pyg_data_type_indicators_and_constraint_edge_patterns():
    """
    Motivation:
    - 'kind' and 'is_x/is_y/is_t' are core node-type features; if these flip, the GNN
      receives the wrong semantics.
    - Selection/disjoint constraints have very specific sparsity patterns that must remain stable.

    Checks:
    - Variable nodes have kind=0; constraint nodes have kind=1.
    - x/y/t indicator slots are correct.
    - Selection constraint receives +1 edges from all x and all y.
    - Each disjoint constraint receives exactly two (-1) incoming edges: x_j and y_j.
    """
    det = [2.0, 5.0, 11.0]
    sce = [[1.0, 2.0, 3.0]]
    p = 2
    inst = _make_sel_instance(det=det, sce=sce, p_select=p)

    problem = SelectionProblem(n_nodes=3, n_scenarios=1, p_select=p)
    d0 = problem.to_pyg_data(inst)[0]

    assert d0.x.shape[1] == 8

    N = 3
    n_vars = 2 * N + 1
    y_start = N
    t_idx = 2 * N
    c_start = n_vars
    idx_c_sel = c_start
    idx_c_dis_start = c_start + 1

    # --- kind feature ---
    assert torch.all(d0.x[:n_vars, 0] == 0.0)
    assert torch.all(d0.x[n_vars:, 0] == 1.0)

    # --- variable-type one-hot-ish indicators ---
    assert torch.all(d0.x[0:N, 1] == 1.0)          # is_x
    assert torch.all(d0.x[0:N, 2] == 0.0)          # not y
    assert torch.all(d0.x[0:N, 3] == 0.0)          # not t

    assert torch.all(d0.x[y_start:2 * N, 2] == 1.0)  # is_y
    assert torch.all(d0.x[y_start:2 * N, 1] == 0.0)
    assert torch.all(d0.x[y_start:2 * N, 3] == 0.0)

    assert d0.x[t_idx, 3].item() == 1.0            # is_t
    assert d0.x[t_idx, 1].item() == 0.0
    assert d0.x[t_idx, 2].item() == 0.0

    # --- selection constraint incoming edges: from all x and all y with +1 ---
    sel_in = dict(_find_edges_to_dst(d0.edge_index, d0.edge_attr, idx_c_sel))
    for j in range(N):
        assert sel_in[j] == 1.0
        assert sel_in[y_start + j] == 1.0

    # --- disjoint constraints: exactly x_j and y_j with -1 ---
    for j in range(N):
        c_idx = idx_c_dis_start + j
        dis_in_edges = _find_edges_to_dst(d0.edge_index, d0.edge_attr, c_idx)
        dis_in = dict(dis_in_edges)

        # Must contain x_j and y_j with -1
        assert dis_in[j] == -1.0
        assert dis_in[y_start + j] == -1.0

        # And nothing else incoming besides those two (from vars); reverse edges exist but are to vars.
        assert set(dis_in.keys()) == {j, y_start + j}


def test_to_pyg_data_cosine_similarity_features_match_closed_form():
    """
    Motivation:
    - Feature slot 7 encodes constraint-row alignment with the objective vector via cosine similarity.
      This is a subtle feature that can silently regress (sign flips, wrong objective vector layout).

    Checks (scenario 0):
    - Selection constraint cos sim:
        row has +1 on all x and y; objective is det on x, 0 on y, 1 on t
        => dot = sum(det), ||row|| = sqrt(2N), ||obj|| = sqrt(sum(det^2) + 1)
    - Disjoint constraint j cos sim:
        row has -1 on x_j and y_j => dot = -det[j], ||row|| = sqrt(2)
    - Budget constraint cos sim:
        row has +1 on t and -sce[j] on y_j => dot = 1, ||row|| = sqrt(1 + sum(sce^2))
    """
    det = [3.0, 4.0, 12.0]
    sce0 = [1.0, 2.0, 2.0]
    sce = [sce0]
    p = 2
    inst = _make_sel_instance(det=det, sce=sce, p_select=p)

    problem = SelectionProblem(n_nodes=3, n_scenarios=1, p_select=p)
    d0 = problem.to_pyg_data(inst)[0]

    N = 3
    n_vars = 2 * N + 1
    c_start = n_vars
    idx_c_sel = c_start
    idx_c_dis_start = c_start + 1
    idx_c_budget = c_start + 1 + N

    det_np = np.asarray(det, dtype=np.float64)
    sce_np = np.asarray(sce0, dtype=np.float64)

    obj_norm = float(np.sqrt(float((det_np ** 2).sum()) + 1.0))

    # Selection constraint
    dot_sel = float(det_np.sum())
    row_norm_sel = float(np.sqrt(2.0 * N))
    expected_sel = dot_sel / (row_norm_sel * obj_norm)
    assert float(d0.x[idx_c_sel, 7].item()) == pytest.approx(expected_sel, abs=1e-6)

    # Disjoint constraints
    row_norm_dis = float(np.sqrt(2.0))
    for j in range(N):
        expected_dis = (-float(det_np[j])) / (row_norm_dis * obj_norm)
        c_idx = idx_c_dis_start + j
        assert float(d0.x[c_idx, 7].item()) == pytest.approx(expected_dis, abs=1e-6)

    # Budget constraint
    dot_bud = 1.0  # only t aligns with objective vector
    row_norm_bud = float(np.sqrt(1.0 + float((sce_np ** 2).sum())))
    expected_bud = dot_bud / (row_norm_bud * obj_norm)
    assert float(d0.x[idx_c_budget, 7].item()) == pytest.approx(expected_bud, abs=1e-6)


def test_solve_robust_bigmilp_trivial_optimum_is_correct():
    """
    Motivation:
    - End-to-end validation that the robust MILP formulation produces a feasible incumbent
      and that objective components (x costs and t) behave as expected.
    - This catches modeling issues that graph-only tests cannot.

    Constructed instance:
    N=2, p=1:
      det = [1, 100]
      sce identical across scenarios so the robust term is stable.
    Optimal: x=[1,0], y all 0, t=0, obj=1.
    """
    gp = _require_gurobi()

    det = [1.0, 100.0]
    sce = [
        [100.0, 1.0],
        [100.0, 1.0],
    ]
    p = 1
    inst = _make_sel_instance(det=det, sce=sce, p_select=p)
    problem = SelectionProblem(n_nodes=2, n_scenarios=2, p_select=p)

    res = problem.solve_robust_bigmilp(inst, time_limit=10.0, threads=1, verbose=False)

    assert res["has_incumbent"] is True
    assert res["x"] == [1.0, 0.0]
    assert math.isfinite(float(res["obj"]))
    assert abs(float(res["t"]) - 0.0) <= 1e-6
    assert abs(float(res["obj"]) - 1.0) <= 1e-6


def test_solve_robust_bigmilp_manual_verifiable_by_enumeration():
    """
    Motivation (manual-verifiable ground truth):
    - We want a tiny 2RO instance where a human can verify the correct answer by hand,
      and the test can *also* verify it automatically without relying on a MILP solver.

    Instance (N=3, S=2, p=2):
      det = [1, 50, 50]
      scenario 0 costs for y: [100, 1, 2]
      scenario 1 costs for y: [100, 2, 1]

    Manual reasoning:
      - If we pick x0=1, we must pick 1 more item per scenario in stage-2 (since p=2).
        In each scenario, we can choose the cheaper of items {1,2}:
          scen0: min(1,2)=1
          scen1: min(2,1)=1
        Worst-case recourse cost t = max(1,1) = 1
        Total obj = det0 + t = 1 + 1 = 2

      - Any solution picking x1=1 or x2=1 in stage-1 has base cost >= 50 already,
        and picking two items in stage-1 costs >= 51, so cannot beat obj=2.

    This test:
      1) Computes best x by brute-force enumeration (solver-independent oracle).
      2) Verifies the MILP solver output matches that oracle exactly.
    """
    gp = _require_gurobi()

    det = [1.0, 50.0, 50.0]
    sce = [
        [100.0, 1.0, 2.0],
        [100.0, 2.0, 1.0],
    ]
    p = 2

    det_np = np.asarray(det, dtype=np.float64)
    sce_np = np.asarray(sce, dtype=np.float64)

    best_obj_oracle, best_x_oracle = _bruteforce_best_x(det=det_np, sce=sce_np, p=p)
    assert best_x_oracle == [1, 0, 0]
    assert abs(best_obj_oracle - 2.0) <= 1e-12

    inst = _make_sel_instance(det=det, sce=sce, p_select=p)
    problem = SelectionProblem(n_nodes=3, n_scenarios=2, p_select=p)
    res = problem.solve_robust_bigmilp(inst, time_limit=10.0, threads=1, verbose=False)

    assert res["has_incumbent"] is True
    assert res["x"] == [1.0, 0.0, 0.0]
    assert abs(float(res["obj"]) - best_obj_oracle) <= 1e-6
    assert abs(float(res["t"]) - 1.0) <= 1e-6


def test_solve_robust_bigmilp_infeasible_when_p_exceeds_n_items():
    """
    Motivation:
    - Common config/data bug: p_select > N makes constraints impossible.
    - The solver path must fail gracefully (no crash) and report no incumbent.

    Why infeasible:
      - For each item j: x_j + y_{i,j} <= 1
      - Summing over j: sum_j (x_j + y_{i,j}) <= N
      - But constraints require sum_j (x_j + y_{i,j}) == p for each scenario i
      - So if p > N, infeasible.
    """
    gp = _require_gurobi()

    det = [1.0, 2.0]
    sce = [
        [1.0, 1.0],
        [1.0, 1.0],
    ]
    inst = _make_sel_instance(det=det, sce=sce, p_select=3)  # N=2, p=3 -> infeasible
    problem = SelectionProblem(n_nodes=2, n_scenarios=2, p_select=3)

    res = problem.solve_robust_bigmilp(inst, time_limit=10.0, threads=1, verbose=False)

    assert res["has_incumbent"] is False
    assert res["obj"] is None
    assert res["x"] is None
    assert res["t"] is None


def test_solve_robust_bigmilp_p_zero_is_feasible_and_zero_solution():
    """
    Motivation:
    - Boundary condition: p_select=0 is a legal but extreme case.
    - Catches sign/index bugs around equality constraints and the epigraph variable.

    Expected:
      - x all zeros
      - y all zeros
      - t=0
      - obj=0
    """
    gp = _require_gurobi()

    det = [5.0, 6.0, 7.0]
    sce = [
        [10.0, 10.0, 10.0],
    ]
    inst = _make_sel_instance(det=det, sce=sce, p_select=0)
    problem = SelectionProblem(n_nodes=3, n_scenarios=1, p_select=0)

    res = problem.solve_robust_bigmilp(inst, time_limit=10.0, threads=1, verbose=False)

    assert res["has_incumbent"] is True
    assert res["x"] == [0.0, 0.0, 0.0]
    assert abs(float(res["t"]) - 0.0) <= 1e-6
    assert abs(float(res["obj"]) - 0.0) <= 1e-6


def test_selection_graphs_are_bipartite():
    """
    Motivation:
    - Ensures selection MILP graphs only connect variables to constraints.
    - Guards against index mixups in the graph encoding.
    """
    problem = SelectionProblem(n_nodes=3, n_scenarios=1, p_select=1)
    inst = _make_sel_instance(det=[1.0, 2.0, 3.0], sce=[[4.0, 5.0, 6.0]], p_select=1)

    data_s0 = problem.to_pyg_data(inst)[0]
    data_det = problem.to_pyg_data_det(inst)

    _assert_bipartite_graph(data_s0)
    _assert_bipartite_graph(data_det)


def test_selection_graph_coefficients_and_rhs():
    """
    Motivation:
    - Verifies constraint placement and signed coefficients for a tiny instance.
    - Catches index/sign bugs in the bipartite MILP encoding.
    """
    problem = SelectionProblem(n_nodes=2, n_scenarios=1, p_select=1)
    inst = _make_sel_instance(det=[1.0, 2.0], sce=[[3.0, 4.0]], p_select=1)

    data = problem._build_pyg_data(inst)[0]
    N = 2
    y_start = N
    t_idx = 2 * N
    n_vars = 2 * N + 1

    c_start = n_vars
    idx_sel = c_start
    idx_dis0 = c_start + 1
    idx_dis1 = c_start + 2
    idx_bud = c_start + 3

    sel_edges = dict(_find_edges_to_dst(data.edge_index, data.edge_attr, idx_sel))
    assert sel_edges[y_start + 0] == 1.0
    assert sel_edges[y_start + 1] == 1.0
    assert sel_edges[0] == 1.0
    assert sel_edges[1] == 1.0
    assert data.x[idx_sel, 6].item() == 1.0

    dis0_edges = dict(_find_edges_to_dst(data.edge_index, data.edge_attr, idx_dis0))
    assert dis0_edges[0] == -1.0
    assert dis0_edges[y_start + 0] == -1.0
    assert data.x[idx_dis0, 6].item() == -1.0

    dis1_edges = dict(_find_edges_to_dst(data.edge_index, data.edge_attr, idx_dis1))
    assert dis1_edges[1] == -1.0
    assert dis1_edges[y_start + 1] == -1.0
    assert data.x[idx_dis1, 6].item() == -1.0

    bud_edges = dict(_find_edges_to_dst(data.edge_index, data.edge_attr, idx_bud))
    assert bud_edges[t_idx] == 1.0
    assert bud_edges[y_start + 0] == -3.0
    assert bud_edges[y_start + 1] == -4.0
    assert data.x[idx_bud, 6].item() == 0.0
