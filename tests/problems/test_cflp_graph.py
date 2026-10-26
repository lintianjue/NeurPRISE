import numpy as np
import torch

from src.core.types import Instance
from src.problems.cflp import CFLPProblem


def _find_edges_to_dst(edge_index: torch.Tensor, edge_attr: torch.Tensor, dst: int) -> dict:
    if edge_index.numel() == 0:
        return {}
    mask = edge_index[1] == int(dst)
    srcs = edge_index[0][mask].tolist()
    coeffs = edge_attr[mask, 0].tolist()
    return dict(zip(srcs, coeffs))


def _make_small_cflp_instance() -> Instance:
    f = np.array([1.0, 2.0], dtype=np.float32)
    a = np.array([3.0, 4.0], dtype=np.float32)
    K = np.array([10.0, 20.0], dtype=np.float32)
    c = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    demand = np.array([[9.0, 11.0]], dtype=np.float32)

    return Instance(
        {
            "n_facilities": np.array([2], dtype=np.int64),
            "n_customers": np.array([2], dtype=np.int64),
            "n_scenarios": np.array([1], dtype=np.int64),
            "f": f,
            "a": a,
            "K": K,
            "c": c,
            "d": demand,
        }
    )


def test_cflp_graph_shapes_and_bipartite():
    """
    Motivation:
    - Verifies variable/constraint counts for scenario and deterministic graphs.
    - Ensures graph edges are strictly bipartite (var <-> constraint).
    """
    problem = CFLPProblem(n_facilities=2, n_customers=2, n_scenarios=1)
    inst = _make_small_cflp_instance()

    data_s0 = problem._build_pyg_data(inst)[0]
    data_det = problem.to_pyg_data_det(inst)

    I, J = 2, 2
    # Vars: y (I) + z (I) + x (I*J) + theta (1 epigraph)
    n_vars = 2 * I + I * J + 1
    # Scenario constraints: install (I) + capacity (I) + demand (J) + epigraph (1)
    n_cons_s = 2 * I + J + 1
    # Deterministic constraints: install (I) + capacity (I) + epigraph (1).
    # Even without demand uncertainty, the epigraph keeps the same variable order
    # as scenario graphs and acts as a placeholder for the recourse-cost link.
    n_cons_det = 2 * I + 1

    assert int(data_s0.n_vars.item()) == n_vars
    assert int(data_s0.n_constraints.item()) == n_cons_s
    assert int(data_det.n_vars.item()) == n_vars
    assert int(data_det.n_constraints.item()) == n_cons_det

    for data in (data_s0, data_det):
        src, dst = data.edge_index
        is_var = src < n_vars
        is_con = dst >= n_vars
        is_var_rev = dst < n_vars
        is_con_rev = src >= n_vars
        assert torch.all((is_var & is_con) | (is_var_rev & is_con_rev))


def test_cflp_graph_coefficients_and_rhs():
    """
    Motivation:
    - Checks key constraint rows: install, demand, and epigraph coefficients.
    - Guards against index/sign errors in the MILP graph encoding.
    """
    problem = CFLPProblem(n_facilities=2, n_customers=2, n_scenarios=1)
    inst = _make_small_cflp_instance()
    data = problem._build_pyg_data(inst)[0]

    I, J = 2, 2
    y_start = 0
    z_start = y_start + I
    x_start = z_start + I
    theta_idx = x_start + I * J
    n_vars = theta_idx + 1

    constr_start = n_vars
    cap_start = constr_start + I
    dem_start = cap_start + I
    epi_idx = dem_start + J

    # Install constraint for facility 0: z0 - K0*y0 <= 0
    install0 = constr_start + 0
    in_edges = _find_edges_to_dst(data.edge_index, data.edge_attr, install0)
    assert in_edges[z_start + 0] == 1.0
    assert in_edges[y_start + 0] == -10.0
    assert data.x[install0, 6].item() == 0.0

    # Demand constraint for customer 1: -x01 - x11 <= -d1
    dem1 = dem_start + 1
    in_edges = _find_edges_to_dst(data.edge_index, data.edge_attr, dem1)
    assert in_edges[x_start + 0 * J + 1] == -1.0
    assert in_edges[x_start + 1 * J + 1] == -1.0
    assert data.x[dem1, 6].item() == -11.0

    # Epigraph constraint: sum c_ij x_ij - theta <= 0
    in_edges = _find_edges_to_dst(data.edge_index, data.edge_attr, epi_idx)
    assert in_edges[theta_idx] == -1.0
    assert in_edges[x_start + 0 * J + 0] == 5.0
    assert in_edges[x_start + 0 * J + 1] == 6.0
    assert in_edges[x_start + 1 * J + 0] == 7.0
    assert in_edges[x_start + 1 * J + 1] == 8.0
