from types import MethodType

import numpy as np
import pytest

from src.problems.vertex_cover import VertexCoverProblem
from src.utils.seed import seed_everything
from src.core.types import Instance


def test_vertex_cover_expected_average_degree_is_ten():
    """
    Motivation:
    - Ensures the ER graph sampling follows p=10/n (expected avg degree ~10).
    - Protects against regressions that change p or edge generation logic.
    """
    seed_everything(123)
    n_nodes = 100
    n_samples = 200
    problem = VertexCoverProblem(n_nodes=n_nodes, n_scenarios=1, p_edge=10.0)

    avg_degrees = []
    for _ in range(n_samples):
        problem._sample_edges()
        degrees = problem.edge_adjacency.sum(axis=1).astype(float)
        avg_degrees.append(float(degrees.mean()))

    mean_degree = float(np.mean(avg_degrees))
    assert abs(mean_degree - 10.0) < 1.5


def test_vertex_cover_solve_exact_passes_keyword_args() -> None:
    """
    Motivation:
    - Prevent accidental positional-arg misalignment in solve_exact.
    - Ensure solve_exact forwards time_limit/threads/verbose by keyword.
    """
    problem = VertexCoverProblem(n_nodes=2, n_scenarios=1, p_edge=1.0)
    inst = Instance({"n_nodes": np.array([2], dtype=np.int64), "n_scenarios": np.array([1], dtype=np.int64)})

    called = {}

    def _fake_solve(
        self, instance, time_limit=None, mip_gap=None, threads=1, verbose=False
    ):
        called["time_limit"] = time_limit
        called["threads"] = threads
        called["verbose"] = verbose
        return {
            "status": 2,
            "has_incumbent": False,
            "obj": 0.0,
            "obj_bound": None,
            "mip_gap": None,
            "x": None,
            "t": None,
            "wc_idx": None,
            "rec_costs": None,
        }

    original = VertexCoverProblem.solve_robust_bigmilp
    VertexCoverProblem.solve_robust_bigmilp = _fake_solve
    try:
        problem.solve_exact(inst, time_limit=12.5, threads=3, verbose=True)
    finally:
        VertexCoverProblem.solve_robust_bigmilp = original

    assert called["time_limit"] == 12.5
    assert called["threads"] == 3
    assert called["verbose"] is True


def test_vertex_cover_solve_exact_forwards_mip_gap() -> None:
    """The achieved-gap report is invalid if VC drops the requested gap."""
    problem = VertexCoverProblem.__new__(VertexCoverProblem)
    problem.exact_backend = "bigmilp"
    called = {}

    def _fake_solve(
        self,
        instance,
        time_limit=None,
        mip_gap=None,
        threads=1,
        verbose=False,
    ):
        called["mip_gap"] = mip_gap
        return {
            "status": 2,
            "has_incumbent": True,
            "obj": 1.0,
            "obj_bound": 1.0,
            "mip_gap": 0.0,
            "x": [1.0],
            "t": 0.0,
            "rec_costs": None,
            "wc_idx": None,
        }

    problem.solve_robust_bigmilp = MethodType(_fake_solve, problem)
    inst = Instance(
        data={
            "adjacency": np.zeros((1, 1)),
            "det": np.ones(1),
            "sce": np.ones((1, 1)),
        }
    )

    problem.solve_exact(inst, mip_gap=1e-4)

    assert called["mip_gap"] == pytest.approx(1e-4)
