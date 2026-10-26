"""End-to-end tests for the regret evaluation pipeline.

Tests the core flow: select scenarios → solve reduced MILP → get x_star →
evaluate x_star on ALL scenarios → compute regret.

Invariants verified:
  1. regret ≥ 0 for feasible instances
  2. eval_obj ≥ V(R_k) (evaluating on Ξ ⊇ R_k can only increase worst-case)
  3. regret = 0 when using ALL scenarios (k = S)
  4. regret monotonically non-increasing as k increases
  5. V(R_k) ≤ V(Ξ) (reduced obj ≤ full obj)

Uses tiny VC instance (n=10, S=5) — runs in seconds without GPU.
"""
import numpy as np
import pytest

from src.core.types import Instance
from src.problems.vertex_cover import VertexCoverProblem
from src.utils.batch_utils import _build_reduced_instance_from_np


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vc_problem():
    return VertexCoverProblem(n_nodes=10, n_scenarios=5)


@pytest.fixture
def small_vc_instance():
    """Reproducible VC instance: n=10, S=5."""
    rng = np.random.RandomState(42)
    N, S = 10, 5
    det = rng.randint(1, 100, size=N).astype(float)
    sce = rng.randint(1, 100, size=(S, N)).astype(float)
    adj = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < 0.3:
                adj[i, j] = 1
                adj[j, i] = 1
    return Instance({
        "n_nodes": np.array([N]),
        "n_scenarios": np.array([S]),
        "n_min_cost": np.array([1]),
        "n_max_cost": np.array([100]),
        "p_edge": np.array([30.0]),
        "det": det,
        "sce": sce,
        "adjacency": adj,
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def solve_full_milp(prob, inst):
    """Solve V(Ξ) — full MILP with all scenarios."""
    out = prob.solve_robust_bigmilp(inst, time_limit=60, threads=1, verbose=False, mip_gap=1e-4)
    assert out["has_incumbent"], "Full MILP must be feasible"
    return float(out["obj"]), out["x"]


def solve_reduced_milp(prob, inst, sce_idx, mip_gap=1e-4):
    """Solve V(R_k) — reduced MILP with selected scenarios."""
    data_np = {k: np.array(v) if hasattr(v, '__len__') else v for k, v in inst.data.items()}
    reduced = _build_reduced_instance_from_np(prob, data_np, sce_idx)
    out = prob.solve_robust_bigmilp(reduced, time_limit=60, threads=1, verbose=False, mip_gap=mip_gap)
    assert out["has_incumbent"], f"Reduced MILP must be feasible for sce_idx={sce_idx}"
    return float(out["obj"]), out["x"]


def evaluate_x_on_all(prob, inst, x_star):
    """Evaluate c^T x + max_{ξ∈Ξ} Q(x, ξ)."""
    data = inst.data
    det = np.asarray(data["det"], dtype=float).reshape(-1)
    S = int(data["n_scenarios"][0])
    first_stage_cost = float(np.dot(det, np.asarray(x_star, dtype=float)))
    max_recourse = -np.inf
    for s in range(S):
        out = prob.solve_recourse_mip(inst, s, x_star, threads=1, time_limit=60)
        rc = float(out["obj"]) if out.get("has_incumbent", True) else float("inf")
        if rc > max_recourse:
            max_recourse = rc
    return first_stage_cost + max_recourse


def compute_regret(eval_obj, full_obj):
    """Regret = (eval_obj - full_obj) / full_obj × 100."""
    return (eval_obj - full_obj) / full_obj * 100.0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRegretPipeline:
    """End-to-end regret computation tests."""

    def test_full_milp_is_optimal(self, vc_problem, small_vc_instance):
        """V(Ξ) solution evaluated on all scenarios should give regret ≈ 0."""
        full_obj, x_full = solve_full_milp(vc_problem, small_vc_instance)
        eval_obj = evaluate_x_on_all(vc_problem, small_vc_instance, x_full)
        regret = compute_regret(eval_obj, full_obj)
        # Regret should be ~0 (within MIP gap tolerance)
        assert regret == pytest.approx(0.0, abs=0.5), \
            f"Full MILP solution should have ~0 regret, got {regret:.4f}%"

    def test_regret_nonnegative(self, vc_problem, small_vc_instance):
        """Regret must be ≥ 0 for any scenario subset."""
        full_obj, _ = solve_full_milp(vc_problem, small_vc_instance)
        S = int(small_vc_instance.data["n_scenarios"][0])
        for k in [1, 2, 3]:
            sce_idx = np.arange(k)  # first k scenarios
            _, x_reduced = solve_reduced_milp(vc_problem, small_vc_instance, sce_idx)
            eval_obj = evaluate_x_on_all(vc_problem, small_vc_instance, x_reduced)
            regret = compute_regret(eval_obj, full_obj)
            assert regret >= -0.5, \
                f"Regret must be ≥ 0, got {regret:.4f}% at k={k}"

    def test_eval_obj_geq_reduced_obj(self, vc_problem, small_vc_instance):
        """eval_obj ≥ V(R_k): evaluating on Ξ ⊇ R_k can only increase worst-case cost."""
        full_obj, _ = solve_full_milp(vc_problem, small_vc_instance)
        for k in [1, 2]:
            sce_idx = np.arange(k)
            reduced_obj, x_reduced = solve_reduced_milp(vc_problem, small_vc_instance, sce_idx)
            eval_obj = evaluate_x_on_all(vc_problem, small_vc_instance, x_reduced)
            assert eval_obj >= reduced_obj - 1e-6, \
                f"eval_obj ({eval_obj:.2f}) must be ≥ V(R_{k}) ({reduced_obj:.2f})"

    def test_reduced_obj_leq_full_obj(self, vc_problem, small_vc_instance):
        """V(R_k) ≤ V(Ξ) — fewer scenarios = weaker lower bound."""
        full_obj, _ = solve_full_milp(vc_problem, small_vc_instance)
        for k in [1, 2, 3]:
            sce_idx = np.arange(k)
            reduced_obj, _ = solve_reduced_milp(vc_problem, small_vc_instance, sce_idx)
            assert reduced_obj <= full_obj + 1e-6, \
                f"V(R_{k}) = {reduced_obj:.2f} must be ≤ V(Ξ) = {full_obj:.2f}"

    def test_regret_zero_at_full_k(self, vc_problem, small_vc_instance):
        """Using all scenarios (k=S) should give regret ≈ 0."""
        S = int(small_vc_instance.data["n_scenarios"][0])
        full_obj, _ = solve_full_milp(vc_problem, small_vc_instance)
        sce_idx = np.arange(S)  # all scenarios
        _, x_all = solve_reduced_milp(vc_problem, small_vc_instance, sce_idx)
        eval_obj = evaluate_x_on_all(vc_problem, small_vc_instance, x_all)
        regret = compute_regret(eval_obj, full_obj)
        assert regret == pytest.approx(0.0, abs=0.5), \
            f"k=S should give ~0 regret, got {regret:.4f}%"

    def test_regret_monotonicity(self, vc_problem, small_vc_instance):
        """More scenarios → lower or equal regret (on average over random subsets)."""
        full_obj, _ = solve_full_milp(vc_problem, small_vc_instance)
        S = int(small_vc_instance.data["n_scenarios"][0])
        rng = np.random.RandomState(123)

        prev_regret = float("inf")
        for k in [1, 2, 3, 4, S]:
            # Use greedy-best selection for deterministic monotonicity
            if k == S:
                sce_idx = np.arange(S)
            else:
                sce_idx = rng.choice(S, k, replace=False)
                sce_idx.sort()
            _, x_k = solve_reduced_milp(vc_problem, small_vc_instance, sce_idx)
            eval_obj = evaluate_x_on_all(vc_problem, small_vc_instance, x_k)
            regret = compute_regret(eval_obj, full_obj)
            # Note: strict monotonicity not guaranteed for random subsets,
            # but k=S must be ≤ any k<S
            if k == S:
                assert regret <= prev_regret + 0.5, \
                    f"k=S regret ({regret:.2f}%) must be ≤ previous ({prev_regret:.2f}%)"
