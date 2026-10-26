import numpy as np
import pytest

from src.core.types import Instance
from src.problems.cflp import CFLPProblem


def _require_gurobi():
    gp = pytest.importorskip("gurobipy")
    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
    except gp.GurobiError as exc:
        pytest.skip(f"Gurobi unavailable: {exc}")
    return gp


@pytest.fixture
def small_lt_instance() -> Instance:
    f = np.array([4.0, 6.0], dtype=np.float32)
    a = np.array([1.0, 1.5], dtype=np.float32)
    K = np.array([5.0, 4.0], dtype=np.float32)
    c = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32)
    demand = np.array([[3.0, 4.0], [6.0, 2.0]], dtype=np.float32)

    return Instance(
        {
            "n_facilities": np.array([2], dtype=np.int64),
            "n_customers": np.array([2], dtype=np.int64),
            "n_scenarios": np.array([2], dtype=np.int64),
            "f": f,
            "a": a,
            "K": K,
            "c": c,
            "d": demand,
        }
    )


@pytest.fixture
def small_lt_problem() -> CFLPProblem:
    return CFLPProblem(n_facilities=2, n_customers=2, n_scenarios=2)


def test_cflp_basic_solve(small_lt_problem, small_lt_instance):
    """
    Motivation:
    - Ensures the expanded robust MIP builds and solves on a tiny instance.
    - Guards against regressions that return infeasible/empty solutions.
    """
    _require_gurobi()
    out = small_lt_problem.solve_robust_bigmilp(
        small_lt_instance,
        scenario_indices=None,
        time_limit=10,
        threads=1,
        verbose=False,
    )

    assert out["has_incumbent"] is True
    assert out["obj"] is not None
    assert np.isfinite(out["obj"])
    assert out["y"] is not None
    assert out["z"] is not None


def test_cflp_ccg_matches_bigmilp_on_tiny_instance(small_lt_problem, small_lt_instance):
    """
    Motivation:
    - Keep CCG objective aligned with BigMILP objective on a tiny deterministic CFLP instance.
    - Protect the intended ablation invariant: backend switch should not change optimum.
    """
    _require_gurobi()
    out_big = small_lt_problem.solve_robust_bigmilp(
        small_lt_instance,
        scenario_indices=None,
        time_limit=10.0,
        threads=1,
        verbose=False,
    )
    out_ccg = small_lt_problem.solve_robust_ccg(
        small_lt_instance,
        scenario_indices=None,
        time_limit=10.0,
        recourse_time_limit=10.0,
        threads=1,
        verbose=False,
        max_iterations=10,
        convergence_tolerance=1e-6,
        duplicate_tolerance=1e-8,
    )

    assert out_big["has_incumbent"] is True
    assert out_ccg["has_incumbent"] is True
    assert np.isfinite(float(out_big["obj"]))
    assert np.isfinite(float(out_ccg["obj"]))
    assert np.isclose(float(out_big["obj"]), float(out_ccg["obj"]), atol=1e-6)
    assert "ccg_converged" in out_ccg
    assert "ccg_iterations" in out_ccg


def test_cflp_solve_exact_uses_ccg_backend_config():
    """
    Motivation:
    - Ensure solve_exact dispatches to CCG when exact_backend is configured to 'ccg'.
    - Prevent silent fallback to BigMILP during backend ablation runs.
    """
    problem = CFLPProblem(n_facilities=2, n_customers=2, n_scenarios=2)
    problem.exact_backend = "ccg"
    problem.configure_ccg_solver(
        max_iterations=7,
        convergence_tolerance=1e-5,
        duplicate_tolerance=1e-7,
        recourse_time_limit=3.0,
    )
    called = {}

    def _fake_ccg(
        self,
        instance,
        scenario_indices=None,
        time_limit=None,
        recourse_time_limit=None,
        mip_gap=None,
        threads=1,
        verbose=False,
        max_iterations=20,
        convergence_tolerance=1e-4,
        duplicate_tolerance=1e-6,
    ):
        called["scenario_indices"] = scenario_indices
        called["time_limit"] = time_limit
        called["recourse_time_limit"] = recourse_time_limit
        called["threads"] = threads
        called["verbose"] = verbose
        called["max_iterations"] = max_iterations
        called["convergence_tolerance"] = convergence_tolerance
        called["duplicate_tolerance"] = duplicate_tolerance
        return {
            "status": 2,
            "has_incumbent": False,
            "obj": None,
            "obj_bound": None,
            "mip_gap": None,
            "y": None,
            "z": None,
            "theta": None,
            "recourse_costs": None,
            "wc_idx": None,
            "runtime_sec": 0.0,
            "ccg_iterations": 1,
            "ccg_converged": False,
            "ccg_termination_reason": "unit_test",
        }

    original = CFLPProblem.solve_robust_ccg
    CFLPProblem.solve_robust_ccg = _fake_ccg
    try:
        dummy = Instance(
            {
                "n_facilities": np.array([2], dtype=np.int64),
                "n_customers": np.array([2], dtype=np.int64),
                "n_scenarios": np.array([2], dtype=np.int64),
                "f": np.array([1.0, 1.0], dtype=np.float32),
                "a": np.array([1.0, 1.0], dtype=np.float32),
                "K": np.array([2.0, 2.0], dtype=np.float32),
                "c": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
                "d": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
            }
        )
        problem.solve_exact(dummy, time_limit=9.0, threads=4, verbose=True)
    finally:
        CFLPProblem.solve_robust_ccg = original

    assert called["scenario_indices"] is None
    assert called["time_limit"] == 9.0
    assert called["recourse_time_limit"] == 3.0
    assert called["threads"] == 4
    assert called["verbose"] is True
    assert called["max_iterations"] == 7
    assert called["convergence_tolerance"] == 1e-5
    assert called["duplicate_tolerance"] == 1e-7


def test_cflp_solve_exact_passes_keyword_args(small_lt_problem, small_lt_instance):
    """
    Motivation:
    - Prevent accidental positional-arg misalignment in solve_exact.
    - Ensure solve_exact forwards time_limit/threads/verbose/mip_gap and scenario_indices by keyword.
    """
    called = {}

    def _fake_solve(
        self,
        instance,
        scenario_indices=None,
        time_limit=None,
        threads=1,
        verbose=False,
        mip_gap=None,
    ):
        called["scenario_indices"] = scenario_indices
        called["mip_gap"] = mip_gap
        called["time_limit"] = time_limit
        called["threads"] = threads
        called["verbose"] = verbose
        return {
            "status": 2,
            "has_incumbent": False,
            "obj": None,
            "obj_bound": None,
            "mip_gap": None,
            "y": None,
            "z": None,
            "theta": None,
            "recourse_costs": None,
            "wc_idx": None,
        }

    original = CFLPProblem.solve_robust_bigmilp
    CFLPProblem.solve_robust_bigmilp = _fake_solve
    try:
        small_lt_problem.solve_exact(small_lt_instance, time_limit=7.0, threads=5, verbose=True,
                                     mip_gap=1e-4)
    finally:
        CFLPProblem.solve_robust_bigmilp = original

    assert called["scenario_indices"] is None
    assert called["time_limit"] == 7.0
    assert called["threads"] == 5
    assert called["verbose"] is True
    assert called["mip_gap"] == 1e-4


def test_cflp_cosine_similarity_install_row():
    """
    Motivation:
    - Cosine similarity feature should match the closed-form row/objective alignment.
    - Uses the install constraint row for facility 0.
    """
    problem = CFLPProblem(n_facilities=2, n_customers=2, n_scenarios=1)

    f = np.array([3.0, 4.0], dtype=np.float32)
    a = np.array([5.0, 6.0], dtype=np.float32)
    K = np.array([7.0, 8.0], dtype=np.float32)
    c = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    demand = np.array([[1.0, 1.0]], dtype=np.float32)

    inst = Instance(
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

    data = problem._build_pyg_data(inst)[0]

    I, J = 2, 2
    y_start = 0
    z_start = y_start + I
    x_start = z_start + I
    theta_idx = x_start + I * J
    n_vars = theta_idx + 1
    constr_start = n_vars

    row = {y_start + 0: -float(K[0]), z_start + 0: 1.0}
    full_obj_vec = np.zeros(n_vars, dtype=float)
    full_obj_vec[y_start:y_start + I] = f
    full_obj_vec[z_start:z_start + I] = a
    full_obj_vec[theta_idx] = 1.0

    dot = row[y_start + 0] * full_obj_vec[y_start + 0] + row[z_start + 0] * full_obj_vec[z_start + 0]
    norm_row = np.sqrt(row[y_start + 0] ** 2 + row[z_start + 0] ** 2)
    norm_obj = float(np.linalg.norm(full_obj_vec))
    expected = float(dot / (norm_row * norm_obj))

    assert np.isclose(data.x[constr_start + 0, 7].item(), expected, atol=1e-6)


def test_cflp_monotonicity(small_lt_problem, small_lt_instance):
    """
    Motivation:
    - Robust objective must be non-decreasing as scenarios are added.
    - This is a core invariant for scenario selection heuristics.
    """
    _require_gurobi()
    out_r = small_lt_problem.solve_robust_bigmilp(
        small_lt_instance,
        scenario_indices=[0],
        time_limit=10,
        threads=1,
        verbose=False,
    )
    out_rp = small_lt_problem.solve_robust_bigmilp(
        small_lt_instance,
        scenario_indices=[0, 1],
        time_limit=10,
        threads=1,
        verbose=False,
    )

    assert out_r["has_incumbent"] is True
    assert out_rp["has_incumbent"] is True
    assert out_r["obj"] <= out_rp["obj"] + 1e-6


def test_cflp_monotonicity_with_solution_change():
    """
    Motivation:
    - Even if the first-stage solution changes when adding scenarios,
      the robust objective should be non-decreasing.
    """
    _require_gurobi()
    problem = CFLPProblem(n_facilities=2, n_customers=2, n_scenarios=2)

    f = np.array([1.0, 3.0], dtype=np.float32)
    a = np.array([0.0, 0.0], dtype=np.float32)
    K = np.array([5.0, 5.0], dtype=np.float32)
    c = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    demand = np.array([[3.0, 2.0], [5.0, 5.0]], dtype=np.float32)

    inst = Instance(
        {
            "n_facilities": np.array([2], dtype=np.int64),
            "n_customers": np.array([2], dtype=np.int64),
            "n_scenarios": np.array([2], dtype=np.int64),
            "f": f,
            "a": a,
            "K": K,
            "c": c,
            "d": demand,
        }
    )

    out_low = problem.solve_robust_bigmilp(inst, scenario_indices=[0], time_limit=10, threads=1)
    out_robust = problem.solve_robust_bigmilp(inst, scenario_indices=[0, 1], time_limit=10, threads=1)

    assert out_low["has_incumbent"] is True
    assert out_robust["has_incumbent"] is True
    assert np.allclose(out_low["y"], np.array([1.0, 0.0], dtype=float))
    assert np.allclose(out_robust["y"], np.array([1.0, 1.0], dtype=float))
    assert out_low["obj"] <= out_robust["obj"] + 1e-6


def test_cflp_single_scenario_equivalence():
    """
    Motivation:
    - With a single scenario, the robust objective should match the deterministic
      two-stage objective for that scenario.
    """
    _require_gurobi()
    problem = CFLPProblem(n_facilities=2, n_customers=1, n_scenarios=1)

    f = np.array([1.0, 10.0], dtype=np.float32)
    a = np.array([1.0, 1.0], dtype=np.float32)
    K = np.array([10.0, 10.0], dtype=np.float32)
    c = np.array([[1.0], [1.0]], dtype=np.float32)
    demand = np.array([[5.0]], dtype=np.float32)

    inst = Instance(
        {
            "n_facilities": np.array([2], dtype=np.int64),
            "n_customers": np.array([1], dtype=np.int64),
            "n_scenarios": np.array([1], dtype=np.int64),
            "f": f,
            "a": a,
            "K": K,
            "c": c,
            "d": demand,
        }
    )

    out = problem.solve_robust_bigmilp(inst, scenario_indices=[0], time_limit=10, threads=1)
    assert out["has_incumbent"] is True

    eval_out = problem.evaluate_full_solution(inst, y=out["y"], z=out["z"], time_limit=10, threads=1)
    assert abs(float(out["obj"]) - float(eval_out["obj"])) <= 1e-6
    assert np.allclose(out["y"], np.array([1.0, 0.0], dtype=float))
    assert np.allclose(out["z"], np.array([5.0, 0.0], dtype=float), atol=1e-6)
    assert abs(float(out["obj"]) - 11.0) <= 1e-6


def test_cflp_two_stage_beats_single_stage_worst_case_plan():
    """
    Motivation:
    - Two-stage CFLP should be strictly better than a "single-stage heavy" plan
      that must satisfy the per-customer worst-case demand simultaneously.
    - This demonstrates the value of deferring assignment decisions to recourse.
    """
    _require_gurobi()
    problem = CFLPProblem(n_facilities=2, n_customers=2, n_scenarios=2)

    f = np.array([0.0, 0.0], dtype=np.float32)
    a = np.array([0.0, 0.0], dtype=np.float32)
    K = np.array([100.0, 100.0], dtype=np.float32)
    c = np.array([[1.0, 100.0], [100.0, 1.0]], dtype=np.float32)
    demand = np.array([[10.0, 1.0], [1.0, 10.0]], dtype=np.float32)

    inst = Instance(
        {
            "n_facilities": np.array([2], dtype=np.int64),
            "n_customers": np.array([2], dtype=np.int64),
            "n_scenarios": np.array([2], dtype=np.int64),
            "f": f,
            "a": a,
            "K": K,
            "c": c,
            "d": demand,
        }
    )

    # Two-stage robust objective (scenario-wise recourse)
    out_2s = problem.solve_robust_bigmilp(inst, scenario_indices=[0, 1], time_limit=10, threads=1)
    assert out_2s["has_incumbent"] is True

    # "Single-stage heavy" plan: satisfy per-customer worst-case demand in one scenario.
    demand_max = np.max(demand, axis=0, keepdims=True)
    inst_max = Instance(
        {
            "n_facilities": np.array([2], dtype=np.int64),
            "n_customers": np.array([2], dtype=np.int64),
            "n_scenarios": np.array([1], dtype=np.int64),
            "f": f,
            "a": a,
            "K": K,
            "c": c,
            "d": demand_max,
        }
    )
    out_single = problem.solve_robust_bigmilp(inst_max, scenario_indices=[0], time_limit=10, threads=1)
    assert out_single["has_incumbent"] is True

    assert float(out_2s["obj"]) < float(out_single["obj"])


def test_cflp_two_stage_beats_worst_case_when_facilities_expensive():
    """
    Motivation:
    - Even with expensive facility openings, two-stage recourse can be cheaper
      than a single-stage plan built for per-customer worst-case demand.
    - Capacity limits make the worst-case plan open more facilities than needed
      for any realized scenario.
    """
    _require_gurobi()
    problem = CFLPProblem(n_facilities=2, n_customers=2, n_scenarios=2)

    f = np.array([1000.0, 1000.0], dtype=np.float32)
    a = np.array([0.0, 0.0], dtype=np.float32)
    K = np.array([15.0, 15.0], dtype=np.float32)
    c = np.array([[1.0, 5.0], [5.0, 1.0]], dtype=np.float32)
    demand = np.array([[10.0, 1.0], [1.0, 10.0]], dtype=np.float32)

    inst = Instance(
        {
            "n_facilities": np.array([2], dtype=np.int64),
            "n_customers": np.array([2], dtype=np.int64),
            "n_scenarios": np.array([2], dtype=np.int64),
            "f": f,
            "a": a,
            "K": K,
            "c": c,
            "d": demand,
        }
    )

    out_2s = problem.solve_robust_bigmilp(inst, scenario_indices=[0, 1], time_limit=10, threads=1)
    assert out_2s["has_incumbent"] is True

    demand_max = np.max(demand, axis=0, keepdims=True)
    inst_max = Instance(
        {
            "n_facilities": np.array([2], dtype=np.int64),
            "n_customers": np.array([2], dtype=np.int64),
            "n_scenarios": np.array([1], dtype=np.int64),
            "f": f,
            "a": a,
            "K": K,
            "c": c,
            "d": demand_max,
        }
    )
    out_single = problem.solve_robust_bigmilp(inst_max, scenario_indices=[0], time_limit=10, threads=1)
    assert out_single["has_incumbent"] is True

    assert float(out_2s["obj"]) < float(out_single["obj"])


def test_cflp_infeasible_recourse():
    """
    Motivation:
    - Without slack, recourse should be infeasible when demand exceeds capacity.
    - Guards against silently accepting under-supply.
    """
    _require_gurobi()
    problem = CFLPProblem(n_facilities=1, n_customers=1, n_scenarios=1)

    f = np.array([1.0], dtype=np.float32)
    a = np.array([1.0], dtype=np.float32)
    K = np.array([2.0], dtype=np.float32)
    c = np.array([[1.0]], dtype=np.float32)
    demand = np.array([[5.0]], dtype=np.float32)  # demand > capacity

    inst = Instance(
        {
            "n_facilities": np.array([1], dtype=np.int64),
            "n_customers": np.array([1], dtype=np.int64),
            "n_scenarios": np.array([1], dtype=np.int64),
            "f": f,
            "a": a,
            "K": K,
            "c": c,
            "d": demand,
        }
    )

    rec_out = problem.solve_recourse_lp(
        inst,
        scenario_index=0,
        y=np.array([1.0], dtype=float),
        z=np.array([2.0], dtype=float),
        time_limit=10,
        threads=1,
        return_solution=True,
    )
    assert rec_out["has_incumbent"] is False


def test_cflp_sample_ranges_and_feasibility():
    """
    Motivation:
    - Default generator should follow Zeng & Zhao parameter ranges.
    - Feasibility guarantee: total capacity >= worst-case scenario total demand.
    """
    problem = CFLPProblem(n_facilities=5, n_customers=4, n_scenarios=3)
    inst = problem.sample()

    f = inst.data["f"]
    a = inst.data["a"]
    K = inst.data["K"]
    c = inst.data["c"]
    demand = inst.data["d"]

    assert np.all(f >= 100.0 - 1e-6)
    assert np.all(f <= 1000.0 + 1e-6)
    assert np.all(a >= 10.0 - 1e-6)
    assert np.all(a <= 100.0 + 1e-6)
    assert np.all(K >= 200.0 - 1e-6)
    assert np.all(K <= 700.0 + 1e-6)
    assert np.all(c >= 1.0 - 1e-6)
    assert np.all(c <= 1000.0 + 1e-6)
    assert np.all(demand >= 5.0 - 1e-6)
    assert np.all(demand <= 750.0 + 1e-6)

    total_capacity = float(np.sum(K))
    max_total_demand = float(np.max(np.sum(demand, axis=1)))
    assert total_capacity + 1e-6 >= max_total_demand


def test_cflp_high_fixed_cost_prefers_single_facility():
    """
    Motivation:
    - High fixed cost should discourage opening an otherwise-viable facility.
    - Confirms first-stage decisions react to fixed costs.
    """
    _require_gurobi()
    problem = CFLPProblem(n_facilities=2, n_customers=2, n_scenarios=1)

    f = np.array([1.0, 100.0], dtype=np.float32)
    a = np.array([0.0, 0.0], dtype=np.float32)
    K = np.array([20.0, 20.0], dtype=np.float32)
    c = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    demand = np.array([[5.0, 5.0]], dtype=np.float32)

    inst = Instance(
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

    out = problem.solve_robust_bigmilp(inst, scenario_indices=[0], time_limit=10, threads=1)
    assert out["has_incumbent"] is True
    assert np.allclose(out["y"], np.array([1.0, 0.0], dtype=float))


def test_cflp_high_capacity_cost_prefers_cheaper_capacity():
    """
    Motivation:
    - Higher unit capacity cost should shift capacity to the cheaper facility.
    - Confirms a_i affects the optimal z_i allocation.
    """
    _require_gurobi()
    problem = CFLPProblem(n_facilities=2, n_customers=2, n_scenarios=1)

    f = np.array([1.0, 1.0], dtype=np.float32)
    a = np.array([10.0, 1.0], dtype=np.float32)
    K = np.array([20.0, 20.0], dtype=np.float32)
    c = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    demand = np.array([[6.0, 4.0]], dtype=np.float32)

    inst = Instance(
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

    out = problem.solve_robust_bigmilp(inst, scenario_indices=[0], time_limit=10, threads=1)
    assert out["has_incumbent"] is True
    assert np.allclose(out["y"], np.array([0.0, 1.0], dtype=float))
    assert np.allclose(out["z"], np.array([0.0, 10.0], dtype=float), atol=1e-6)


def test_cflp_transport_cost_opens_both():
    """
    Motivation:
    - Divergent transport costs should open both facilities to avoid expensive routes.
    - Validates recourse costs drive first-stage choices.
    """
    _require_gurobi()
    problem = CFLPProblem(n_facilities=2, n_customers=2, n_scenarios=1)

    f = np.array([1.0, 1.0], dtype=np.float32)
    a = np.array([0.0, 0.0], dtype=np.float32)
    K = np.array([10.0, 10.0], dtype=np.float32)
    c = np.array([[0.0, 100.0], [100.0, 0.0]], dtype=np.float32)
    demand = np.array([[5.0, 5.0]], dtype=np.float32)

    inst = Instance(
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

    out = problem.solve_robust_bigmilp(inst, scenario_indices=[0], time_limit=10, threads=1)
    assert out["has_incumbent"] is True
    assert np.allclose(out["y"], np.array([1.0, 1.0], dtype=float))


def test_cflp_demand_volatility_opens_extra_facility():
    """
    Motivation:
    - Volatile demand should force opening extra capacity to cover the worst case.
    - Compares low-demand vs robust (low+high) decisions.
    """
    _require_gurobi()
    problem = CFLPProblem(n_facilities=2, n_customers=2, n_scenarios=2)

    f = np.array([1.0, 3.0], dtype=np.float32)
    a = np.array([0.0, 0.0], dtype=np.float32)
    K = np.array([5.0, 5.0], dtype=np.float32)
    c = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    demand = np.array([[3.0, 2.0], [5.0, 5.0]], dtype=np.float32)

    inst = Instance(
        {
            "n_facilities": np.array([2], dtype=np.int64),
            "n_customers": np.array([2], dtype=np.int64),
            "n_scenarios": np.array([2], dtype=np.int64),
            "f": f,
            "a": a,
            "K": K,
            "c": c,
            "d": demand,
        }
    )

    out_low = problem.solve_robust_bigmilp(inst, scenario_indices=[0], time_limit=10, threads=1)
    out_robust = problem.solve_robust_bigmilp(inst, scenario_indices=[0, 1], time_limit=10, threads=1)

    assert out_low["has_incumbent"] is True
    assert out_robust["has_incumbent"] is True
    assert np.allclose(out_low["y"], np.array([1.0, 0.0], dtype=float))
    assert np.allclose(out_robust["y"], np.array([1.0, 1.0], dtype=float))


def test_cflp_high_demand_increases_capacity():
    """
    Motivation:
    - Higher demand scenarios should increase installed capacity in the robust solution.
    - Validates that z adapts to the worst-case demand level.
    """
    _require_gurobi()
    problem = CFLPProblem(n_facilities=2, n_customers=2, n_scenarios=2)

    f = np.array([1.0, 5.0], dtype=np.float32)
    a = np.array([1.0, 1.0], dtype=np.float32)
    K = np.array([20.0, 5.0], dtype=np.float32)
    c = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    demand = np.array([[2.0, 3.0], [6.0, 4.0]], dtype=np.float32)

    inst = Instance(
        {
            "n_facilities": np.array([2], dtype=np.int64),
            "n_customers": np.array([2], dtype=np.int64),
            "n_scenarios": np.array([2], dtype=np.int64),
            "f": f,
            "a": a,
            "K": K,
            "c": c,
            "d": demand,
        }
    )

    out_low = problem.solve_robust_bigmilp(inst, scenario_indices=[0], time_limit=10, threads=1)
    out_robust = problem.solve_robust_bigmilp(inst, scenario_indices=[0, 1], time_limit=10, threads=1)

    assert out_low["has_incumbent"] is True
    assert out_robust["has_incumbent"] is True
    assert float(np.sum(out_low["z"])) < float(np.sum(out_robust["z"])) - 1e-6


def _zeng2013_vertices() -> np.ndarray:
    a_list = []
    b_list = []
    # -g_i <= 0
    a_list.append([-1.0, 0.0, 0.0])
    b_list.append(0.0)
    a_list.append([0.0, -1.0, 0.0])
    b_list.append(0.0)
    a_list.append([0.0, 0.0, -1.0])
    b_list.append(0.0)
    # g_i <= 1
    a_list.append([1.0, 0.0, 0.0])
    b_list.append(1.0)
    a_list.append([0.0, 1.0, 0.0])
    b_list.append(1.0)
    a_list.append([0.0, 0.0, 1.0])
    b_list.append(1.0)
    # g0 + g1 + g2 <= 1.8
    a_list.append([1.0, 1.0, 1.0])
    b_list.append(1.8)
    # g0 + g1 <= 1.2
    a_list.append([1.0, 1.0, 0.0])
    b_list.append(1.2)

    A = np.asarray(a_list, dtype=float)
    b = np.asarray(b_list, dtype=float)
    n_constraints = A.shape[0]

    vertices = []
    tol = 1e-8
    for idx in range(n_constraints):
        for jdx in range(idx + 1, n_constraints):
            for kdx in range(jdx + 1, n_constraints):
                A_eq = np.vstack([A[idx], A[jdx], A[kdx]])
                b_eq = np.array([b[idx], b[jdx], b[kdx]], dtype=float)
                if np.linalg.matrix_rank(A_eq) < 3:
                    continue
                try:
                    sol = np.linalg.solve(A_eq, b_eq)
                except np.linalg.LinAlgError:
                    continue
                if np.all(A @ sol <= b + tol) and np.all(sol >= -tol):
                    vertices.append(sol)

    if not vertices:
        raise RuntimeError("No vertices found for Zeng2013 uncertainty set.")

    unique = np.unique(np.round(np.asarray(vertices), 8), axis=0)
    return unique


def test_cflp_zeng2013_discrete_extremes_matches_optimal():
    """
    Motivation:
    - Uses extreme points of Zeng & Zhao's polyhedral demand set as scenarios.
    - Robust discrete solution should match the reported optimal value (33680).
    """
    _require_gurobi()
    vertices = _zeng2013_vertices()
    demand = 206.0 + 40.0 * vertices[:, 0]
    demand = np.stack([demand, 274.0 + 40.0 * vertices[:, 1], 220.0 + 40.0 * vertices[:, 2]], axis=1)

    f = np.array([400.0, 414.0, 326.0], dtype=np.float32)
    a = np.array([18.0, 25.0, 20.0], dtype=np.float32)
    K = np.array([800.0, 800.0, 800.0], dtype=np.float32)
    c = np.array(
        [
            [22.0, 33.0, 24.0],
            [33.0, 23.0, 30.0],
            [20.0, 25.0, 27.0],
        ],
        dtype=np.float32,
    )

    inst = Instance(
        {
            "n_facilities": np.array([3], dtype=np.int64),
            "n_customers": np.array([3], dtype=np.int64),
            "n_scenarios": np.array([demand.shape[0]], dtype=np.int64),
            "f": f,
            "a": a,
            "K": K,
            "c": c,
            "d": demand.astype(np.float32),
        }
    )

    problem = CFLPProblem(n_facilities=3, n_customers=3, n_scenarios=int(demand.shape[0]))
    out = problem.solve_robust_bigmilp(inst, scenario_indices=None, time_limit=30, threads=1)

    assert out["has_incumbent"] is True
    assert abs(float(out["obj"]) - 33680.0) <= 1e-3
