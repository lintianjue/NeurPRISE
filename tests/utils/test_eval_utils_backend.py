import numpy as np

from src.core.types import Labels
from src.utils.eval_utils import _eval_single_reduced_instance


class _DummyProblem:
    def __init__(self, *, backend: str, full_obj_source: str):
        self.exact_backend = backend
        self.eval_full_obj_source = full_obj_source
        self.solve_calls = []

    def get_scenarios(self, data_np):
        return np.asarray(data_np["sce"])

    def solve_exact(self, instance, time_limit=None, threads=1, verbose=False, mip_gap=None):
        sce = np.asarray(instance.data["sce"])
        n_scenarios = int(sce.shape[0])
        self.solve_calls.append(n_scenarios)
        obj = 100.0 if n_scenarios == 3 else 90.0
        return Labels(
            obj=obj,
            feasible=True,
            obj_bound=obj - 1.0,
            mip_gap=0.0,
        )


def _make_data_np():
    return {
        "sce": np.array(
            [
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
            ],
            dtype=float,
        ),
        "n_scenarios": np.array([3], dtype=int),
    }


def test_eval_single_reduced_instance_uses_label_full_obj_by_default():
    # Motivation: preserve historical behavior where full objective comes from labels unless overridden.
    problem = _DummyProblem(backend="bigmilp", full_obj_source="label")
    full_obj, red_obj, regret, feasible, obj_bound, mip_gap = _eval_single_reduced_instance(
        problem=problem,
        data_np=_make_data_np(),
        sce_idx=np.array([0, 1], dtype=int),
        full_obj_val=55.0,
        milp_time_limit=10.0,
        threads=1,
        verbose=False,
    )
    assert full_obj == 55.0
    assert red_obj == 90.0
    assert regret == 35.0
    assert feasible is True
    assert obj_bound == 89.0
    assert mip_gap == 0.0
    assert problem.solve_calls == [2]


def test_eval_single_reduced_instance_recomputes_full_obj_for_ccg_backend():
    # Motivation: when backend is CCG, full objective should also be solved by CCG for backend-consistent regrets.
    problem = _DummyProblem(backend="ccg", full_obj_source="label")
    full_obj, red_obj, regret, feasible, obj_bound, mip_gap = _eval_single_reduced_instance(
        problem=problem,
        data_np=_make_data_np(),
        sce_idx=np.array([0, 1], dtype=int),
        full_obj_val=55.0,
        milp_time_limit=10.0,
        threads=1,
        verbose=False,
    )
    assert full_obj == 100.0
    assert red_obj == 90.0
    assert regret == -10.0
    assert feasible is True
    assert obj_bound == 89.0
    assert mip_gap == 0.0
    assert problem.solve_calls == [3, 2]


def test_eval_single_reduced_instance_recomputes_full_obj_when_source_is_solve():
    # Motivation: explicit full_obj_source=solve should trigger full solve even under BigMILP backend.
    problem = _DummyProblem(backend="bigmilp", full_obj_source="solve")
    full_obj, red_obj, regret, *_ = _eval_single_reduced_instance(
        problem=problem,
        data_np=_make_data_np(),
        sce_idx=np.array([0, 1], dtype=int),
        full_obj_val=55.0,
        milp_time_limit=10.0,
        threads=1,
        verbose=False,
    )
    assert full_obj == 100.0
    assert red_obj == 90.0
    assert regret == -10.0
    assert problem.solve_calls == [3, 2]
