"""Fast tests for shifted-family dispatch and true-regret solver binding."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


THREADS = 40
TIME_LIMIT = 600


def _load_eval_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "04_05_eval_ml_2ro_e2e.py"
    spec = importlib.util.spec_from_file_location("eval_script_for_binding_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _reduced_consumer_reads_globals():
    return THREADS, TIME_LIMIT


def _recourse_consumer_reads_threads():
    return THREADS


def _consumer_binds_defaults(threads=THREADS, time_limit=TIME_LIMIT):
    return threads, time_limit


@pytest.mark.parametrize(
    ("config_name", "family"),
    [
        ("cflp_ebp_normal", "cflp_ebp"),
        ("cflp_ebp_mm", "cflp_ebp"),
        ("vertex_cover_normal", "vertex_cover"),
        ("selection_mm", "selection"),
    ],
)
def test_shifted_config_dispatches_to_true_regret_family(config_name, family):
    # Motivation: shifted CFLP previously took the VC/SEL branch and looked plausibly infeasible.
    module = _load_eval_module()
    assert module.canonical_true_regret_problem_name(config_name) == family


def test_unknown_true_regret_family_fails_closed():
    # Motivation: an unrecognised problem must not silently enter a structurally wrong branch.
    module = _load_eval_module()
    with pytest.raises(ValueError, match="Unsupported true-regret problem family"):
        module.canonical_true_regret_problem_name("mystery_shift")


def test_solver_globals_are_set_and_read_back():
    # Motivation: Hydra solver values alone do not reach eval_regret's module-global consumers.
    module = _load_eval_module()
    fake = SimpleNamespace(
        THREADS=40,
        TIME_LIMIT=600,
        get_neurprise_x=_reduced_consumer_reads_globals,
        evaluate_x_on_all_scenarios=_recourse_consumer_reads_threads,
    )
    effective = module.configure_true_regret_solver(fake, threads=12, time_limit=900)
    assert effective == {"threads": 12, "time_limit": 900.0}
    assert (fake.THREADS, fake.TIME_LIMIT) == (12, 900.0)


def test_real_true_regret_consumers_accept_and_read_binding():
    # Motivation: prove the production consumers, not only the fake, load globals at call time.
    from scripts.eval import eval_regret

    module = _load_eval_module()
    previous = (eval_regret.THREADS, eval_regret.TIME_LIMIT)
    try:
        effective = module.configure_true_regret_solver(
            eval_regret, threads=12, time_limit=900
        )
        assert effective == {"threads": 12, "time_limit": 900.0}
    finally:
        eval_regret.THREADS, eval_regret.TIME_LIMIT = previous


def test_default_bound_consumer_is_rejected_before_setting_globals():
    # Motivation: this planted silent failure ignores later module-global assignments.
    module = _load_eval_module()
    fake = SimpleNamespace(
        THREADS=40,
        TIME_LIMIT=600,
        get_neurprise_x=_consumer_binds_defaults,
        evaluate_x_on_all_scenarios=_recourse_consumer_reads_threads,
    )
    with pytest.raises(RuntimeError, match="does not read"):
        module.configure_true_regret_solver(fake, threads=12, time_limit=900)
    assert (fake.THREADS, fake.TIME_LIMIT) == (40, 600)
