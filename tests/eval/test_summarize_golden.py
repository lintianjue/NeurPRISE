"""Golden tests: ``save_results``'s regret summary statistics come from
``src.eval.regret.summarize`` instead of being hand-computed inline.

``tests/fixtures/regret_summary_golden/gen.py`` ran ONCE, before the statistics moved into
``summarize``, and froze ``save_results``'s output (minus the three volatile
timestamp/host keys) for six per-instance record sets. These tests re-run the
identical procedure through the refactored code and demand byte-identical
output — any change to the statistics (a different denominator, a dropped
key, a different feasibility filter) fails here first. The retired scenario-gap
fields have been removed from these synthetic fixtures; the remaining values are frozen.
"""
import importlib.util
import json
import os

import numpy as np
import pytest

from scripts.eval import eval_regret as er
from src.eval import regret

_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures", "regret_summary_golden")
_VOLATILE_KEYS = ("eval_date", "eval_server", "eval_machine_id")


def _load_cases():
    """Load ``records.CASES`` by file path — see ``gen.py`` for why not
    ``from tests.fixtures.regret_summary_golden.records import CASES``: a sibling repo on
    ``PYTHONPATH`` may ship its own regular ``tests/__init__.py``,
    which PEP 420 lets win over this repo's namespace-package ``tests/`` dir.
    """
    path = os.path.join(_FIXTURE_DIR, "records.py")
    spec = importlib.util.spec_from_file_location("regret_summary_golden_records", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CASES


CASES = _load_cases()


def _run_case(tmp_path, monkeypatch, rows):
    """Same procedure as gen.py: save via save_results, read the file back,
    strip the volatile keys. Returns the parsed object.

    ``save_results`` writes under ``eval_regret.BASE`` via ``assert_within_base``,
    so BASE is redirected to ``tmp_path`` too, not just REGRET_DIR.
    """
    monkeypatch.setattr(er, "BASE", str(tmp_path))
    monkeypatch.setattr(er, "REGRET_DIR", os.path.join(str(tmp_path), "eval_regret"))
    monkeypatch.setattr(er, "RUN", None)
    name = er._result_name("selection", "small", "neurprise", k=1)
    er.save_results(rows, "selection", "small", "neurprise", k=1)
    out_path = os.path.join(er.REGRET_DIR, "selection", name)
    with open(out_path) as f:
        obj = json.loads(f.read())
    for key in _VOLATILE_KEYS:
        obj.pop(key, None)
    return obj


@pytest.mark.parametrize("case", sorted(CASES))
def test_save_results_matches_golden(tmp_path, monkeypatch, case):
    """(M) Kill: change np.std to np.var in summarize, or drop regret_median."""
    obj = _run_case(tmp_path, monkeypatch, CASES[case])

    fixture_path = os.path.join(_FIXTURE_DIR, f"{case}.json")
    with open(fixture_path) as f:
        fixture_text = f.read()
    fixture_obj = json.loads(fixture_text)

    assert obj == fixture_obj
    assert json.dumps(obj, indent=2, sort_keys=True) == fixture_text


def test_summarize_keys_and_infeasible_semantics():
    """(M) Kill: `feasible = regrets` (dropping the isfinite filter)."""
    with_inf = CASES["with_inf"]
    summary = regret.summarize(with_inf)
    assert summary["n_infeasible"] == 1
    assert summary["regret_mean"] is None
    assert summary["regret_mean_feasible_only"] is not None
    assert np.isfinite(summary["regret_mean_feasible_only"])

    empty = regret.summarize([])
    assert empty["n_instances"] == 0
    assert empty["n_infeasible"] == 0
    stat_keys = ("regret_mean", "regret_std",
                 "regret_median", "regret_mean_feasible_only")
    for key in stat_keys:
        assert empty[key] is None, f"{key} should be None for an empty result set, got {empty[key]!r}"


def test_reexports_are_the_library_objects():
    """Kill: a local redefinition (of either name) in eval_regret.py."""
    assert er._round_or_none is regret.round_or_none
    assert er._time_components is regret.time_components
