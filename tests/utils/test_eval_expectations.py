"""`eval.expect.*` in the ML evaluator: declared expectations about the evaluated split.

An evaluation that loads a smaller label file than the one it was meant to (e.g. a
50-record file under the 500-record file's name) must refuse to run, not report a
regret number computed on 10% of the population. Unset expectations stay inert.
"""
from __future__ import annotations

import pytest
from omegaconf import OmegaConf


@pytest.fixture(scope="session")
def evaluator(producer_module):
    """The real evaluator, imported by file location via the repo's producer fixture.

    Its module name starts with a digit, so `import scripts.04_05_...` is impossible;
    `tests/conftest.py::producer_module` is the repo's single way to reach it, and using
    it keeps this test anchored on the producer rather than on a copy of its logic.
    """
    return producer_module("scripts/04_05_eval_ml_2ro_e2e.py")


class _DS(list):
    pass


def _dm(tmp_path, population):
    class _DM:
        data_file = str(tmp_path / "absent.pkl")
        full_ds = _DS(range(population))
    return _DM()


def test_unset_expectations_stay_inert(tmp_path, evaluator):
    """Every pre-existing command must keep behaving exactly as before."""
    cfg = OmegaConf.create({"data": {"num_workers": 12}, "eval": {"num_workers": 12}})
    observed = evaluator.assert_eval_expectations(cfg, _dm(tmp_path, 7), 3)
    assert observed["population"] == 7 and observed["split_instances"] == 3
    assert "label_sha256" not in observed  # never hashed when unasserted


def test_declared_expectations_reject_a_short_split(tmp_path, evaluator):
    cfg = OmegaConf.create({"data": {"num_workers": 0}, "eval": {
        "num_workers": 0, "expect": {"population": 500, "test_instances": 50}}})
    with pytest.raises(RuntimeError, match="eval.expect assertions failed"):
        evaluator.assert_eval_expectations(cfg, _dm(tmp_path, 50), 5)


def test_num_workers_is_asserted_not_merely_set(tmp_path, evaluator):
    cfg = OmegaConf.create({"data": {"num_workers": 12}, "eval": {
        "num_workers": 12, "expect": {"num_workers": 0}}})
    with pytest.raises(RuntimeError, match=r"eval\.expect\.num_workers=0"):
        evaluator.assert_eval_expectations(cfg, _dm(tmp_path, 7), 3)
