"""Tables 15/16 (scenario-count runtime): truncation keeps the first S scenarios; ratios are of sums."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import scripts.eval.scenario_runtime as sr  # noqa: E402
from src.core.types import Instance  # noqa: E402


def test_truncate_keeps_first_s_scenarios_and_rest_of_instance():
    sce = np.arange(12.0).reshape(6, 2)
    got = sr.truncate(Instance({"sce": sce, "det": np.ones(2), "n_scenarios": np.array([6])}), 4)
    assert np.array_equal(got.data["sce"], sce[:4]) and got.data["n_scenarios"].tolist() == [4]
    assert np.array_equal(got.data["det"], np.ones(2))
    with pytest.raises(AssertionError):
        sr.truncate(got, 5)


def test_ratios_divide_component_sums_by_the_smallest_s():
    cond = lambda inf, sol: {"stats": {"inference_sum": inf, "solve_sum": sol, "total_sum": inf + sol}}
    got = sr.ratios({"20": cond(1.0, 50.0), "100": cond(5.0, 60.0)})  # key order is not S order
    assert got["20"] == {"inference": 1.0, "solve": 1.0, "total": 1.0}
    assert got["100"] == pytest.approx({"inference": 5.0, "solve": 1.2, "total": 65 / 51})
