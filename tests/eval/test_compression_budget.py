"""Table 7 (compression budget k̂): the k̂ rule, the table statistics, and the PRISE rows."""
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import scripts.eval.compression_budget as cb  # noqa: E402


def test_khat_is_first_step_within_one_percent():
    assert cb.khat([90, 98, 99.5, 100], 100) == 3          # gap 10%, 2%, 0.5%
    assert cb.khat([99], 100) == 1                         # exactly 1% converges
    assert cb.khat([90, 102, 100], 100) == 3               # |gap|: a 2% overshoot is not within 1%
    assert cb.khat([90] * 8, 100) is None


def test_gains_nonincreasing_pools_consecutive_pairs():
    trace = lambda g: {"label": type("L", (), {"forward_trace": type("F", (), {"gains": g})})}
    got = cb.gains_nonincreasing([trace([5, 3, 3, 4]), trace([2, 1])])  # pairs: >=, >=, <; >=
    assert (got["hits"], got["total"], got["frac"]) == (3, 4, 0.75)


def test_row_counts_non_converged_as_K_and_converged_at_K_as_converged():
    # |Ξ| = 50: k̂ <= 1, 2, 4, 6 are the 2/4/8/12% columns.
    row = cb.row("T", "M", [1, 2, 8, None], n_scenarios=50)
    assert row == {"problem": "T", "method": "M",
                   "mean": 100 * (1 + 2 + 8 + 8) / 4 / 50,  # 9.5
                   "median": 100 * 5 / 50,                 # median of 1, 2, 8, 8
                   "pct_le_2": 25.0, "pct_le_4": 50.0, "pct_le_8": 50.0, "pct_le_12": 50.0,
                   "not_converged": 25.0}


# Table 7 as printed (Mean, Med., <=2, <=4, <=8, <=12, N/C).
PRISE_ROWS = {"SEL-20-50": [4.2, 4.0, 34.4, 79.6, 95.2, 96.4, 3.6],
              "VC-20-50": [7.8, 8.0, 1.2, 18.8, 71.6, 91.6, 4.4]}


def test_prise_rows_reproduce_from_release_labels(tmp_path, monkeypatch, capsys):
    root = os.environ["NEURPRISE_INPUT_ROOT"]
    needed = ["eval_regret/full_objs_cache.json",
              *(cb.PROBLEM_CONFIGS[p]["data"].replace("/inst_", "/label_prise_") for p in cb.PROBLEMS),
              *(cb.PROBLEM_CONFIGS[p]["data"] for p in cb.PROBLEMS)]
    if not all(os.path.isfile(os.path.join(root, f)) for f in needed):
        pytest.skip("released data not under $NEURPRISE_INPUT_ROOT (scripts/reproduce.sh download)")
    monkeypatch.setenv("NEURPRISE_RUN_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["compression_budget.py", "--input-root", root,
                                      "--run-id", "t", "--methods", "PRISE"])
    assert cb.main() == 0
    out = json.loads((tmp_path / "compression_budget/t/compression_budget.json").read_text())
    got = {r["problem"]: [round(r[c], 1) for c in out["columns"]] for r in out["rows"]}
    assert got == PRISE_ROWS
    # "Gains are empirically non-increasing in 93--99% of consecutive PRISE steps."
    assert [round(100 * out["gains_nonincreasing"][t]["frac"]) for t in PRISE_ROWS] == [99, 93]
    assert out["gains_nonincreasing"]["VC-20-50"]["total"] == 2500 * 7  # all labels, not the test split
