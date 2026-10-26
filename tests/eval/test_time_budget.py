"""Table 14 (time-budgeted MILP): the incumbent's gap to V(Xi), and an infinite gap without one."""
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import scripts.eval.time_budget as tb  # noqa: E402


class Stub:
    def __init__(self, out):
        self.out, self.calls = out, []

    def solve_robust_bigmilp(self, inst, **kw):
        self.calls.append(kw)
        return self.out


def test_regret_of_the_incumbent_under_the_given_limit():
    stub = Stub({"has_incumbent": True, "obj": 110.0, "status": 9, "runtime_sec": 1.0})
    got = tb.solve_one(stub, None, 100.0, tb.BUDGET_S[4], threads=40, mip_gap=1e-4)
    assert stub.calls == [{"time_limit": 1, "mip_gap": 1e-4, "threads": 40}]
    assert got["gap_pct"] == pytest.approx(10.0) and got["status"] == 9 and got["full_obj"] == 100.0


def test_no_incumbent_has_an_infinite_gap():
    stub = Stub({"has_incumbent": False, "obj": None, "status": 9, "runtime_sec": 2.0})
    got = tb.solve_one(stub, None, 100.0, tb.BUDGET_S[6], threads=40, mip_gap=1e-4)
    assert got["obj_value"] == got["gap_pct"] == math.inf and stub.calls[0]["time_limit"] == 2
