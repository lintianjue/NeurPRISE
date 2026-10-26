"""``run_neurprise_regret`` resolves ``MIP_GAP`` at call time.

Its two sibling wrappers (``get_neurprise_x`` / ``true_regret_single``, and the
library functions in ``src/eval/regret.py``) already read the module globals
``THREADS`` / ``TIME_LIMIT`` / ``MIP_GAP`` at call time instead of binding a
default at import time -- the same trap this closes for ``mip_gap``. No caller
assigns ``eval_regret.MIP_GAP`` today (so no current result changes), but a
default of ``mip_gap=MIP_GAP`` would silently ignore one that did.

No Gurobi, no real data: every collaborator ``run_neurprise_regret`` calls is
monkeypatched so only the mip_gap plumbing is under test.
"""
import os
import pickle

import numpy as np
import pytest

import scripts.eval.eval_regret as er


def test_run_neurprise_regret_reads_MIP_GAP_at_call_time(tmp_path, monkeypatch):
    """(M) Kill: restoring `mip_gap=MIP_GAP` in the signature (recorded value
    becomes 1e-4, the import-time default, instead of the patched 0.01).
    """
    monkeypatch.setattr(er, "INPUT_ROOT", str(tmp_path))

    cfg = er.PROBLEM_CONFIGS[("selection", "small")]
    data_path = tmp_path / cfg["data"]
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_path, "wb") as f:
        pickle.dump([object()], f)

    k_ratio = er.K_RATIOS[cfg["n_scenarios"]][1]
    eval_base = tmp_path / "eb"
    idx_dir = eval_base / f"k_{k_ratio}"
    idx_dir.mkdir(parents=True)
    np.save(idx_dir / "all_idx.npy", np.array([0]))

    monkeypatch.setattr(er, "make_problem", lambda *a, **k: None)
    monkeypatch.setattr(er, "find_neurprise_eval_base", lambda *a, **k: str(eval_base))
    monkeypatch.setattr(er, "load_full_objs", lambda *a, **k: [1.0])

    recorded = {}

    def _fake_solve_reduced_milp(prob, inst, sce_idx, problem_name="vertex_cover", *,
                                 mip_gap=None, threads=None, time_limit=None, return_raw=False):
        # Fakes the LIBRARY solver under get_neurprise_x, so the test sees the
        # value get_neurprise_x resolved, not the value run_neurprise_regret passed.
        recorded["mip_gap"] = mip_gap
        # sol=None => the (problem_name != "cflp_ebp") branch `continue`s,
        # so run_neurprise_regret never touches red_obj/solve_out below.
        return (None, 0.0, {})

    monkeypatch.setattr(er, "_core_solve_reduced_milp", _fake_solve_reduced_milp)
    monkeypatch.setattr(er, "MIP_GAP", 0.01)

    result = er.run_neurprise_regret("selection", "small", 1)

    assert result == []  # every row skipped (sol is None)
    assert recorded["mip_gap"] == pytest.approx(0.01)
