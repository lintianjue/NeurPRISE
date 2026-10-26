"""``--eval-base`` lets the T2 re-run point the regret evaluator directly at
the e2e run directory that just produced the selections, instead of falling
back to the glob search in ``find_neurprise_eval_base`` over
``INPUT_ROOT/eval/...`` (which can pick a different, stale directory).

No Gurobi, no real data: same fake-collaborator idiom as
``tests/eval/test_regret_timing_fields.py`` / ``tests/scripts/test_eval_regret_mip_gap_call_time.py``.
"""
import pytest
import argparse
import json
import pickle

import numpy as np

from scripts.eval import eval_regret as er


def _base_args(**overrides):
    """The subset of ``main()``'s argparse namespace ``_print_dry_run`` and
    ``_run_neurprise`` read."""
    ns = argparse.Namespace(
        problem="selection", scale="small", method="neurprise", k=1,
        mip_gap=None, threads=40, server=None, eval_base=None,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_eval_base_flag_reaches_run_neurprise_regret_and_read_selection_time(tmp_path, monkeypatch):
    """(M) flag parsed but not forwarded: ``_run_neurprise`` keeps calling
    ``run_neurprise_regret`` without ``eval_base_override``, so a fake,
    unrelated ``find_neurprise_eval_base`` result would be used instead and
    the explicit directory below would never be read.
    """
    monkeypatch.setattr(er, "INPUT_ROOT", str(tmp_path))
    cfg = er.PROBLEM_CONFIGS[("selection", "small")]
    data_path = tmp_path / cfg["data"]
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_path, "wb") as f:
        pickle.dump([object()], f)

    k_ratio = er.K_RATIOS[cfg["n_scenarios"]][1]
    eval_base = tmp_path / "explicit_e2e_base"
    idx_dir = eval_base / f"k_{k_ratio}"
    idx_dir.mkdir(parents=True)
    np.save(idx_dir / "all_idx.npy", np.array([0]))
    (idx_dir / "timing.json").write_text(json.dumps({"selection_time_total_sec": 2.5}))

    monkeypatch.setattr(er, "make_problem", lambda *a, **k: None)
    # A DIFFERENT directory the glob search would have picked -- if
    # --eval-base is not forwarded, the run reads this one instead and the
    # assertions below fail (no all_idx.npy under it -> ERROR/None results).
    monkeypatch.setattr(er, "find_neurprise_eval_base", lambda *a, **k: str(tmp_path / "wrong"))
    monkeypatch.setattr(er, "load_full_objs", lambda *a, **k: [10.0])
    monkeypatch.setattr(er, "assert_within_base", lambda *_a, **_k: None)

    def _fake_solve(prob, inst, sce_idx, problem_name="vertex_cover", *,
                     mip_gap=None, threads=None, time_limit=None, return_raw=False):
        return (np.array([1.0, 0.0]), 9.0,
                {"runtime_sec": 0.0, "obj_bound": None, "mip_gap": None, "status": "OPTIMAL"})

    monkeypatch.setattr(er, "_core_solve_reduced_milp", _fake_solve)
    monkeypatch.setattr(er, "_core_evaluate_x_on_all_scenarios", lambda *a, **k: 11.0)

    regret_dir = tmp_path / "eval_regret"
    regret_dir.mkdir()
    monkeypatch.setattr(er, "REGRET_DIR", str(regret_dir))

    args = _base_args(eval_base=str(eval_base))
    er._run_neurprise(args)

    out_path = regret_dir / "selection" / "SEL-20-50_neurprise_k1.json"
    summary = json.loads(out_path.read_text())
    assert summary["n_instances"] == 1
    assert summary["selection_time_total_sec"] == 2.5
    assert summary["selection_time_source"].endswith("timing.json")


def test_dry_run_prints_the_explicit_eval_base(tmp_path, monkeypatch, capsys):
    """(M) dry-run ignores it: ``_print_dry_run`` keeps calling
    ``find_neurprise_eval_base`` unconditionally, so the printed EVAL_BASE
    would be the glob result, not the explicit directory.
    """
    monkeypatch.setattr(er, "RUN", None)
    explicit = tmp_path / "explicit_dry_run_base"
    explicit.mkdir()
    monkeypatch.setattr(
        er, "find_neurprise_eval_base",
        lambda *a, **k: str(tmp_path / "should_not_be_printed"))

    args = _base_args(eval_base=str(explicit))
    er._print_dry_run(args)

    out = capsys.readouterr().out
    assert f"EVAL_BASE={explicit}" in out
    assert "should_not_be_printed" not in out


def test_explicit_eval_base_outside_input_root_is_refused(tmp_path, monkeypatch):
    """--eval-base is a read path: it must sit inside the declared input root (it is not
    a repo write, so the write guard does not apply). (M) drop the `_assert_inside`
    call in run_neurprise_regret -> the run proceeds and returns None/ERROR instead."""
    monkeypatch.setattr(er, "INPUT_ROOT", str(tmp_path / "in"))
    cfg = er.PROBLEM_CONFIGS[("selection", "small")]
    data_path = tmp_path / "in" / cfg["data"]
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_path, "wb") as f:
        pickle.dump([object()], f)
    monkeypatch.setattr(er, "make_problem", lambda *a, **k: None)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    with pytest.raises(RuntimeError, match="inside the input root"):
        er.run_neurprise_regret("selection", "small", 1, eval_base_override=str(outside))


def test_short_full_objs_is_a_clear_error_not_an_index_error(tmp_path, monkeypatch, capsys):
    """CFLP-70-70-100, T2 pass 2: an e2e csv with fewer rows than the indices raised
    IndexError deep in the loop. (M) drop the `len(full_objs) < n_avail` guard."""
    monkeypatch.setattr(er, "INPUT_ROOT", str(tmp_path))
    cfg = er.PROBLEM_CONFIGS[("selection", "small")]
    data_path = tmp_path / cfg["data"]
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_path, "wb") as f:
        pickle.dump([object(), object()], f)
    k_ratio = er.K_RATIOS[cfg["n_scenarios"]][1]
    eval_base = tmp_path / "e2e_base"
    (eval_base / f"k_{k_ratio}").mkdir(parents=True)
    np.save(eval_base / f"k_{k_ratio}" / "all_idx.npy", np.array([0, 1]))
    monkeypatch.setattr(er, "make_problem", lambda *a, **k: None)
    monkeypatch.setattr(er, "load_full_objs", lambda *a, **k: [10.0])  # 1 row for 2 instances
    monkeypatch.setattr(er, "_core_solve_reduced_milp",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not solve")))
    assert er.run_neurprise_regret("selection", "small", 1, eval_base_override=str(eval_base)) is None
    assert "full_objs has 1 rows < 2" in capsys.readouterr().out
