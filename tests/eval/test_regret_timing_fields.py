"""NeurPRISE's Time(s) column.

The paper's NeurPRISE Time(s) = scoring time + reduced-solve time. This
evaluator only performs the reduced-solve half (scoring/selection is
precomputed by ``scripts/04_05_eval_ml_2ro_e2e.py`` and read back as
``all_idx.npy``/``test_idx.npy``), so:

- the reduced-solve half is a real measurement, taken as wall clock over
  build+solve (``reduced_eval_wall_sec``);
- the selection half is READ from the e2e run's ``timing.json``, never
  recomputed, and is ``null`` (not ``0``) when that file is missing or
  unparseable -- an absent measurement is not a zero measurement.

No Gurobi, no real data: every collaborator is monkeypatched or faked, same
idiom as ``tests/eval/test_regret.py`` / ``tests/scripts/test_eval_regret_mip_gap_call_time.py``.
"""
import json
import pickle
import time

import numpy as np
import pytest

from scripts.eval import eval_regret as er


def _fake_row(instance_id, reduced_eval_wall_sec):
    """Minimal per-instance record: the fields ``summarize``/``annotate_record``
    read, plus the one field this change adds.
    """
    return {
        "instance_id": instance_id,
        "full_obj": 10.0,
        "reduced_obj": 9.0,
        "eval_on_xi": 11.0,
        "regret": 10.0,
        "reduced_eval_wall_sec": reduced_eval_wall_sec,
    }


def _redirect_save_results(monkeypatch, tmp_path):
    """save_results' legacy (no --run-id) path writes under REGRET_DIR;
    point it at tmp_path and disable the BASE-containment guard, same as
    tests/test_eval_server_provenance.py.
    """
    regret_dir = tmp_path / "eval_regret"
    regret_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(er, "REGRET_DIR", str(regret_dir))
    monkeypatch.setattr(er, "assert_within_base", lambda *_a, **_k: None)
    return regret_dir


def test_per_instance_records_reduced_eval_wall_sec(tmp_path, monkeypatch, capsys):
    """(M) span excludes the solve: start the clock after get_neurprise_x
    returns instead of before it -- this test's sleep would then not be
    inside the measured span and ``reduced_eval_wall_sec`` would read ~0.
    """
    monkeypatch.setattr(er, "INPUT_ROOT", str(tmp_path))
    cfg = er.PROBLEM_CONFIGS[("selection", "small")]
    data_path = tmp_path / cfg["data"]
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_path, "wb") as f:
        pickle.dump([object()] * 10, f)

    k_ratio = er.K_RATIOS[cfg["n_scenarios"]][1]
    eval_base = tmp_path / "eb"
    idx_dir = eval_base / f"k_{k_ratio}"
    idx_dir.mkdir(parents=True)
    np.save(idx_dir / "all_idx.npy", np.zeros((10, 1), dtype=int))

    monkeypatch.setattr(er, "make_problem", lambda *a, **k: None)
    monkeypatch.setattr(er, "find_neurprise_eval_base", lambda *a, **k: str(eval_base))
    monkeypatch.setattr(er, "load_full_objs", lambda *a, **k: [10.0] * 10)

    def _fake_solve(prob, inst, sce_idx, problem_name="vertex_cover", *,
                     mip_gap=None, threads=None, time_limit=None, return_raw=False):
        time.sleep(0.01)
        return (np.array([1.0, 0.0]), 9.0,
                {"runtime_sec": 0.0, "obj_bound": None, "mip_gap": None, "status": "OPTIMAL"})

    monkeypatch.setattr(er, "_core_solve_reduced_milp", _fake_solve)
    monkeypatch.setattr(er, "_core_evaluate_x_on_all_scenarios", lambda *a, **k: 11.0)

    results = er.run_neurprise_regret("selection", "small", 1)

    assert len(results) == 10
    assert all(r["regret"] == pytest.approx(10.0) for r in results)
    assert all("scen_gap" not in r for r in results)
    output = capsys.readouterr().out
    assert "[10/10] regret=10.00%" in output
    assert "scengap" not in output.lower()
    row = results[0]
    assert row["reduced_eval_wall_sec"] >= 0.01
    assert row["reduced_solve_sec"] == 0.0  # unchanged: solve's own runtime, not the wall clock


def test_summary_sums_wall_time_over_instances(tmp_path, monkeypatch):
    """(M) total = last instance only: read the last row's value instead of
    summing.
    """
    _redirect_save_results(monkeypatch, tmp_path)
    fake_results = [_fake_row(i, v) for i, v in enumerate([0.10, 0.20, 0.30])]

    summary = er.save_results(fake_results, "selection", "small", "neurprise", k=1, threads=40)

    assert summary["reduced_eval_time_total_sec"] == pytest.approx(0.60)
    assert summary["reduced_eval_time_per_instance_sec"] == pytest.approx(0.20)
    assert summary["reduced_eval_time_n_recorded"] == 3


def test_selection_time_is_read_from_timing_json(tmp_path, monkeypatch):
    """(M) read the wrong key / never read the file: pull
    ``selection_time_total_sec`` from somewhere else, or skip the read and
    leave the field at its ``_time_components`` default (``None``, since
    these rows carry no ``selection_sec``).
    """
    monkeypatch.setattr(er, "INPUT_ROOT", str(tmp_path))
    eval_base = tmp_path / "eb"
    k_ratio = 0.02
    (eval_base / f"k_{k_ratio}").mkdir(parents=True)
    (eval_base / f"k_{k_ratio}" / "timing.json").write_text(
        json.dumps({"selection_time_total_sec": 4.5}))

    fake_results = [_fake_row(0, 1.5)]
    sel_info = er._read_selection_time(str(eval_base), k_ratio, len(fake_results))
    _redirect_save_results(monkeypatch, tmp_path)

    summary = er.save_results(fake_results, "selection", "small", "neurprise", k=1,
                              threads=40, selection_time_info=sel_info)

    assert summary["selection_time_total_sec"] == pytest.approx(4.5)
    assert summary["time_column_sec"] == pytest.approx(4.5 + summary["reduced_eval_time_total_sec"])
    assert summary["selection_time_source"].endswith("timing.json")
    assert summary["selection_is_precomputed"] is True


def test_missing_timing_json_is_null_not_zero(tmp_path, monkeypatch):
    """(M) default 0.0: fall back to 0.0 instead of None when the file is
    absent, which would read as a real (fast) measurement.
    """
    monkeypatch.setattr(er, "INPUT_ROOT", str(tmp_path))
    eval_base = tmp_path / "eb_missing"
    k_ratio = 0.02
    (eval_base / f"k_{k_ratio}").mkdir(parents=True)  # no timing.json inside

    sel_info = er._read_selection_time(str(eval_base), k_ratio, 1)
    assert sel_info["selection_time_total_sec"] is None
    assert sel_info["selection_time_source"] is None

    fake_results = [_fake_row(0, 1.5)]
    _redirect_save_results(monkeypatch, tmp_path)

    summary = er.save_results(fake_results, "selection", "small", "neurprise", k=1,
                              threads=40, selection_time_info=sel_info)

    assert summary["selection_time_total_sec"] is None
    assert summary["time_column_sec"] is None
    assert summary["selection_time_source"] is None


def test_strict_json_writes_null_for_missing_selection_time(tmp_path, monkeypatch):
    """(M) skip None keys: omit the field entirely instead of writing it as
    JSON ``null`` -- src/utils/regret_json.dump_strict* already passes plain
    None through untouched, so this pins that the summary dict actually
    carries the key rather than relying on json.dump's default.
    """
    monkeypatch.setattr(er, "INPUT_ROOT", str(tmp_path))
    eval_base = tmp_path / "eb_missing2"
    k_ratio = 0.02
    (eval_base / f"k_{k_ratio}").mkdir(parents=True)

    sel_info = er._read_selection_time(str(eval_base), k_ratio, 1)
    fake_results = [_fake_row(0, 1.5)]
    regret_dir = _redirect_save_results(monkeypatch, tmp_path)

    er.save_results(fake_results, "selection", "small", "neurprise", k=1,
                    threads=40, selection_time_info=sel_info)

    out_path = regret_dir / "selection" / "SEL-20-50_neurprise_k1.json"
    text = out_path.read_text()
    assert '"selection_time_total_sec": null' in text
