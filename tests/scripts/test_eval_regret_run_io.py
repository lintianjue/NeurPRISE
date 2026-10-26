"""eval_regret.py's run_io migration: writes go through an exclusive-create run
directory (never overwrite), reads come from INPUT_ROOT (never the historical
BASE once an isolated run supplies its own copy), and ``smoke_test`` no longer
writes at all.

No Gurobi, no real data: results are synthetic per-instance rows in the same
shape ``run_neurprise_regret`` appends (see its ``results.append`` in ``eval_regret.py``).
"""
import ast
import os
import json
import sys

import pytest

import scripts.eval.eval_regret as er
from src.utils.run_io import open_run


def _row(instance_id, full_obj, reduced_obj, eval_on_xi, regret):
    return {
        "instance_id": instance_id, "full_obj": full_obj, "reduced_obj": reduced_obj,
        "eval_on_xi": eval_on_xi, "regret": regret,
        "reduced_obj_bound": None, "reduced_mip_gap": None,
        "reduced_status": "OPTIMAL", "reduced_solve_sec": 0.1,
    }


_RESULTS = [
    _row(0, 100.0, 90.0, 105.0, 5.0),
    _row(1, 50.0, 45.0, 52.0, 4.0),
]


def _parse_strict(text):
    """json.loads that raises on any Infinity/NaN literal, like is_strict_file."""
    def _reject(name):
        raise ValueError(f"non-standard JSON constant: {name}")
    return json.loads(text, parse_constant=_reject)


# --------------------------------------------------------------------------
# 1. RUN set: exclusive-create write under the run directory.
# --------------------------------------------------------------------------

def test_save_results_run_mode_writes_under_run_dir_and_refuses_second_write(tmp_path, monkeypatch):
    """Mutation killed: dump_strict_new calling dump_strict (plain overwrite)
    instead of publish_new -- the second save would silently succeed."""
    rp = open_run("eval_regret", "run1", [], base=tmp_path / "runs", env={})
    monkeypatch.setattr(er, "RUN", rp)

    er.save_results(_RESULTS, "selection", "small", "neurprise", k=1, mip_gap=None)

    name = er._result_name("selection", "small", "neurprise", k=1, mip_gap=None)
    out_path = rp.run_dir / "eval_regret" / "selection" / name
    assert out_path.exists()

    parsed = _parse_strict(out_path.read_text())
    assert parsed["n_instances"] == 2
    assert parsed["regret_mean_feasible_only"] == 4.5
    assert "scen_gap_mean" not in parsed
    assert all("scen_gap" not in row for row in parsed["per_instance"])

    before = out_path.read_bytes()
    with pytest.raises(FileExistsError):
        er.save_results(_RESULTS, "selection", "small", "neurprise", k=1, mip_gap=None)
    assert out_path.read_bytes() == before


# --------------------------------------------------------------------------
# 2. RUN None: legacy path refuses to overwrite (replaces the old NOTE print).
# --------------------------------------------------------------------------

def test_save_results_legacy_mode_refuses_overwrite(tmp_path, monkeypatch):
    """Mutation killed: restoring the old 'NOTE: overwriting' + clobber behaviour."""
    monkeypatch.setattr(er, "RUN", None)
    monkeypatch.setattr(er, "REGRET_DIR", str(tmp_path / "eval_regret"))
    monkeypatch.setattr(er, "assert_within_base", lambda p: p)

    name = er._result_name("vertex_cover", "small", "neurprise", k=1, mip_gap=None)
    out_dir = tmp_path / "eval_regret" / "vertex_cover"
    out_dir.mkdir(parents=True)
    target = out_dir / name
    target.write_bytes(b"PRE-EXISTING")
    before = target.read_bytes()

    with pytest.raises(FileExistsError):
        er.save_results(_RESULTS, "vertex_cover", "small", "neurprise", k=1, mip_gap=None)

    assert target.read_bytes() == before


# --------------------------------------------------------------------------
# 3. Readers use INPUT_ROOT, not BASE.
# --------------------------------------------------------------------------

def test_find_neurprise_eval_base_reads_input_root_not_base(tmp_path, monkeypatch):
    """Mutation killed: reverting find_neurprise_eval_base's directory joins
    back to BASE (both the primary-pattern join and the fallback join)."""
    eval_leaf = (
        tmp_path / "in" / "eval" / "vertex_cover" / "s42_i2500_n20_s50" /
        "ML_s42_i2500_t-train_scale_norm_m-det_source_sce_mean_l-score_weighted_x_test" /
        "e2e_2ro"
    )
    eval_leaf.mkdir(parents=True)

    monkeypatch.setattr(er, "INPUT_ROOT", str(tmp_path / "in"))
    monkeypatch.setattr(er, "BASE", str(tmp_path / "nonexistent_base"))

    result = er.find_neurprise_eval_base("vertex_cover", "s42_i2500_n20_s50", scale="small")
    assert result == str(eval_leaf)


# --------------------------------------------------------------------------
# 4. smoke_test no longer writes anything.
# --------------------------------------------------------------------------

def _smoke_test_node():
    import inspect
    src = inspect.getsource(sys.modules["scripts.eval.eval_regret"])
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "smoke_test":
            return node
    raise AssertionError("smoke_test function not found")


def test_smoke_test_contains_no_write_open_and_no_os_remove():
    """Mutation killed: restoring the deleted 'Step 4: test save' block."""
    node = _smoke_test_node()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            if isinstance(func, ast.Name) and func.id == "open":
                mode = None
                if len(sub.args) >= 2 and isinstance(sub.args[1], ast.Constant):
                    mode = sub.args[1].value
                for kw in sub.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        mode = kw.value.value
                assert not (isinstance(mode, str) and set(mode) & set("wxa")), (
                    "smoke_test must not open() a file for writing")
            if (isinstance(func, ast.Attribute) and func.attr == "remove"
                    and isinstance(func.value, ast.Name) and func.value.id == "os"):
                pytest.fail("smoke_test must not call os.remove")


# --------------------------------------------------------------------------
# 5. main() --dry-run prints resolved paths and returns before any solve.
# --------------------------------------------------------------------------

def test_main_dry_run_prints_paths_and_returns_before_solve(tmp_path, monkeypatch, capsys):
    """Mutation killed: dropping the early ``return`` in the --dry-run branch."""
    cfg = er.PROBLEM_CONFIGS[("selection", "small")]
    data_path = tmp_path / "in" / cfg["data"]
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_bytes(b"")

    def _open_run_wrapper(batch, run_id, input_roots, *, base=None, env=None):
        return open_run(batch, run_id, input_roots, base=tmp_path / "runs", env={})

    def _must_not_be_called(*a, **k):
        raise AssertionError("run_neurprise_regret must not be called under --dry-run")

    # main() reassigns these module globals directly (``global INPUT_ROOT, RUN``),
    # bypassing monkeypatch's own setattr tracking; register the pre-test value
    # here so monkeypatch's teardown restores it regardless of what main() sets.
    monkeypatch.setattr(er, "INPUT_ROOT", er.INPUT_ROOT)
    monkeypatch.setattr(er, "RUN", er.RUN)
    monkeypatch.setattr(er, "open_run", _open_run_wrapper)
    monkeypatch.setattr(er, "run_neurprise_regret", _must_not_be_called)
    monkeypatch.setattr(sys, "argv", [
        "eval_regret.py", "--problem", "selection", "--scale", "small",
        "--method", "neurprise", "--k", "1",
        "--input-root", str(tmp_path / "in"), "--run-id", "t1",
        "--batch", "t", "--dry-run",
    ])

    er.main()  # must return normally, not raise/exit

    out = capsys.readouterr().out
    lines = out.splitlines()
    fields = dict(l.split("=", 1) for l in lines if "=" in l and l.split("=", 1)[0] in ("DATA", "OUT"))
    # The printed paths are the evidence the operator reads before any real run, so
    # pin WHERE they point, not just that the lines exist (mutations killed: DATA
    # built from BASE, OUT built from REGRET_DIR).
    run_dir = (tmp_path / "runs" / "t" / "t1").resolve()
    assert fields["DATA"] == os.path.join(os.path.realpath(tmp_path / "in"), cfg["data"]), out
    assert os.path.commonpath([fields["OUT"], str(run_dir)]) == str(run_dir), out
    assert (run_dir / "RUN.json").exists()


# --------------------------------------------------------------------------
# 6. main() requires --input-root/--run-id outside --smoke-test.
# --------------------------------------------------------------------------

def test_main_missing_input_root_and_run_id_exits_2(monkeypatch):
    """Mutation killed: dropping the parser.error() for missing --input-root/--run-id."""
    monkeypatch.setattr(sys, "argv", [
        "eval_regret.py", "--problem", "selection", "--scale", "small", "--method", "neurprise",
    ])
    with pytest.raises(SystemExit) as exc:
        er.main()
    assert exc.value.code == 2
