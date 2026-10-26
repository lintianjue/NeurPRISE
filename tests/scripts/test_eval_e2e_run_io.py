"""scripts/04_05_eval_ml_2ro_e2e.py writes only through run_io, into its own run dir.
Byte-identical JSON output is pinned against the goldens in tests/fixtures/output_path_golden/.

The module is loaded by file path.
"""
import ast
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.utils.run_io import RunIOError, open_run

REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO / "scripts" / "04_05_eval_ml_2ro_e2e.py"
GOLDEN_DIR = REPO / "tests" / "fixtures" / "output_path_golden"


def _load_module():
    spec = importlib.util.spec_from_file_location("eval_e2e_run_io_mod", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---- _resolve_out_dir ----------------------------------------------------------------

def test_resolve_out_dir_grid_matches_head(tmp_path):
    """(M) re-add the Path(out_dir).mkdir(...) call.
    (M) change "e2e_2ro" to any other literal."""
    module = _load_module()
    rows = json.loads((GOLDEN_DIR / "out_dir_grid.json").read_text())
    assert rows, "golden grid is empty"

    for row in rows:
        output_dir = str(tmp_path / row["output_dir"])
        cfg = OmegaConf.create(
            {"output_dir": output_dir, "name": row["name"], "k_ratio": row["k_ratio"]}
        )
        result = module._resolve_out_dir(cfg)
        assert os.path.relpath(result, tmp_path) == row["result_relpath"]

    assert not any(tmp_path.iterdir()), "_resolve_out_dir created something under tmp_path"


def test_resolve_out_dir_default_is_run_dir_eval(tmp_path):
    """(M) drop the "eval" segment from the run_dir default."""
    module = _load_module()
    run_dir = tmp_path / "run"
    cfg = OmegaConf.create({"output_dir": "", "name": "N", "k_ratio": 0.2})

    result = module._resolve_out_dir(cfg, run_dir)
    assert os.path.relpath(result, run_dir) == os.path.join("eval", "N", "e2e_2ro", "k_0.2")

    with pytest.raises(ValueError):
        module._resolve_out_dir(cfg, None)


# ---- _publish_json ---------------------------------------------------------------------

def test_publish_json_bytes_match_head_and_refuse_overwrite(tmp_path):
    """(M) replace publish_new with open(path, "w") inside _publish_json."""
    module = _load_module()
    in_root = tmp_path / "in"
    in_root.mkdir()
    rp = open_run("b", "r", [in_root], base=tmp_path / "runs", env={})

    p1 = {"b": 1, "a": [1.5, None, "x"], "n": {"z": True, "y": [3, 2]}}
    p2 = {"nan": float("nan"), "inf": float("inf"), "k": 2, "s": "é"}

    out1 = module._publish_json(rp, "p1.json", p1, allow_nan=False)
    assert out1.read_bytes() == (GOLDEN_DIR / "json_atomic_1.bin").read_bytes()

    out2 = module._publish_json(rp, "p2.json", p2, allow_nan=True)
    assert out2.read_bytes() == (GOLDEN_DIR / "json_atomic_2.bin").read_bytes()

    before = out1.read_bytes()
    with pytest.raises(FileExistsError):
        module._publish_json(rp, "p1.json", {"different": True}, allow_nan=False)
    assert out1.read_bytes() == before
    assert list(rp.run_dir.glob("*.partial.*")) == []


# ---- structural (AST) locks -------------------------------------------------------------

def _load_tree():
    return ast.parse(MODULE_PATH.read_text(encoding="utf-8"))


def _find_main(tree):
    return next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main"
    )


def _find_publish_json(tree):
    return next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_publish_json"
    )


def test_main_has_no_direct_write_calls():
    """(M) revert one np.save call to a bare path argument (no create_new)."""
    tree = _load_tree()
    main_func = _find_main(tree)
    publish_func = _find_publish_json(tree)

    for node in ast.walk(main_func):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "replace" \
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
            raise AssertionError("os.replace(...) call remains in main()")
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            mode = None
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                mode = node.args[1].value
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = kw.value.value
            if mode is not None and mode.startswith(("w", "a")):
                raise AssertionError(f"write-mode open() call remains in main(): mode={mode!r}")
        if isinstance(node.func, ast.Attribute) and node.func.attr == "mkdir":
            raise AssertionError(".mkdir(...) call remains in main()")

    with_nodes = [n for n in ast.walk(main_func) if isinstance(n, ast.With)]

    def _is_create_new_call(node) -> bool:
        return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "create_new"

    dump_calls = []
    for n in ast.walk(main_func):
        if not isinstance(n, ast.Call):
            continue
        if isinstance(n.func, ast.Attribute) and n.func.attr == "dump" \
                and isinstance(n.func.value, ast.Name) and n.func.value.id == "json":
            dump_calls.append(n)
        if isinstance(n.func, ast.Attribute) and n.func.attr == "save" \
                and isinstance(n.func.value, ast.Name) and n.func.value.id == "np":
            dump_calls.append(n)
        if isinstance(n.func, ast.Attribute) and n.func.attr == "to_csv":
            dump_calls.append(n)

    assert dump_calls, "no np.save/.to_csv/json.dump call found in main() -- test needs rewriting"

    for call in dump_calls:
        candidates = [w for w in with_nodes if any(call is n for n in ast.walk(w))]
        assert candidates, f"{ast.dump(call)} is not inside any `with` block in main()"
        innermost = next(
            w for w in candidates
            if not any(w is not other and any(w is n for n in ast.walk(other)) for other in candidates)
        )
        assert len(innermost.items) == 1
        assert _is_create_new_call(innermost.items[0].context_expr), (
            f"write call's enclosing with-context is not create_new(...): {ast.dump(call)}"
        )

    # _publish_json's own json.dump call is exempt (it is create_new/publish_new's own writer).
    publish_dump_calls = [
        n for n in ast.walk(publish_func)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "dump"
    ]
    assert publish_dump_calls, "_publish_json no longer calls json.dump"


def test_output_dir_outside_run_dir_is_refused_before_solving():
    """(M) delete the output_path(...) line that follows a _resolve_out_dir call."""
    tree = _load_tree()
    main_func = _find_main(tree)

    def _is_resolve_out_dir_call(node) -> bool:
        return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_resolve_out_dir"

    def _is_output_path_call(node) -> bool:
        return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "output_path"

    def _statement_lists(node):
        """Every statement list (a block body) anywhere under `node`, including nested
        ones (if/with/try/for/...) -- a call inside an `if` block is not a top-level
        statement of main() but must still be checked in its own block."""
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(node, field, None)
            if isinstance(stmts, list) and stmts and isinstance(stmts[0], ast.stmt):
                yield stmts
                for stmt in stmts:
                    yield from _statement_lists(stmt)
        for handler in getattr(node, "handlers", []):
            yield from _statement_lists(handler)

    found = 0
    for body in _statement_lists(main_func):
        for i, stmt in enumerate(body):
            if isinstance(stmt, ast.Assign) and _is_resolve_out_dir_call(stmt.value):
                found += 1
                assert i + 1 < len(body), "_resolve_out_dir is the last statement of its block"
                nxt = body[i + 1]
                if isinstance(nxt, (ast.Assign, ast.Expr)):
                    assert _is_output_path_call(nxt.value), (
                        f"statement after _resolve_out_dir(...) is not output_path(...): {ast.dump(nxt)}"
                    )
                else:
                    raise AssertionError(f"unexpected statement after _resolve_out_dir(...): {ast.dump(nxt)}")
    assert found >= 1, "no _resolve_out_dir(...) assignment in main(); this check would pass vacuously"
