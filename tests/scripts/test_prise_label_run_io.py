"""Dataset writers retain isolated output layout when a custom input root is used."""
import ast
import importlib.util
from pathlib import Path

from src.utils.run_io import open_run

REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO / "scripts" / "02_prise_label.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("prise_label_run_io_mod", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_label_output_path_mirrors_input_layout(tmp_path):
    """(M) drop the "data" segment from _label_output_path."""
    module = _load_module()
    in_root = tmp_path / "in"
    in_root.mkdir()
    rp = open_run("b", "r", [in_root], base=tmp_path / "runs", env={})

    out = module._label_output_path(rp, "selection", "/in/data/selection/label_x.pkl")

    assert out == rp.run_dir / "data" / "selection" / "label_x.pkl"


def test_main_writes_only_through_create_new():
    """AST lock: every pickle.dump/json.dump inside main() sits in a `with` whose context
    expression calls create_new; no write-mode open()/.mkdir() call remains in main() (a
    read-mode open() of an input file, e.g. the instance-spec pickle, is untouched --
    reads are not run_io's concern here).
    (M) revert the pickle write to open(path, "wb")."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    main_func = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main"
    )

    def _is_create_new_call(node) -> bool:
        return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "create_new"

    dump_calls = [
        n for n in ast.walk(main_func)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr in ("dump",)
        and isinstance(n.func.value, ast.Name) and n.func.value.id in ("pickle", "json")
    ]
    assert dump_calls, "no pickle.dump/json.dump call found in main() -- test needs rewriting"

    with_nodes = [n for n in ast.walk(main_func) if isinstance(n, ast.With)]
    for call in dump_calls:
        # Candidates: every `with` whose subtree contains this call. The innermost is the
        # one not itself contained in any other candidate's subtree.
        candidates = [w for w in with_nodes if any(call is n for n in ast.walk(w))]
        assert candidates, f"{ast.dump(call)} is not inside any `with` block"
        innermost = next(
            w for w in candidates
            if not any(w is not other and any(w is n for n in ast.walk(other)) for other in candidates)
        )
        assert len(innermost.items) == 1
        ctx_expr = innermost.items[0].context_expr
        assert _is_create_new_call(ctx_expr), (
            f"dump call's enclosing with-context is not create_new(...): {ast.dump(ctx_expr)}"
        )

    def _mode_arg(call: ast.Call) -> str | None:
        if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant):
            return call.args[1].value
        for kw in call.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                return kw.value.value
        return None

    for node in ast.walk(main_func):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            mode = _mode_arg(node)
            if mode is not None and mode.startswith(("w", "a")):
                raise AssertionError(f"write-mode open() call remains in main(): mode={mode!r}")
        if isinstance(node.func, ast.Attribute) and node.func.attr == "mkdir":
            raise AssertionError(".mkdir(...) call remains in main()")
