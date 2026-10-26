"""Dataset writers retain isolated output layout when a custom input root is used."""
import ast
import importlib.util
from pathlib import Path

from src.utils.run_io import open_run

REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO / "scripts" / "01_create_inst_spec.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("create_inst_spec_run_io", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_instance_output_path_mirrors_input_layout(tmp_path):
    """(M) drop the "data" segment from _instance_output_path."""
    in_root = tmp_path / "in"
    in_root.mkdir()
    rp = open_run("b", "r", [in_root], base=tmp_path / "runs", env={})

    out = _load_module()._instance_output_path(rp, "selection", "inst_x.pkl")

    assert out == rp.run_dir / "data" / "selection" / "inst_x.pkl"


def test_main_opens_its_run_before_seeding_consumers_and_writes_via_create_new():
    """(M) revert the pickle write to open(path, "wb") / reintroduce os.makedirs."""
    main = next(n for n in ast.walk(ast.parse(MODULE_PATH.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)]
    names = {c.func.id for c in calls if isinstance(c.func, ast.Name)}
    attrs = {c.func.attr for c in calls if isinstance(c.func, ast.Attribute)}
    assert {"open_hydra_run", "create_new"} <= names
    assert "open" not in names and "makedirs" not in attrs
