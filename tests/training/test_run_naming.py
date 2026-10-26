"""Checkpoint naming has one source (src/training/run_naming.py).

Goldens are tests/fixtures/filename_chains_golden/{naming_grid,naming_w16_cells}.json, produced by
tests/fixtures/filename_chains_golden/gen_naming.py BEFORE naming moved into run_naming -- see that
script's docstring for the two independent producers and the cross-check between them.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig

from src.training import run_naming

_REPO = Path(__file__).resolve().parents[2]
_GOLDEN_DIR = _REPO / "tests" / "fixtures" / "filename_chains_golden"
_CONFIGS_DIR = _REPO / "configs"

_GRID_ROWS = json.loads((_GOLDEN_DIR / "naming_grid.json").read_text())


def _load_train_module():
    module_path = _REPO / "scripts" / "03_train.py"
    spec = importlib.util.spec_from_file_location("train_script_naming", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


@pytest.mark.parametrize("row", _GRID_ROWS, ids=lambda r: f"{r['config_name']}-{r['problem']}-{r['out_tag']}")
def test_checkpoint_name_matches_head_grid(row: dict) -> None:
    """(M) in checkpoint_name drop the leading '_' of cfg_part
    (f"_c-{config_name}" -> f"c-{config_name}"): the double underscore vanishes."""
    with initialize_config_dir(config_dir=str(_CONFIGS_DIR), version_base=None):
        cfg = compose(config_name=row["config_name"], return_hydra_config=True, overrides=row["overrides"])
        HydraConfig.instance().set_config(cfg)
        assert run_naming.checkpoint_name(cfg) == row["ckpt_name"]


def test_train_forwarders_are_the_library_functions() -> None:
    """The three naming helpers in 03_train.py are one-line forwarders to run_naming, and
    main() no longer assigns best_model_filename (the naming statements moved out).
    (M) paste the old _config_name_fragment body back into 03_train.py."""
    tree = _module_ast(_REPO / "scripts" / "03_train.py")
    forwarders = {"_config_name_fragment", "_hydra_choice_fragment", "_hydra_config_name_fragment"}
    seen = set()
    main_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in forwarders:
            seen.add(node.name)
            assert len(node.body) == 1, f"{node.name} body is not a single statement"
            stmt = node.body[0]
            assert isinstance(stmt, ast.Return), f"{node.name} body is not a bare return"
            call = stmt.value
            assert isinstance(call, ast.Call), f"{node.name} does not return a call"
            assert isinstance(call.func, ast.Attribute) and call.func.attr == node.name, (
                f"{node.name} does not forward to run_naming.{node.name}"
            )
            assert isinstance(call.func.value, ast.Name) and call.func.value.id == "run_naming", (
                f"{node.name} does not forward to the run_naming module"
            )
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            main_func = node
    assert seen == forwarders, f"missing forwarders: {forwarders - seen}"
    assert main_func is not None
    for stmt in ast.walk(main_func):
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    assert target.id != "best_model_filename", (
                        "main() still assigns best_model_filename; the naming statements "
                        "were supposed to move into run_naming.checkpoint_name"
                    )
