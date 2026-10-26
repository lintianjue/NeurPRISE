"""Characterisation of the three live DataModule filename chains (record, do not merge).

Three chains build the label filename from a Hydra config:
  (a) training side -- scripts/03_train.py::_train_label_file_name
  (b) eval side, in-sample -- src.utils.omega_conf.create_problem_module
  (c) eval side, overridden shape -- src.utils.omega_conf.create_problem_module_with_data_override

(b) and (c) are untouched code: this test checks them against the HEAD golden
(tests/fixtures/filename_chains_golden/chains_pre_refactor.json, produced by gen_chains.py BEFORE (a) was
extracted) and must never diverge from it. (a) is NOT compared against a pre-existing
golden of its own -- 03_train.py's naming statements were only ever callable inside
main() before they were extracted -- so this test RECORDS (a) into
tests/fixtures/filename_chains_golden/chains_characterised.json on its first run and freezes it
thereafter: report the chains, don't merge them.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig

import src.utils.omega_conf as omega_conf_mod
from src.utils.omega_conf import create_problem_module, create_problem_module_with_data_override

_REPO = Path(__file__).resolve().parents[2]
_GOLDEN_DIR = _REPO / "tests" / "fixtures" / "filename_chains_golden"
_CONFIGS_DIR = _REPO / "configs"
_CHARACTERISED_PATH = _GOLDEN_DIR / "chains_characterised.json"

_HEAD_ROWS = json.loads((_GOLDEN_DIR / "chains_pre_refactor.json").read_text())

_SHAPE_BY_PROBLEM = {
    "vertex_cover": dict(n_nodes=50, n_facilities=None, n_customers=None),
    "selection":    dict(n_nodes=50, n_facilities=None, n_customers=None),
    "cflp_ebp":     dict(n_nodes=None, n_facilities=70, n_customers=70),
}


def _load_train_module():
    module_path = _REPO / "scripts" / "03_train.py"
    spec = importlib.util.spec_from_file_location("train_script_chains", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _RecorderDataModule:
    calls: list[dict] = []

    def __init__(self, **kwargs):
        _RecorderDataModule.calls.append(kwargs)


def _compose(config_name: str, problem: str, overrides: list[str]):
    with initialize_config_dir(config_dir=str(_CONFIGS_DIR), version_base=None):
        cfg = compose(config_name=config_name, return_hydra_config=True,
                       overrides=[f"problem={problem}", *overrides])
        HydraConfig.instance().set_config(cfg)
        return cfg


@pytest.fixture(scope="module")
def train_module():
    return _load_train_module()


def _build_table(train_module) -> list[dict]:
    orig_dm = omega_conf_mod.DataModule
    omega_conf_mod.DataModule = _RecorderDataModule
    table = []
    try:
        for head_row in _HEAD_ROWS:
            problem = head_row["problem"]
            overrides = head_row["overrides"]
            entry = dict(problem=problem, overrides=overrides)
            if "error" in head_row:
                entry["error"] = head_row["error"]
                table.append(entry)
                continue

            cfg_train = _compose("exp", problem, overrides)
            entry["a"] = train_module._train_label_file_name(cfg_train)

            cfg_eval = _compose("eval", problem, overrides)
            _RecorderDataModule.calls.clear()
            create_problem_module(cfg_eval)
            entry["b"] = _RecorderDataModule.calls[-1]["file_name"]
            assert entry["b"] == head_row["create_problem_module"]["file_name"], (
                f"create_problem_module (untouched code) diverged from the HEAD golden for "
                f"problem={problem} overrides={overrides}"
            )

            shape = _SHAPE_BY_PROBLEM[problem]
            _RecorderDataModule.calls.clear()
            create_problem_module_with_data_override(
                cfg_eval, instance_cnt=500, n_nodes=shape["n_nodes"], n_scenarios=100,
                n_facilities=shape["n_facilities"], n_customers=shape["n_customers"],
            )
            entry["c"] = _RecorderDataModule.calls[-1]["file_name"]
            assert entry["c"] == head_row["create_problem_module_with_data_override"]["file_name"], (
                f"create_problem_module_with_data_override (untouched code) diverged from the "
                f"HEAD golden for problem={problem} overrides={overrides}"
            )
            table.append(entry)
    finally:
        omega_conf_mod.DataModule = orig_dm
    return table


def test_chains_characterised(train_module) -> None:
    table = _build_table(train_module)

    if _CHARACTERISED_PATH.is_file():
        frozen = json.loads(_CHARACTERISED_PATH.read_text())
        assert table == frozen, (
            "the characterised three-column table changed; if that's expected, delete "
            f"{_CHARACTERISED_PATH} and re-run once to re-freeze it (review the diff)"
        )
    else:
        _CHARACTERISED_PATH.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")

    # Documented difference #1: a bigmilp-output row ends in _bigmilp.pkl on both chains.
    for row in table:
        if row.get("overrides") == ["label.prise.output=bigmilp"] and "error" not in row:
            assert row["a"].endswith("_bigmilp.pkl")
            assert row["b"].endswith("_bigmilp.pkl")

