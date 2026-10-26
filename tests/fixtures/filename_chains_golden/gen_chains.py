"""Generate the pre-refactor goldens for the eval-side DataModule filename chains.

Run ONCE, before the chains were refactored:

    PYTHONPATH=$PWD python tests/fixtures/filename_chains_golden/gen_chains.py

Monkeypatches ``src.utils.omega_conf.DataModule`` with a recorder class that stores its
constructor kwargs and returns immediately (no real dataset file is touched); the rest of
``create_problem_module`` / ``create_problem_module_with_data_override`` runs for real,
including ``load_problem`` (no data needed to build the problem object itself).

For each problem P x label-variant override, records ``file_name`` / ``data_path`` /
``role_contract`` (repr) from ``create_problem_module``, ``file_name`` from
``create_problem_module_with_data_override``, and the two config values
the two chains disagree on: ``cfg.label.type`` and
``cfg.problem.data.get("label_type", "prise")``. If composing a row fails, the exception
class is recorded in the row instead of stopping the generator.
"""
import json
import os
import sys

_FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_FIXTURE_DIR, "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402

import src.utils.omega_conf as omega_conf_mod  # noqa: E402

CONFIGS_DIR = os.path.join(_REPO, "configs")

PROBLEMS = ("vertex_cover", "selection", "cflp_ebp")
LABEL_VARIANT_OVERRIDES = (
    [],
    ["label.prise.output=bigmilp"],
    ["problem.data.label_variant=merged500"],
)

# Shape kwargs for create_problem_module_with_data_override, dispatched by problem family
# (node-count problems vs facility/customer problems -- see configs/problem/*.yaml).
_SHAPE_BY_PROBLEM = {
    "vertex_cover": dict(n_nodes=50, n_facilities=None, n_customers=None),
    "selection":    dict(n_nodes=50, n_facilities=None, n_customers=None),
    "cflp_ebp":     dict(n_nodes=None, n_facilities=70, n_customers=70),
}


class _RecorderDataModule:
    """Records constructor kwargs on a module-level list and returns; no real init."""
    calls: list[dict] = []

    def __init__(self, **kwargs):
        _RecorderDataModule.calls.append(kwargs)


def _compose(problem: str, extra_overrides: list[str]):
    overrides = [f"problem={problem}", *extra_overrides]
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="eval", return_hydra_config=True, overrides=overrides)
        HydraConfig.instance().set_config(cfg)
        return cfg


def _row(problem: str, extra_overrides: list[str]) -> dict:
    row = dict(problem=problem, overrides=extra_overrides)
    try:
        cfg = _compose(problem, extra_overrides)
    except Exception as exc:
        row["error"] = type(exc).__name__
        return row

    row["label_type_cfg_label"] = cfg.label.type
    row["label_type_cfg_problem_data"] = cfg.problem.data.get("label_type", "prise")

    _RecorderDataModule.calls.clear()
    try:
        omega_conf_mod.create_problem_module(cfg)
        call = _RecorderDataModule.calls[-1]
        row["create_problem_module"] = dict(
            file_name=call.get("file_name"),
            data_path=call.get("data_path"),
            role_contract=repr(call.get("role_contract")),
        )
    except Exception as exc:
        row["create_problem_module"] = {"error": type(exc).__name__}

    shape = _SHAPE_BY_PROBLEM[problem]
    _RecorderDataModule.calls.clear()
    try:
        omega_conf_mod.create_problem_module_with_data_override(
            cfg,
            instance_cnt=500,
            n_nodes=shape["n_nodes"],
            n_scenarios=100,
            n_facilities=shape["n_facilities"],
            n_customers=shape["n_customers"],
        )
        call = _RecorderDataModule.calls[-1]
        row["create_problem_module_with_data_override"] = dict(file_name=call.get("file_name"))
    except Exception as exc:
        row["create_problem_module_with_data_override"] = {"error": type(exc).__name__}

    return row


def main() -> None:
    orig_dm = omega_conf_mod.DataModule
    omega_conf_mod.DataModule = _RecorderDataModule
    try:
        rows = [
            _row(problem, list(overrides))
            for problem in PROBLEMS
            for overrides in LABEL_VARIANT_OVERRIDES
        ]
    finally:
        omega_conf_mod.DataModule = orig_dm

    out_path = os.path.join(_FIXTURE_DIR, "chains_pre_refactor.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"chains_pre_refactor.json: {len(rows)} rows")


if __name__ == "__main__":
    main()
