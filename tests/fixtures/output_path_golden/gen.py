"""Generate the pre-refactor goldens for the eval/label output-path helpers.

Run ONCE, before the helpers were refactored::

    PYTHONPATH=$PWD python tests/fixtures/output_path_golden/gen.py

Three independent pre-refactor producers are recorded, each loaded by file path (never
``import scripts...``):

0.1 ``scripts/04_05_eval_ml_2ro_e2e.py::_write_json_atomic`` -- exact bytes for two
    payloads (one allow_nan=False, one allow_nan=True) -> ``json_atomic_1.bin`` /
    ``json_atomic_2.bin``. The refactor replaces this with ``_publish_json``; the golden pins
    that the new writer must produce byte-identical output.
0.2 ``scripts/04_05_eval_ml_2ro_e2e.py::_resolve_out_dir`` -- the output-dir grid ->
    ``out_dir_grid.json``. The pre-refactor ``_resolve_out_dir`` mkdirs as a side effect, hence the
    tmp dir.
0.3 ``scripts/02_prise_label.py::_build_eval_dir`` -- the eval-dir tag grid ->
    ``eval_dir_tags.json``.
"""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

_FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_FIXTURE_DIR, "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from omegaconf import OmegaConf  # noqa: E402

from src.utils.utils import _find_project_root  # noqa: E402


def _load_by_path(rel_path: str, module_name: str):
    path = os.path.join(_REPO, rel_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gen_json_atomic(eval_module) -> None:
    p1 = {"b": 1, "a": [1.5, None, "x"], "n": {"z": True, "y": [3, 2]}}
    p2 = {"nan": float("nan"), "inf": float("inf"), "k": 2, "s": "é"}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        path1 = tmp / "p1.json"
        eval_module._write_json_atomic(path1, p1, allow_nan=False)
        data1 = path1.read_bytes()

        path2 = tmp / "p2.json"
        eval_module._write_json_atomic(path2, p2, allow_nan=True)
        data2 = path2.read_bytes()

    (Path(_FIXTURE_DIR) / "json_atomic_1.bin").write_bytes(data1)
    (Path(_FIXTURE_DIR) / "json_atomic_2.bin").write_bytes(data2)
    print(f"json_atomic_1.bin: {len(data1)} bytes")
    print(f"json_atomic_2.bin: {len(data2)} bytes")


def _gen_out_dir_grid(eval_module) -> list[dict]:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        output_dirs = [str(tmp / "x"), str(tmp / "x") + "/"]
        names = ["", "N90", " pad "]
        k_ratios = [0.2, 0.01, 1, "0.20"]
        for output_dir in output_dirs:
            for name in names:
                for k_ratio in k_ratios:
                    cfg = OmegaConf.create(
                        {"output_dir": output_dir, "name": name, "k_ratio": k_ratio}
                    )
                    result = eval_module._resolve_out_dir(cfg)
                    rows.append(
                        {
                            "output_dir": os.path.relpath(output_dir, tmp),
                            "name": name,
                            "k_ratio": k_ratio,
                            "result_relpath": os.path.relpath(result, tmp),
                        }
                    )
    return rows


def _gen_eval_dir_tags(label_module) -> list[dict]:
    rows = []

    cfg_selection = OmegaConf.create(
        {
            "seed": 42,
            "problem": {
                "name": "selection",
                "instance": {"n_nodes": 20, "n_scenarios": 50},
            },
        }
    )
    path = label_module._build_eval_dir(cfg_selection, "selection", instance_cnt=2500)
    rows.append(
        {
            "cfg": {"seed": 42, "problem": "selection", "n_nodes": 20, "n_scenarios": 50},
            "problem_name": "selection",
            "instance_cnt": 2500,
            "relpath": str(path.relative_to(_find_project_root())),
        }
    )

    cfg_cflp = OmegaConf.create(
        {
            "seed": 42,
            "problem": {
                "name": "cflp_ebp",
                "instance": {"n_facilities": 70, "n_customers": 80, "n_scenarios": 100},
            },
        }
    )
    path = label_module._build_eval_dir(cfg_cflp, "cflp_ebp", instance_cnt=500)
    rows.append(
        {
            "cfg": {"seed": 42, "problem": "cflp_ebp", "n_facilities": 70, "n_customers": 80, "n_scenarios": 100},
            "problem_name": "cflp_ebp",
            "instance_cnt": 500,
            "relpath": str(path.relative_to(_find_project_root())),
        }
    )

    cfg_vc = OmegaConf.create(
        {
            "seed": 7,
            "problem": {
                "name": "vertex_cover",
                "instance": {"n_nodes": 50, "n_scenarios": 10},
            },
        }
    )
    path = label_module._build_eval_dir(cfg_vc, "vertex_cover", instance_cnt=100)
    rows.append(
        {
            "cfg": {"seed": 7, "problem": "vertex_cover", "n_nodes": 50, "n_scenarios": 10},
            "problem_name": "vertex_cover",
            "instance_cnt": 100,
            "relpath": str(path.relative_to(_find_project_root())),
        }
    )

    return rows


def main() -> None:
    eval_module = _load_by_path("scripts/04_05_eval_ml_2ro_e2e.py", "output_path_golden_eval_e2e")
    label_module = _load_by_path("scripts/02_prise_label.py", "output_path_golden_prise_label")

    _gen_json_atomic(eval_module)

    out_dir_rows = _gen_out_dir_grid(eval_module)
    with open(os.path.join(_FIXTURE_DIR, "out_dir_grid.json"), "w") as f:
        json.dump(out_dir_rows, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"out_dir_grid.json: {len(out_dir_rows)} rows")

    eval_dir_rows = _gen_eval_dir_tags(label_module)
    with open(os.path.join(_FIXTURE_DIR, "eval_dir_tags.json"), "w") as f:
        json.dump(eval_dir_rows, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"eval_dir_tags.json: {len(eval_dir_rows)} rows")


if __name__ == "__main__":
    main()
