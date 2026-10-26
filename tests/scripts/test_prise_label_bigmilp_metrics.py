import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

_GOLDEN_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "output_path_golden"


def _load_prise_label_module():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    module_path = root / "scripts" / "02_prise_label.py"
    spec = importlib.util.spec_from_file_location("prise_label", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_collect_bigmilp_metrics_main_flow():
    """Motivation: summarize BigMILP timing/objectives for a full set of instances."""
    module = _load_prise_label_module()
    results = [
        {"obj": 10.0, "runtime_sec": 2.0},
        {"obj": 14.0, "runtime_sec": 4.0},
        {"obj": 12.0, "runtime_sec": 6.0},
    ]

    metrics = module._collect_bigmilp_metrics(results, process_cnt=3)

    assert metrics["num_instances"] == 3
    assert np.isclose(metrics["full_obj_mean"], 12.0)
    assert np.isclose(metrics["full_obj_std"], np.std([10.0, 14.0, 12.0]))
    assert np.isclose(metrics["eval_walltime_sec"], 12.0)
    assert np.isclose(metrics["eval_walltime_per_instance_sec"], 4.0)


def test_collect_bigmilp_metrics_filters_missing_values():
    """Motivation: ignore None/NaN values while computing BigMILP metrics."""
    module = _load_prise_label_module()
    results = [
        {"obj": None, "runtime_sec": None},
        {"obj": float("nan"), "runtime_sec": 1.5},
        {"obj": 5.0, "runtime_sec": float("nan")},
    ]

    metrics = module._collect_bigmilp_metrics(results, process_cnt=3)

    assert metrics["num_instances"] == 3
    assert np.isclose(metrics["full_obj_mean"], 5.0)
    assert np.isclose(metrics["full_obj_std"], 0.0)
    assert np.isclose(metrics["eval_walltime_sec"], 1.5)
    assert np.isclose(metrics["eval_walltime_per_instance_sec"], 1.5)


def test_build_eval_dir_selection(tmp_path):
    """Motivation: mirror eval output naming for selection/vertex_cover problems.
    (M) drop root from the join (return the eval/.../tag path unrooted)."""
    module = _load_prise_label_module()
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "problem": {
                "name": "selection",
                "instance": {"n_nodes": 20, "n_scenarios": 50},
            },
        }
    )
    root = tmp_path

    path = module._build_eval_dir(cfg, "selection", instance_cnt=2500, root=root)

    assert path == root / "eval/selection/s42_i2500_n20_s50"


def test_build_eval_dir_cflp(tmp_path):
    """Motivation: mirror eval output naming for CFLP problems.
    (M) swap the m/n fields of the cflp tag."""
    module = _load_prise_label_module()
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "problem": {
                "name": "cflp_ebp",
                "instance": {"n_facilities": 70, "n_customers": 80, "n_scenarios": 100},
            },
        }
    )
    root = tmp_path

    path = module._build_eval_dir(cfg, "cflp_ebp", instance_cnt=500, root=root)

    assert path == root / "eval/cflp_ebp/s42_i500_m70_n80_s100"


def test_build_eval_dir_matches_golden(tmp_path):
    """Cross-check against tests/fixtures/output_path_golden/eval_dir_tags.json, generated
    from src.utils.utils._find_project_root()."""
    module = _load_prise_label_module()
    rows = json.loads((_GOLDEN_DIR / "eval_dir_tags.json").read_text())
    root = tmp_path

    for row in rows:
        cfg = OmegaConf.create(
            {
                "seed": row["cfg"]["seed"],
                "problem": {"name": row["cfg"]["problem"], "instance": dict(row["cfg"])},
            }
        )
        path = module._build_eval_dir(
            cfg, row["problem_name"], instance_cnt=row["instance_cnt"], root=root
        )
        assert str(path.relative_to(root)) == row["relpath"]
