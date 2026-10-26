"""Label filename suffixes have one function; dead label helpers are gone."""
from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig

import src.utils.label_utils as label_utils
from src.utils.label_utils import (
    _resolve_data_folder,
    _resolve_instance_spec_path,
    _resolve_label_path,
    label_filename,
)

_REPO = Path(__file__).resolve().parents[2]
_CONFIGS_DIR = _REPO / "configs"


def _compose(problem: str):
    with initialize_config_dir(config_dir=str(_CONFIGS_DIR), version_base=None):
        cfg = compose(config_name="exp", return_hydra_config=True, overrides=[f"problem={problem}"])
        HydraConfig.instance().set_config(cfg)
        return cfg


_BASE = "/d/sel/label_prise_seed42_cnt2500_n20_s50_p10_scen_uniform.pkl"

_GRID_EXPECTED = {
    ("both", None): "/d/sel/label_prise_seed42_cnt2500_n20_s50_p10_scen_uniform.pkl",
    ("both", (0, 100)): "/d/sel/label_prise_seed42_cnt2500_n20_s50_p10_scen_uniform_shard0-100.pkl",
    ("bigmilp", None): "/d/sel/label_prise_seed42_cnt2500_n20_s50_p10_scen_uniform_bigmilp.pkl",
    ("bigmilp", (0, 100)): "/d/sel/label_prise_seed42_cnt2500_n20_s50_p10_scen_uniform_bigmilp_shard0-100.pkl",
    ("forward_trace", None): "/d/sel/label_prise_seed42_cnt2500_n20_s50_p10_scen_uniform_forward_trace.pkl",
    ("forward_trace", (0, 100)): "/d/sel/label_prise_seed42_cnt2500_n20_s50_p10_scen_uniform_forward_trace_shard0-100.pkl",
}


@pytest.mark.parametrize("output_mode,shard", list(_GRID_EXPECTED))
def test_label_filename_grid(output_mode, shard) -> None:
    """(M) swap the order of the two suffix steps."""
    assert label_filename(_BASE, output_mode, shard) == _GRID_EXPECTED[(output_mode, shard)]


def test_merge_label_shards_regex_accepts_the_new_name() -> None:
    """(M) change _shard{a}-{b} to _shard{a}_{b}."""
    module_path = _REPO / "scripts" / "pipeline" / "merge_label_shards.py"
    spec = importlib.util.spec_from_file_location("merge_label_shards_mod", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    SHARD_RE = module.SHARD_RE

    sharded = Path(label_filename(_BASE, "bigmilp", (0, 100))).stem
    m = SHARD_RE.search(sharded)
    assert m is not None
    assert m.groups() == ("0", "100")

    unsharded = Path(label_filename(_BASE, "bigmilp", None)).stem
    assert SHARD_RE.search(unsharded) is None


def test_resolve_instance_spec_path_mirrors_p_select_and_p_edge() -> None:
    """The two real configs pin the current names; neither exercises the mirroring (vertex_cover
    declares both p_edge and p_select; selection's pattern only reads p_select, and
    get_instance_file_name's own alias table already maps p_select -> p_edge). The synthetic
    config below is the one case the mirroring is for: a pattern that reads {p_edge} on an
    instance that only declares p_select (the reverse direction, which the alias table does
    not cover).
    (M) delete the two mirroring `if` statements (label_utils.py `_resolve_instance_spec_path`)."""
    from omegaconf import OmegaConf

    cfg_syn = OmegaConf.create({
        "seed": 1,
        "data": {"instance_cnt": 5},
        "problem": {
            "name": "synthetic",
            "instance": {"n_nodes": 4, "n_scenarios": 3, "p_select": 7, "scenario_name": "u"},
            "data": {"inst_spec_pattern": "inst_s{seed}_c{instance_cnt}_n{n_nodes}_s{n_scenarios}_p{p_edge}_{scenario_name}.pkl"},
        },
    })
    assert _resolve_instance_spec_path(cfg_syn, "/root", "synthetic") == "/root/synthetic/inst_s1_c5_n4_s3_p7_u.pkl"

    cfg_sel = _compose("selection")
    assert _resolve_instance_spec_path(cfg_sel, "/root", cfg_sel.problem.name) == (
        "/root/selection/inst_seed42_cnt100_n50_s10_p25_scen_uniform.pkl"
    )
    cfg_vc = _compose("vertex_cover")
    assert _resolve_instance_spec_path(cfg_vc, "/root", cfg_vc.problem.name) == (
        "/root/vertex_cover/inst_seed42_cnt100_n8_s10_p10_scen_uniform.pkl"
    )


def test_resolve_label_path_uses_label_type_and_creates_nothing(tmp_path: Path) -> None:
    """No directory-creation side effect: an input root is read-only, so
    this function only computes a path.
    (M) paste the makedirs back.
    (M) hard-code label_type="prise" in the get_instance_file_name call."""
    cfg = _compose("selection")
    path = _resolve_label_path(cfg, str(tmp_path), cfg.problem.name)
    assert path == str(tmp_path / "selection" / "label_prise_seed42_cnt100_n50_s10_p25_scen_uniform.pkl")
    assert not (tmp_path / "selection").exists()

    path_other = _resolve_label_path(cfg, str(tmp_path), cfg.problem.name, label_type="bigmilp")
    assert path_other == str(tmp_path / "selection" / "label_bigmilp_seed42_cnt100_n50_s10_p25_scen_uniform.pkl")


def test_resolve_data_folder_joins_project_root(tmp_path: Path, monkeypatch) -> None:
    """(M) return cfg_data_path without the join.
    (M) paste the makedirs back."""
    monkeypatch.setattr(label_utils, "_find_project_root", lambda: str(tmp_path))
    cfg = _compose("selection")
    cfg.problem.data.path = "data/"
    result = _resolve_data_folder(cfg)
    assert os.path.normpath(result) == os.path.normpath(str(tmp_path / "data"))
    assert not os.path.exists(result)


def test_dead_label_helpers_are_gone() -> None:
    """(M) paste one back."""
    for path in (_REPO / "src" / "utils" / "label_utils.py", _REPO / "scripts" / "02_prise_label.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert "_get_label_top_k" not in names, f"{path} still defines _get_label_top_k"
        assert "_eval_red_obj_only" not in names, f"{path} still defines _eval_red_obj_only"
