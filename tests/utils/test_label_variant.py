"""Which physical label file a run reads.

The suffix that turns ``label_..._scen_uniform.pkl`` into ``..._forward_trace.pkl``
or ``..._merged500.pkl`` used to be copy-pasted at three call sites. These tests pin
the single helper's behaviour and, critically, that the *default* resolution is
unchanged, so the 450-record 360/45/45 runs stay reproducible.

See src/utils/label_roles.py and scripts/pipeline/merge_label_roles.py.
"""
import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.utils import omega_conf
from src.utils.vertex_cover.vertex_cover_utils import apply_label_variant_suffix

BASE = "label_prise_seed42_cnt500_n50_s50_p10_scen_uniform.pkl"


def _cfg(output="both", label_variant="__absent__"):
    data = {"path": "data"}
    if label_variant != "__absent__":
        data["label_variant"] = label_variant
    return OmegaConf.create({"problem": {"data": data}, "label": {"prise": {"output": output}}})


@pytest.mark.parametrize(
    "output,expected",
    [
        ("both", BASE),
        ("forward_trace", "label_prise_seed42_cnt500_n50_s50_p10_scen_uniform_forward_trace.pkl"),
        ("bigmilp", "label_prise_seed42_cnt500_n50_s50_p10_scen_uniform_bigmilp.pkl"),
    ],
)
def test_production_knob_still_selects_the_file_it_always_did(output, expected):
    # Motivation: every published medium/large run resolves through label.prise.output.
    # If this drifts, W6's reproduction reads a different file and silently retrains.
    assert apply_label_variant_suffix(BASE, _cfg(output=output)) == expected
    # ... and declaring label_variant: null must not change it either.
    assert apply_label_variant_suffix(BASE, _cfg(output=output, label_variant=None)) == expected


def test_label_variant_selects_a_variant_the_labeller_cannot_produce():
    # Motivation: 02_prise_label.py rejects output=merged500 because it cannot write one.
    # Reading a merged file must not have to impersonate a labelling request.
    expected = "label_prise_seed42_cnt500_n50_s50_p10_scen_uniform_merged500.pkl"
    assert apply_label_variant_suffix(BASE, _cfg(label_variant="merged500")) == expected


def test_label_variant_takes_precedence_over_the_production_knob():
    # Motivation: the two knobs can disagree; consumption must win, deterministically.
    got = apply_label_variant_suffix(BASE, _cfg(output="forward_trace", label_variant="merged500"))
    assert got.endswith("_merged500.pkl")
    # and an explicit "both" variant means the undecorated name, overriding output=
    assert apply_label_variant_suffix(BASE, _cfg(output="forward_trace", label_variant="both")) == BASE


def test_suffix_is_not_applied_twice():
    # Motivation: callers pass an already-decorated name on some paths.
    once = apply_label_variant_suffix(BASE, _cfg(label_variant="merged500"))
    assert apply_label_variant_suffix(once, _cfg(label_variant="merged500")) == once


def test_missing_label_section_is_not_an_error():
    # Motivation: several eval configs carry no label.prise block at all.
    assert apply_label_variant_suffix(BASE, OmegaConf.create({"problem": {"data": {}}})) == BASE


def _capture_file_name(monkeypatch, cfg, **override):
    captured = {}

    class DummyDataModule:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(omega_conf, "DataModule", DummyDataModule)
    if override:
        omega_conf.create_problem_module_with_data_override(cfg, **override)
    else:
        omega_conf.create_problem_module(cfg)
    return captured["file_name"]


@pytest.fixture
def selection_cfg():
    return OmegaConf.create(
        {
            "seed": 42,
            "problem": {
                "name": "selection",
                "module": "src.problems.selection.SelectionProblem",
                "instance": {
                    "n_nodes": 50,
                    "n_scenarios": 100,
                    "p_select": 25,
                    "n_min_cost": 1,
                    "n_max_cost": 100,
                    "scenario_name": "uniform",
                },
                "data": {
                    "path": "data",
                    "label_variant": None,
                    "inst_spec_label_pattern": (
                        "label_{label_type}_seed{seed}_cnt{instance_cnt}_n{n_nodes}_s{n_scenarios}_"
                        "p{p_select}_scen_{scenario_name}.pkl"
                    ),
                },
            },
            "data": {"instance_cnt": 500, "val_split": 0.2, "test_split": 0.0},
            "label": {"type": "prise", "prise": {"output": "both"}},
            "eval": {"batch_size": 1, "num_workers": 0},
        }
    )


def test_both_omega_conf_entry_points_apply_the_same_variant(selection_cfg, monkeypatch):
    # Motivation: create_problem_module and ..._with_data_override are separate code
    # paths that must resolve the same file; they carried duplicated suffix blocks.
    selection_cfg.problem.data.label_variant = "merged500"
    expected = "label_prise_seed42_cnt500_n50_s100_p25_scen_uniform_merged500.pkl"

    assert _capture_file_name(monkeypatch, selection_cfg) == expected
    assert _capture_file_name(
        monkeypatch,
        selection_cfg,
        instance_cnt=500,
        n_nodes=50,
        n_scenarios=100,
        p_edge=10,
        scenario_name=None,
        seed=42,
    ) == expected


def test_third_call_site_has_not_been_re_inlined():
    # Motivation: the drift this helper exists to prevent is someone pasting the
    # block back into the training entry point, where no unit test would see it.
    src = Path(__file__).resolve().parents[2] / "scripts" / "03_train.py"
    text = src.read_text()
    assert "apply_label_variant_suffix(file_name, cfg)" in text
    assert "output_mode" not in text, (
        "scripts/03_train.py resolves the label variant inline again; it must call "
        "apply_label_variant_suffix so the three call sites cannot disagree."
    )
