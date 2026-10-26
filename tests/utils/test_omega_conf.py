import pytest
from omegaconf import OmegaConf

from src.utils import omega_conf


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
                    "inst_spec_label_pattern": (
                        "label_{label_type}_seed{seed}_cnt{instance_cnt}_n{n_nodes}_s{n_scenarios}_"
                        "p{p_select}_scen_{scenario_name}.pkl"
                    ),
                },
            },
            "data": {
                "instance_cnt": 500,
                "val_split": 0.2,
                "test_split": 0.0,
            },
            "label": {"type": "prise"},
            "eval": {"batch_size": 1, "num_workers": 0},
        }
    )


def test_create_problem_module_with_data_override_selection_uses_half_nodes(selection_cfg, monkeypatch):
    # Motivation: ensure selection override filenames use p_select derived from n_nodes.
    captured = {}

    class DummyDataModule:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(omega_conf, "DataModule", DummyDataModule)

    omega_conf.create_problem_module_with_data_override(
        selection_cfg,
        instance_cnt=500,
        n_nodes=50,
        n_scenarios=100,
        p_edge=10,
        scenario_name=None,
        seed=42,
    )

    assert captured["file_name"] == "label_prise_seed42_cnt500_n50_s100_p25_scen_uniform.pkl"


def test_bigmilp_test_eval_selects_eval_only_role_contract(selection_cfg, monkeypatch):
    # Motivation: a BigMILP file can label the whole population, but the test
    # consumer must validate only the configured held-out role.
    captured = {}

    class DummyDataModule:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(omega_conf, "DataModule", DummyDataModule)
    selection_cfg.label["prise"] = {"output": "bigmilp"}
    selection_cfg.eval.split = "test"

    omega_conf.create_problem_module(selection_cfg)

    assert captured["role_contract"] == "eval_only"


@pytest.mark.parametrize("split", ["train", "val"])
def test_bigmilp_train_or_val_keeps_fail_closed_role_contract(
    selection_cfg,
    monkeypatch,
    split,
):
    # Motivation: choosing BigMILP must not turn missing training supervision
    # into an implicit exception when the requested consumer is train or val.
    captured = {}

    class DummyDataModule:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(omega_conf, "DataModule", DummyDataModule)
    selection_cfg.label["prise"] = {"output": "bigmilp"}
    selection_cfg.eval.split = split

    omega_conf.create_problem_module(selection_cfg)

    assert captured["role_contract"] == "train_val_test"
