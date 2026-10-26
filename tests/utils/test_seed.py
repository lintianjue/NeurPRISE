from omegaconf import OmegaConf

from src.utils.seed import resolve_eval_seed


# Motivation: eval.seed should override cfg.seed for evaluation RNG control.
def test_resolve_eval_seed_uses_eval_seed_override():
    cfg = OmegaConf.create({"seed": 42, "eval": {"seed": 3}})

    assert resolve_eval_seed(cfg) == 3


# Motivation: missing/empty eval.seed should preserve cfg.seed to keep dataset identity stable.
def test_resolve_eval_seed_falls_back_to_base_seed_when_missing_or_empty():
    cfg_missing = OmegaConf.create({"seed": 42, "eval": {}})
    cfg_empty = OmegaConf.create({"seed": 42, "eval": {"seed": ""}})

    assert resolve_eval_seed(cfg_missing) == 42
    assert resolve_eval_seed(cfg_empty) == 42
