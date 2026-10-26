"""Regression tests for the training/DataModule normalization boundary."""

import importlib.util
from pathlib import Path

from omegaconf import OmegaConf


def _load_train_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "03_train.py"
    spec = importlib.util.spec_from_file_location("train_script_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_scale_mode_reaches_datamodule_kwargs():
    # Motivation: scale normalization must reach the DataModule; a resolved-only value is not evidence.
    module = _load_train_module()
    cfg = OmegaConf.create({"data": {"normalize": True, "normalize_mode": "scale"}})
    assert module._data_normalization_kwargs(cfg) == {
        "normalize": True,
        "normalize_mode": "scale",
    }


def test_missing_mode_preserves_legacy_minmax_default():
    # Motivation: configs without normalize_mode must remain backward compatible.
    module = _load_train_module()
    cfg = OmegaConf.create({"data": {"normalize": False}})
    assert module._data_normalization_kwargs(cfg) == {
        "normalize": False,
        "normalize_mode": "minmax",
    }
