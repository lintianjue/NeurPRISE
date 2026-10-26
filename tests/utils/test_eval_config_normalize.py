from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


@pytest.fixture
def config_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs"


@pytest.fixture
def compose_cfg(config_dir: Path) -> Callable[[str], DictConfig]:
    def _compose(config_name: str) -> DictConfig:
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            return compose(config_name=config_name)

    return _compose


def test_eval_scale_norm_sets_data_normalize_true(compose_cfg: Callable[[str], DictConfig]) -> None:
    # Motivation: ensure eval_scale_norm enables data normalization for 2RO eval runs.
    cfg = compose_cfg("eval_scale_norm")
    assert bool(cfg.data.normalize) is True
