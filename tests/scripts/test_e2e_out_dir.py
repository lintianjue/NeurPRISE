"""Shared e2e output-dir rule (src.utils.eval_helpers.resolve_e2e_out_dir) and the SOR
(ab-approx) entry's run_io wiring. The ML entry's wiring is pinned in test_eval_e2e_run_io."""
import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.utils.eval_helpers import resolve_e2e_out_dir

REPO = Path(__file__).resolve().parents[2]


def _cfg(output_dir):
    return OmegaConf.create({"output_dir": output_dir, "name": "N", "k_ratio": 0.2})


def test_relative_output_dir_sits_under_the_run_dir(tmp_path):
    """(M) resolve a relative output_dir against the cwd even when run_dir is given."""
    out = resolve_e2e_out_dir(_cfg("eval/selection/s42"), tmp_path / "run")
    assert out == str(tmp_path / "run" / "eval" / "selection" / "s42" / "N" / "e2e_2ro" / "k_0.2")


def test_absolute_output_dir_is_kept_and_empty_defaults_to_run_dir_eval(tmp_path):
    assert resolve_e2e_out_dir(_cfg(str(tmp_path / "abs")), tmp_path / "run") == \
        str(tmp_path / "abs" / "N" / "e2e_2ro" / "k_0.2")
    assert resolve_e2e_out_dir(_cfg(""), tmp_path / "run") == \
        str(tmp_path / "run" / "eval" / "N" / "e2e_2ro" / "k_0.2")
    with pytest.raises(ValueError):
        resolve_e2e_out_dir(_cfg(""), None)

