import importlib.util
import sys
from pathlib import Path

from omegaconf import OmegaConf


def _load_train_module():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    module_path = root / "scripts" / "03_train.py"
    spec = importlib.util.spec_from_file_location("train_script_ddp", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ddp_find_unused_defaults_true():
    # Motivation: multi-GPU should be robust when some params are unused.
    module = _load_train_module()
    cfg = OmegaConf.create({"train": {}})
    assert module._resolve_ddp_find_unused_parameters(cfg) is True


def test_ddp_find_unused_respects_override():
    # Motivation: allow explicit opt-out for performance.
    module = _load_train_module()
    cfg = OmegaConf.create({"train": {"ddp_find_unused_parameters": False}})
    assert module._resolve_ddp_find_unused_parameters(cfg) is False
