import os
import random
import numpy as np
import torch
from omegaconf import DictConfig

# Shared global RNG instance
_GLOBAL_RNG = np.random.default_rng()
_GLOBAL_SEED = None

def seed_everything(seed: int):
    global _GLOBAL_SEED, _GLOBAL_RNG
    _GLOBAL_SEED = seed

    # Seed all libraries
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    # Also seed the shared RNG stream
    _GLOBAL_RNG = np.random.default_rng(seed)

def get_rng():
    """
    Return the shared global RNG.
    All randomness in the project should come from this.
    """
    return _GLOBAL_RNG


def resolve_eval_seed(cfg: DictConfig) -> int:
    """
    Resolve RNG seed for evaluation scripts.

    - cfg.seed controls dataset identity (default: 42)
    - eval.seed optionally overrides runtime RNG for evaluation only
    """
    base_seed = int(cfg.get("seed", 42))
    eval_cfg = cfg.get("eval", {})
    eval_seed = eval_cfg.get("seed", None) if eval_cfg is not None else None
    if eval_seed in (None, ""):
        return base_seed
    return int(eval_seed)
