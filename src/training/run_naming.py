"""The single source of the checkpoint filename/path naming used by
``scripts/03_train.py``. Anything that predicts a launch's checkpoint name before training
calls this module rather than re-implementing the statements.
"""
from __future__ import annotations

import os

from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig


def _config_name_fragment(node: DictConfig | None, default: str) -> str:
    """
    Turn a Hydra/OmegaConf node into a short, filesystem-safe name fragment.
    Priority:
      1) 'name' field if present
      2) last component of '_target_'
      3) provided default
    """
    if node is None:
        return default

    # Try explicit name
    try:
        if "name" in node and node.name:
            frag = str(node.name)
        # Otherwise try _target_
        elif "_target_" in node and node._target_:
            frag = str(node._target_).split(".")[-1]
        else:
            frag = default
    except Exception:
        frag = default

    # Make filesystem-safe
    frag = frag.replace("/", "-").replace(".", "-")
    return frag


def _hydra_choice_fragment(group: str, default: str) -> str:
    """
    Prefer the Hydra config choice (file name) for a config group.
    Falls back to the provided default when unavailable.
    """
    try:
        choices = HydraConfig.get().runtime.choices
    except Exception:
        choices = {}

    if not choices or group not in choices:
        return default

    choice = choices.get(group)
    if not choice:
        return default

    choice = str(choice).replace("/", "-").replace(".", "-")
    return choice


def _hydra_config_name_fragment(default: str = "") -> str:
    """
    Name of the active Hydra config (e.g., 'train_teacher' when using --config-name train_teacher
    or hydra.job.config_name override). Returns `default` if not available.
    """
    try:
        cfg_name = HydraConfig.get().job.config_name
    except Exception:
        cfg_name = default

    if not cfg_name:
        return default

    cfg_name = str(cfg_name).replace("/", "-").replace(".", "-")
    return cfg_name


def checkpoint_name(cfg: DictConfig) -> str:
    """The best-model checkpoint BASENAME a launch of ``cfg`` writes.

    ``name_suffix`` already begins with ``cfg_part``'s leading underscore, so
    the join below yields a DOUBLE underscore when ``config_name`` is set (e.g.
    ``…_s50__c-train_scale_norm_…``) -- keep it: collapsing it to one underscore names a
    path nothing ever writes.
    """
    seed = cfg.seed
    instance_cnt = cfg.data.get("instance_cnt", 1000)
    inst_cfg = cfg.problem.instance
    n_nodes = getattr(inst_cfg, "n_nodes", None)
    n_facilities = getattr(inst_cfg, "n_facilities", None)
    n_customers = getattr(inst_cfg, "n_customers", None)
    n_scenarios = cfg.problem.instance.n_scenarios

    config_name = _hydra_config_name_fragment(default="")
    model_name = _hydra_choice_fragment(
        "model", _config_name_fragment(cfg.get("model", None), "model"))
    loss_name = _hydra_choice_fragment(
        "losses", _config_name_fragment(cfg.get("losses", None), "loss"))

    if n_facilities is not None and n_customers is not None:
        base = f"s{seed}_i{instance_cnt}_m{n_facilities}_n{n_customers}_s{n_scenarios}"
    else:
        base = f"s{seed}_i{instance_cnt}_n{n_nodes}_s{n_scenarios}"
    cfg_part = f"_c-{config_name}" if config_name else ""
    name_suffix = f"{cfg_part}_m-{model_name}_l-{loss_name}"
    # Optional unique tag (transfer runs: distinguishes mode/budget cells that
    # would otherwise collide, since seed/model/loss are identical across them).
    output_tag = str(cfg.train.get("output_tag", "") or "")
    tag_part = f"_{output_tag}" if output_tag else ""
    best_model_filename = f"{base}_{name_suffix}{tag_part}_best.pt"
    return best_model_filename


def checkpoint_path(cfg: DictConfig) -> str:
    """The full best-model checkpoint path a launch of ``cfg`` writes."""
    return os.path.join(cfg.train.model_output_dir, checkpoint_name(cfg))
