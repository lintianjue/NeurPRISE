#!/usr/bin/env python3
import pickle
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig

from src.core.types import Instance
from src.utils.label_utils import build_problem_from_cfg
from src.utils.omega_conf import register_omega_conf_custom_resolvers
from src.utils.run_io import create_new, open_hydra_run, data_paths, data_output_path
from src.utils.seed import seed_everything
from src.utils.vertex_cover.vertex_cover_utils import get_instance_file_name

import logging

logger = logging.getLogger(__name__)


def _instance_output_path(rp, problem_name: str, file_name: str) -> Path:
    """Dataset destination shared with the next stage under the default layout."""
    return data_output_path(rp, problem_name, file_name)


@hydra.main(config_path="../configs", config_name="exp", version_base=None)
def main(cfg: DictConfig):
    logger.info("Starting instance generation (Hydra)")

    seed = cfg.get("seed", 42)
    instance_cnt = cfg.data.get("instance_cnt", 1000)

    problem_name = cfg.problem.name
    log_every = cfg.get("log_every", 100)

    seed_everything(seed)
    rp = open_hydra_run(cfg, __file__)

    problem = build_problem_from_cfg(cfg.problem)

    inst_cfg = cfg.problem.instance
    n_nodes = getattr(inst_cfg, "n_nodes", None)
    n_facilities = getattr(inst_cfg, "n_facilities", None)
    n_customers = getattr(inst_cfg, "n_customers", None)

    # Compose deterministic file name (consistent with training)
    dump_file_name = get_instance_file_name(
        seed=cfg.seed,
        instance_cnt=cfg.data.get("instance_cnt", 1000),
        n_nodes=n_nodes,
        n_scenarios=inst_cfg.n_scenarios,
        n_facilities=n_facilities,
        n_customers=n_customers,
        pattern=cfg.problem.data.get('inst_spec_pattern', None),
        # Optional extras; only used if pattern references them
        p_edge=getattr(cfg, "p_edge", 10),
        p_select=getattr(inst_cfg, "p_select", None),
        scenario_name=inst_cfg.scenario_name,
    )
    dump_path = _instance_output_path(rp, problem_name, dump_file_name)

    # Generate
    all_instances: List[Instance] = []
    for i in range(instance_cnt):
        inst = problem.sample()
        all_instances.append(inst)

        if (i + 1) % max(1, log_every) == 0:
            pct = (i + 1) / instance_cnt * 100.0
            logger.info(f"[create_instance_data] {i + 1}/{instance_cnt} ({pct:.1f}%) generated")

    with create_new(data_paths(rp), dump_path, mode="wb") as f:
        pickle.dump(all_instances, f)

    logger.info(f"[create_instance_data] Saved {instance_cnt} instances to {dump_path}")


if __name__ == "__main__":
    register_omega_conf_custom_resolvers()
    main()
