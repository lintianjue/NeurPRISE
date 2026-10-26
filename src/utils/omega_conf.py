from omegaconf import OmegaConf, DictConfig
import os
import logging

import sys

from src.training.datamodule import (
    ROLE_CONTRACT_EVAL_ONLY,
    ROLE_CONTRACT_TRAIN_VAL_TEST,
    DataModule,
)
from src.utils.utils import load_problem
from src.utils.vertex_cover.vertex_cover_utils import (
    apply_label_variant_suffix,
    get_instance_file_name,
)


def register_omega_conf_custom_resolvers():
    from src.utils.run_io import REPO, DEFAULT_BASE, INPUT_ROOT_VAR, RUN_ROOT_VAR
    os.environ[INPUT_ROOT_VAR] = os.environ.get(INPUT_ROOT_VAR) or str(REPO)
    os.environ[RUN_ROOT_VAR] = os.environ.get(RUN_ROOT_VAR) or str(DEFAULT_BASE)
    def _python_name(path: str, default: str = "train") -> str:
        if path is None:
            path = sys.argv[0] if sys.argv else "03_train.py"
        try:
            return os.path.splitext(os.path.basename(path))[0]
        except Exception:
            return default

    OmegaConf.register_new_resolver("python_name", _python_name, replace=True)

def _normalize_selection_p_select(problem_name: str, problem_kwargs: dict) -> None:
    """Ensure selection p_select is valid when n_nodes changes."""
    if not str(problem_name).startswith("selection"):
        return
    n_nodes = problem_kwargs.get("n_nodes", None)
    if n_nodes is None:
        return
    new_p = int(n_nodes) // 2
    prev = problem_kwargs.get("p_select", None)
    if prev is None or int(prev) != int(new_p):
        logging.warning(
            "Selection p_select=%s overridden for n_nodes=%s; using p_select=%s",
            str(prev),
            str(n_nodes),
            str(new_p),
        )
    problem_kwargs["p_select"] = new_p


def _resolve_data_role_contract(cfg: DictConfig) -> str:
    """Opt into the all-test contract only for explicit BigMILP evaluation."""
    label_output = str(cfg.get("label", {}).get("prise", {}).get("output", ""))
    eval_split = str(cfg.get("eval", {}).get("split", ""))
    if label_output == "bigmilp" and eval_split in {"test", "all"}:
        return ROLE_CONTRACT_EVAL_ONLY
    return ROLE_CONTRACT_TRAIN_VAL_TEST


def create_problem_module(cfg: DictConfig):
    # Load problem using Hydra config
    problem_module = cfg.problem.module
    problem_kwargs = OmegaConf.to_container(cfg.problem.instance, resolve=True)
    _normalize_selection_p_select(cfg.problem.name, problem_kwargs)
    problem = load_problem(problem_module, problem_kwargs)
    inst_cfg = cfg.problem.instance
    data_cfg = cfg.problem.data
    label_type = cfg.label.type

    n_nodes = getattr(inst_cfg, "n_nodes", None)
    n_facilities = getattr(inst_cfg, "n_facilities", None)
    n_customers = getattr(inst_cfg, "n_customers", None)

    p_edge = getattr(inst_cfg, "p_edge", None)
    if p_edge is None and str(cfg.problem.name).startswith("selection") and n_nodes is not None:
        # Hardcode p as n/2 for selection (used only for filename pattern)
        p_edge = int(n_nodes) // 2

    file_name = get_instance_file_name(
        seed=cfg.seed,
        instance_cnt=cfg.data.get("instance_cnt", 1000),
        n_nodes=n_nodes,
        n_scenarios=inst_cfg.n_scenarios,
        n_facilities=n_facilities,
        n_customers=n_customers,
        pattern=data_cfg.get("inst_spec_label_pattern", None),
        # Optional extras; only used if pattern references them
        p_edge=p_edge,
        scenario_name=getattr(inst_cfg, "scenario_name", "default"),
        label_type=label_type
    )
    file_name = apply_label_variant_suffix(file_name, cfg)

    # 3. Problem + DataModule
    dm = DataModule(
        problem=problem,
        data_path=cfg.problem.data.path,
        problem_name=cfg.problem.name,
        file_name=file_name,
        batch_size=cfg.eval.batch_size,
        num_workers=cfg.eval.num_workers,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        split_mode=cfg.data.get("split_mode", "ratio"),
        val_split_number=cfg.data.get("val_split_number", 0),
        test_abs_split_number=cfg.data.get("test_abs_split_number", 0),
        train_split_number=cfg.data.get("train_split_number", 0),
        normalize=bool(cfg.data.get("normalize", False)),
        normalize_mode=str(cfg.data.get("normalize_mode", "minmax")),
        role_contract=_resolve_data_role_contract(cfg),
    )
    return dm, problem


def create_problem_module_with_data_override(
    cfg: DictConfig,
    *,
    instance_cnt: int,
    n_nodes: int | None,
    n_scenarios: int,
    n_facilities: int | None = None,
    n_customers: int | None = None,
    p_edge: int | None = None,
    scenario_name: str | None = None,
    seed: int | None = None,
):
    """
    Build (dm, problem) like create_problem_module(cfg), but allow overriding the
    dataset-identifying parameters used to locate the pickle file.

    This is meant for generalization evaluation where:
      - eval.ckpt_path points to the trained model
      - eval data file is intentionally different from training data
    """
    # 1) Build problem from cfg (same as default)
    problem_module = cfg.problem.module
    problem_kwargs = OmegaConf.to_container(cfg.problem.instance, resolve=True)
    _normalize_selection_p_select(cfg.problem.name, problem_kwargs)
    problem = load_problem(problem_module, problem_kwargs)

    # 2) Resolve filename inputs (override-only affects file selection)
    inst_cfg = cfg.problem.instance
    data_cfg = cfg.problem.data
    label_type = cfg.label.type

    eff_seed = int(cfg.seed if seed is None else seed)
    eff_scenario_name = getattr(inst_cfg, "scenario_name", "default") if scenario_name is None else str(scenario_name)

    p_edge_for_filename = p_edge
    if str(cfg.problem.name) == "selection" and n_nodes is not None:
        # Match create_problem_module(): selection filenames always use p_select = n_nodes//2.
        p_edge_for_filename = int(n_nodes) // 2

    file_name = get_instance_file_name(
        seed=eff_seed,
        instance_cnt=int(instance_cnt),
        n_nodes=None if n_nodes is None else int(n_nodes),
        n_scenarios=int(n_scenarios),
        n_facilities=None if n_facilities is None else int(n_facilities),
        n_customers=None if n_customers is None else int(n_customers),
        pattern=data_cfg.get("inst_spec_label_pattern", None),
        p_edge=p_edge_for_filename,
        scenario_name=eff_scenario_name,
        label_type=label_type,
    )
    file_name = apply_label_variant_suffix(file_name, cfg)

    # 3) DataModule uses same base path + problem_name
    dm = DataModule(
        problem=problem,
        data_path=cfg.problem.data.path,
        problem_name=cfg.problem.name,
        file_name=file_name,
        batch_size=cfg.eval.batch_size,
        num_workers=cfg.eval.num_workers,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        split_mode=cfg.data.get("split_mode", "ratio"),
        val_split_number=cfg.data.get("val_split_number", 0),
        test_abs_split_number=cfg.data.get("test_abs_split_number", 0),
        train_split_number=cfg.data.get("train_split_number", 0),
        normalize=bool(cfg.data.get("normalize", False)),
        normalize_mode=str(cfg.data.get("normalize_mode", "minmax")),
        role_contract=_resolve_data_role_contract(cfg),
    )
    return dm, problem
