#!/usr/bin/env python3
"""Generate PRISE labels and (optionally) record BigMILP timing metrics."""
import json
import pickle
from typing import List, Optional, Literal, Any
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from src.core.types import Instance, Labels  # ForwardTrace lives in types.py too if you added it there
from src.utils.label_utils import (
    _resolve_data_folder,
    _resolve_instance_spec_path,
    build_problem_from_cfg,
    _resolve_label_path,
    label_filename,
)
from src.utils.omega_conf import register_omega_conf_custom_resolvers
from src.utils.run_io import open_hydra_run, create_new, data_paths, data_output_path
from src.utils.seed import seed_everything

from src.heuristics.eval import (
    _instance_to_numpy_dict,
    _forward_with_trace,        # <<< NEW
    _compute_effective_K,
)

import logging
logger = logging.getLogger(__name__)


OutputMode = Literal["bigmilp", "forward_trace", "both"]


def _round_value(val: Any, decimals: int) -> Any:
    if val is None:
        return None
    if isinstance(val, (float, np.floating)):
        return float(np.round(val, decimals))
    if isinstance(val, np.ndarray):
        if np.issubdtype(val.dtype, np.floating):
            return np.round(val, decimals)
        return val
    if isinstance(val, dict):
        return {k: _round_value(v, decimals) for k, v in val.items()}
    if isinstance(val, list):
        return [_round_value(v, decimals) for v in val]
    return val


def _round_forward_trace(trace, decimals: int):
    if trace is None:
        return None
    trace.candidate_scores = _round_value(trace.candidate_scores, decimals)
    trace.best_scores = _round_value(trace.best_scores, decimals)
    trace.gains = _round_value(trace.gains, decimals)
    return trace


def _round_label_floats(label: Labels, decimals: int) -> Labels:
    label.obj = _round_value(label.obj, decimals)
    label.feasible = _round_value(label.feasible, decimals)
    label.x = _round_value(label.x, decimals)
    label.sce_target_idx = _round_value(label.sce_target_idx, decimals)
    label.sce_ranking = _round_value(label.sce_ranking, decimals)
    label.sce_scores = _round_value(label.sce_scores, decimals)
    label.sce_top_idx = _round_value(label.sce_top_idx, decimals)
    label.obj_bound = _round_value(label.obj_bound, decimals)
    label.mip_gap = _round_value(label.mip_gap, decimals)
    label.meta = _round_value(label.meta, decimals)
    label.forward_trace = _round_forward_trace(label.forward_trace, decimals)
    return label


def _get_output_mode(cfg: DictConfig) -> OutputMode:
    """
    Controls what to compute/store per instance.

    Supported:
      - "bigmilp":        only robust big-MILP output (obj/x); no forward trace fields
      - "forward_trace":  only forward trace fields (sce_top_idx/ranking/forward_trace); no big-MILP solve
      - "both":           current behavior
    """
    # Prefer a dedicated knob if present; fall back to current behavior.
    mode = None
    if "label" in cfg and "prise" in cfg.label and "output" in cfg.label.prise:
        mode = str(cfg.label.prise.output).strip().lower()

    if mode in (None, ""):
        return "both"

    if mode in ("both", "forward_trace", "bigmilp"):
        return mode
    raise ValueError(f"Unknown cfg.label.prise.output='{mode}'. Expected bigmilp|forward_trace|both.")


def _get_process_first_pct(cfg: DictConfig) -> Optional[float]:
    """
    Process only the first X% instances.
    Exclusive with cfg.label.prise.process_last_pct.
    """
    if "label" in cfg and "prise" in cfg.label and "process_first_pct" in cfg.label.prise:
        val = cfg.label.prise.process_first_pct
        if val is None:
            return None
        pct = float(val)
        if not (0.0 < pct <= 100.0):
            raise ValueError(f"cfg.label.prise.process_first_pct must be in (0, 100], got {pct}")
        return pct
    return None


def _get_process_last_pct(cfg: DictConfig) -> Optional[float]:
    """
    Process only the last X% instances.
    Exclusive with cfg.label.prise.process_first_pct.
    """
    if "label" in cfg and "prise" in cfg.label and "process_last_pct" in cfg.label.prise:
        val = cfg.label.prise.process_last_pct
        if val is None:
            return None
        pct = float(val)
        if not (0.0 < pct <= 100.0):
            raise ValueError(f"cfg.label.prise.process_last_pct must be in (0, 100], got {pct}")
        return pct
    return None


def _get_index_shard(cfg: DictConfig) -> Optional[tuple[int, int]]:
    """Explicit 0-based half-open shard [index_start, index_end).

    Used to split one labeling job across parallel processes. Returns None
    unless both bounds are set, so existing callers are unaffected.
    """
    if "label" not in cfg or "prise" not in cfg.label:
        return None
    start = cfg.label.prise.get("index_start", None)
    end = cfg.label.prise.get("index_end", None)
    if start is None and end is None:
        return None
    if start is None or end is None:
        raise ValueError(
            "label.prise.index_start and index_end must be set together."
        )
    start, end = int(start), int(end)
    if start < 0 or end <= start:
        raise ValueError(
            f"Need 0 <= index_start < index_end; got [{start}, {end})."
        )
    return start, end


def _get_process_range(cfg: DictConfig, instance_cnt: int) -> tuple[int, int, str]:
    """
    Returns (start_idx0_inclusive, end_idx0_exclusive, mode_desc).

    Default: full range.
    Mutually exclusive:
      - label.prise.process_first_pct
      - label.prise.process_last_pct
      - label.prise.{index_start,index_end}
    """
    first_pct = _get_process_first_pct(cfg)
    last_pct = _get_process_last_pct(cfg)
    shard = _get_index_shard(cfg)

    if shard is not None:
        if first_pct is not None or last_pct is not None:
            raise ValueError(
                "label.prise.index_start/index_end are exclusive with "
                "process_first_pct / process_last_pct."
            )
        start, end = shard
        if start >= instance_cnt:
            raise ValueError(
                f"index_start={start} is beyond instance_cnt={instance_cnt}."
            )
        end = min(end, instance_cnt)
        return start, end, f"index shard [{start}, {end})"

    if first_pct is not None and last_pct is not None:
        raise ValueError(
            "Configs are mutually exclusive: set only one of "
            "cfg.label.prise.process_first_pct or cfg.label.prise.process_last_pct"
        )

    if instance_cnt <= 0:
        return 0, 0, "empty"

    if first_pct is None and last_pct is None:
        return 0, instance_cnt, "all (default)"

    if first_pct is not None:
        process_cnt = int(np.ceil(instance_cnt * (first_pct / 100.0)))
        end_idx0 = min(instance_cnt, max(0, process_cnt))
        return 0, end_idx0, f"first {first_pct:.2f}%"

    # last_pct is not None
    process_cnt = int(np.ceil(instance_cnt * (last_pct / 100.0)))
    start_idx0 = max(0, instance_cnt - max(0, process_cnt))
    return start_idx0, instance_cnt, f"last {last_pct:.2f}%"


def _get_k_for_forward(cfg: DictConfig, S: int) -> int:
    k_ratio = float(cfg.label.prise.k_ratio)
    if k_ratio is None or k_ratio <= 0.0:
        raise ValueError(f"k_ratio must be positive, got {k_ratio}")
    return _compute_effective_K(S, k_ratio)


def _build_eval_dir(cfg: DictConfig, problem_name: str, instance_cnt: int, root: Path) -> Path:
    inst_cfg = cfg.problem.instance
    seed = int(cfg.get("seed", 42))
    n_scenarios = int(inst_cfg.n_scenarios)
    n_nodes = getattr(inst_cfg, "n_nodes", None)
    n_facilities = getattr(inst_cfg, "n_facilities", None)
    n_customers = getattr(inst_cfg, "n_customers", None)

    if n_facilities is not None and n_customers is not None:
        tag = f"s{seed}_i{instance_cnt}_m{n_facilities}_n{n_customers}_s{n_scenarios}"
    else:
        if n_nodes is None:
            raise ValueError("Expected problem.instance.n_nodes for non-CFLP problems.")
        tag = f"s{seed}_i{instance_cnt}_n{n_nodes}_s{n_scenarios}"

    return Path(root) / "eval" / problem_name / tag


def _label_output_path(rp, problem_name: str, label_path: str) -> Path:
    """Dataset destination shared with the next stage under the default layout."""
    return data_output_path(rp, problem_name, Path(label_path).name)


def _collect_bigmilp_metrics(
    results: List[dict],
    process_cnt: int,
) -> dict:
    obj_vals = []
    time_vals = []
    for res in results:
        obj = res.get("obj", None)
        if obj is not None and np.isfinite(obj):
            obj_vals.append(float(obj))
        runtime_val = res.get("runtime_sec", None)
        if runtime_val is not None and np.isfinite(runtime_val):
            time_vals.append(float(runtime_val))

    obj_mean = float(np.mean(obj_vals)) if obj_vals else None
    obj_std = float(np.std(obj_vals)) if obj_vals else None
    walltime = float(np.sum(time_vals)) if time_vals else None
    per_instance = float(walltime / max(1, len(time_vals))) if time_vals else None

    return {
        "num_instances": int(process_cnt),
        "full_obj_mean": obj_mean,
        "full_obj_std": obj_std,
        "eval_walltime_sec": walltime,
        "eval_walltime_per_instance_sec": per_instance,
    }


def _run_bigmilp(
    *,
    problem,
    inst: Instance,
    time_limit: float,
    mip_gap: Optional[float],
    threads: int,
    verbose: bool,
    inst_idx: Optional[int] = None,
    inst_total: Optional[int] = None,
) -> dict:
    res = problem.solve_robust_bigmilp(
        inst, time_limit=time_limit, threads=threads, verbose=verbose, mip_gap=mip_gap
    )

    prefix = ""
    if inst_idx is not None and inst_total is not None:
        prefix = f"[BIGMILP {inst_idx}/{inst_total}] "

    logger.info(
        prefix
        + f"status={res.get('status', None)} | "
        f"has_incumbent={res.get('has_incumbent', None)} | "
        f"obj={res.get('obj', None)} | "
        f"obj_bound={res.get('obj_bound', None)} | "
        f"mip_gap={res.get('mip_gap', None)} | "
        f"runtime_sec={res.get('runtime_sec', None)} | "
        f"t_var={res.get('t', None)}"
    )
    return res


def _run_forward_trace(
    *,
    problem,
    data_np: dict,
    K: int,
    milp_time_limit: float,
    threads: int,
    verbose: bool,
    log_prefix: str,
    milp_mip_gap: Optional[float] = None,
    eps_gain: Optional[float] = None,
    stop_M: Optional[int] = None,
):
    # full_obj_val is currently unused in your forward eval path (you pass 0.0)
    # keep it as-is for compatibility.
    full_obj_val = 0.0
    # eps_gain / stop_M default to None => _forward_with_trace's own defaults
    # (0.0001 / 1), i.e. existing behaviour byte-for-byte. Passing eps_gain=0.0
    # with a large stop_M disables the plateau cutoff.
    extra = {}
    if eps_gain is not None:
        extra["eps_gain"] = float(eps_gain)
    if stop_M is not None:
        extra["stop_M"] = int(stop_M)
    trace = _forward_with_trace(
        problem=problem,
        data_np=data_np,
        full_obj_val=full_obj_val,
        K=K,
        milp_time_limit=milp_time_limit,
        log_prefix=log_prefix,
        threads=threads,
        verbose=verbose,
        milp_mip_gap=milp_mip_gap,
        **extra,
    )
    return trace


@hydra.main(config_path="../configs", config_name="exp", version_base=None)
def main(cfg: DictConfig):
    logger.info("Starting PRISE labelling (Hydra, from inst_spec)")

    seed = cfg.get("seed", 42)
    seed_everything(seed)
    rp = open_hydra_run(cfg, __file__)

    data_folder = _resolve_data_folder(cfg)
    problem_name = cfg.problem.name

    inst_spec_path = _resolve_instance_spec_path(cfg, data_folder, problem_name)
    inst_spec_label_path = _resolve_label_path(cfg, data_folder, problem_name)

    with open(inst_spec_path, "rb") as f:
        instances: List[Instance] = pickle.load(f)

    instance_cnt = len(instances)
    log_every = cfg.get("log_every", 10)
    problem = build_problem_from_cfg(cfg.problem)

    # Optional solver time limit for per-step 2RO solves
    big_milp_time_limit = float(cfg.problem.solver.time_limit)
    big_milp_mip_gap = cfg.problem.solver.get("mip_gap", None)
    if big_milp_mip_gap is not None:
        big_milp_mip_gap = float(big_milp_mip_gap)
    prise_time_limit = float(cfg.problem.solver.prise_time_limit)

    # Cheap-label knob B: optional MIP gap for the forward-pass subset solves
    # (the supervision-producing path). None => exact Gurobi default.
    forward_mip_gap = None
    if "label" in cfg and "prise" in cfg.label and "forward_mip_gap" in cfg.label.prise:
        _fmg = cfg.label.prise.forward_mip_gap
        forward_mip_gap = float(_fmg) if _fmg is not None else None

    # Reproduce the ORACLE's forward greedy, which runs to k_max with the
    # plateau cutoff disabled. Both default to None => unchanged behaviour.
    forward_eps_gain = None
    forward_stop_M = None
    if "label" in cfg and "prise" in cfg.label:
        _eg = cfg.label.prise.get("forward_eps_gain", None)
        _sm = cfg.label.prise.get("forward_stop_M", None)
        forward_eps_gain = float(_eg) if _eg is not None else None
        forward_stop_M = int(_sm) if _sm is not None else None
        if forward_eps_gain is not None or forward_stop_M is not None:
            logger.info("[prise_label] forward-greedy stopping OVERRIDDEN: "
                        "eps_gain=%s stop_M=%s", str(forward_eps_gain), str(forward_stop_M))

    threads = cfg.solver.threads
    big_milp_model_verbose = cfg.solver.verbose
    prise_verbose = cfg.solver.prise_verbose

    output_mode: OutputMode = _get_output_mode(cfg)

    start_idx0, end_idx0, range_desc = _get_process_range(cfg, instance_cnt)
    process_cnt = max(0, end_idx0 - start_idx0)
    logger.info(
        "[prise_label] instance_range=%s => processing %d/%d instances (range %d..%d)",
        range_desc,
        process_cnt,
        instance_cnt,
        (start_idx0 + 1 if process_cnt > 0 else 0),
        end_idx0,
    )

    inst_spec_label_path = label_filename(
        inst_spec_label_path, output_mode,
        (start_idx0, end_idx0) if _get_index_shard(cfg) is not None else None,
    )

    out_path = _label_output_path(rp, problem_name, inst_spec_label_path)

    logger.info(
        f"[prise_label] Generating PRISE labels problem={problem_name} "
        f"output_mode={output_mode} "
        f"(big_milp_time_limit={big_milp_time_limit}, mip_gap={big_milp_mip_gap}) for {instance_cnt} instances "
        f"big_milp solver threads={threads} verbose={big_milp_model_verbose}... "
        f"prise_time_limit={prise_time_limit} prise_verbose={prise_verbose} "
        f"forward_mip_gap={forward_mip_gap}"
    )
    logger.info(
        "[prise_label] Solver settings: threads=%s, time_limit=%s, mip_gap=%s",
        str(threads),
        str(big_milp_time_limit),
        str(big_milp_mip_gap),
    )

    paired_labels = []
    bigmilp_results: List[dict] = []

    for i0 in range(start_idx0, end_idx0):
        i = i0 + 1  # 1-based global index for logs/compat
        inst = instances[i0]
        data_np = _instance_to_numpy_dict(inst, 0)
        sce = problem.get_scenarios(data_np)
        S = sce.shape[0]

        res = None
        if output_mode in ("bigmilp", "both"):
            res = _run_bigmilp(
                problem=problem,
                inst=inst,
                time_limit=big_milp_time_limit,
                mip_gap=big_milp_mip_gap,
                threads=threads,
                verbose=big_milp_model_verbose,
                inst_idx=i,
                inst_total=instance_cnt,
            )
            bigmilp_results.append(res)

        trace = None
        selected_idx = None
        ranking = None
        worst_sce_idx = None

        if output_mode in ("forward_trace", "both"):
            K_i = _get_k_for_forward(cfg, S)
            try:
                trace = _run_forward_trace(
                    problem=problem,
                    data_np=data_np,
                    K=K_i,
                    milp_time_limit=prise_time_limit,
                    threads=threads,
                    verbose=prise_verbose,
                    log_prefix=f"[FWD Inst {i}] ",
                    milp_mip_gap=forward_mip_gap,
                    eps_gain=forward_eps_gain,
                    stop_M=forward_stop_M,
                )
            except Exception as e:
                logger.exception(
                    "[prise_label] Forward heuristic failed on instance %d/%d: %s",
                    i, instance_cnt, str(e),
                )
                # If we're in forward-trace-only mode, we cannot produce anything useful here.
                if output_mode == "forward_trace":
                    continue
                # If we're in "both", still allow saving the bigmilp-only label.
                trace = None

            if trace is not None:
                selected_idx = np.asarray(trace.selected_idx, dtype=int)

                # ---- Build DP-L1 ranking (chosen first, then remaining) ----
                chosen_list = selected_idx.tolist()
                chosen_set = set(chosen_list)
                remaining = [s for s in range(S) if s not in chosen_set]
                ranking = np.array(chosen_list + remaining, dtype=int)

                worst_sce_idx = int(ranking[0]) if len(ranking) > 0 else None

        # ---- Build Labels (populate only what's available in the chosen mode) ----
        if res is not None:
            obj_val = res.get("obj", None)
            x_val = res.get("x", None)
            feasible_val = bool(res.get("has_incumbent", False))
        else:
            obj_val = float("nan")
            x_val = None
            feasible_val = True  # forward-trace-only supervision may still be valid

        lab = Labels(
            obj=obj_val,
            feasible=feasible_val,
            x=x_val,
            sce_target_idx=(np.array([worst_sce_idx], dtype=int) if worst_sce_idx is not None else None),
            sce_ranking=ranking,
            sce_scores=None,          # not used for now
            sce_top_idx=selected_idx,
            forward_trace=trace,      # attach only if computed
        )
        lab = _round_label_floats(lab, decimals=4)

        paired_labels.append({"instance": inst, "label": lab})

        if (i % max(1, log_every)) == 0 or i == end_idx0:
            pct = i / max(1, instance_cnt) * 100.0
            head = None if ranking is None else ranking[: min(5, len(ranking))].tolist()
            logger.info(
                "[prise_label] %d/%d (%.1f%%) processed | mode=%s | "
                "selected_idx=%s, ranking_head=%s",
                i, instance_cnt, pct, output_mode,
                None if selected_idx is None else selected_idx.tolist(),
                head,
            )

    # Save instance and label pairs for subsequent training.
    with create_new(data_paths(rp), out_path, mode="wb") as f:
        pickle.dump(paired_labels, f)

    logger.info(
        f"[prise_label] Saved instance+label pairs to: {out_path}"
    )

    if output_mode in ("bigmilp", "both") and process_cnt > 0:
        metrics = _collect_bigmilp_metrics(bigmilp_results, process_cnt)
        eval_dir = _build_eval_dir(cfg, problem_name, instance_cnt, rp.run_dir)
        metrics_path = eval_dir / "bigmilp.json"
        logger.info("[prise_label] BigMILP metrics output: %s", metrics_path)
        logger.info("[prise_label] BigMILP metrics: %s", metrics)
        with create_new(rp, metrics_path, mode="w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("[prise_label] BigMILP metrics saved.")


if __name__ == "__main__":
    register_omega_conf_custom_resolvers()
    main()
