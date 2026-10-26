#!/usr/bin/env python3
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.core.types import Instance
from src.utils.batch_utils import _instance_to_numpy_dict
from src.utils.eval_helpers import load_model_checkpoint, evaluate_loader, resolve_e2e_out_dir as _resolve_out_dir
from src.utils.eval_utils import _eval_single_reduced_instance
from src.utils.omega_conf import register_omega_conf_custom_resolvers, create_problem_module, create_problem_module_with_data_override
from src.utils.run_io import open_hydra_run, create_new, publish_new, output_path
from src.utils.seed import seed_everything, resolve_eval_seed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_GUROBI_STATUS_NAMES = {
    2: "OPTIMAL",
    3: "INFEASIBLE",
    4: "INF_OR_UNBD",
    5: "UNBOUNDED",
    9: "TIME_LIMIT",
    13: "SUBOPTIMAL",
}


def _scalar(value: Any, field: str) -> Any:
    """Return one scalar and reject missing or non-scalar solver evidence."""
    if value is None:
        raise RuntimeError(f"reduced solve evidence missing {field}")
    array = np.asarray(value)
    if array.size != 1:
        raise RuntimeError(
            f"reduced solve evidence {field} must be scalar, got shape {array.shape}"
        )
    return array.reshape(-1)[0].item()


def validate_reduced_solve_evidence(
    raw: Dict[str, Any], *, instance_index: int, accepted_mip_gap: float
) -> Dict[str, Any]:
    """Normalize one reduced solve certificate and reject anything non-optimal."""
    accepted_mip_gap = float(accepted_mip_gap)
    if not np.isfinite(accepted_mip_gap) or accepted_mip_gap < 0:
        raise RuntimeError(
            f"reduced solve acceptance gap must be finite and nonnegative, got "
            f"{accepted_mip_gap}"
        )
    status = int(_scalar(raw.get("status"), "status"))
    feasible = bool(_scalar(raw.get("feasible"), "feasible"))
    obj_bound = float(_scalar(raw.get("obj_bound"), "obj_bound"))
    achieved_gap = float(_scalar(raw.get("mip_gap"), "mip_gap"))
    if not np.isfinite(obj_bound):
        raise RuntimeError(
            f"reduced solve {instance_index} has non-finite objective bound {obj_bound}"
        )
    if not np.isfinite(achieved_gap) or achieved_gap < 0:
        raise RuntimeError(
            f"reduced solve {instance_index} has invalid achieved MIP gap {achieved_gap}"
        )
    if not feasible:
        raise RuntimeError(f"reduced solve {instance_index} has no feasible incumbent")
    if status != 2:
        name = _GUROBI_STATUS_NAMES.get(status, "UNKNOWN")
        raise RuntimeError(
            f"reduced solve {instance_index} is non-OPTIMAL: status={status} ({name})"
        )
    if achieved_gap > accepted_mip_gap + 1e-12:
        raise RuntimeError(
            f"reduced solve {instance_index} achieved gap {achieved_gap} exceeds "
            f"accepted {accepted_mip_gap}"
        )
    return {
        "instance_index": int(instance_index),
        "status": status,
        "status_name": "OPTIMAL",
        "feasible": True,
        "obj_bound": obj_bound,
        "achieved_mip_gap": achieved_gap,
        "accepted_mip_gap": accepted_mip_gap,
        "optimality_accepted": True,
    }


def reconcile_reduced_solve_evidence(
    rows: list[Dict[str, Any]], *, expected_count: int
) -> Dict[str, Any]:
    """Build a self-reconciling aggregate or fail before any result is written."""
    if len(rows) != expected_count:
        raise RuntimeError(
            f"reduced solve evidence count {len(rows)} != expected {expected_count}"
        )
    indices = [row.get("instance_index") for row in rows]
    if indices != list(range(expected_count)):
        raise RuntimeError(
            "reduced solve evidence indices are missing, duplicated, or out of order"
        )
    if any(row.get("optimality_accepted") is not True for row in rows):
        raise RuntimeError("reduced solve evidence contains an unaccepted row")
    status_counts: Dict[str, int] = {}
    for row in rows:
        name = str(row.get("status_name"))
        status_counts[name] = status_counts.get(name, 0) + 1
    if status_counts != {"OPTIMAL": expected_count}:
        raise RuntimeError(f"reduced solve status aggregate is not all OPTIMAL: {status_counts}")
    gaps = [float(row["achieved_mip_gap"]) for row in rows]
    return {
        "schema_version": "reduced-solve-evidence/1.0",
        "expected_count": int(expected_count),
        "observed_count": len(rows),
        "accepted_count": len(rows),
        "rejected_count": 0,
        "status_counts": status_counts,
        "max_achieved_mip_gap": max(gaps) if gaps else None,
        "all_optimality_accepted": True,
    }


def _publish_json(rp, path, payload: Dict[str, Any], *, allow_nan: bool) -> Path:
    """Write a JSON artifact atomically (via run_io.publish_new) so a partial file cannot
    become a skip sentinel."""
    def _write(fh):
        json.dump(payload, fh, indent=2, sort_keys=True, allow_nan=allow_nan)
        fh.write("\n")
    return publish_new(rp, path, _write, mode="w")


def canonical_true_regret_problem_name(problem_name: str) -> str:
    """Map distribution-specific config names to eval-regret problem families."""
    for prefix, family in (
        ("cflp", "cflp_ebp"),
        ("vertex_cover", "vertex_cover"),
        ("selection", "selection"),
    ):
        if problem_name == prefix or problem_name.startswith(prefix + "_"):
            return family
    raise ValueError(f"Unsupported true-regret problem family: {problem_name!r}")


def configure_true_regret_solver(
    eval_regret_module: Any,
    *,
    threads: int,
    time_limit: Optional[float],
) -> Dict[str, float | int]:
    """Set, read back, and prove consumption of true-regret solver globals."""
    threads = int(threads)
    if threads <= 0:
        raise ValueError(f"true-regret threads must be positive, got {threads}")
    if time_limit is None or float(time_limit) <= 0:
        raise ValueError(
            f"true-regret time_limit must be positive, got {time_limit!r}"
        )
    time_limit = float(time_limit)

    required_names = {
        "get_neurprise_x": {"THREADS", "TIME_LIMIT"},
        "evaluate_x_on_all_scenarios": {"THREADS"},
    }
    for function_name, globals_used in required_names.items():
        function = getattr(eval_regret_module, function_name, None)
        code_names = set(getattr(getattr(function, "__code__", None), "co_names", ()))
        missing = globals_used - code_names
        if missing:
            raise RuntimeError(
                f"{function_name} does not read {sorted(missing)} as module globals at "
                "call time; requested true-regret solver settings would be silent"
            )

    eval_regret_module.THREADS = threads
    eval_regret_module.TIME_LIMIT = time_limit
    effective_threads = int(eval_regret_module.THREADS)
    effective_time_limit = float(eval_regret_module.TIME_LIMIT)
    if (effective_threads, effective_time_limit) != (threads, time_limit):
        raise RuntimeError(
            "true-regret solver setting read-back mismatch: "
            f"requested=({threads}, {time_limit}) "
            f"effective=({effective_threads}, {effective_time_limit})"
        )
    return {"threads": effective_threads, "time_limit": effective_time_limit}


def _format_np_array(name: str, value: Any) -> str:
    arr = np.asarray(value)
    return (
        f"{name} shape={arr.shape} values=\n"
        f"{np.array2string(arr, threshold=np.inf, max_line_width=200)}"
    )


def _log_instance_spec(tag: str, data_np: Dict[str, Any]) -> None:
    logger.info("%s instance keys: %s", tag, sorted(list(data_np.keys())))
    for key in ("det", "cost", "x", "c", "n_nodes", "n_scenarios", "p_select"):
        if key in data_np:
            logger.info("%s %s", tag, _format_np_array(key, data_np[key]))


def _sha256_file(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_eval_expectations(
    cfg: DictConfig, datamodule: Any, split_size: int
) -> Dict[str, Any]:
    """Check the caller's declared expectations about *what was actually loaded*.

    Every key under ``eval.expect`` is null by default, so this is inert for every
    pre-existing command.  A caller that sets one is not merely configuring the run --
    it is asserting an outcome, and a mismatch stops the run before the first solve.

    This exists because the failure it guards is silent.  `data/<problem>/` holds a
    50-record BigMILP label file under the same 500-record *name* as the 500-record
    merged population; loading it hands ``split=test`` five instances and a
    plausible-looking regret number, with no error, whenever the role contract is
    ``eval_only`` (`src/utils/omega_conf.py:49-55`).  Declaring the population, the split
    size and the file digest converts that into a refusal.

    Unset keys are logged as ``<unset>`` rather than skipped quietly, so a launch log
    shows which guards a command chose not to arm.
    """
    expect = cfg.get("eval", {}).get("expect", {}) or {}
    observed: Dict[str, Any] = {
        "label_file": str(getattr(datamodule, "data_file", "")),
        "population": int(len(datamodule.full_ds)),
        "split_instances": int(split_size),
        "data_num_workers": int(cfg.data.num_workers),
        "eval_num_workers": int(cfg.get("eval", {}).get("num_workers", cfg.data.num_workers)),
    }
    failures: list[str] = []

    def _check(key: str, actual: Any, *, cast=int) -> None:
        wanted = expect.get(key, None)
        if wanted is None:
            logger.info("eval.expect.%s = <unset> (observed %s)", key, actual)
            return
        wanted = cast(wanted)
        logger.info("eval.expect.%s = %s (observed %s)", key, wanted, actual)
        if actual != wanted:
            failures.append(f"eval.expect.{key}={wanted!r} but observed {actual!r}")

    _check("num_workers", observed["data_num_workers"])
    _check("num_workers", observed["eval_num_workers"])
    _check("population", observed["population"])
    _check("test_instances", observed["split_instances"])

    wanted_sha = expect.get("label_sha256", None)
    if wanted_sha is None:
        logger.info("eval.expect.label_sha256 = <unset> (file %s)", observed["label_file"])
    else:
        observed["label_sha256"] = _sha256_file(observed["label_file"])
        logger.info(
            "eval.expect.label_sha256 = %s (observed %s for %s)",
            wanted_sha, observed["label_sha256"], observed["label_file"],
        )
        if observed["label_sha256"] != str(wanted_sha):
            failures.append(
                f"eval.expect.label_sha256={wanted_sha} but {observed['label_file']} "
                f"hashes to {observed['label_sha256']}"
            )

    if failures:
        raise RuntimeError(
            "eval.expect assertions failed before any solve: " + "; ".join(failures)
        )
    return observed


def _evaluate_2ro_with_fixed_indices(
    loader,
    problem,
    sce_idx_all: np.ndarray,
    num_instances: int,
    K: int,
    milp_time_limit: Optional[float],
    threads: int,
    mip_gap: Optional[float] = None,
    true_regret: bool = False,
    problem_name: str = "vertex_cover",
    n_scenarios: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[Dict[str, np.ndarray]]]:
    """
    Mirrors the reduced evaluation loop but stays local to this E2E script.
    Returns:
      full_objs, reduced_objs, regrets, per_instance_eval_times_sec, true_regret_out,
      per-instance reduced-solve evidence

    true_regret_out is None unless true_regret=True, in which case it is a dict of
    per-instance arrays {eval_on_xi, regret_pct, feasible} in Table-1 out-of-sample
    convention (100*(eval_on_xi-full_obj)/full_obj, worst-case recourse over ALL
    scenarios). When true_regret=False the loop and the first four return values are
    byte-identical to the legacy behavior.
    """
    full_objs = []
    reduced_objs = []
    regrets = []
    eval_times = []
    tr_eval_on_xi = []
    tr_regret_pct = []
    tr_feasible = []
    solve_evidence = []
    if true_regret:
        from scripts.eval.eval_regret import true_regret_single  # reuse, don't duplicate
        if n_scenarios is None:
            raise ValueError("true_regret=True requires n_scenarios.")

    global_idx = 0
    n_idx, k_idx = sce_idx_all.shape
    if n_idx != num_instances or k_idx != K:
        raise ValueError(f"idx shape mismatch: idx={sce_idx_all.shape}, expected=({num_instances},{K})")

    for batch_id, batch in enumerate(loader):
        inst_b: Instance = batch["instance"]
        labels_b = batch["label"]
        B = labels_b.obj.shape[0]

        if global_idx + B > num_instances:
            raise ValueError(
                f"Mismatch: idx array has {num_instances} rows but dataloader yields at least {global_idx + B}."
            )

        sce_idx_batch = sce_idx_all[global_idx : global_idx + B]  # [B, K]

        for i in range(B):
            full_obj_val = float(labels_b.obj[i].detach().cpu().item())
            data_np = _instance_to_numpy_dict(inst_b, i)
            sce_idx_i = sce_idx_batch[i]

            t1 = time.perf_counter()
            # feasible / obj_bound / achieved_mip_gap are re-read from raw_evidence by
            # validate_reduced_solve_evidence below, so the loose copies are unused.
            full_obj, red_obj, regret, _, _, _, raw_evidence = _eval_single_reduced_instance(
                problem=problem,
                data_np=data_np,
                sce_idx=sce_idx_i,
                full_obj_val=full_obj_val,
                milp_time_limit=milp_time_limit,
                threads=threads,
                mip_gap=mip_gap,
                return_solve_evidence=True,
            )
            eval_times.append(time.perf_counter() - t1)

            if mip_gap is None:
                raise RuntimeError("reduced solve acceptance requires solver.mip_gap")
            solve_evidence.append(
                validate_reduced_solve_evidence(
                    raw_evidence,
                    instance_index=global_idx + i,
                    accepted_mip_gap=float(mip_gap),
                )
            )

            full_objs.append(full_obj)
            reduced_objs.append(red_obj)
            regrets.append(regret)

            if true_regret:
                tr = true_regret_single(
                    problem, Instance(data=data_np), sce_idx_i, full_obj,
                    n_scenarios, problem_name=problem_name, mip_gap=mip_gap,
                )
                tr_eval_on_xi.append(tr["eval_on_xi"])
                tr_regret_pct.append(tr["regret_pct"])
                tr_feasible.append(tr["feasible"])

        global_idx += B
        if (batch_id + 1) % 10 == 0:
            logger.info("2RO: processed %d batches (%d instances).", batch_id + 1, global_idx)

    true_regret_out = None
    if true_regret:
        true_regret_out = {
            "eval_on_xi": np.asarray(tr_eval_on_xi, dtype=float),
            "regret_pct": np.asarray(tr_regret_pct, dtype=float),
            "feasible": np.asarray(tr_feasible, dtype=bool),
        }

    return (
        np.asarray(full_objs, dtype=float),
        np.asarray(reduced_objs, dtype=float),
        np.asarray(regrets, dtype=float),
        np.asarray(eval_times, dtype=float),
        true_regret_out,
        solve_evidence,
    )


def _basic_metrics(full_objs: np.ndarray, reduced_objs: np.ndarray, regrets: np.ndarray, K: int, k_ratio: float) -> Dict[str, Any]:
    # Keep it minimal and robust. Add more fields if you want.
    full_mean = float(np.mean(full_objs)) if len(full_objs) else float("nan")
    regret_mean = float(np.mean(regrets)) if len(regrets) else float("nan")
    if np.isfinite(full_mean) and full_mean != 0.0 and np.isfinite(regret_mean):
        regret_error = round(abs(regret_mean / full_mean) * 100.0, 4)
    else:
        regret_error = float("nan")
    return {
        "num_instances": int(len(regrets)),
        "K": int(K),
        "k_ratio": float(k_ratio),
        "full_obj_mean": full_mean,
        "reduced_obj_mean": float(np.mean(reduced_objs)) if len(reduced_objs) else float("nan"),
        "regret_mean": regret_mean,
        "regret_std": float(np.std(regrets)) if len(regrets) else float("nan"),
        "regret_median": float(np.median(regrets)) if len(regrets) else float("nan"),
        "regret_error": regret_error,
    }

def _log_normalization_probe(dataloader, tag: str) -> None:
    """Log a quick sanity check on det/sce ranges for the first batch only."""
    try:
        batch = next(iter(dataloader))
    except Exception as e:
        logger.warning("Normalize probe (%s) failed to read batch: %s", tag, str(e))
        return
    inst = batch.get("instance", None)
    if inst is None or not hasattr(inst, "data"):
        logger.warning("Normalize probe (%s) missing instance data.", tag)
        return
    det = inst.data.get("det", None)
    sce = inst.data.get("sce", None)
    p_sel = inst.data.get("p_select", None)
    n_nodes = inst.data.get("n_nodes", None)
    if det is not None:
        det_t = torch.as_tensor(det)
        logger.info(
            "Normalize probe (%s) det: min=%.6f max=%.6f",
            tag,
            float(det_t.min().item()),
            float(det_t.max().item()),
        )
    if sce is not None:
        sce_t = torch.as_tensor(sce)
        logger.info(
            "Normalize probe (%s) sce: min=%.6f max=%.6f",
            tag,
            float(sce_t.min().item()),
            float(sce_t.max().item()),
        )
    if p_sel is not None and n_nodes is not None:
        try:
            p_val = int(torch.as_tensor(p_sel).reshape(-1)[0].item())
            n_val = int(torch.as_tensor(n_nodes).reshape(-1)[0].item())
            logger.info("Normalize probe (%s) p_select=%d n_nodes=%d", tag, p_val, n_val)
        except Exception:
            pass


def _verify_full_objectives(
    loader,
    problem,
    *,
    max_instances: int,
    milp_time_limit: Optional[float],
    threads: int,
) -> Optional[Dict[str, Any]]:
    """Recompute full objectives on raw instances to sanity-check label alignment."""
    if max_instances <= 0:
        return None
    use_bigmilp = max_instances == 1
    checked = 0
    diffs = []
    debug_info: Optional[Dict[str, Any]] = None
    for batch in loader:
        inst_b: Instance = batch["instance"]
        labels_b = batch["label"]
        B = labels_b.obj.shape[0]
        for i in range(B):
            if checked >= max_instances:
                break
            data_np = _instance_to_numpy_dict(inst_b, i)
            inst_single = Instance(data=data_np)
            lab_obj = float(labels_b.obj[i].detach().cpu().item())
            full_obj_val = None
            if use_bigmilp and hasattr(problem, "solve_robust_bigmilp"):
                res = problem.solve_robust_bigmilp(
                    inst_single, time_limit=milp_time_limit, threads=threads, verbose=False, mip_gap=None
                )
                obj_val = res.get("obj", None)
                if obj_val is not None:
                    full_obj_val = float(obj_val)
                else:
                    logger.warning(
                        "Full-obj verify (bigmilp) missing obj; status=%s has_incumbent=%s",
                        str(res.get("status", None)),
                        str(res.get("has_incumbent", None)),
                    )
            if full_obj_val is None:
                labels_re = problem.solve_exact(
                    inst_single, time_limit=milp_time_limit, threads=threads, verbose=False
                )
                full_obj_val = float(labels_re.obj)
            diffs.append(full_obj_val - lab_obj)
            checked += 1
            if debug_info is None:
                debug_info = {
                    "data_np": data_np,
                    "label_obj": lab_obj,
                    "full_obj": full_obj_val,
                }
        if checked >= max_instances:
            break
    if diffs:
        diffs_np = np.asarray(diffs, dtype=float)
        verify_tag = "bigmilp" if use_bigmilp else "exact"
        logger.info(
            "Full-obj verify (%s): n=%d, diff mean=%.6f std=%.6f min=%.6f max=%.6f",
            verify_tag,
            int(diffs_np.size),
            float(diffs_np.mean()),
            float(diffs_np.std()),
            float(diffs_np.min()),
            float(diffs_np.max()),
        )
        if debug_info is not None:
            logger.info(
                "Full-obj verify (%s): label_obj=%.6f full_obj=%.6f diff=%.6f",
                verify_tag,
                float(debug_info["label_obj"]),
                float(debug_info["full_obj"]),
                float(debug_info["full_obj"] - debug_info["label_obj"]),
            )
    return debug_info


@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    rp = open_hydra_run(cfg, __file__)
    eval_cfg = cfg.get("eval", {})
    eval_seed = resolve_eval_seed(cfg)
    seed_everything(eval_seed)
    logger.info("Seed setup: data_seed=%d eval_seed=%d", int(cfg.get("seed", 42)), eval_seed)

    if eval_cfg.get("k_ratio", None) is None:
        raise ValueError("This E2E script requires eval.k_ratio (no eval.k).")

    device = torch.device(str(eval_cfg.get("device", "cpu")))
    solver_cfg = cfg.get("solver", {})

    threads = int(solver_cfg.get("threads", 1))
    milp_time_limit = solver_cfg.get("time_limit", None)

    # --- Solve backend selection (bigmilp or ccg) ---
    solve_backend = str(eval_cfg.get("solve_backend", "bigmilp")).strip().lower()
    if solve_backend not in {"bigmilp", "ccg"}:
        raise ValueError(f"Unsupported eval.solve_backend='{solve_backend}' (expected bigmilp|ccg).")
    full_obj_source = str(eval_cfg.get("full_obj_source", "label")).strip().lower()
    if full_obj_source not in {"label", "solve"}:
        raise ValueError(f"Unsupported eval.full_obj_source='{full_obj_source}' (expected label|solve).")
    # Opt-in Table-1 out-of-sample true regret (default off => byte-identical output).
    true_regret = bool(eval_cfg.get("true_regret", False))
    true_regret_problem_name = str(cfg.problem.name)
    if true_regret:
        from scripts.eval import eval_regret as eval_regret_module

        true_regret_problem_name = canonical_true_regret_problem_name(
            str(cfg.problem.name)
        )
        effective = configure_true_regret_solver(
            eval_regret_module,
            threads=threads,
            time_limit=milp_time_limit,
        )
        logger.info(
            "True-regret binding: problem=%s -> family=%s, THREADS=%d, TIME_LIMIT=%s",
            str(cfg.problem.name),
            true_regret_problem_name,
            effective["threads"],
            effective["time_limit"],
        )

    # 1) Build problem + dataloader(s)
    # If data.normalize=true, use normalized instances for ML, but keep raw instances for 2RO eval.
    # This avoids scaling the original problem parameters in the solver stage.
    use_alt_data = bool(eval_cfg.get("use_alt_data", False))
    normalize_data = bool(cfg.data.get("normalize", False))
    if use_alt_data:
        data_ovr = eval_cfg.get("data_override", {})
        dm_ml, _ = create_problem_module_with_data_override(
            cfg,
            instance_cnt=int(data_ovr.get("instance_cnt")),
            n_nodes=(None if data_ovr.get("n_nodes", None) in (None, "") else int(data_ovr.get("n_nodes"))),
            n_facilities=(None if data_ovr.get("n_facilities", None) in (None, "") else int(data_ovr.get("n_facilities"))),
            n_customers=(None if data_ovr.get("n_customers", None) in (None, "") else int(data_ovr.get("n_customers"))),
            n_scenarios=int(data_ovr.get("n_scenarios")),
            p_edge=(None if data_ovr.get("p_edge", None) in (None, "") else data_ovr.get("p_edge")),
            scenario_name=(
                None if data_ovr.get("scenario_name", None) in (None, "") else str(data_ovr.get("scenario_name"))),
            seed=(None if data_ovr.get("seed", None) in (None, "") else int(data_ovr.get("seed"))),
        )
        if normalize_data:
            cfg_raw = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
            cfg_raw.data.normalize = False
            dm_2ro, _ = create_problem_module_with_data_override(
                cfg_raw,
                instance_cnt=int(data_ovr.get("instance_cnt")),
                n_nodes=(None if data_ovr.get("n_nodes", None) in (None, "") else int(data_ovr.get("n_nodes"))),
                n_facilities=(None if data_ovr.get("n_facilities", None) in (None, "") else int(data_ovr.get("n_facilities"))),
                n_customers=(None if data_ovr.get("n_customers", None) in (None, "") else int(data_ovr.get("n_customers"))),
                n_scenarios=int(data_ovr.get("n_scenarios")),
                p_edge=(None if data_ovr.get("p_edge", None) in (None, "") else data_ovr.get("p_edge")),
                scenario_name=(
                    None if data_ovr.get("scenario_name", None) in (None, "") else str(data_ovr.get("scenario_name"))),
                seed=(None if data_ovr.get("seed", None) in (None, "") else int(data_ovr.get("seed"))),
            )
        else:
            dm_2ro = dm_ml
        logger.info(
            "EVAL DATA OVERRIDE ENABLED: instance_cnt=%s, n_nodes=%s, n_facilities=%s, n_customers=%s, n_scenarios=%s, p_edge=%s, scenario_name=%s, seed=%s",
            str(data_ovr.get("instance_cnt")),
            str(data_ovr.get("n_nodes")),
            str(data_ovr.get("n_facilities", None)),
            str(data_ovr.get("n_customers", None)),
            str(data_ovr.get("n_scenarios")),
            str(data_ovr.get("p_edge", None)),
            str(data_ovr.get("scenario_name", None)),
            str(data_ovr.get("seed", None)),
        )
        n_scenarios = int(data_ovr.get("n_scenarios"))
    else:
        dm_ml, _ = create_problem_module(cfg)
        if normalize_data:
            cfg_raw = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
            cfg_raw.data.normalize = False
            dm_2ro, _ = create_problem_module(cfg_raw)
        else:
            dm_2ro = dm_ml
        n_scenarios = int(getattr(dm_ml.problem, "n_scenarios", cfg.problem.instance.n_scenarios))
    split = str(eval_cfg.get("split", "test"))
    if split == "train":
        # train_eval_dataloader, NOT train_dataloader: evaluation must never shuffle, or
        # the `<split>_idx.npy` written below -- which carries no instance ids -- cannot be
        # lined back up with the instances. See datamodule.train_eval_dataloader.
        dataloader_ml = dm_ml.train_eval_dataloader()
        dataloader_2ro = dm_2ro.train_eval_dataloader()
    elif split == "val":
        dataloader_ml = dm_ml.val_dataloader()
        dataloader_2ro = dm_2ro.val_dataloader()
    elif split == "test":
        dataloader_ml = dm_ml.test_dataloader()
        dataloader_2ro = dm_2ro.test_dataloader()
    elif split == "all":
        dataloader_ml = dm_ml.full_dataloader()
        dataloader_2ro = dm_2ro.full_dataloader()
    elif split == "val_test":
        dataloader_ml = dm_ml.val_test_dataloader()
        dataloader_2ro = dm_2ro.val_test_dataloader()
    else:
        raise ValueError(f"Unknown eval.split='{split}'")

    assert_eval_expectations(cfg, dm_ml, len(dataloader_ml.dataset))

    if normalize_data:
        _log_normalization_probe(dataloader_ml, "ml_normalized")
        _log_normalization_probe(dataloader_2ro, "2ro_raw")

    verify_cfg = eval_cfg.get("verify_full_obj", {})
    verify_max = int(verify_cfg.get("max_instances", 0))
    verify_debug_info = None
    if verify_max > 0:
        logger.info("Running full-obj verification on %d instances (raw data).", verify_max)
        verify_debug_info = _verify_full_objectives(
            loader=dataloader_2ro,
            problem=dm_2ro.problem,
            max_instances=verify_max,
            milp_time_limit=milp_time_limit,
            threads=threads,
        )

    logger.info(
        "Split=%s, device=%s, threads=%d, milp_time_limit=%s, use_alt_data=%s",
        split, device, threads, str(milp_time_limit), str(use_alt_data)
    )

    # --- Configure solve backend on the 2RO problem ---
    problem_2ro = dm_2ro.problem
    if hasattr(problem_2ro, "exact_backend"):
        problem_2ro.exact_backend = solve_backend
    elif solve_backend != "bigmilp":
        # Problem doesn't have exact_backend attr — set it dynamically
        problem_2ro.exact_backend = solve_backend
    if hasattr(problem_2ro, "eval_full_obj_source"):
        problem_2ro.eval_full_obj_source = full_obj_source

    if solve_backend == "ccg":
        if not hasattr(problem_2ro, "solve_robust_ccg"):
            raise ValueError(
                f"eval.solve_backend=ccg requires solve_robust_ccg; "
                f"got problem={type(problem_2ro).__name__}."
            )
        ccg_cfg = solver_cfg.get("ccg", {})
        ccg_max_iterations = int(eval_cfg.get("ccg_max_iterations", ccg_cfg.get("max_iterations", 20)))
        ccg_convergence_tolerance = float(
            eval_cfg.get("ccg_convergence_tolerance", ccg_cfg.get("convergence_tolerance", 1e-4))
        )
        ccg_duplicate_tolerance = float(
            eval_cfg.get("ccg_duplicate_tolerance", ccg_cfg.get("duplicate_tolerance", 1e-6))
        )
        ccg_recourse_time_limit = eval_cfg.get(
            "ccg_recourse_time_limit",
            ccg_cfg.get("recourse_time_limit", None),
        )
        if ccg_recourse_time_limit is not None:
            ccg_recourse_time_limit = float(ccg_recourse_time_limit)
        if hasattr(problem_2ro, "configure_ccg_solver"):
            problem_2ro.configure_ccg_solver(
                max_iterations=ccg_max_iterations,
                convergence_tolerance=ccg_convergence_tolerance,
                duplicate_tolerance=ccg_duplicate_tolerance,
                recourse_time_limit=ccg_recourse_time_limit,
                mip_gap=solver_cfg.get("mip_gap", None),
            )
        logger.info(
            "Configured CCG backend: max_iterations=%d, tol=%.6g, dup_tol=%.6g, recourse_tl=%s, mip_gap=%s",
            ccg_max_iterations,
            ccg_convergence_tolerance,
            ccg_duplicate_tolerance,
            str(ccg_recourse_time_limit),
            str(solver_cfg.get("mip_gap", None)),
        )
    logger.info(
        "Solve backend: %s, full_obj_source: %s",
        solve_backend,
        full_obj_source,
    )

    # 2) ML eval: compute scores/idx + selection timing
    model = instantiate(cfg.model).to(device)
    if bool(eval_cfg.get("load_ckpt", True)):
        load_model_checkpoint(model, str(eval_cfg.get("ckpt_path", "")), device)

    t_ml0 = time.perf_counter()
    ml_results = evaluate_loader(
        model=model,
        dataloader=dataloader_ml,
        device=device,
        problem=dm_ml.problem,
        K=None,
        k_ratio=float(eval_cfg.k_ratio),
        enable_cheat_swap=bool(eval_cfg.get("cheat_swap_worst", False)),
        use_label_ranking=bool(eval_cfg.get("use_label_ranking", False)),
        debug_force_teacher=bool(eval_cfg.get("debug_force_teacher", False)),
    )
    ml_wall = time.perf_counter() - t_ml0

    if ml_results.get("idx", None) is None:
        raise RuntimeError("ML stage did not return 'idx'. This E2E pipeline needs indices.")
    idx_t = ml_results["idx"].detach().cpu().numpy().astype(int)  # [N, K]
    num_instances, K = idx_t.shape

    selection_time_total_sec = float(ml_results.get("selection_time_total_sec", 0.0))
    selection_time_per_instance_sec = float(ml_results.get("selection_time_per_instance_sec", 0.0))

    if verify_max == 1 and verify_debug_info is not None and num_instances > 0:
        data_np = verify_debug_info["data_np"]
        label_obj = float(verify_debug_info["label_obj"])
        full_obj_val = float(verify_debug_info["full_obj"])
        sce_idx0 = idx_t[0]
        logger.info("Verify instance: using ML-selected indices for instance 0: %s", sce_idx0)
        _log_instance_spec("Verify instance", data_np)
        if hasattr(dm_2ro.problem, "get_scenarios"):
            full_sce = dm_2ro.problem.get_scenarios(data_np)
            logger.info("%s", _format_np_array("Full scenarios", full_sce))
            logger.info("%s", _format_np_array("Reduced scenarios", full_sce[sce_idx0]))
        full_obj_val, red_obj, regret, feasible, obj_bound, mip_gap = _eval_single_reduced_instance(
            problem=dm_2ro.problem,
            data_np=data_np,
            sce_idx=sce_idx0,
            full_obj_val=full_obj_val,
            milp_time_limit=milp_time_limit,
            threads=threads,
        )
        logger.info(
            "Verify reduced: label_obj=%.6f full_obj=%.6f reduced_obj=%.6f regret=%.6f feasible=%s obj_bound=%s mip_gap=%s",
            label_obj,
            full_obj_val,
            red_obj,
            regret,
            str(feasible),
            str(obj_bound),
            str(mip_gap),
        )

    # 3) 2RO reduced evaluation (in-memory indices)
    logger.info(
        "E2E 2RO eval config: milp_time_limit=%s, threads=%d, num_instances=%d, K=%d, k_ratio=%s",
        str(milp_time_limit), threads, num_instances, K, str(eval_cfg.k_ratio),
    )
    if milp_time_limit is not None and float(milp_time_limit) < 5.0:
        logger.warning(
            "milp_time_limit=%.1f is very short — reduced BigMILP solves may be suboptimal, "
            "especially for large instances (n>=50) with K>=2. Consider time_limit>=60.",
            float(milp_time_limit),
        )
    t_2ro0 = time.perf_counter()
    solver_mip_gap = solver_cfg.get("mip_gap", None)
    full_objs, reduced_objs, regrets, eval_times, tr_out, solve_evidence = _evaluate_2ro_with_fixed_indices(
        loader=dataloader_2ro,
        problem=dm_2ro.problem,
        sce_idx_all=idx_t,
        num_instances=num_instances,
        K=K,
        milp_time_limit=milp_time_limit,
        threads=threads,
        mip_gap=solver_mip_gap,
        true_regret=true_regret,
        problem_name=true_regret_problem_name,
        n_scenarios=n_scenarios,
    )
    two_ro_wall = time.perf_counter() - t_2ro0
    solve_evidence_summary = reconcile_reduced_solve_evidence(
        solve_evidence, expected_count=num_instances
    )

    # 4) Save combined outputs
    out_dir = _resolve_out_dir(eval_cfg, rp.run_dir)
    out_dir = output_path(rp, out_dir)
    logger.info("Writing E2E outputs to: %s", out_dir)

    # Save indices for inspection/reuse (optional but useful)
    with create_new(rp, os.path.join(out_dir, f"{split}_idx.npy")) as fh:
        np.save(fh, idx_t)

    # Save timing breakdown
    timing = {
        "k_ratio": float(eval_cfg.k_ratio),
        "num_instances": int(num_instances),
        "n_scenarios": int(n_scenarios),
        "K": int(K),
        "ml_walltime_sec": float(ml_wall),
        "selection_time_total_sec": float(selection_time_total_sec),
        "selection_time_per_instance_sec": float(selection_time_per_instance_sec),
        "reduced_eval_walltime_sec": float(two_ro_wall),
        "reduced_eval_sum_per_instance_sec": float(np.sum(eval_times)),
        "reduced_eval_per_instance_sec_mean": float(np.mean(eval_times)) if len(eval_times) else 0.0,
        "end_to_end_walltime_sec": float(ml_wall + two_ro_wall),
        "solve_backend": solve_backend,
        "full_obj_source": full_obj_source,
    }
    with create_new(rp, os.path.join(out_dir, "timing.json"), mode="w") as f:
        json.dump(timing, f, indent=2, sort_keys=True)

    # Save metrics
    metrics = _basic_metrics(full_objs, reduced_objs, regrets, K=K, k_ratio=float(eval_cfg.k_ratio))
    metrics.update(timing)
    metrics["reduced_solve_evidence"] = solve_evidence_summary

    df_cols = {
        "full_obj": full_objs,
        "reduced_obj": reduced_objs,
        "regret": regrets,
        "reduced_eval_time_sec": eval_times,
        "reduced_status": [row["status"] for row in solve_evidence],
        "reduced_status_name": [row["status_name"] for row in solve_evidence],
        "reduced_feasible": [row["feasible"] for row in solve_evidence],
        "reduced_obj_bound": [row["obj_bound"] for row in solve_evidence],
        "reduced_mip_gap": [row["achieved_mip_gap"] for row in solve_evidence],
        "reduced_optimality_accepted": [
            row["optimality_accepted"] for row in solve_evidence
        ],
    }

    # Opt-in Table-1 out-of-sample true regret (added only when enabled).
    if tr_out is not None:
        # NOTE: not named `rp`, to avoid colliding with the RunPaths `rp` opened at the
        # top of main() (every write below depends on that one).
        regret_pct = tr_out["regret_pct"]
        feas = tr_out["feasible"]
        feas_mask = feas & np.isfinite(regret_pct)
        rp_feas = regret_pct[feas_mask]
        metrics["true_regret_pct_mean"] = float(np.mean(rp_feas)) if rp_feas.size else float("nan")
        metrics["true_regret_pct_median"] = float(np.median(rp_feas)) if rp_feas.size else float("nan")
        metrics["true_regret_pct_std"] = float(np.std(rp_feas)) if rp_feas.size else float("nan")
        metrics["n_feasible"] = int(np.sum(feas))
        metrics["n_infeasible"] = int(np.sum(~feas))
        metrics["true_regret_convention"] = "100*(eval_on_xi-full_obj)/full_obj, worst-case recourse over all scenarios, feasible-only mean"
        df_cols["eval_on_xi"] = tr_out["eval_on_xi"]
        df_cols["regret_pct"] = regret_pct
        df_cols["feasible"] = feas
        logger.info(
            "True regret (Table-1 conv): mean=%.4f%% median=%.4f%% (feasible-only) | n_feasible=%d n_infeasible=%d",
            metrics["true_regret_pct_mean"], metrics["true_regret_pct_median"],
            metrics["n_feasible"], metrics["n_infeasible"],
        )

    # Per-instance CSV (handy for plots)
    df = pd.DataFrame(df_cols)
    with create_new(rp, os.path.join(out_dir, f"{split}_per_instance.csv"), mode="w") as fh:
        df.to_csv(fh, index=False)
    _publish_json(
        rp,
        Path(out_dir) / "reduced_solve_evidence.json",
        {"summary": solve_evidence_summary, "per_instance": solve_evidence},
        allow_nan=False,
    )
    # metrics.json is the runner's completion sentinel, so publish it last and atomically.
    _publish_json(rp, Path(out_dir) / "metrics.json", metrics, allow_nan=True)

    # Print a clear, single summary block
    logger.info("=== E2E SUMMARY (k_ratio=%s, K=%d, N=%d, Scenario=%d) ===", str(eval_cfg.k_ratio), K, num_instances, n_scenarios)
    logger.info("Selection time: total=%.6f sec (%.6f sec/inst)", selection_time_total_sec, selection_time_per_instance_sec)
    logger.info("Reduced eval:   total=%.6f sec (%.6f sec/inst mean)", float(np.sum(eval_times)), float(np.mean(eval_times)))
    logger.info("Wall times:     ML=%.6f sec, 2RO=%.6f sec, E2E=%.6f sec", ml_wall, two_ro_wall, ml_wall + two_ro_wall)
    logger.info("Objective gap: full minus reduced (objective units; the paper's regret is computed "
                "by scripts/eval/eval_regret.py): mean/std %.6f / %.6f",
                -metrics["regret_mean"], metrics["regret_std"])


if __name__ == "__main__":
    register_omega_conf_custom_resolvers()
    main()
