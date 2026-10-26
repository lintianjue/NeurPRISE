#!/usr/bin/env python3
"""Runtime growth with the scenario count S (paper Tables 15/16, tab:scenario_runtime_med/_lrg).

The 50 VC test instances of one scale (medium: VC-50-50, large: VC-50-100) are truncated to their
first S scenarios, S in {10, 20, 50} (medium) or {20, 50, 100} (large). The cross-scale VC-20-50 model scores
the S scenarios on --device (the paper: cuda), one instance per forward pass; the top k = 6 are kept and the reduced MILP
is solved. Per instance it records the forward-pass time, the reduced-solve time and their sum.
The tables print sum_i t_i(S) / sum_i t_i(smallest S) for each of the three.

usage:
  python scripts/eval/scenario_runtime.py --scale medium --run-id table15
  python scripts/eval/scenario_runtime.py --scale large --run-id table16
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import socket
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, REPO)

from repro.runs import checkpoints, ckpt_overrides  # noqa: E402
from scripts.eval.eval_regret import PROBLEM_CONFIGS, make_problem  # noqa: E402
from src.core.types import Instance, Labels  # noqa: E402
from src.training.datamodule import DataModule  # noqa: E402
from src.utils.batch_utils import _build_reduced_instance_from_np, to_device  # noqa: E402
from src.utils.eval_helpers import load_model_checkpoint  # noqa: E402
from src.utils.run_io import add_run_args, create_new, input_file, open_run  # noqa: E402

K = 6
S_VALUES = {"medium": (10, 20, 50), "large": (20, 50, 100)}
CHECKPOINT = "main/VC-20-50"
COMPONENTS = ("inference", "solve", "total")


def load_model(path: str, device: torch.device):
    """The checkpoint's model, built under the eval config its training config maps to."""
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from src.utils.omega_conf import register_omega_conf_custom_resolvers

    _, config_name, *overrides = ckpt_overrides(checkpoints()[CHECKPOINT])
    register_omega_conf_custom_resolvers()
    with initialize_config_dir(version_base=None, config_dir=os.path.join(REPO, "configs")):
        cfg = compose(config_name=config_name, overrides=[*overrides, "problem=vertex_cover"])
    model = instantiate(cfg.model)
    load_model_checkpoint(model, path, device)
    return model.to(device).eval()


def truncate(inst: Instance, s: int) -> Instance:
    sce = np.asarray(inst.data["sce"])
    assert sce.shape[0] >= s, (sce.shape, s)
    return Instance(dict(inst.data, sce=sce[:s], n_scenarios=np.array([s])))


def run_one(model, problem, inst: Instance, threads: int, mip_gap: float, time_limit: float,
            device: torch.device) -> dict:
    """Top-k of one instance, timing the model forward pass and the reduced solve."""
    norm = DataModule._normalize_instance_shared_pairs(inst, mode="scale")  # eval_scale_norm
    batch = to_device(problem.collate_fn([{"instance": norm, "label": Labels()}])["instance"], device)
    with torch.no_grad():
        t0 = time.perf_counter()
        scores = model(batch, problem=problem)["scores"].cpu()  # .cpu() waits for the GPU
        t_inf = time.perf_counter() - t0
    idx = torch.topk(scores, k=K, dim=1).indices[0].numpy()
    reduced = _build_reduced_instance_from_np(problem, inst.data, idx)
    t0 = time.perf_counter()
    labels = problem.solve_exact(reduced, time_limit=time_limit, threads=threads, mip_gap=mip_gap)
    t_solve = time.perf_counter() - t0
    if labels.meta["status"] != 2:
        raise RuntimeError(f"reduced solve on scenarios {idx.tolist()} ended with status {labels.meta['status']}")
    return {"inference_time_s": t_inf, "solve_time_s": t_solve, "total_time_s": t_inf + t_solve,
            "reduced_obj": float(labels.obj), "selected_scenarios": idx.tolist()}


def ratios(conditions: dict) -> dict:
    """sum_i t_i(S) / sum_i t_i(smallest S), per component."""
    base = conditions[min(conditions, key=int)]["stats"]
    return {s: {c: cond["stats"][f"{c}_sum"] / base[f"{c}_sum"] for c in COMPONENTS}
            for s, cond in conditions.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_run_args(ap)
    ap.add_argument("--batch", default="scenario_runtime",
                    help="run_io batch name for --run-id (default: %(default)s)")
    ap.add_argument("--scale", choices=("medium", "large"), required=True)
    ap.add_argument("--threads", type=int, default=40, help="Gurobi threads (default: %(default)s)")
    ap.add_argument("--mip-gap", type=float, default=1e-4)
    ap.add_argument("--time-limit", type=float, default=900)
    ap.add_argument("--device", default="cpu", help="NeurPRISE scoring device (default: %(default)s)")
    args = ap.parse_args()
    device = torch.device(args.device)

    run = open_run(args.batch, args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
                   [args.input_root])
    commit = json.loads((run.run_dir / "RUN.json").read_text())["commit"]
    host, load0 = socket.gethostname(), os.getloadavg()
    print(f"RUN_DIR={run.run_dir}\nhost={host} commit={commit} loadavg_start={load0} "
          f"cpus={os.cpu_count()} gurobi_threads={args.threads} torch_threads={torch.get_num_threads()}", flush=True)
    cfg = PROBLEM_CONFIGS[("vertex_cover", args.scale)]
    tag, n = cfg["tag"], cfg["n_test"]
    ckpt = checkpoints()[CHECKPOINT]["path"]
    model = load_model(str(input_file(run, os.path.join(args.input_root, ckpt))), device)
    with open(input_file(run, os.path.join(args.input_root, cfg["data"])), "rb") as fh:
        instances = pickle.load(fh)
    assert len(instances) == cfg["n_total"], (tag, len(instances))
    instances = instances[-n:]
    s_values = S_VALUES[args.scale]

    # Untimed warm-up: the first forward pass on a GPU pays one-off kernel setup, which would
    # otherwise land in the smallest-S baseline of every ratio.
    run_one(model, make_problem("vertex_cover", {**cfg, "n_scenarios": s_values[0]}),
            truncate(instances[0], s_values[0]), args.threads, args.mip_gap, args.time_limit, device)
    conditions = {}
    for s in s_values:
        problem = make_problem("vertex_cover", {**cfg, "n_scenarios": s})
        rows = [{"instance_id": i, **run_one(model, problem, truncate(inst, s), args.threads,
                                             args.mip_gap, args.time_limit, device)}
                for i, inst in enumerate(instances)]
        stats = {f"{c}_sum": math.fsum(r[f"{c}_time_s"] for r in rows) for c in COMPONENTS}
        conditions[str(s)] = {"n_scenarios": s, "k": K, "n_instances": n, "stats": stats, "per_instance": rows}
        print(f"{tag} S={s:<3} sum " + " ".join(f"{c}={stats[c + '_sum']:.3f}s" for c in COMPONENTS), flush=True)
    out = ratios(conditions)
    for s, r in out.items():
        print(f"{tag} S={s:<3} ratio vs S={s_values[0]}: " + " ".join(f"{c}={v:.3f}x" for c, v in r.items()))

    load1 = os.getloadavg()
    print(f"loadavg_end={load1}")
    name = f"{tag}_truncated_k{K}.json"
    with create_new(run, name, mode="w") as fh:
        json.dump({"schema": "runtime-scaling/1", "tag": tag,
                   "eval_date": datetime.now(timezone.utc).isoformat(), "host": host, "commit": commit,
                   "loadavg_start": load0, "loadavg_end": load1,
                   "config": {"n_test": n, "n_nodes": cfg["n_nodes"], "n_scenarios_orig": cfg["n_scenarios"],
                              "k": K, "threads": args.threads, "mip_gap": args.mip_gap,
                              "time_limit": args.time_limit, "s_values": [int(s) for s in conditions],
                              "torch_threads": torch.get_num_threads(), "device": args.device,
                              "data": cfg["data"], "checkpoint": ckpt},
                   "ratios": out, "conditions": conditions}, fh, indent=1)
    print(f"Saved: {run.run_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
