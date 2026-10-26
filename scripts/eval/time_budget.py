#!/usr/bin/env python3
"""Time-budgeted full MILP (paper Table 14, tab:time_budget).

Gurobi solves the full-scenario robust MILP of each of the 50 VC test instances of one scale
(medium: VC-50-50, large: VC-50-100) under a fixed per-instance time limit, 1 s for the k = 4 row
and 2 s for the k = 6 row, and keeps its incumbent. Its regret is gap_pct = (obj - V(Xi)) / V(Xi) * 100,
with V(Xi) from eval_regret/full_objs_cache.json, the full_obj of the NeurPRISE regret rows. The output
has the shape of the April 2026 time-budget files (<tag>_fullmilp_k<K>_budget<B>s.json).

usage:
  python scripts/eval/time_budget.py --scale large --k 4 --run-id VC-50-100_fullmilp_k4_budget1s
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import socket
import sys
import time
from datetime import datetime, timezone

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, REPO)

from scripts.eval.eval_regret import PROBLEM_CONFIGS, make_problem  # noqa: E402
from src.utils.run_io import add_run_args, create_new, input_file, open_run  # noqa: E402

BUDGET_S = {4: 1, 6: 2}  # fixed constants, the Budget column of Table 14
FULL_OBJS = "eval_regret/full_objs_cache.json"


def solve_one(problem, inst, full_obj: float, budget: float, threads: int, mip_gap: float) -> dict:
    """The incumbent of the full MILP under the time limit, and its gap to V(Xi) in percent."""
    t0 = time.perf_counter()
    out = problem.solve_robust_bigmilp(inst, time_limit=budget, mip_gap=mip_gap, threads=threads)
    solve_time = time.perf_counter() - t0
    obj = out["obj"] if out["has_incumbent"] else math.inf
    return {"obj_value": obj, "has_incumbent": out["has_incumbent"], "status": out["status"],
            "solve_time": solve_time, "full_obj": full_obj, "gap_pct": (obj - full_obj) / full_obj * 100}


def sha256(path) -> str:
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_run_args(ap)
    ap.add_argument("--batch", default="time_budget",
                    help="run_io batch name for --run-id (default: %(default)s)")
    ap.add_argument("--scale", choices=("medium", "large"), required=True)
    ap.add_argument("--k", type=int, choices=sorted(BUDGET_S), required=True)
    ap.add_argument("--threads", type=int, default=40, help="Gurobi threads (default: %(default)s)")
    ap.add_argument("--mip-gap", type=float, default=1e-4)
    args = ap.parse_args()

    cfg = PROBLEM_CONFIGS[("vertex_cover", args.scale)]
    tag, n, budget = cfg["tag"], cfg["n_test"], BUDGET_S[args.k]
    run = open_run(args.batch, args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
                   [args.input_root])
    commit = json.loads((run.run_dir / "RUN.json").read_text())["commit"]
    data, cache = (input_file(run, os.path.join(args.input_root, p)) for p in (cfg["data"], FULL_OBJS))
    host, load0 = socket.gethostname(), os.getloadavg()
    print(f"RUN_DIR={run.run_dir}\nhost={host} commit={commit} loadavg_start={load0} "
          f"gurobi_threads={args.threads} mip_gap={args.mip_gap} {tag} k={args.k} budget={budget}s\n"
          f"sha256 {cfg['data']}={sha256(data)}\nsha256 {FULL_OBJS}={sha256(cache)}", flush=True)
    with open(data, "rb") as fh:
        instances = pickle.load(fh)
    assert len(instances) == cfg["n_total"], (tag, len(instances))
    full_objs = json.loads(cache.read_text())[tag]
    assert len(full_objs) == n, (tag, len(full_objs))

    problem = make_problem("vertex_cover", cfg)
    rows = [{**solve_one(problem, inst, fo, budget, args.threads, args.mip_gap), "instance_idx": i}
            for i, (inst, fo) in enumerate(zip(instances[-n:], full_objs))]
    n_feasible = sum(r["has_incumbent"] for r in rows)
    summary = {"num_instances": n, "num_feasible": n_feasible, "num_infeasible": n - n_feasible,
               "gap_pct_mean": math.fsum(r["gap_pct"] for r in rows) / n,  # inf if any has no incumbent
               "solve_time_mean": math.fsum(r["solve_time"] for r in rows) / n}
    n_time_limit = sum(r["status"] == 9 for r in rows)  # GRB.TIME_LIMIT
    print(f"{tag} k={args.k} budget={budget}s " + " ".join(f"{k}={v}" for k, v in summary.items())
          + f" n_time_limit={n_time_limit}")

    load1 = os.getloadavg()
    print(f"loadavg_end={load1}")
    name = f"{tag}_fullmilp_k{args.k}_budget{budget}s.json"
    with create_new(run, name, mode="w") as fh:
        json.dump({"problem": "vertex_cover", "scale": args.scale, "tag": tag, "K": args.k,
                   "time_budget_sec": budget, **summary, "threads": args.threads, "mip_gap": args.mip_gap,
                   "eval_date": datetime.now(timezone.utc).isoformat(), "host": host, "commit": commit,
                   "loadavg_start": load0, "loadavg_end": load1, "data": cfg["data"], "full_objs": FULL_OBJS,
                   "per_instance": rows}, fh, indent=1)
    print(f"Saved: {run.run_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
