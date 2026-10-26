#!/usr/bin/env python3
"""Compression budget k̂ of PRISE, MaxSum and Random (paper Table 7, tab:effective_k_stats).

    k̂ = min{k <= K : |V(Ξ) - V(R^(k))| / V(Ξ) <= 1%},   K = 8,

where R^(k) is the first k scenarios of a method's order and V(Ξ) is the exact objective
(eval_regret/full_objs_cache.json). An instance with no such k did not converge (N/C).

  PRISE   V(R^(k)) = forward_trace.best_scores of the released labels; no MILP is solved.
  MaxSum  scenarios sorted by sum |ξ| descending, V(R^(k)) solved incrementally.
  Random  one uniform permutation per instance from one rng (seed 42) per problem.

Test split: the last 250 of the 2,500 instances. Mean and Med. are of min(k̂, K) / |Ξ|, so a
non-converged instance counts as K; the "<= r%" columns count converged instances only.

With PRISE it also reports, over all 2,500 labelled instances, the fraction of consecutive PRISE
steps whose marginal gain does not increase (g[t] >= g[t+1], forward_trace.gains, pooled over
instances): the "gains are non-increasing in 93-99% of steps" statistic of the same appendix.

usage:
  python scripts/eval/compression_budget.py --run-id table7 --threads 1
  python scripts/eval/compression_budget.py --run-id table7_prise --methods PRISE   # no MILP
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import statistics
import sys
from datetime import datetime, timezone

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, REPO)

from scripts.eval.eval_regret import PROBLEM_CONFIGS, make_problem  # noqa: E402
from src.heuristics.search_framework import Incremental2ROSolver  # noqa: E402
from src.utils.run_io import add_run_args, create_new, input_file, open_run  # noqa: E402

EPS = 0.01
K = 8
RANDOM_SEED = 42
RATIOS_PCT = (2, 4, 8, 12)
PROBLEMS = (("selection", "small"), ("vertex_cover", "small"))  # SEL-20-50, VC-20-50
METHODS = ("Random", "MaxSum", "PRISE")


def khat(scores, v_full: float, eps: float = EPS):
    """First k (1-based) with |V(Ξ) - V(R^(k))| / V(Ξ) <= eps, or None."""
    return next((k for k, v in enumerate(scores, 1) if abs(v_full - v) / abs(v_full) <= eps), None)


def solve_order(problem, instance, order, v_full: float, threads: int, mip_gap: float):
    """V(R^(1)), V(R^(2)), ... along `order`, stopping at convergence or after K steps."""
    solver = Incremental2ROSolver(problem, instance, time_limit=None, threads=threads,
                                  verbose=False, mip_gap=mip_gap)
    try:
        scores = []
        for s in order[:K]:
            obj, _, _, feasible = solver.evaluate_subset([int(s)])
            if not feasible:
                raise RuntimeError(f"reduced problem with scenarios {list(order[:len(scores) + 1])} "
                                   "has no optimal solution")
            scores.append(obj)
            if khat([obj], v_full):
                break
            solver.commit_subset([int(s)])
        return scores
    finally:
        solver.cleanup()


def gains_nonincreasing(labels) -> dict:
    """Consecutive PRISE steps with g[t] >= g[t+1], pooled over all labelled instances."""
    pairs = [np.asarray(l["label"].forward_trace.gains, dtype=float) for l in labels]
    hits = sum(int((g[:-1] >= g[1:]).sum()) for g in pairs)
    total = sum(len(g) - 1 for g in pairs)
    return {"hits": hits, "total": total, "frac": hits / total, "n_instances": len(labels)}


def row(tag: str, method: str, khats: list, n_scenarios: int) -> dict:
    budget = [K if k is None else k for k in khats]
    out = {"problem": tag, "method": method,
           "mean": 100 * statistics.fmean(budget) / n_scenarios,
           "median": 100 * statistics.median(budget) / n_scenarios}
    for r in RATIOS_PCT:
        out[f"pct_le_{r}"] = 100 * sum(k is not None and 100 * k <= r * n_scenarios for k in khats) / len(khats)
    out["not_converged"] = 100 * sum(k is None for k in khats) / len(khats)
    return out


def load(run, input_root: str, rel: str):
    with open(input_file(run, os.path.join(input_root, rel)), "rb") as fh:
        return pickle.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_run_args(ap)
    ap.add_argument("--batch", default="compression_budget",
                    help="run_io batch name for --run-id (default: %(default)s)")
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--threads", type=int, default=1, help="Gurobi threads (default: %(default)s)")
    ap.add_argument("--mip-gap", type=float, default=1e-4)
    args = ap.parse_args()

    run = open_run(args.batch, args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
                   [args.input_root])
    print(f"RUN_DIR={run.run_dir}")
    with open(input_file(run, os.path.join(args.input_root, "eval_regret/full_objs_cache.json"))) as fh:
        full_objs = json.load(fh)

    rows, per_instance, gains = [], {}, {}
    for problem_name, scale in PROBLEMS:
        cfg = PROBLEM_CONFIGS[(problem_name, scale)]
        tag, n, size = cfg["tag"], cfg["n_test"], cfg["n_scenarios"]
        v_full = full_objs[tag]
        assert len(v_full) == n, (tag, len(v_full))
        instances = load(run, args.input_root, cfg["data"])[-n:]
        problem = make_problem(problem_name, cfg)
        rng = np.random.default_rng(RANDOM_SEED)
        per_instance[tag] = {}
        for method in [m for m in METHODS if m in args.methods]:
            if method == "PRISE":
                labels = load(run, args.input_root, cfg["data"].replace("/inst_", "/label_prise_"))
                g = gains[tag] = gains_nonincreasing(labels)
                print(f"gains_nonincreasing_frac {tag} {g['frac']:.4f} ({g['hits']}/{g['total']})", flush=True)
                labels = labels[-n:]
                assert {l["label"].forward_trace.K for l in labels} == {K}, tag
                scores = [[float(x) for x in l["label"].forward_trace.best_scores] for l in labels]
            else:
                scores = []
                for i, inst in enumerate(instances):
                    if method == "Random":
                        order = rng.permutation(size)
                    else:
                        sce = problem.get_scenarios(inst.data)
                        order = np.argsort(-np.abs(sce.reshape(size, -1)).sum(axis=1))
                    scores.append(solve_order(problem, inst, order, v_full[i], args.threads, args.mip_gap))
                    if (i + 1) % 25 == 0:
                        print(f"  {tag} {method}: {i + 1}/{n}", flush=True)
            khats = [khat(s[:K], v) for s, v in zip(scores, v_full)]
            per_instance[tag][method] = {"khat": khats, "scores": scores}
            rows.append(row(tag, method, khats, size))
            r = rows[-1]
            print(f"{tag:<10} {method:<7}" + "".join(f" {r[c]:5.1f}" for c in list(r)[2:]), flush=True)

    with create_new(run, "compression_budget.json", mode="w") as fh:
        json.dump({"eps": EPS, "K": K, "random_seed": RANDOM_SEED, "threads": args.threads,
                   "mip_gap": args.mip_gap, "columns": list(rows[0])[2:], "rows": rows,
                   "gains_nonincreasing": gains,
                   "per_instance": per_instance}, fh, indent=1)
    print(f"Saved: {run.run_dir / 'compression_budget.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
