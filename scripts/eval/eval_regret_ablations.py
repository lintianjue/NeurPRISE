#!/usr/bin/env python3
"""Regret evaluator for NeurPRISE ablation variants (loss, architecture, seed, training size).

Loads scenario indices from an ablation model's eval directory, solves the
reduced MILP, evaluates the first-stage solution on ALL scenarios, and
computes regret on the full scenario set.

SAFETY: Writes ONLY to eval_regret/. NEVER touches eval/.

The model_dir name encodes the ablation variant.  Short names are derived
automatically (see ABLATION_NAME_MAP) and used in the output filename:
    eval_regret/{problem}/{tag}_{ablation_name}_k{k}.json

Usage:
    # Single run
    python eval_regret_ablations.py --problem vertex_cover --scale small \
        --model-dir ML_s42_i2500_n20_s50_t-train_scale_norm_m-det_source_sce_mean_l-log_bce_loss_test \
        --k 2

    # With explicit ablation name
    python eval_regret_ablations.py --problem cflp_ebp --scale small \
        --model-dir ML_s42_i2500_m30_n30_s50_t-train_scale_norm_m-deepsets_l-score_weighted_loss_flat_test \
        --k 4 --ablation-name deepsets

"""
import argparse
import json
import os
import pickle
import re
import sys
import time
import numpy as np

# The project root is derived from this file's location, never a hardcoded path.
_PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
sys.path.insert(0, _PROJECT_ROOT)

BASE = _PROJECT_ROOT
REGRET_DIR = os.path.join(BASE, "eval_regret")
THREADS = 40
MIP_GAP = 1e-4
TIME_LIMIT = 600

# Import shared infrastructure from eval_regret.py
from scripts.eval.eval_regret import (
    PROBLEM_CONFIGS, EVAL_DIRS, K_RATIOS,
    make_problem, evaluate_x_on_all_scenarios, get_neurprise_x, save_results,
)

# ── Ablation name extraction ──
# Maps substrings found in model_dir → short ablation names.
# Checked in order; first match wins.
ABLATION_NAME_MAP = [
    # Loss ablations (check before architecture since both can appear)
    ("l-score_weighted_loss_flat", "swlf"),
    ("l-log_bce_loss", "bce"),
    ("l-multilabel_bce_loss", "mlbce"),
    ("l-regression_loss", "r2r"),
    # Architecture ablations
    ("m-config_early_fusion_mlp", "ef"),
    ("m-one_shot_crossattn_edge_dim1_small", "crossattn"),
    ("m-deepsets", "deepsets"),
    ("m-det_source_sce_mean", "ours"),
]

# Seed pattern: ML_seed{N}_... or ML_s42_e{N}_...
SEED_RE = re.compile(r"ML_seed(\d+)_")
EPOCH_RE = re.compile(r"ML_s42_e(\d+)_")
# Training size pattern: ablation_t{N}_
TRAIN_SIZE_RE = re.compile(r"ablation_t(\d+)_")
# Normalization pattern
NORMALIZE_RE = re.compile(r"t-train_normalize_")
SCALE_NORM_RE = re.compile(r"t-train_scale_norm_")
# Cross-scale eval: tr-{train_spec}_ev-{eval_spec}
CROSS_SCALE_RE = re.compile(r"tr-.*?_ev-")


def extract_ablation_name(model_dir):
    """Derive a short, human-readable ablation label from the model directory name.

    Priority:
    1. Seed variants:         seed1, seed2, ...
    2. Epoch checkpoints:     e1, e2, ...
    3. Training size:         t500, t1500, t2000
    4. Cross-scale eval:      gen_{spec}  (generalization)
    5. Loss variant:          bce, mlbce, r2r, swlf
    6. Architecture variant:  ef, crossattn, deepsets, ours
    7. Normalization:         norm (normalize) vs snorm (scale_norm)

    If architecture is "ours" (det_source_sce_mean) and loss is "swlf"
    (score_weighted_loss_flat), this is the default model — return "ours".
    """
    parts = []

    # Seed variant
    m = SEED_RE.search(model_dir)
    if m:
        parts.append(f"seed{m.group(1)}")

    # Epoch checkpoint
    m = EPOCH_RE.search(model_dir)
    if m:
        parts.append(f"e{m.group(1)}")

    # Training size
    m = TRAIN_SIZE_RE.search(model_dir)
    if m:
        parts.append(f"t{m.group(1)}")

    # Cross-scale (generalization)
    if CROSS_SCALE_RE.search(model_dir):
        parts.append("gen")

    # Architecture + loss
    arch = None
    loss = None
    for substr, short in ABLATION_NAME_MAP:
        if substr in model_dir:
            if substr.startswith("l-"):
                loss = short
            elif substr.startswith("m-"):
                arch = short

    # Normalization variant (only flag if using plain "normalize" instead of "scale_norm")
    uses_plain_norm = NORMALIZE_RE.search(model_dir) and not SCALE_NORM_RE.search(model_dir)

    # Compose the name
    if not parts:
        # No seed/epoch/trainsize/gen — pure architecture or loss ablation
        if arch == "ours" and loss == "swlf" and not uses_plain_norm:
            return "ours"

        # Normalization as a standalone differentiator
        if uses_plain_norm:
            parts.append("norm")

        # Architecture (only if non-default)
        if arch and arch != "ours":
            parts.append(arch)

        # Loss (only if non-default)
        if loss and loss != "swlf":
            parts.append(loss)

        return "_".join(parts) if parts else model_dir[:30]
    else:
        # Has seed/epoch/trainsize/gen prefix — add non-default arch/loss
        if uses_plain_norm:
            parts.append("norm")
        if arch and arch != "ours":
            parts.append(arch)
        if loss and loss != "swlf":
            parts.append(loss)
        return "_".join(parts)


def find_model_eval_dir(problem_name, scale, model_dir):
    """Locate the eval directory for a given model variant.

    Searches two layouts:
      A. eval/{problem}/{inst_tag}/{model_dir}/e2e_2ro/         (standard)
      B. eval/{problem}/{ablation_dir}/{model_dir}/e2e_2ro/     (ablation grouping)

    Returns the path to the e2e_2ro/ directory, or None.
    """
    eval_dir_key = EVAL_DIRS.get((problem_name, scale), "")
    if not eval_dir_key:
        return None

    # Layout A: standard inst_tag based
    candidate = os.path.join(BASE, eval_dir_key, model_dir, "e2e_2ro")
    if os.path.isdir(candidate):
        return candidate

    # Layout B: search all subdirs of eval/{problem}/ for {model_dir}/e2e_2ro
    prob_eval_root = os.path.join(BASE, "eval", problem_name)
    if os.path.isdir(prob_eval_root):
        for subdir in os.listdir(prob_eval_root):
            candidate = os.path.join(prob_eval_root, subdir, model_dir, "e2e_2ro")
            if os.path.isdir(candidate):
                return candidate

    return None


def load_full_objs_cached(problem_name, scale, n_test):
    """Load full_objs from the portable JSON cache.

    Falls back to eval_regret.load_full_objs() if cache miss.
    """
    cfg = PROBLEM_CONFIGS[(problem_name, scale)]
    tag = cfg["tag"]

    cache_path = os.path.join(REGRET_DIR, "full_objs_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        if tag in cache:
            objs = cache[tag][:n_test]
            print(f"  full_objs loaded from cache ({len(objs)} instances)")
            return objs

    # Fallback: load from eval dirs
    from scripts.eval.eval_regret import find_neurprise_eval_base, load_full_objs
    eval_dir_key = EVAL_DIRS.get((problem_name, scale), "")
    inst_tag = os.path.basename(eval_dir_key)
    eval_base = find_neurprise_eval_base(problem_name, inst_tag, scale=scale)
    if eval_base is not None:
        S = cfg["n_scenarios"]
        k_ratios = K_RATIOS.get(S, {})
        for kr in sorted(k_ratios.values()):
            try:
                objs = load_full_objs(eval_base, kr, n_test)
                if objs:
                    print(f"  full_objs loaded from eval dir (k_ratio={kr})")
                    return objs
            except Exception:
                continue

    return None


def run_ablation_regret(problem_name, scale, model_dir, k, ablation_name=None):
    """Compute regret for a NeurPRISE ablation variant.

    Mirrors run_neurprise_regret() but targets an arbitrary model_dir.
    """
    cfg = PROBLEM_CONFIGS[(problem_name, scale)]
    prob = make_problem(problem_name, cfg)
    S = cfg["n_scenarios"]
    n_test = cfg["n_test"]

    # Resolve ablation short name
    if ablation_name is None:
        ablation_name = extract_ablation_name(model_dir)
    print(f"  Ablation name: {ablation_name}")

    # Find the eval directory
    eval_base = find_model_eval_dir(problem_name, scale, model_dir)
    if eval_base is None:
        print(f"ERROR: eval directory not found for {model_dir}")
        print(f"  Searched under eval/{problem_name}/*/")
        return None, ablation_name

    # Resolve k → k_ratio
    k_ratios = K_RATIOS.get(S, {})
    if k not in k_ratios:
        print(f"ERROR: k={k} not in K_RATIOS[{S}] = {k_ratios}")
        return None, ablation_name
    k_ratio = k_ratios[k]

    # Load scenario indices
    k_dir = os.path.join(eval_base, f"k_{k_ratio}")
    if not os.path.isdir(k_dir):
        # Try alternate k_ratio formatting (e.g., 0.04 vs 0.040)
        k_dir_alt = os.path.join(eval_base, f"k_{k_ratio:.2f}")
        if os.path.isdir(k_dir_alt):
            k_dir = k_dir_alt
        else:
            print(f"ERROR: k directory not found: {k_dir}")
            print(f"  Available: {os.listdir(eval_base) if os.path.isdir(eval_base) else 'N/A'}")
            return None, ablation_name

    idx_path = os.path.join(k_dir, "all_idx.npy")
    if not os.path.exists(idx_path):
        idx_path = os.path.join(k_dir, "test_idx.npy")
    if not os.path.exists(idx_path):
        print(f"ERROR: no idx file found in {k_dir}")
        print(f"  Contents: {os.listdir(k_dir)}")
        return None, ablation_name

    loaded_idx = np.load(idx_path)
    print(f"  Loaded indices from {os.path.basename(idx_path)}: shape={loaded_idx.shape}")

    # Load test instances
    data_path = os.path.join(BASE, cfg["data"])
    if not os.path.exists(data_path):
        print(f"ERROR: data file not found: {data_path}")
        return None, ablation_name

    with open(data_path, "rb") as f:
        all_instances = pickle.load(f)

    # Determine index alignment: _test suffix → last n_test, _all → all instances
    is_test_split = "test_idx" in os.path.basename(idx_path)
    if is_test_split:
        test_instances = all_instances[-n_test:]
    else:
        test_instances = all_instances[-n_test:]
        # all_idx.npy may cover all instances; take last n_test
        if loaded_idx.shape[0] > n_test:
            loaded_idx = loaded_idx[-n_test:]

    n_avail = min(n_test, loaded_idx.shape[0], len(test_instances))

    # GUARD: catch instance count mismatch that has caused bugs repeatedly
    if loaded_idx.shape[0] < n_test:
        print(f"  WARNING: loaded_idx has {loaded_idx.shape[0]} rows but n_test={n_test}.")
        print(f"  This likely means the wrong eval directory was used (_test instead of _all).")
        print(f"  ABORTING to prevent misaligned results. Fix: use the _all model dir.")
        print(f"  FAILED: no results for {ablation_name}")
        return None, ablation_name

    # Load full_objs
    full_objs = load_full_objs_cached(problem_name, scale, n_avail)
    if not full_objs:
        print("ERROR: could not load full_objs (no cache, no eval dir)")
        return None, ablation_name

    # --- PRE-CHECKS: data alignment safety ---

    # 1. full_obj magnitude sanity per problem type
    fo_mean = np.mean(full_objs[:n_avail])
    if problem_name == "cflp_ebp" and fo_mean > 2_000_000:
        print(f"WARNING: full_obj mean={fo_mean:.0f} seems too high for CFLP — possible scale mismatch")
    if problem_name == "vertex_cover" and fo_mean > 10_000:
        print(f"WARNING: full_obj mean={fo_mean:.0f} seems too high for VC — possible scale mismatch")
    if problem_name == "selection" and fo_mean > 100_000:
        print(f"WARNING: full_obj mean={fo_mean:.0f} seems too high for SEL — possible scale mismatch")

    # 2. Instance count match
    if len(test_instances) < n_avail:
        print(f"ERROR: expected {n_avail} test instances but loaded {len(test_instances)}")
        return None, ablation_name
    if len(full_objs) < n_avail:
        print(f"ERROR: full_objs has {len(full_objs)} entries but need {n_avail}")
        return None, ablation_name

    # 3. Scenario index range check
    max_idx = int(np.max(loaded_idx[:n_avail]))
    if max_idx >= S:
        print(f"ERROR: scenario index {max_idx} >= S={S} — model evaluated on different S?")
        return None, ablation_name

    # --- END PRE-CHECKS ---

    print(f"  Evaluating {n_avail} instances, k={k} (k_ratio={k_ratio}), S={S}")
    t0 = time.time()
    results = []

    for i in range(n_avail):
        inst = test_instances[i]
        fo = full_objs[i]
        sce_idx = loaded_idx[i]

        # Solve reduced MILP
        sol, red_obj = get_neurprise_x(prob, inst, sce_idx, mip_gap=MIP_GAP, problem_name=problem_name)

        # Check for failed solve
        if problem_name == "cflp_ebp":
            if sol is None or sol[0] is None:
                print(f"  WARNING: inst {i} solve failed, skipping")
                continue
            x = sol  # (y, z) tuple
        else:
            if sol is None:
                print(f"  WARNING: inst {i} solve failed, skipping")
                continue
            x = sol

        eval_obj = evaluate_x_on_all_scenarios(prob, inst, x, S, problem_name=problem_name)
        regret = (eval_obj - fo) / fo * 100 if fo != 0 else float("nan")

        # --- POST-CHECKS (per-instance) ---

        # 1. Regret non-negative check (allow small float noise)
        if regret < -1.0:
            print(f"  WARNING: inst {i} has negative regret {regret:.2f}% — possible data mismatch. Skipping.")
            continue

        # --- END POST-CHECKS ---

        results.append({
            "instance_id": i,
            "full_obj": fo,
            "reduced_obj": red_obj,
            "eval_on_xi": eval_obj,
            "regret": round(regret, 4),
        })

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_avail - i - 1)
            print(f"  [{i+1}/{n_avail}] regret={regret:.2f}% "
                  f"(elapsed={elapsed:.0f}s, ETA={eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"  Done: {len(results)}/{n_avail} instances in {elapsed:.0f}s")

    # --- POST-BATCH CHECK: systematic negative regret ---
    if results:
        neg_count = sum(1 for r in results if r["regret"] < -0.5)
        if neg_count > len(results) * 0.1:
            print(f"ERROR: {neg_count}/{len(results)} instances have negative regret "
                  f"— likely data mismatch. NOT saving.")
            return None, ablation_name

    return results, ablation_name


def main():
    parser = argparse.ArgumentParser(
        description="Regret evaluator for NeurPRISE ablation model variants")
    parser.add_argument("--problem", required=True,
                        choices=["vertex_cover", "selection", "cflp_ebp"])
    parser.add_argument("--scale", required=True,
                        choices=["small", "medium", "large", "ultra",
                                 "s100", "s200", "normal_x", "mm_x",
                                 "normal_id", "mm_id"])
    parser.add_argument("--model-dir", required=True,
                        help="Model directory name under eval/{problem}/{inst_tag}/")
    parser.add_argument("--k", type=int, required=True,
                        help="Number of selected scenarios")
    parser.add_argument("--ablation-name", type=str, default=None,
                        help="Override auto-detected ablation short name")
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--server", type=str, default=None,
                        help="Server name for traceability logging")
    args = parser.parse_args()

    # Override module-level THREADS for solvers
    import scripts.eval.eval_regret as _er
    _er.THREADS = args.threads
    sys.modules[__name__].THREADS = args.threads

    print(f"=== Ablation regret: {args.model_dir}")
    print(f"    problem={args.problem} scale={args.scale} k={args.k} threads={args.threads}")
    print()

    results, ablation_name = run_ablation_regret(
        args.problem, args.scale, args.model_dir, args.k,
        ablation_name=args.ablation_name,
    )

    if results:
        # Save with ablation-specific naming: {tag}_{ablation_name}_k{k}.json
        summary = save_results(
            results, args.problem, args.scale,
            method=ablation_name,
            k=args.k,
            threads=args.threads,
            server=args.server,
        )
        print(f"\n  Ablation: {ablation_name}")
        print(f"  Regret:   {summary['regret_mean']:.2f}% +/- {summary['regret_std']:.2f}%")
        print(f"  Median:   {summary['regret_median']:.2f}%")
    else:
        print(f"\n  FAILED: no results for {ablation_name}")
        sys.exit(1)


if __name__ == "__main__":
    from src.utils.log_utils import timestamp_stdout
    timestamp_stdout()   # script runs only: tests call main() and parse its output
    main()
