#!/usr/bin/env python3
"""Perturb the PRISE supervision of a label file (supervision-sensitivity study, VC-20-50).

The loss (score_weighted_loss_flat) reads only forward_trace.{selected_idx, gains, step_mask}:
the selected scenarios and their marginal gains. Each arm perturbs one of the two:

  noise    g' = g * exp(sigma * z), z ~ N(0, 1) on every valid step: Gaussian noise of std sigma
           on the log-gains the loss uses. Which scenarios are selected is unchanged.
  permute  the gain values are permuted uniformly among the selected scenarios.
  remap    one uniform permutation of all S scenarios is applied to the scenario indices
           (selected_idx, sce_ranking, sce_top_idx, sce_target_idx); gains are unchanged, so
           the supervision points at random scenarios. forward_trace.candidate_scores (only
           logged by the trainer) is left as it is.

Each arm is written as data/<problem>/label_prise_<arm>_<rest of the input name>.
Training selects it with problem.data.label_type=prise_<arm>. The RNG seeds are the ones the paper's labels used;
with them the output equals the paper's perturbed labels.

usage:
  python scripts/pipeline/perturb_labels.py \
      --label data/vertex_cover/label_prise_seed42_cnt2500_n20_s50_p10_scen_uniform.pkl \
      --noise 0.1 0.7 --permute --remap
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import sys

import numpy as np
from scipy.stats import kendalltau

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, REPO)

from src.utils.run_io import add_run_args, create_new, input_file, open_script_run, data_paths, data_output_path  # noqa: E402

NOISE_SEED = 20260724  # + round(1000 * sigma)
PERMUTE_SEED = 20260724999
REMAP_SEED = 20260725111


def add_noise(label, sigma: float, rng: np.random.Generator) -> None:
    ft = label.forward_trace
    gains = np.asarray(ft.gains, dtype=float)
    mask = np.asarray(ft.step_mask, dtype=bool)
    z = rng.standard_normal(len(gains))
    ft.gains = np.where(mask, gains * np.exp(sigma * z), gains)


def permute_gains(label, rng: np.random.Generator) -> None:
    ft = label.forward_trace
    gains = np.asarray(ft.gains, dtype=float).copy()
    valid = np.flatnonzero(np.asarray(ft.step_mask, dtype=bool))
    gains[valid] = gains[valid][rng.permutation(len(valid))]
    ft.gains = gains


def remap_scenarios(label, rng: np.random.Generator) -> None:
    ft = label.forward_trace
    pi = rng.permutation(ft.S)
    ft.selected_idx = pi[np.asarray(ft.selected_idx)]
    for field in ("sce_ranking", "sce_top_idx", "sce_target_idx"):
        setattr(label, field, pi[np.asarray(getattr(label, field))])
    # The paper's labels drew tie-break noise here for a top-4 overlap metric; drawing it
    # keeps the random stream, and so every later instance's permutation, the same.
    rng.uniform(0, 1e-9, 2 * ft.S)


def fidelity(clean, new) -> dict[str, float]:
    """How much of the supervision survived: gain order among the selected scenarios
    (Kendall tau), the selected set, and the top-gain scenario."""
    taus, overlap, top_kept = [], [], []
    for a, b in zip(clean, new):
        fa, fb = a["label"].forward_trace, b["label"].forward_trace
        mask = np.asarray(fa.step_mask, dtype=bool)
        ga, gb = np.asarray(fa.gains, dtype=float)[mask], np.asarray(fb.gains, dtype=float)[mask]
        sa, sb = np.asarray(fa.selected_idx)[mask], np.asarray(fb.selected_idx)[mask]
        if mask.sum() >= 2:
            taus.append(kendalltau(ga, gb).statistic)
        overlap.append(len(set(sa) & set(sb)) / len(sa))
        top_kept.append(sa[np.argmax(ga)] == sb[np.argmax(gb)])
    return {"gain_kendall_tau": float(np.mean(taus)), "selected_set_overlap": float(np.mean(overlap)),
            "top_scenario_kept": float(np.mean(top_kept)), "n_valid_steps": float(np.mean(
                [np.asarray(r["label"].forward_trace.step_mask).sum() for r in clean]))}


def arms(args) -> list[tuple[str, int, callable]]:
    out = [(f"noise{round(100 * s):03d}", NOISE_SEED + round(1000 * s),
            lambda lab, rng, s=s: add_noise(lab, s, rng)) for s in args.noise]
    if args.permute:
        out.append(("permute", PERMUTE_SEED, permute_gains))
    if args.remap:
        out.append(("remap", REMAP_SEED, remap_scenarios))
    return out


def sha256(path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--label", required=True, help="label file, relative to the input root")
    ap.add_argument("--noise", type=float, nargs="*", default=[], metavar="SIGMA")
    ap.add_argument("--permute", action="store_true")
    ap.add_argument("--remap", action="store_true")
    add_run_args(ap)
    args = ap.parse_args()
    folder, name = os.path.split(args.label)
    if not name.startswith("label_prise_"):
        ap.error(f"expected a label_prise_* file, got {name}")
    if not arms(args):
        ap.error("choose at least one of --noise, --permute, --remap")

    run = open_script_run(args, __file__)
    src = input_file(run, os.path.join(args.input_root, args.label))
    with open(src, "rb") as f:
        clean = pickle.load(f)

    record = {"input": args.label, "input_sha256": sha256(src), "arms": {}}
    for arm, seed, perturb in arms(args):
        out = os.path.join(folder, name.replace("label_prise_", f"label_prise_{arm}_", 1))
        out_path = (data_output_path(run, os.path.basename(folder), os.path.basename(out))
                    if data_paths(run) is not run else run.run_dir / out)
        data = copy.deepcopy(clean)
        rng = np.random.default_rng(seed)
        for item in data:
            perturb(item["label"], rng)
        with create_new(data_paths(run), out_path) as fh:
            pickle.dump(data, fh)
        record["arms"][arm] = {"label_type": f"prise_{arm}", "file": out, "seed": seed,
                               "sha256": sha256(out_path), **fidelity(clean, data)}
        print(json.dumps({arm: record["arms"][arm]}))
    with create_new(run, "perturb_labels.json", mode="w") as fh:
        json.dump(record, fh, indent=2)
    print(f"wrote {len(record['arms'])} label files; manifest: {run.run_dir / 'perturb_labels.json'}")


if __name__ == "__main__":
    main()
