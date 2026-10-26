#!/usr/bin/env python3
"""Generate evaluation commands from runnable experiment recipes.

Read instances, labels and checkpoints from NEURPRISE_INPUT_ROOT and write
new experiment results under NEURPRISE_RUN_ROOT. No paper values are compared.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import shlex
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "repro", "runs.json")
BATCH = "repro"
E2E_DEVICE = "cpu"  # NeurPRISE scoring device; the paper's timings use --device cuda

# Label file variant the e2e datamodule must load: for these sets the undecorated label file
# holds only part of the instances; the 500-record role-merged file (forward-trace train/val
# labels + exact test labels, see scripts/pipeline/merge_label_roles.py) is the paper's split.
E2E_LABEL_VARIANT = {("cflp_ebp", "medium"): "merged500", ("cflp_ebp", "large"): "merged500",
                     ("selection", "large"): "merged500", ("vertex_cover", "medium"): "merged500",
                     ("vertex_cover", "large"): "merged500"}

# Sets whose e2e input is the exact (bigmilp) label file, and the split to evaluate: for the
# cross-distribution sets it holds every instance; for the cross-size sets (more scenarios or
# nodes than the model was trained on) only the 50 test instances.
BIGMILP_SPLIT = {"normal_x": "test", "mm_x": "test", "normal_id": "test", "mm_id": "test",
                 "s100": "all", "s200": "all", "ultra": "all"}

CHECKPOINTS = os.path.join(REPO, "repro", "checkpoints.json")


@functools.lru_cache(maxsize=None)
def checkpoints() -> dict[str, dict]:
    """repro/checkpoints.json by key (a recipe names its model by key)."""
    return {c["key"]: c for c in json.load(open(CHECKPOINTS))["checkpoints"]}


def ckpt_overrides(ckpt: dict) -> list[str]:
    """The evaluator must build the model the checkpoint was trained with, under the eval
    config that matches its training config (train_<x> -> eval_<x>)."""
    t = ckpt["train"]
    return ["--config-name", "eval_" + t["config"].removeprefix("train_"),
            f"model={t['model']}", f"losses={t['loss']}"]


def _configs():
    sys.path.insert(0, REPO)
    import scripts.eval.eval_regret as er
    return er.PROBLEM_CONFIGS, er.K_RATIOS


def _instance_dims(cfg: dict) -> list[str]:
    out = [f"problem.instance.n_scenarios={cfg['n_scenarios']}"]
    for key in ("n_nodes", "n_facilities", "n_customers", "p_edge", "p_select"):
        if key in cfg:
            out.append(f"problem.instance.{key}={cfg[key]}")
    return out


def e2e_problem(cfg: dict, ckpt: dict) -> list[str]:
    """Hydra overrides that make the e2e evaluator read the instances the regret stage reads
    (cfg["data"] = data/<problem config>/..._scen_<scenario>.pkl). A checkpoint trained under
    another config of the same instances (the CV encoding, configs/problem/cflp_cv.yaml)
    keeps its own config."""
    _, config, name = cfg["data"].split("/")
    scenario = name.rsplit("_scen_", 1)[1].removesuffix(".pkl")
    return [f"problem={ckpt['train'].get('problem', config)}",
            f"problem.instance.scenario_name={scenario}"]


def build(recipe: dict, threads: int, mip_gap: float, time_limit: float,
          device: str = E2E_DEVICE) -> list[dict]:
    """Commands for one recipe; paths use $NEURPRISE_INPUT_ROOT / $NEURPRISE_RUN_ROOT."""
    if recipe["kind"] == "compression":  # Table 7: no checkpoint, no reduced-solve time limit
        return [{"id": recipe["id"], "argv": [
            "python", "scripts/eval/compression_budget.py", "--input-root", "$NEURPRISE_INPUT_ROOT",
            "--run-id", recipe["id"], "--batch", BATCH, "--threads", str(threads),
            "--mip-gap", str(mip_gap)]}]
    if recipe["kind"] == "runtime":  # Tables 15/16: scenario-count timing, one scale per recipe
        return [{"id": recipe["id"], "argv": [
            "python", "scripts/eval/scenario_runtime.py", "--input-root", "$NEURPRISE_INPUT_ROOT",
            "--run-id", recipe["id"], "--batch", BATCH, "--scale", recipe["scale"],
            "--threads", str(threads), "--mip-gap", str(mip_gap), "--time-limit", f"{time_limit:g}",
            "--device", device]}]
    if recipe["kind"] == "time_budget":  # Table 14: full MILP under the fixed 1 s / 2 s limit
        return [{"id": recipe["id"], "argv": [
            "python", "scripts/eval/time_budget.py", "--input-root", "$NEURPRISE_INPUT_ROOT",
            "--run-id", recipe["id"], "--batch", BATCH, "--scale", recipe["scale"],
            "--k", str(recipe["k"]), "--threads", str(threads), "--mip-gap", str(mip_gap)]}]
    problem_configs, k_ratios = _configs()
    problem, scale, k = recipe["problem"], recipe["scale"], recipe.get("k")
    rid = recipe["id"]
    common = ["--input-root", "$NEURPRISE_INPUT_ROOT", "--run-id", rid, "--batch", BATCH,
              "--threads", str(threads), "--time-limit", f"{time_limit:g}"]
    cfg = problem_configs[(problem, scale)]
    k_ratio = k_ratios[cfg["n_scenarios"]][k]
    e2e_id = f"{rid}_e2e"
    entry = checkpoints()[recipe["checkpoint"]]
    ckpt = f"$NEURPRISE_INPUT_ROOT/{entry['path']}"
    split = BIGMILP_SPLIT.get(scale, "test")
    e2e = ["python", "scripts/04_05_eval_ml_2ro_e2e.py", *ckpt_overrides(entry),
           *e2e_problem(cfg, entry), f"run_name={e2e_id}", "eval.name=repro",
           f"eval.k_ratio={k_ratio}", f"eval.ckpt_path={ckpt}",
           f"solver.threads={threads}", f"eval.split={split}", "eval.load_ckpt=true",
           f"eval.device={device}", *_instance_dims(cfg),
           f"data.instance_cnt={cfg['n_total']}"]
    if (problem, scale) in E2E_LABEL_VARIANT:
        e2e.append(f"problem.data.label_variant={E2E_LABEL_VARIANT[(problem, scale)]}")
    if scale in BIGMILP_SPLIT:
        e2e.append("label.prise.output=bigmilp")
    e2e_dir = f"$NEURPRISE_RUN_ROOT/04_05_eval_ml_2ro_e2e/{e2e_id}"
    regret = ["python", "scripts/eval/eval_regret.py", "--problem", problem, "--scale", scale,
              "--method", "neurprise", "--k", str(k), *common, "--mip-gap", str(mip_gap),
              "--eval-root", e2e_dir, "--eval-base", f"{e2e_dir}/eval/repro/e2e_2ro"]
    return [{"id": e2e_id, "argv": e2e}, {"id": rid, "argv": regret}]


def e2e_label_file(e2e_argv: list[str]) -> str:
    """The label pickle the e2e evaluator will load, relative to the input root."""
    from hydra import compose, initialize_config_dir
    from src.utils.omega_conf import register_omega_conf_custom_resolvers
    from src.utils.vertex_cover.vertex_cover_utils import apply_label_variant_suffix, get_instance_file_name

    i = e2e_argv.index("--config-name")
    config_name = e2e_argv[i + 1]
    overrides = [a for a in e2e_argv[2:] if "=" in a and not a.startswith("eval.ckpt_path=")]
    register_omega_conf_custom_resolvers()
    with initialize_config_dir(version_base=None, config_dir=os.path.join(REPO, "configs")):
        cfg = compose(config_name=config_name, overrides=overrides)
    inst = cfg.problem.instance
    n_nodes = inst.get("n_nodes")
    p_edge = inst.get("p_edge")
    if p_edge is None and str(cfg.problem.name).startswith("selection") and n_nodes is not None:
        p_edge = int(n_nodes) // 2
    name = get_instance_file_name(
        seed=cfg.seed, instance_cnt=cfg.data.get("instance_cnt", 1000), n_nodes=n_nodes,
        n_scenarios=inst.n_scenarios, n_facilities=inst.get("n_facilities"),
        n_customers=inst.get("n_customers"), pattern=cfg.problem.data.get("inst_spec_label_pattern"),
        p_edge=p_edge, scenario_name=inst.get("scenario_name", "default"), label_type=cfg.label.type)
    return f"data/{cfg.problem.name}/{apply_label_variant_suffix(name, cfg)}"


def inputs(recipe: dict) -> list[str]:
    """Files under the input root needed to evaluate one NeurPRISE recipe."""
    problem_configs, _ = _configs()
    cfg = problem_configs[(recipe["problem"], recipe["scale"])]
    need = [cfg["data"], "eval_regret/full_objs_cache.json"]
    if recipe.get("checkpoint"):
        need.append(checkpoints()[recipe["checkpoint"]]["path"])
    need.append(e2e_label_file(build(recipe, 40, 1e-4, 900)[0]["argv"]))
    return need


def produces(recipe: dict) -> str:
    """Where the regenerated result lands, relative to $NEURPRISE_RUN_ROOT."""
    if recipe["kind"] == "compression":
        return f"{BATCH}/{recipe['id']}/compression_budget.json"
    problem_configs, _ = _configs()
    tag = problem_configs[(recipe["problem"], recipe["scale"])]["tag"]
    if recipe["kind"] == "runtime":
        return f"{BATCH}/{recipe['id']}/{tag}_truncated_k6.json"
    k = recipe.get("k")
    if recipe["kind"] == "time_budget":
        return f"{BATCH}/{recipe['id']}/{recipe['output']}"
    method = "neurprise_id" if recipe["problem"] == "cflp_ebp" and recipe["scale"].endswith("_id") \
        else "neurprise"
    name = f"{tag}_{method}_k{k}.json"
    return f"{BATCH}/{recipe['id']}/eval_regret/{recipe['problem']}/{name}"


def load(tables=None) -> list[dict]:
    recipes = [r for r in json.load(open(RUNS))["runs"] if r["auto"]]
    if tables:
        recipes = [r for r in recipes if set(r["tables"]) & set(tables)]
    return recipes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--table", type=int, nargs="*")
    ap.add_argument("--threads", type=int, default=40)
    ap.add_argument("--mip-gap", type=float, default=1e-4)
    ap.add_argument("--time-limit", type=float, default=900)
    ap.add_argument("--device", default=E2E_DEVICE, help="NeurPRISE scoring device, e.g. cuda (default: %(default)s)")
    ap.add_argument("--emit", choices=["sh", "json"], default="sh")
    args = ap.parse_args()

    recipes = load()
    available = sorted({table for recipe in recipes for table in recipe["tables"]})
    if args.table is not None:
        if not args.table:
            ap.error("--table requires at least one table number")
        unsupported = sorted(set(args.table) - set(available))
        if unsupported:
            ap.error(
                f"No NeurPRISE evaluation recipes for table(s): {', '.join(map(str, unsupported))}. "
                f"Available table groups: {', '.join(map(str, available))}"
            )
        recipes = load(args.table)
    if not recipes:
        ap.error("No runnable NeurPRISE evaluation recipes found")

    runs = []
    for recipe in recipes:
        for r in build(recipe, args.threads, args.mip_gap, args.time_limit, args.device):
            runs.append({**r, "output": recipe["output"]})
    if args.emit == "json":
        print(json.dumps(runs, indent=1))
        return 0
    print("set -euo pipefail")
    for r in runs:
        print(" ".join(f'"{a}"' if "$NEURPRISE" in a else shlex.quote(a) for a in r["argv"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
