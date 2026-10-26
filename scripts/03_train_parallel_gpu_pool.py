#!/usr/bin/env python3
"""
Parallel GPU-pool runner for scripts/03_train.sh.

Key idea:
- Maintain an asyncio.Queue of GPU ids (tokens).
- Each experiment acquires one GPU id (await queue.get()).
- Runs training with CUDA_VISIBLE_DEVICES set to that GPU.
- When the process ends, returns the token (queue.put_nowait()).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional


DEFAULT_EXPERIMENTS: List[str] = [
    "selection|det_source_sce_mean|score_weighted_loss_flat",
]


@dataclass(frozen=True)
class Experiment:
    problem: str
    model: str
    loss: str


def parse_experiment(spec: str, default_problem: str) -> Experiment:
    parts = spec.split("|")
    if len(parts) == 3:
        return Experiment(problem=parts[0], model=parts[1], loss=parts[2])
    if len(parts) == 2:
        return Experiment(problem=default_problem, model=parts[0], loss=parts[1])
    raise ValueError(f"Bad experiment spec: {spec!r}. Expected 'PROBLEM|MODEL|LOSS' or 'MODEL|LOSS'.")


def parse_gpus(gpus_csv: str) -> List[str]:
    gpus = [g.strip() for g in gpus_csv.split(",") if g.strip() != ""]
    if not gpus:
        raise ValueError("No GPUs provided. Use e.g. --gpus 0,2,3")
    # keep as strings (CUDA_VISIBLE_DEVICES expects strings)
    return gpus


def parse_nvidia_smi_list(output: str) -> List[str]:
    gpus: List[str] = []
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("GPU "):
            try:
                idx = line.split(":", 1)[0].split()[1]
            except (IndexError, ValueError):
                continue
            if idx.isdigit():
                gpus.append(idx)
    return gpus


def list_available_gpus() -> Optional[List[str]]:
    try:
        out = subprocess.check_output(["nvidia-smi", "--list-gpus"], text=True)
    except FileNotFoundError:
        return None
    except subprocess.CalledProcessError:
        return None
    gpus = parse_nvidia_smi_list(out)
    return gpus if gpus else None


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_tag(s: str) -> str:
    return s.replace("/", "-").replace(" ", "_")


def split_extra(extra_str: str) -> List[str]:
    extra_str = (extra_str or "").strip()
    if not extra_str:
        return []
    # matches the bash behavior (space-separated Hydra overrides)
    return shlex.split(extra_str)


async def stream_to_file(stream: asyncio.StreamReader, fp) -> None:
    while True:
        chunk = await stream.readline()
        if not chunk:
            break
        fp.buffer.write(chunk)
        fp.flush()


async def run_one_job(
    *,
    gpu_ids: List[str],
    n_gpus_per_job: int,
    exp_index: int,
    exp: Experiment,
    run_one: Path,
    n_nodes: int,
    n_customers: int | None,
    n_scenarios: int,
    n_instances: int,
    batch: int,
    dist_name: str,
    train_config: str,
    log_name_format: str,
    run_id: str,
    device_in_container: str,
    extra_overrides: List[str],
    log_dir: Path,
) -> int:
    ts = now_ts()
    cuda_devices = ",".join(gpu_ids)
    if log_name_format:
        log_name = log_name_format.format(
            i=exp_index,
            problem=safe_tag(exp.problem),
            model=safe_tag(exp.model),
            loss=safe_tag(exp.loss),
            run_id=run_id or ts,
        )
        log_file = log_dir / log_name
    else:
        tag = safe_tag(f"p-{exp.problem}_m-{exp.model}_l-{exp.loss}_gpus-{cuda_devices}_{ts}")
        log_file = log_dir / f"{tag}.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if n_gpus_per_job > 1:
        env["DDP"] = "1"
        env["NPROC_PER_NODE"] = str(n_gpus_per_job)
        env["MASTER_PORT"] = str(29500 + (exp_index % 1000))

    if exp.problem in {"cflp", "cflp_ebp"}:
        if n_customers is None:
            raise ValueError("n_customers is required for cflp variants.")
        cmd = [
            str(run_one),
            exp.problem,
            str(n_nodes),
            str(n_customers),
            str(n_scenarios),
            str(n_instances),
            f"model={exp.model}",
            f"losses={exp.loss}",
            device_in_container,
            str(batch),
            dist_name,
        ]
        if train_config:
            cmd += ["--config-name", train_config]
        cmd += extra_overrides
    else:
        cmd = [
            str(run_one),
            exp.problem,
            str(n_nodes),
            str(n_scenarios),
            str(n_instances),
            f"model={exp.model}",
            f"losses={exp.loss}",
            device_in_container,
            str(batch),
            dist_name,
        ]
        if train_config:
            cmd += ["--config-name", train_config]
        cmd += extra_overrides

    print(
        f"[LAUNCH] gpus={cuda_devices} problem={exp.problem} model={exp.model} loss={exp.loss} "
        f"(per-job-gpus={n_gpus_per_job})"
    )
    print(f"         log={log_file}")
    print(f"         CMD: CUDA_VISIBLE_DEVICES={cuda_devices} " + " ".join(shlex.quote(c) for c in cmd))

    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_file, "x", encoding="utf-8") as f:  # never clobber an earlier job log
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        assert proc.stdout is not None
        await stream_to_file(proc.stdout, f)
        rc = await proc.wait()

    if rc == 0:
        print(f"[DONE] gpus={cuda_devices} problem={exp.problem} model={exp.model} loss={exp.loss}")
    else:
        print(f"[FAIL] gpus={cuda_devices} rc={rc} problem={exp.problem} model={exp.model} loss={exp.loss}")

    return rc


def resolve_gpu_pool(args: argparse.Namespace) -> List[str]:
    gpus_arg = args.gpus.strip()
    if args.n_gpus is not None:
        available = list_available_gpus()
        if available is None:
            raise RuntimeError("nvidia-smi not found or failed; cannot validate --n_gpus.")
        if len(available) < args.n_gpus:
            raise RuntimeError(
                f"Requested n_gpus={args.n_gpus}, but only {len(available)} GPU(s) available."
            )
        if gpus_arg.lower() == "auto":
            gpus = available
        else:
            gpus = parse_gpus(gpus_arg)
        if len(gpus) < args.n_gpus:
            raise ValueError(
                f"--n_gpus={args.n_gpus} requires at least {args.n_gpus} GPU ids; got {len(gpus)} ({gpus_arg})."
            )
        return gpus
    return parse_gpus(gpus_arg)


async def acquire_gpus(queue: asyncio.Queue[str], count: int) -> List[str]:
    gpus: List[str] = []
    for _ in range(count):
        gpus.append(await queue.get())
    return gpus


def release_gpus(queue: asyncio.Queue[str], gpus: Iterable[str]) -> None:
    for g in gpus:
        queue.put_nowait(g)


async def scheduler_main(args: argparse.Namespace) -> int:
    gpus = resolve_gpu_pool(args)
    experiments = [parse_experiment(s, args.default_problem) for s in (args.experiments or DEFAULT_EXPERIMENTS)]
    extra = split_extra(args.extra)

    run_one = Path(args.run_one).resolve()
    if not run_one.exists():
        raise FileNotFoundError(f"--run_one not found: {run_one}")

    log_dir = Path(args.log_dir)

    print("=== Parallel TRAIN runs (GPU pool) ===")
    print(f"RUN_ONE={run_one}")
    gpus_per_job = int(args.n_gpus) if args.n_gpus is not None else 1
    print(f"GPUS={','.join(gpus)} (pool size={len(gpus)}, per-job={gpus_per_job})")
    print(
        f"n_nodes={args.n_nodes}, n_customers={args.n_customers}, "
        f"n_scenarios={args.n_scenarios}, n_instances={args.n_instances}, batch={args.batch}"
    )
    print(f"extra={args.extra}")
    print(f"log_dir={log_dir}")
    print(f"runs={len(experiments)}")
    print("=====================================")

    gpu_queue: asyncio.Queue[str] = asyncio.Queue()
    for g in gpus:
        gpu_queue.put_nowait(g)

    # Track return codes
    results: List[Optional[int]] = [None] * len(experiments)

    stop_event = asyncio.Event()

    def _request_stop() -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            # Windows / limited env
            pass

    async def run_exp(i: int, exp: Experiment) -> None:
        gpu_ids = await acquire_gpus(gpu_queue, gpus_per_job)
        try:
            if stop_event.is_set():
                results[i] = 130
                return
            rc = await run_one_job(
                gpu_ids=gpu_ids,
                n_gpus_per_job=gpus_per_job,
                exp_index=i,
                exp=exp,
                run_one=run_one,
                n_nodes=args.n_nodes,
                n_customers=args.n_customers,
                n_scenarios=args.n_scenarios,
                n_instances=args.n_instances,
                batch=args.batch,
                dist_name=args.dist_name,
                train_config=args.train_config,
                log_name_format=args.log_name_format,
                run_id=args.run_id,
                device_in_container=args.device_in_container,
                extra_overrides=extra,
                log_dir=log_dir,
            )
            results[i] = rc
        finally:
            release_gpus(gpu_queue, gpu_ids)

    tasks = [asyncio.create_task(run_exp(i, exp)) for i, exp in enumerate(experiments)]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        for t in tasks:
            t.cancel()
        raise

    failed = any((rc is None or rc != 0) for rc in results)
    if failed:
        print(f"Some runs failed. Check logs in: {log_dir}", file=sys.stderr)
        return 1

    print(f"All runs completed. Logs in: {log_dir}")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parallel GPU-pool runner for scripts/03_train.sh")
    p.add_argument(
        "--gpus",
        default="0,2,3",
        help='Comma-separated physical GPU ids, e.g. "0,2,3" (or "auto" with --n_gpus)',
    )
    p.add_argument(
        "--n_gpus",
        type=int,
        default=None,
        help="Number of GPUs per job (requires availability via nvidia-smi).",
    )
    p.add_argument("--n_nodes", type=int, default=20)
    p.add_argument("--n_customers", type=int, default=None)
    p.add_argument("--n_scenarios", type=int, default=50)
    p.add_argument("--n_instances", type=int, default=2500)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--extra", default="", help='Extra Hydra overrides as a single string, e.g. "train.epochs=50 seed=42"')
    p.add_argument("--log_dir", default="logs/train_parallel", help="Directory for per-run logs")
    p.add_argument("--run_one", default="scripts/03_train.sh", help="Path to 03_train.sh")
    p.add_argument("--device_in_container", default="cuda:0", help="Device string passed to 03_train.sh (use cuda:0)")
    p.add_argument("--dist_name", default="uniform", help="Scenario name passed to 03_train.sh")
    p.add_argument("--train_config", default="train_normalize", help="Hydra train config name for 03_train.py")
    p.add_argument("--log_name_format", default="", help="Optional log filename format with {i},{problem},{model},{loss},{run_id}")
    p.add_argument("--run_id", default="", help="Run id used in log_name_format")
    p.add_argument("--list_default_exps", action="store_true", help="Print default experiments and exit")
    p.add_argument("--default_problem", default="vertex_cover")

    # Optional: override experiments from CLI (repeatable).
    # Example: --exp "vertex_cover|det_source_sce_mean|score_weighted_loss_flat"
    p.add_argument(
        "--exp",
        dest="experiments",
        action="append",
        help="Add an experiment spec. Repeatable. Format: PROBLEM|MODEL|LOSS (or MODEL|LOSS). "
             "If omitted, uses the built-in DEFAULT_EXPERIMENTS list.",
    )
    return p


def main() -> int:
    args = build_argparser().parse_args()
    if args.n_gpus is None:
        env_n_gpus = os.environ.get("N_GPUS")
        if env_n_gpus:
            try:
                args.n_gpus = int(env_n_gpus)
            except ValueError as exc:
                raise RuntimeError(f"Invalid N_GPUS={env_n_gpus!r}; expected integer.") from exc
    if args.list_default_exps:
        for spec in DEFAULT_EXPERIMENTS:
            print(spec)
        return 0
    return asyncio.run(scheduler_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
