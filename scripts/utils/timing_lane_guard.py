#!/usr/bin/env python3
"""Detect active, timing-reportable solver jobs from process structure and CPU use.

Searching flattened ``ps`` text for words such as ``gurobi`` is unreliable: an idle
command carrying ``--solver gurobipy`` becomes a blocker, while an unrelated CPU-active
solver whose name lacks that word is invisible.

This guard has two independent requirements:

* the process must execute one of the explicitly listed timing entry-point paths;
* its user/system CPU counter must advance during a short sample.

Exit status is 0 when the timing lane is free, 1 when an active timing process is
found, and 2 when the process inventory cannot be checked reliably.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Callable, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_SECONDS = 0.25
DEFAULT_TIMING_ENTRYPOINTS = (
    ROOT / "scripts/04_05_eval_ml_2ro_e2e.py",
    ROOT / "scripts/eval/time_budget_medium.py",
    ROOT / "scripts/eval/mipgap_regret_sweep.py",
    ROOT / "scripts/eval/eval_regret.py",
    ROOT / "scripts/eval/eval_regret_ablations.py",
)

COMMAND_WRAPPERS = frozenset({"env", "nohup", "setsid"})
SHELL_COMMANDS = frozenset({"bash", "dash", "sh", "zsh"})


@dataclasses.dataclass(frozen=True)
class ProcessSample:
    """The evidence used for one admission decision."""

    pid: int
    entrypoint: Path
    argv: tuple[str, ...]
    ticks_before: int
    ticks_after: int

    @property
    def cpu_ticks_delta(self) -> int:
        return self.ticks_after - self.ticks_before


def _unwrap_command(argv: Sequence[str]) -> list[str]:
    remaining = list(argv)
    while remaining and Path(remaining[0]).name in COMMAND_WRAPPERS:
        wrapper = Path(remaining.pop(0)).name
        while remaining and remaining[0].startswith("-"):
            remaining.pop(0)
        if wrapper == "env":
            while remaining and "=" in remaining[0] and not remaining[0].startswith("="):
                remaining.pop(0)
    return remaining


def _script_argument(argv: Sequence[str], *, shell: bool) -> str | None:
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == "--":
            index += 1
            break
        if shell and token.startswith("-") and "c" in token[1:]:
            return None
        if not shell and token in {"-c", "-m"}:
            return None
        if token.startswith("-"):
            index += 1
            continue
        break
    return argv[index] if index < len(argv) else None


def resolve_entrypoint(argv: Sequence[str], cwd: Path) -> Path | None:
    """Resolve only the executable/script command position, never arbitrary text."""
    command = _unwrap_command(argv)
    if not command:
        return None
    executable = Path(command[0])
    name = executable.name
    if name.startswith("python"):
        raw_path = _script_argument(command, shell=False)
    elif name in SHELL_COMMANDS:
        raw_path = _script_argument(command, shell=True)
    else:
        raw_path = command[0]
    if raw_path is None:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = cwd / path
    return path.resolve(strict=False)


def _read_argv(process_dir: Path) -> tuple[str, ...]:
    raw = (process_dir / "cmdline").read_bytes()
    return tuple(part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part)


def _read_cpu_ticks(process_dir: Path) -> int:
    # /proc/PID/stat's comm field is parenthesized and may itself contain spaces.
    fields = (process_dir / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
    return int(fields[11]) + int(fields[12])  # fields 14 and 15 in procfs numbering


def _candidate_processes(
    proc_root: Path,
    timing_entrypoints: set[Path],
    ignored_pids: set[int],
) -> dict[int, tuple[Path, tuple[str, ...], int]]:
    candidates: dict[int, tuple[Path, tuple[str, ...], int]] = {}
    for process_dir in proc_root.iterdir():
        if not process_dir.name.isdigit():
            continue
        pid = int(process_dir.name)
        if pid in ignored_pids:
            continue
        try:
            argv = _read_argv(process_dir)
            cwd = (process_dir / "cwd").resolve(strict=True)
            entrypoint = resolve_entrypoint(argv, cwd)
            if entrypoint in timing_entrypoints:
                candidates[pid] = (entrypoint, argv, _read_cpu_ticks(process_dir))
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            # Processes may exit between any two procfs reads. An unreadable unrelated
            # process is not evidence that a named timing entry point is active.
            continue
    return candidates


def find_active_timing_processes(
    *,
    proc_root: Path = Path("/proc"),
    timing_entrypoints: Iterable[Path] = DEFAULT_TIMING_ENTRYPOINTS,
    ignored_pids: Iterable[int] = (),
    sample_seconds: float = DEFAULT_SAMPLE_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[ProcessSample]:
    """Return path-matched processes whose CPU counters advance during the sample."""
    if sample_seconds < 0:
        raise ValueError("sample_seconds must be non-negative")
    targets = {Path(path).resolve(strict=False) for path in timing_entrypoints}
    missing = sorted(str(p) for p in targets if not p.is_file())
    if missing:
        raise FileNotFoundError(
            "timing entry point(s) not found; the guard cannot see a lane it cannot name: "
            + ", ".join(missing))
    before = _candidate_processes(proc_root, targets, set(ignored_pids))
    sleep_fn(sample_seconds)
    active: list[ProcessSample] = []
    for pid, (entrypoint, argv, ticks_before) in before.items():
        try:
            ticks_after = _read_cpu_ticks(proc_root / str(pid))
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
        sample = ProcessSample(pid, entrypoint, argv, ticks_before, ticks_after)
        if sample.cpu_ticks_delta > 0:
            active.append(sample)
    return sorted(active, key=lambda sample: sample.pid)


def _ancestor_pids(pid: int, proc_root: Path) -> set[int]:
    ancestors = {pid}
    current = pid
    while current > 1:
        try:
            fields = (proc_root / str(current) / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
            current = int(fields[1])
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            break
        if current in ancestors:
            break
        ancestors.add(current)
    return ancestors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=DEFAULT_SAMPLE_SECONDS,
        help=f"CPU-counter sampling interval (default: {DEFAULT_SAMPLE_SECONDS}s)",
    )
    parser.add_argument(
        "--entrypoint",
        action="append",
        type=Path,
        default=[],
        help="additional exact timing entry-point path (repeatable)",
    )
    args = parser.parse_args(argv)
    targets = (*DEFAULT_TIMING_ENTRYPOINTS, *args.entrypoint)
    try:
        active = find_active_timing_processes(
            timing_entrypoints=targets,
            ignored_pids=_ancestor_pids(os.getpid(), Path("/proc")),
            sample_seconds=args.sample_seconds,
        )
    except (OSError, ValueError) as exc:
        print(f"timing-lane guard failed: {exc}", file=sys.stderr)
        return 2
    if not active:
        print(f"timing lane free: 0 active processes across {len(targets)} exact entry points")
        return 0
    for sample in active:
        print(
            f"active timing process: pid={sample.pid} cpu_ticks_delta={sample.cpu_ticks_delta} "
            f"entrypoint={sample.entrypoint} argv={shlex.join(sample.argv)}",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
