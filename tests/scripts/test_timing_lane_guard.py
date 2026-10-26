"""Regressions for the structural, CPU-measured timing-lane guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.utils.timing_lane_guard import find_active_timing_processes, main, resolve_entrypoint


def _stat(pid: int, *, parent: int = 1, user_ticks: int = 0, system_ticks: int = 0) -> str:
    # procfs fields through stime (15); the parser deliberately handles spaces in comm.
    return (
        f"{pid} (test worker) R {parent} 0 0 0 0 0 0 0 0 0 "
        f"{user_ticks} {system_ticks} 0 0 0 0 0\n"
    )


def _process(proc_root: Path, pid: int, cwd: Path, argv: list[str], *, ticks: int) -> Path:
    process_dir = proc_root / str(pid)
    process_dir.mkdir(parents=True)
    (process_dir / "cmdline").write_bytes(b"\0".join(token.encode() for token in argv) + b"\0")
    (process_dir / "stat").write_text(_stat(pid, user_ticks=ticks), encoding="utf-8")
    (process_dir / "cwd").symlink_to(cwd, target_is_directory=True)
    return process_dir


def test_solver_name_in_an_inert_argument_is_not_an_entrypoint(tmp_path: Path) -> None:
    """Motivation: a solver name in an argument (`--solver gurobipy`) is not an entry point."""
    argv = ["bash", "scripts/run_eval.sh", "--solver", "gurobipy"]
    assert resolve_entrypoint(argv, tmp_path) == (tmp_path / "scripts/run_eval.sh").resolve()
    assert resolve_entrypoint(argv, tmp_path) != (tmp_path / "scripts/formal_eval.py").resolve()


def test_same_basename_from_another_project_is_not_the_pinned_path(tmp_path: Path) -> None:
    """Motivation: identity is the resolved script path, not a familiar filename."""
    ours = tmp_path / "ours/scripts/formal_eval.py"
    theirs = tmp_path / "theirs/scripts/formal_eval.py"
    argv = ["python3", "scripts/formal_eval.py"]
    assert resolve_entrypoint(argv, theirs.parents[1]) == theirs.resolve()
    assert resolve_entrypoint(argv, theirs.parents[1]) != ours.resolve()


def test_path_match_without_measured_cpu_does_not_block(tmp_path: Path) -> None:
    """Motivation: an idle orchestration shell must not hold the lane for hours."""
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "scripts/formal_eval.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch()
    process_dir = _process(proc_root, 101, repo, ["python3", "scripts/formal_eval.py"], ticks=12)

    def remain_idle(_: float) -> None:
        (process_dir / "stat").write_text(_stat(101, user_ticks=12), encoding="utf-8")

    assert find_active_timing_processes(
        proc_root=proc_root,
        timing_entrypoints=[target],
        sample_seconds=0,
        sleep_fn=remain_idle,
    ) == []


def test_exact_path_with_measured_cpu_blocks(tmp_path: Path) -> None:
    """Motivation: the positive control proves the guard can still fail."""
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "scripts/formal_eval.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch()
    process_dir = _process(proc_root, 202, repo, ["python3", "scripts/formal_eval.py"], ticks=12)

    def consume_cpu(_: float) -> None:
        (process_dir / "stat").write_text(_stat(202, user_ticks=19), encoding="utf-8")

    active = find_active_timing_processes(
        proc_root=proc_root,
        timing_entrypoints=[target],
        sample_seconds=0,
        sleep_fn=consume_cpu,
    )
    assert [(item.pid, item.entrypoint, item.cpu_ticks_delta) for item in active] == [
        (202, target.resolve(), 7)
    ]


def test_cpu_active_non_target_is_not_a_competitor(tmp_path: Path) -> None:
    """Motivation: overlap is permitted among jobs that publish no timing values."""
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "scripts/formal_eval.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch()
    process_dir = _process(proc_root, 303, repo, ["python3", "scripts/non_timing_train.py"], ticks=4)

    def consume_cpu(_: float) -> None:
        (process_dir / "stat").write_text(_stat(303, user_ticks=40), encoding="utf-8")

    assert find_active_timing_processes(
        proc_root=proc_root,
        timing_entrypoints=[target],
        sample_seconds=0,
        sleep_fn=consume_cpu,
    ) == []


def test_missing_entrypoint_raises_not_free(tmp_path: Path) -> None:
    """Motivation: a moved/renamed entry point must not silently report
    "timing lane free"; the guard must refuse to answer for a lane it cannot name.
    (M) delete the `if missing: raise` block in find_active_timing_processes."""
    gone = tmp_path / "gone.py"
    with pytest.raises(FileNotFoundError, match="gone.py"):
        find_active_timing_processes(
            proc_root=tmp_path,
            timing_entrypoints=[gone],
            sample_seconds=0,
            sleep_fn=lambda s: None,
        )


def test_main_exits_2_when_an_entrypoint_is_missing(tmp_path: Path, capsys) -> None:
    """Motivation: main() must map a missing timing entry point to exit code 2
    (the existing OSError->2 mapping) instead of exit 0 "timing lane free"; the default
    11 entry points exist in this checkout, so only the extra --entrypoint is missing.
    (M) delete the `if missing: raise` block in find_active_timing_processes."""
    gone = tmp_path / "gone.py"
    rc = main(["--entrypoint", str(gone), "--sample-seconds", "0"])
    assert rc == 2
    captured = capsys.readouterr()
    assert "gone.py" in captured.err
