"""Exclusive-create run-directory I/O guard.

Models and results use fresh run directories. Generated datasets use the default
data folder so subsequent stages can read them without path configuration.
Existing files are never overwritten.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Callable, Iterable, Mapping

REPO = Path(__file__).resolve().parents[2]
DEFAULT_BASE = REPO / "runs"

INPUT_ROOT_VAR = "NEURPRISE_INPUT_ROOT"
RUN_ROOT_VAR = "NEURPRISE_RUN_ROOT"

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class RunIOError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunPaths:
    input_roots: tuple[Path, ...]
    run_dir: Path


def _realpath(p: str | Path) -> Path:
    return Path(os.path.realpath(str(p)))


def _is_equal_or_inside(target: Path, root: Path) -> bool:
    target = _realpath(target)
    root = _realpath(root)
    if target == root:
        return True
    return os.path.commonpath([str(target), str(root)]) == str(root)


def guard_env(repo: Path = REPO, env: Mapping[str, str] | None = None) -> None:
    if env is None:
        env = os.environ
    repo_real = _realpath(repo)
    for var in ("PROJECT_DIR",):
        val = env.get(var)
        if val:
            val_real = _realpath(val)
            if not _is_equal_or_inside(val_real, repo_real):
                raise RunIOError(
                    f"{var} escapes repo: {val_real} is not inside {repo_real}"
                )
    import src

    src_real = _realpath(src.__file__)
    if not _is_equal_or_inside(src_real, repo_real):
        raise RunIOError(
            f"src is not inside repo: {src_real} is not inside {repo_real}"
        )


def _repository_run(target: Path, root: Path) -> bool:
    return (_realpath(root) == _realpath(REPO)
            and not (REPO / "runs").is_symlink()
            and _is_equal_or_inside(target, REPO / "runs"))


def data_paths(rp: RunPaths) -> RunPaths:
    """Use the shared data folder for default inputs; custom inputs keep isolated outputs."""
    if rp.input_roots == (_realpath(REPO),):
        root = REPO / "data"
        if root.is_symlink():
            raise RunIOError(f"symlink in data path: {root}")
        return RunPaths(input_roots=(), run_dir=root)
    return rp


def data_output_path(rp: RunPaths, problem: str, filename: str) -> Path:
    """Resolve a new dataset file and reject collisions before expensive generation."""
    _validate_name(problem, "problem")
    _validate_name(filename, "data filename")
    target_rp = data_paths(rp)
    relative = Path(problem) / filename
    if target_rp is rp:
        relative = Path("data") / relative
    target = output_path(target_rp, relative)
    if target.exists():
        raise FileExistsError(f"Data already exists: {target}. Reuse it or choose a different dataset configuration.")
    return target


def _validate_name(name: str, label: str) -> None:
    if not _NAME_RE.match(name) or ".." in name:
        raise RunIOError(f"invalid {label}: {name!r}")


def _loaded_versions() -> dict[str, str | None]:
    """Versions of the numeric libraries this process has ALREADY imported (never imports one)."""
    out: dict[str, str | None] = {"python": sys.version.split()[0]}
    for name in ("numpy", "torch", "gurobipy"):
        mod = sys.modules.get(name)
        if mod is None:
            continue
        v = getattr(mod, "__version__", None)
        if v is None and name == "gurobipy":
            try:
                v = ".".join(map(str, mod.gurobi.version()))
            except Exception:
                v = None
        out[name] = None if v is None else str(v)
    return out


def open_run(
    batch: str,
    run_id: str,
    input_roots: Iterable[str | Path],
    *,
    base: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> RunPaths:
    guard_env(env=env)

    _validate_name(batch, "batch")
    _validate_name(run_id, "run_id")

    if env is None:
        env = os.environ

    roots: list[Path] = []
    for root in input_roots:
        root_path = Path(root)
        if not root_path.exists():
            raise RunIOError(f"input root does not exist: {root_path}")
        roots.append(_realpath(root_path))
    roots_t = tuple(roots)

    if base is None:
        # A launcher that set NEURPRISE_RUN_ROOT expects every run under it,
        # not under the repository default (direct regret callers pass no base).
        run_root = env.get(RUN_ROOT_VAR)
        base = Path(run_root) if run_root else DEFAULT_BASE

    would_be_run_dir = _realpath(base / batch) / run_id
    for root in roots_t:
        if (_is_equal_or_inside(would_be_run_dir, root) or _is_equal_or_inside(
            root, would_be_run_dir
        )) and not _repository_run(would_be_run_dir, root):
            raise RunIOError(
                f"run dir overlaps input root: {would_be_run_dir} vs {root}"
            )

    os.makedirs(base / batch, exist_ok=True)
    run_dir = base / batch / run_id
    os.mkdir(run_dir)  # no exist_ok: FileExistsError must propagate
    run_dir = _realpath(run_dir)

    rp = RunPaths(input_roots=roots_t, run_dir=run_dir)

    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        commit = None

    record = {
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "commit": commit,
        "sys_path_head": sys.path[:4],
        "input_roots": [str(r) for r in roots_t],
        "run_dir": str(run_dir),
        "env": {
            k: env.get(k)
            for k in ("PROJECT_DIR", "PYTHONPATH", INPUT_ROOT_VAR, RUN_ROOT_VAR)
        },
        # What a reportable run must log: thread settings (e.g. OPENBLAS_NUM_THREADS=4).
        # Solver threads themselves are in argv; these are the process-level knobs.
        "thread_env": {
            k: env.get(k)
            for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")
        },
        "cpu_count": os.cpu_count(),
        "loadavg_at_open": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "versions": _loaded_versions(),
        "host": socket.gethostname(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with create_new(rp, "RUN.json", mode="w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)

    return rp


def open_hydra_run(cfg, entry_file: str, *, env: Mapping[str, str] | None = None) -> RunPaths:
    """Open a named run, using the repository inputs and runs/ unless overridden."""
    if env is None:
        env = os.environ
    input_root = env.get(INPUT_ROOT_VAR) or str(REPO)
    run_root = env.get(RUN_ROOT_VAR) or str(DEFAULT_BASE)

    batch = Path(entry_file).stem
    run_id = str(cfg.run_name)
    return open_run(batch, run_id, [input_root], base=Path(run_root), env=env)


def add_run_args(parser) -> None:
    """The two launcher arguments of an argparse entry point (see open_script_run)."""
    parser.add_argument("--input-root", default=os.environ.get(INPUT_ROOT_VAR) or str(REPO),
                        help=f"input tree (default: repository root; override with ${INPUT_ROOT_VAR})")
    parser.add_argument("--run-id", default=None,
                        help=f"run dir name under ${RUN_ROOT_VAR}/<entry>/ (default: UTC timestamp)")


def open_script_run(args, entry_file: str, extra_input_roots: Iterable[str | Path] = ()) -> RunPaths:
    """Open the run directory of an argparse entry point: batch = the entry's file stem,
    run_id = --run-id or a UTC timestamp, inputs = --input-root (+ extra_input_roots)."""
    if not args.input_root:
        raise RunIOError(f"--input-root or ${INPUT_ROOT_VAR} is required")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return open_run(Path(entry_file).stem, run_id, [args.input_root, *extra_input_roots])


def output_path(rp: RunPaths, path: str | Path) -> Path:
    path = Path(path)
    target = path if path.is_absolute() else rp.run_dir / path

    # (a) any existing path component from run_dir down to target, including
    # target itself, must not be a symlink. (A target lexically outside
    # run_dir is caught by check (b) below via realpath instead.)
    try:
        rel_parts = target.relative_to(rp.run_dir).parts
    except ValueError:
        rel_parts = ()
    cur = rp.run_dir
    for part in rel_parts:
        cur = cur / part
        if cur.is_symlink():
            raise RunIOError(f"symlink in path: {cur}")

    target_real = _realpath(target)
    run_dir_real = _realpath(rp.run_dir)
    if target_real == run_dir_real or not _is_equal_or_inside(
        target_real, run_dir_real
    ):
        raise RunIOError(f"path escapes run dir: {target} -> {target_real}")

    for root in rp.input_roots:
        if _is_equal_or_inside(target_real, root) and not _repository_run(target_real, root):
            raise RunIOError(f"path aliases input root: {target} -> {root}")

    return target


def input_file(rp: RunPaths, path: str | Path) -> Path:
    """realpath of an input file that must exist, be a regular file, and lie inside one of
    rp.input_roots; otherwise RunIOError. Symlinks are followed (realpath) before the check."""
    real = _realpath(path)
    if not real.is_file():
        raise RunIOError(f"not a regular file: {path} -> {real}")
    if not any(_is_equal_or_inside(real, root) for root in rp.input_roots):
        raise RunIOError(f"path is not inside an input root: {path} -> {real}")
    return real


def create_new(rp: RunPaths, path: str | Path, mode: str = "wb") -> IO:
    if mode not in ("wb", "w"):
        raise ValueError(f"unsupported mode: {mode!r}")
    p = output_path(rp, path)
    os.makedirs(p.parent, exist_ok=True)
    p = output_path(rp, path)  # re-check after mkdir (parent symlink race)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    if mode == "w":
        return os.fdopen(fd, mode, encoding="utf-8")
    return os.fdopen(fd, mode)


def publish_new(
    rp: RunPaths,
    path: str | Path,
    write_fn: Callable[[IO], None],
    mode: str = "wb",
) -> Path:
    final = output_path(rp, path)
    tmp = final.parent / f".{final.name}.partial.{os.getpid()}.{secrets.token_hex(4)}"
    tmp_fh = create_new(rp, tmp, mode=mode)
    try:
        write_fn(tmp_fh)
        tmp_fh.flush()
        os.fsync(tmp_fh.fileno())
        tmp_fh.close()
        try:
            os.link(tmp, final)
        except FileExistsError:
            raise
        except OSError as exc:
            raise RunIOError(f"hard link unsupported on this filesystem: {exc}")
    finally:
        if not tmp_fh.closed:
            tmp_fh.close()
        if tmp.exists():
            os.unlink(tmp)
    return final


def mutable_path(rp: RunPaths, path: str | Path) -> Path:
    return output_path(rp, path)

