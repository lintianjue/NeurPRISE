"""Prove that OpenBLAS (not Gurobi) causes the 64-thread CPU burst.

Hypothesis: When OPENBLAS_NUM_THREADS is unset, numpy matrix ops spawn
up to MAX_THREADS (= nproc = 64) threads. Gurobi's Threads parameter
correctly limits Gurobi's own threads, but BLAS threads add on top.

Test 1: numpy matmul without OPENBLAS_NUM_THREADS → ~nproc threads
Test 2: numpy matmul with OPENBLAS_NUM_THREADS=1 → ~1 BLAS thread
Test 3: Gurobi Threads=N correctly limits Gurobi model threads to N
Test 4: Gurobi solve + unrestricted OpenBLAS → threads > Gurobi Threads
Test 5: Gurobi solve + OPENBLAS_NUM_THREADS=1 → threads ≈ Gurobi Threads
"""

import os
import subprocess
import sys
import textwrap
import time

import pytest

NPROC = os.cpu_count() or 1


def _count_threads_in_subprocess(code: str, env_overrides: dict | None = None) -> int:
    """Run *code* in a subprocess and return peak thread count it reports."""
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    # Remove vars that might interfere
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        if key not in (env_overrides or {}):
            env.pop(key, None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=120, env=env,
    )
    if result.returncode != 0:
        pytest.fail(f"Subprocess failed:\n{result.stderr}")
    return int(result.stdout.strip().split("\n")[-1])


# ── Test 1 & 2: OpenBLAS thread scaling ──────────────────────────────

NUMPY_THREAD_CODE = textwrap.dedent("""\
    import os, time, threading, numpy as np

    # Trigger OpenBLAS thread pool with a large matmul
    a = np.random.randn(2000, 2000)
    b = np.random.randn(2000, 2000)
    _ = a @ b

    # Small sleep to let thread pool stabilise
    time.sleep(0.1)
    n_threads = len(os.listdir(f'/proc/{os.getpid()}/task'))
    print(n_threads)
""")


def test_openblas_default_spawns_many_threads():
    """Without OPENBLAS_NUM_THREADS, numpy matmul spawns ~nproc threads."""
    n_threads = _count_threads_in_subprocess(NUMPY_THREAD_CODE)
    # Expect significantly more than 1 (typically nproc or close)
    assert n_threads > 8, (
        f"Expected many BLAS threads without OPENBLAS_NUM_THREADS, got {n_threads}"
    )
    print(f"  OpenBLAS default: {n_threads} threads (nproc={NPROC})")


def test_openblas_limited_to_1():
    """With OPENBLAS_NUM_THREADS=1, numpy matmul uses ~1 BLAS thread."""
    n_threads = _count_threads_in_subprocess(
        NUMPY_THREAD_CODE, env_overrides={"OPENBLAS_NUM_THREADS": "1"}
    )
    # Should be much fewer — just Python main + 1 BLAS thread
    assert n_threads < 8, (
        f"Expected few threads with OPENBLAS_NUM_THREADS=1, got {n_threads}"
    )
    print(f"  OpenBLAS limited: {n_threads} threads")


# ── Test 3: Gurobi Threads param works ───────────────────────────────

GUROBI_PARAM_CODE = textwrap.dedent("""\
    import gurobipy as gp
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.setParam('Threads', {threads})
    env.start()
    m = gp.Model(env=env)
    print(m.Params.Threads)
    m.dispose()
    env.dispose()
""")


def test_gurobi_threads_param_respected():
    """Gurobi model.Params.Threads matches what we set on the env."""
    for target in [1, 4, 40]:
        code = GUROBI_PARAM_CODE.format(threads=target)
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=30,
        )
        # Filter out Gurobi license messages, get last line
        lines = [l for l in result.stdout.strip().split("\n") if l.strip().isdigit()]
        actual = int(lines[-1])
        assert actual == target, f"Gurobi Threads: set {target}, got {actual}"
    print("  Gurobi Threads param correctly respected for 1, 4, 40")


# ── Test 4 & 5: Gurobi + OpenBLAS combined ──────────────────────────

# Solve a moderate CFLP to trigger real Gurobi threading + BLAS
GUROBI_SOLVE_THREAD_CODE = textwrap.dedent("""\
    import os, time, threading
    import numpy as np
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_THREADS = 8  # small enough to distinguish from BLAS burst

    # Build a moderate-size LP/MIP that takes a few seconds
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.setParam('Threads', GUROBI_THREADS)
    env.start()
    m = gp.Model(env=env)

    N = 200  # variables
    M = 400  # constraints
    rng = np.random.default_rng(42)

    x = m.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="x")
    # Random objective
    m.setObjective(gp.quicksum(rng.standard_normal() * x[j] for j in range(N)), GRB.MINIMIZE)
    # Random constraints (dense)
    A = rng.standard_normal((M, N))
    b = rng.standard_normal(M) * 10
    for i in range(M):
        m.addConstr(gp.quicksum(A[i, j] * x[j] for j in range(N)) >= b[i])
    m.update()

    # Also do a big numpy matmul to trigger BLAS pool
    big = rng.standard_normal((2000, 2000))
    _ = big @ big

    m.optimize()
    time.sleep(0.2)

    n_threads = len(os.listdir(f'/proc/{{pid}}/task'.format(pid=os.getpid())))
    print(n_threads)
    m.dispose()
    env.dispose()
""")


def test_gurobi_plus_openblas_unrestricted():
    """Gurobi(Threads=8) + unrestricted OpenBLAS → more than 8 threads."""
    n_threads = _count_threads_in_subprocess(GUROBI_SOLVE_THREAD_CODE)
    # Should be well above Gurobi's 8 threads because OpenBLAS adds ~nproc
    assert n_threads > 16, (
        f"Expected Gurobi(8) + OpenBLAS to exceed 16 threads, got {n_threads}"
    )
    print(f"  Gurobi(8) + OpenBLAS default: {n_threads} threads")


def test_gurobi_plus_openblas_restricted():
    """Gurobi(Threads=8) + OPENBLAS_NUM_THREADS=1 → close to 8 threads."""
    n_threads = _count_threads_in_subprocess(
        GUROBI_SOLVE_THREAD_CODE,
        env_overrides={"OPENBLAS_NUM_THREADS": "1"},
    )
    # Should be close to Gurobi's 8 (plus a few for Python/GC/etc)
    assert n_threads < 20, (
        f"Expected ~8-15 threads with OPENBLAS restricted, got {n_threads}"
    )
    print(f"  Gurobi(8) + OpenBLAS(1): {n_threads} threads")
