import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest


def _load_pool_module():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    module_path = root / "scripts" / "03_train_parallel_gpu_pool.py"
    spec = importlib.util.spec_from_file_location("train_pool_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_nvidia_smi_list_extracts_gpu_ids():
    # Motivation: ensure GPU id parsing is stable for availability checks.
    module = _load_pool_module()
    output = "GPU 0: Tesla V100\nGPU 1: Tesla V100\n"
    assert module.parse_nvidia_smi_list(output) == ["0", "1"]


def test_resolve_gpu_pool_auto_uses_all_available():
    # Motivation: auto mode should select all available GPUs for the pool.
    module = _load_pool_module()
    module.list_available_gpus = lambda: ["0", "1", "2"]
    args = module.argparse.Namespace(gpus="auto", n_gpus=2)
    assert module.resolve_gpu_pool(args) == ["0", "1", "2"]


def test_resolve_gpu_pool_rejects_pool_smaller_than_ngpus():
    # Motivation: prevent starting a job when the pool is smaller than per-job GPU count.
    module = _load_pool_module()
    module.list_available_gpus = lambda: ["0", "1", "2"]
    args = module.argparse.Namespace(gpus="0", n_gpus=2)
    with pytest.raises(ValueError):
        module.resolve_gpu_pool(args)


def test_resolve_gpu_pool_rejects_insufficient_available_gpus():
    # Motivation: fail fast when requested GPUs exceed availability.
    module = _load_pool_module()
    module.list_available_gpus = lambda: ["0"]
    args = module.argparse.Namespace(gpus="auto", n_gpus=2)
    with pytest.raises(RuntimeError):
        module.resolve_gpu_pool(args)


def test_resolve_gpu_pool_no_ngpus_returns_requested_list():
    # Motivation: keep existing behavior when per-job GPU count is not specified.
    module = _load_pool_module()
    args = module.argparse.Namespace(gpus="0,2", n_gpus=None)
    assert module.resolve_gpu_pool(args) == ["0", "2"]


def test_run_one_job_sets_ddp_env_for_multi_gpu(tmp_path, monkeypatch):
    # Motivation: ensure multi-GPU jobs enable DDP and set CUDA_VISIBLE_DEVICES correctly.
    module = _load_pool_module()
    captured = {}

    async def _noop_stream(*_args, **_kwargs):
        return None

    class DummyProc:
        def __init__(self):
            self.stdout = asyncio.StreamReader()

        async def wait(self):
            return 0

    async def _fake_exec(*_cmd, **kwargs):
        captured["env"] = kwargs.get("env", {})
        captured["cmd"] = _cmd
        return DummyProc()

    monkeypatch.setattr(module, "stream_to_file", _noop_stream)
    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", _fake_exec)

    rc = asyncio.run(
        module.run_one_job(
            gpu_ids=["0", "1"],
            n_gpus_per_job=2,
            exp_index=3,
            exp=module.Experiment(problem="vertex_cover", model="m", loss="l"),
            run_one=Path("scripts/03_train.sh"),
            n_nodes=20,
            n_customers=None,
            n_scenarios=50,
            n_instances=2500,
            batch=32,
            dist_name="uniform",
            train_config="train_normalize",
            log_name_format="",
            run_id="",
            device_in_container="cuda:0",
            extra_overrides=[],
            log_dir=tmp_path,
        )
    )

    assert rc == 0
    env = captured["env"]
    assert env.get("CUDA_VISIBLE_DEVICES") == "0,1"
    assert env.get("DDP") == "1"
    assert env.get("NPROC_PER_NODE") == "2"
    assert "MASTER_PORT" in env


def test_scheduler_acquires_two_gpus_per_job(tmp_path, monkeypatch):
    # Motivation: ensure scheduler reserves N_GPUS tokens per job before launch.
    module = _load_pool_module()
    captured = {}

    async def _fake_run_one_job(**kwargs):
        captured["gpu_ids"] = kwargs.get("gpu_ids", [])
        captured["n_gpus_per_job"] = kwargs.get("n_gpus_per_job")
        return 0

    monkeypatch.setattr(module, "run_one_job", _fake_run_one_job)
    monkeypatch.setattr(module, "list_available_gpus", lambda: ["0", "1"])

    args = module.argparse.Namespace(
        gpus="auto",
        n_gpus=2,
        experiments=["vertex_cover|m|l"],
        default_problem="vertex_cover",
        extra="",
        run_one="scripts/03_train.sh",
        n_nodes=20,
        n_customers=None,
        n_scenarios=50,
        n_instances=2500,
        batch=32,
        dist_name="uniform",
        train_config="train_normalize",
        log_name_format="",
        run_id="",
        device_in_container="cuda:0",
        log_dir=str(tmp_path),
    )

    rc = asyncio.run(module.scheduler_main(args))
    assert rc == 0
    assert captured["n_gpus_per_job"] == 2
    assert captured["gpu_ids"] == ["0", "1"]
