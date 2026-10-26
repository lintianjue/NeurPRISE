"""scripts/03_train.py opens its run through run_io (label_path, init-checkpoint record,
best.pt inside the run dir, DDP rank-0-opens/others-receive).

03_train.py is loaded by file path; sys.argv is never touched. env dicts are built
explicitly and passed to the functions under test -- os.environ is never mutated except
through monkeypatch (none of these tests need it).
"""
import ast
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.utils.run_io import INPUT_ROOT_VAR, RUN_ROOT_VAR, RunIOError, RunPaths, open_run

REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO / "scripts" / "03_train.py"


def _load_train_module():
    spec = importlib.util.spec_from_file_location("train_script_run_io", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _env(tmp_path):
    in_root = tmp_path / "in"
    in_root.mkdir()
    run_root = tmp_path / "root"
    return {INPUT_ROOT_VAR: str(in_root), RUN_ROOT_VAR: str(run_root)}, in_root, run_root


# ---- _open_training_run ------------------------------------------------------------

def test_single_process_opens_run_under_entry_stem(tmp_path):
    """(M) pass "train" as batch in _open_training_run's open_hydra_run call."""
    module = _load_train_module()
    env, _, run_root = _env(tmp_path)
    cfg = OmegaConf.create({"run_name": "vc_x", "train": {}})

    rp = module._open_training_run(cfg, ddp_enabled=False, rank=0, env=env)

    import os
    assert rp.run_dir == Path(os.path.realpath(run_root / "03_train" / "vc_x"))
    assert (rp.run_dir / "RUN.json").is_file()


class _FakeDistReturns:
    """rank-1-side fake: overwrites lst[0] with a payload built from a run the test opened
    itself as "rank 0"."""

    def __init__(self, payload_value):
        self.payload_value = payload_value
        self.calls = []

    def broadcast_object_list(self, lst, src=0):
        self.calls.append((list(lst), src))
        lst[0] = self.payload_value


def test_ddp_non_zero_rank_receives_the_run_dir_and_creates_nothing(tmp_path):
    """(M) make rank 1 also call open_hydra_run (rank!=0 branch calls it too)."""
    module = _load_train_module()
    env, in_root, run_root = _env(tmp_path)
    rp0 = open_run("03_train", "vc_x", [in_root], base=run_root, env=env)
    payload_value = (tuple(str(r) for r in rp0.input_roots), str(rp0.run_dir))
    fake_dist = _FakeDistReturns(payload_value)

    cfg = OmegaConf.create({"run_name": "vc_x", "train": {}})
    rp1 = module._open_training_run(
        cfg, ddp_enabled=True, rank=1, dist_mod=fake_dist, env=env
    )

    assert rp1.input_roots == rp0.input_roots
    assert rp1.run_dir == rp0.run_dir
    assert len(list((run_root / "03_train").iterdir())) == 1


class _FakeDistRecords:
    """rank-0-side fake: records exactly what was broadcast, without mutating it."""

    def __init__(self):
        self.calls = []

    def broadcast_object_list(self, lst, src=0):
        self.calls.append((list(lst), src))


def test_ddp_rank_zero_broadcasts_what_it_opened(tmp_path):
    """(M) broadcast before opening (broadcast the placeholder [None] instead of the
    real payload)."""
    module = _load_train_module()
    env, _, run_root = _env(tmp_path)
    fake_dist = _FakeDistRecords()
    cfg = OmegaConf.create({"run_name": "vc_x", "train": {}})

    rp = module._open_training_run(
        cfg, ddp_enabled=True, rank=0, dist_mod=fake_dist, env=env
    )

    assert len(fake_dist.calls) == 1
    broadcast_payload, src = fake_dist.calls[0]
    assert src == 0
    assert broadcast_payload[0][1] == str(rp.run_dir)


# ---- _model_dir ----------------------------------------------------------------------

def test_model_dir_default_and_bounds(tmp_path):
    """(M) drop the mutable_path call from _model_dir (use run_dir/value directly)."""
    module = _load_train_module()
    in_root = tmp_path / "in"
    in_root.mkdir()
    rp = open_run("03_train", "vc_x", [in_root], base=tmp_path / "root", env={})

    cfg_default = OmegaConf.create({"train": {"model_output_dir": ""}})
    result = module._model_dir(cfg_default, rp)
    assert result == rp.run_dir / "model"
    assert cfg_default.train.model_output_dir == str(result)

    cfg_inside = OmegaConf.create({"train": {"model_output_dir": str(rp.run_dir / "m2")}})
    result2 = module._model_dir(cfg_inside, rp)
    assert result2 == rp.run_dir / "m2"

    cfg_outside = OmegaConf.create({"train": {"model_output_dir": str(tmp_path / "elsewhere")}})
    with pytest.raises(RunIOError):
        module._model_dir(cfg_outside, rp)


# ---- _init_checkpoint_record -----------------------------------------------------------

def test_init_checkpoint_record(tmp_path):
    """(M) return the sha256 of an empty string regardless of the checkpoint's real bytes."""
    module = _load_train_module()
    in_root = tmp_path / "in"
    in_root.mkdir()
    rp = open_run("03_train", "vc_x", [in_root], base=tmp_path / "root", env={})

    assert module._init_checkpoint_record(rp, None) is None
    assert module._init_checkpoint_record(rp, "") is None

    ckpt = in_root / "src.pt"
    data = b"not a real checkpoint, but real bytes"
    ckpt.write_bytes(data)
    record = module._init_checkpoint_record(rp, str(ckpt))
    assert record["sha256"] == hashlib.sha256(data).hexdigest()
    assert Path(record["realpath"]).is_relative_to(Path(str(in_root)).resolve())

    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"x")
    with pytest.raises(RunIOError):
        module._init_checkpoint_record(rp, str(outside))


# ---- sidecar (structural) --------------------------------------------------------------

def test_sidecar_records_datamodule_file_and_init_checkpoint():
    """(M) restore os.path.join(cfg.problem.data.path, cfg.problem.name, file_name) as
    label_path."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    emit_calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "emit_loudly"
    ]
    assert emit_calls, "emit_loudly is no longer called"
    call = emit_calls[0]

    label_path_kw = next(kw for kw in call.keywords if kw.arg == "label_path")
    assert isinstance(label_path_kw.value, ast.Attribute)
    assert label_path_kw.value.attr == "data_file"
    assert isinstance(label_path_kw.value.value, ast.Name)
    assert label_path_kw.value.value.id == "dm"

    extra_kw = next(kw for kw in call.keywords if kw.arg == "extra")
    assert isinstance(extra_kw.value, ast.Dict)
    keys = [k.value for k in extra_kw.value.keys if isinstance(k, ast.Constant)]
    assert "init_from_checkpoint" in keys
