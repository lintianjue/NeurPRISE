import logging

import torch

from src.utils import ddp_trace as ddp_trace_module


def test_ddp_trace_wraps_and_restores_barrier(monkeypatch, caplog):
    """Motivation: enabling DDP trace should log and still call the original op."""
    calls = {"n": 0}

    def fake_barrier():
        calls["n"] += 1
        return None

    monkeypatch.setattr(ddp_trace_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(ddp_trace_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(ddp_trace_module.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(ddp_trace_module.dist, "barrier", fake_barrier)

    caplog.set_level(logging.INFO)
    restore = ddp_trace_module.enable_ddp_trace(logging.getLogger("ddp-test"))

    ddp_trace_module.dist.barrier()

    assert calls["n"] == 1
    assert any("DDP trace: rank=1 enter barrier" in rec.message for rec in caplog.records)
    assert any("DDP trace: rank=1 exit barrier" in rec.message for rec in caplog.records)

    restore()
    ddp_trace_module.dist.barrier()
    assert calls["n"] == 2
