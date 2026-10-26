from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

import torch
import torch.distributed as dist


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        try:
            return int(dist.get_rank())
        except Exception:
            return -1
    return -1


def _tensor_summary(arg: Any) -> str:
    if torch.is_tensor(arg):
        return f"tensor(shape={tuple(arg.shape)}, dtype={arg.dtype}, device={arg.device})"
    return ""


def enable_ddp_trace(logger: logging.Logger, *, rank0_only: bool = False) -> Callable[[], None]:
    """
    Monkeypatch common torch.distributed collectives to log entry/exit.
    Returns a restore function to undo the patches.
    """
    if not dist.is_available():
        return lambda: None

    originals: dict[str, Callable[..., Any]] = {}
    wrapped: dict[str, Callable[..., Any]] = {}

    def should_log() -> bool:
        return (not rank0_only) or _rank() == 0

    def wrap(name: str) -> None:
        if not hasattr(dist, name):
            return
        orig = getattr(dist, name)
        if not callable(orig):
            return
        originals[name] = orig

        def _wrapped(*args: Any, **kwargs: Any):
            if should_log():
                tensor_info = ""
                if args:
                    tensor_info = _tensor_summary(args[0])
                logger.info("DDP trace: rank=%s enter %s %s", _rank(), name, tensor_info)
            out = orig(*args, **kwargs)
            if should_log():
                logger.info("DDP trace: rank=%s exit %s", _rank(), name)
            return out

        wrapped[name] = _wrapped
        setattr(dist, name, _wrapped)

    for op in [
        "barrier",
        "broadcast",
        "all_reduce",
        "reduce",
        "all_gather",
        "gather",
        "scatter",
        "reduce_scatter",
        "all_to_all",
    ]:
        wrap(op)

    def restore() -> None:
        for name, orig in originals.items():
            setattr(dist, name, orig)

    return restore
