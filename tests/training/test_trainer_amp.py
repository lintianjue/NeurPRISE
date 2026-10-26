import pytest
import torch

from src.training.trainer import Trainer


@pytest.fixture()
def model_and_opt():
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, opt


def test_trainer_amp_off_cpu(model_and_opt):
    # Motivation: keep default training behavior when AMP is disabled.
    model, opt = model_and_opt
    trainer = Trainer(model, opt, metrics={}, device="cpu", amp_dtype="off")
    assert trainer.amp_enabled is False
    assert trainer._grad_scaler is None


def test_trainer_amp_fp16_cpu_disables_amp(model_and_opt):
    # Motivation: avoid enabling CUDA AMP paths on CPU-only setups.
    model, opt = model_and_opt
    trainer = Trainer(model, opt, metrics={}, device="cpu", amp_dtype="fp16")
    assert trainer.amp_enabled is False
    assert trainer._grad_scaler is None


def test_trainer_amp_invalid_dtype_raises(model_and_opt):
    # Motivation: reject unexpected AMP dtype values early.
    model, opt = model_and_opt
    with pytest.raises(ValueError):
        Trainer(model, opt, metrics={}, device="cpu", amp_dtype="float32")


# ---------------------------------------------------------------------------
# TC-1: test_normalize_amp_dtype_aliases
# ---------------------------------------------------------------------------
def test_normalize_amp_dtype_aliases(model_and_opt):
    """float16/bfloat16/none aliases are normalized correctly."""
    model, opt = model_and_opt

    # "float16" -> "fp16"
    t = Trainer(model, opt, metrics={}, device="cpu", amp_dtype="float16")
    assert t.amp_dtype == "fp16"

    # "bfloat16" -> "bf16"
    t2 = Trainer(model, opt, metrics={}, device="cpu", amp_dtype="bfloat16")
    assert t2.amp_dtype == "bf16"

    # None -> "off"
    t3 = Trainer(model, opt, metrics={}, device="cpu", amp_dtype=None)
    assert t3.amp_dtype == "off"

    # "none" -> "off"
    t4 = Trainer(model, opt, metrics={}, device="cpu", amp_dtype="none")
    assert t4.amp_dtype == "off"


# ---------------------------------------------------------------------------
# TC-6: test_strip_ddp_prefix_strips_exactly_once
# ---------------------------------------------------------------------------
def test_strip_ddp_prefix_strips_once(model_and_opt):
    """_strip_ddp_prefix removes exactly one 'module.' prefix."""
    model, opt = model_and_opt
    trainer = Trainer(model, opt, metrics={}, device="cpu", amp_dtype="off")

    state_dict = {
        "module.weight": torch.ones(2),
        "module.bias": torch.zeros(2),
    }
    stripped = trainer._strip_ddp_prefix(state_dict)

    assert "weight" in stripped
    assert "bias" in stripped
    assert "module.weight" not in stripped
    # Verify double prefix is stripped exactly once (not recursively)
    double = {"module.module.layer": torch.ones(3)}
    result = trainer._strip_ddp_prefix(double)
    assert "module.layer" in result   # stripped once: "module." prefix removed once
