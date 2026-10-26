"""
TC-2: test_use_teacher_forcing_prob_boundaries
TC-3: test_extract_teacher_idx_shape_validation
"""
import pytest
import torch
import types

from src.training.trainer import Trainer


@pytest.fixture()
def trainer_prob():
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, opt


# ---------------------------------------------------------------------------
# TC-2: _use_teacher_forcing boundary probabilities
# ---------------------------------------------------------------------------
def test_use_teacher_forcing_prob_boundaries(trainer_prob):
    """prob=0 always False, prob=1 always True."""
    model, opt = trainer_prob

    # mode="prob", teacher_prob=0.0 => always False
    t_never = Trainer(model, opt, metrics={}, device="cpu", amp_dtype="off",
                      teacher_mode="prob", teacher_prob=0.0)
    for _ in range(20):
        assert t_never._use_teacher_forcing() is False

    # mode="prob", teacher_prob=1.0 => always True
    t_always = Trainer(model, opt, metrics={}, device="cpu", amp_dtype="off",
                       teacher_mode="prob", teacher_prob=1.0)
    for _ in range(20):
        assert t_always._use_teacher_forcing() is True


def test_use_teacher_forcing_mode_none(trainer_prob):
    """teacher_mode='none' always returns False."""
    model, opt = trainer_prob
    t = Trainer(model, opt, metrics={}, device="cpu", amp_dtype="off",
                teacher_mode="none", teacher_prob=1.0)
    for _ in range(5):
        assert t._use_teacher_forcing() is False


def test_use_teacher_forcing_mode_always(trainer_prob):
    """teacher_mode='always' always returns True."""
    model, opt = trainer_prob
    t = Trainer(model, opt, metrics={}, device="cpu", amp_dtype="off",
                teacher_mode="always", teacher_prob=0.0)
    for _ in range(5):
        assert t._use_teacher_forcing() is True


def test_use_teacher_forcing_invalid_mode_raises(trainer_prob):
    """Invalid teacher_mode raises ValueError."""
    model, opt = trainer_prob
    with pytest.raises(ValueError):
        Trainer(model, opt, metrics={}, device="cpu", amp_dtype="off",
                teacher_mode="invalid")


# ---------------------------------------------------------------------------
# TC-3: _extract_teacher_idx shape validation
# ---------------------------------------------------------------------------
def test_extract_teacher_idx_shape_validation_1d_returns_none(trainer_prob):
    """If teacher tensor has dim != 2, return None."""
    model, opt = trainer_prob
    t = Trainer(model, opt, metrics={}, device="cpu", amp_dtype="off",
                teacher_mode="always", teacher_key="sce_top_idx")

    # 1D tensor (invalid shape)
    labels = types.SimpleNamespace(sce_top_idx=torch.tensor([0, 1, 2]))  # 1D
    result = t._extract_teacher_idx(labels, T=3, device=torch.device("cpu"))
    assert result is None


def test_extract_teacher_idx_trims_to_T(trainer_prob):
    """If teacher has more columns than T, trim to T."""
    model, opt = trainer_prob
    t = Trainer(model, opt, metrics={}, device="cpu", amp_dtype="off",
                teacher_mode="always", teacher_key="sce_top_idx")

    # B=2, 5 columns but T=3
    labels = types.SimpleNamespace(sce_top_idx=torch.zeros((2, 5), dtype=torch.long))
    result = t._extract_teacher_idx(labels, T=3, device=torch.device("cpu"))
    assert result is not None
    assert result.shape == (2, 3)


def test_extract_teacher_idx_exact_T(trainer_prob):
    """If teacher has exactly T columns, returned unchanged."""
    model, opt = trainer_prob
    t = Trainer(model, opt, metrics={}, device="cpu", amp_dtype="off",
                teacher_mode="always", teacher_key="sce_top_idx")

    labels = types.SimpleNamespace(sce_top_idx=torch.ones((3, 4), dtype=torch.long))
    result = t._extract_teacher_idx(labels, T=4, device=torch.device("cpu"))
    assert result is not None
    assert result.shape == (3, 4)


def test_extract_teacher_idx_too_short_returns_none(trainer_prob):
    """If teacher has fewer columns than T, return None."""
    model, opt = trainer_prob
    t = Trainer(model, opt, metrics={}, device="cpu", amp_dtype="off",
                teacher_mode="always", teacher_key="sce_top_idx")

    # B=2, only 2 columns but T=5
    labels = types.SimpleNamespace(sce_top_idx=torch.ones((2, 2), dtype=torch.long))
    result = t._extract_teacher_idx(labels, T=5, device=torch.device("cpu"))
    assert result is None
