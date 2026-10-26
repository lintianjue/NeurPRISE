"""
TC-19: test_extract_scalar_int_all_types
Tests for eval_helpers._extract_scalar_int.
"""
import pytest
import numpy as np
import torch

from src.utils.eval_helpers import _extract_scalar_int


# ---------------------------------------------------------------------------
# TC-19: _extract_scalar_int handles all supported types
# ---------------------------------------------------------------------------
def test_extract_scalar_int_from_torch_tensor_scalar():
    """Scalar 1-element tensor returns int."""
    v = torch.tensor(42)
    assert _extract_scalar_int(v) == 42


def test_extract_scalar_int_from_torch_tensor_1d():
    """1D tensor [1] returns int of first element."""
    v = torch.tensor([7])
    assert _extract_scalar_int(v) == 7


def test_extract_scalar_int_from_numpy_scalar():
    """numpy scalar returns int."""
    v = np.int64(99)
    assert _extract_scalar_int(v) == 99


def test_extract_scalar_int_from_numpy_array():
    """1D numpy array [1] returns int of first element."""
    v = np.array([5])
    assert _extract_scalar_int(v) == 5


def test_extract_scalar_int_from_python_int():
    """Plain Python int returns int."""
    assert _extract_scalar_int(3) == 3


def test_extract_scalar_int_from_none():
    """None returns None."""
    assert _extract_scalar_int(None) is None


def test_extract_scalar_int_empty_tensor_returns_none():
    """Empty tensor returns None."""
    v = torch.tensor([])
    assert _extract_scalar_int(v) is None


def test_extract_scalar_int_empty_numpy_returns_none():
    """Empty numpy array returns None."""
    v = np.array([])
    assert _extract_scalar_int(v) is None


def test_extract_scalar_int_unsupported_type_returns_none():
    """Unsupported type (str) returns None."""
    assert _extract_scalar_int("hello") is None
