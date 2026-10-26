"""
TC-22: test_average_meter_update_reset_and_empty
TC-23: test_to_numpy_all_types_including_error
TC-24: test_find_project_root_returns_valid_dir
"""
import os
import pytest
import numpy as np
import torch

from src.utils.utils import to_numpy, _find_project_root


# ---------------------------------------------------------------------------
# TC-23: to_numpy all types
# ---------------------------------------------------------------------------
def test_to_numpy_from_tensor():
    t = torch.tensor([1.0, 2.0, 3.0])
    arr = to_numpy(t)
    assert isinstance(arr, np.ndarray)
    np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])


def test_to_numpy_from_numpy_array():
    a = np.array([4.0, 5.0])
    result = to_numpy(a)
    assert result is a  # should return the same object


def test_to_numpy_from_list():
    result = to_numpy([1, 2, 3])
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [1, 2, 3])


def test_to_numpy_from_tuple():
    result = to_numpy((1.0, 2.0))
    assert isinstance(result, np.ndarray)


def test_to_numpy_from_python_float():
    result = to_numpy(3.14)
    assert isinstance(result, np.ndarray)
    assert float(result) == pytest.approx(3.14, abs=1e-9)


def test_to_numpy_from_none():
    assert to_numpy(None) is None


def test_to_numpy_unsupported_raises_type_error():
    """Unsupported type raises TypeError."""
    with pytest.raises(TypeError):
        to_numpy("unsupported_string")


# ---------------------------------------------------------------------------
# TC-24: _find_project_root
# ---------------------------------------------------------------------------
def test_find_project_root_returns_valid_dir():
    """_find_project_root returns a directory that exists."""
    root = _find_project_root()
    assert os.path.isdir(root)


def test_find_project_root_contains_git_or_readme():
    """The returned root contains either .git or README.md."""
    root = _find_project_root()
    has_git = os.path.exists(os.path.join(root, ".git"))
    has_readme = os.path.exists(os.path.join(root, "README.md"))
    assert has_git or has_readme, f"Neither .git nor README.md found in {root}"


def test_find_project_root_is_absolute():
    """The returned root is an absolute path."""
    root = _find_project_root()
    assert os.path.isabs(root)
