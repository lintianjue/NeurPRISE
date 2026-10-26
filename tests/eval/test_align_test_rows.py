"""Selection rows are matched to the LAST n_test instances (the test split)."""
import numpy as np
import pytest

from scripts.eval.eval_regret import IndexAlignmentError, _align_test_rows


def test_test_split_rows_pass_through():
    idx = np.arange(50 * 2).reshape(50, 2)
    out, n = _align_test_rows(idx, 50, 500)
    assert n == 50 and (out == idx).all()


def test_whole_file_rows_take_the_last_n_test():
    """Mutation killed: ``loaded_idx[:n_test]`` (first rows = training instances)."""
    idx = np.arange(500).reshape(500, 1)
    out, n = _align_test_rows(idx, 50, 500)
    assert n == 50 and out[0, 0] == 450 and out[-1, 0] == 499


def test_partial_test_run_keeps_first_rows():
    idx = np.arange(5).reshape(5, 1)
    out, n = _align_test_rows(idx, 50, 500)
    assert n == 5 and (out == idx).all()


def test_unplaceable_row_count_refuses():
    with pytest.raises(IndexAlignmentError):
        _align_test_rows(np.zeros((450, 1)), 50, 500)
