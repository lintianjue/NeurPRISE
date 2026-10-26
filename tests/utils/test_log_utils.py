"""
TC-20: test_setup_custom_logging_filename_and_dir
Tests for log_utils.setup_custom_logging.
"""
import os
import pytest
import tempfile

from src.utils.log_utils import setup_custom_logging


# ---------------------------------------------------------------------------
# TC-20: setup_custom_logging creates log dir and returns a valid path
# ---------------------------------------------------------------------------
def test_setup_custom_logging_creates_dir_and_returns_path():
    """setup_custom_logging creates the log dir if needed and returns a .log path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_folder = os.path.join(tmpdir, "test_logs")
        path = setup_custom_logging(
            script_name="my_script",
            params_dict=None,
            log_folder_path=log_folder,
        )

        # Directory should have been created
        assert os.path.isdir(log_folder)

        # Returned path ends with .log
        assert path.endswith(".log")

        # Path should be inside the log folder
        assert path.startswith(log_folder)


def test_setup_custom_logging_filename_contains_script_name():
    """The returned log filename contains the script_name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = setup_custom_logging(
            script_name="unitest",
            log_folder_path=tmpdir,
        )
        basename = os.path.basename(path)
        assert "unitest" in basename


def test_setup_custom_logging_with_params_dict():
    """params_dict keys/values appear encoded in the filename."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = setup_custom_logging(
            script_name="test",
            params_dict={"lr": 0.01, "epoch": 5},
            log_folder_path=tmpdir,
        )
        basename = os.path.basename(path)
        # The params are encoded as "first_char_of_key + value" => "l0.01" and "e5"
        assert "l0.01" in basename
        assert "e5" in basename


def test_setup_custom_logging_none_params_no_crash():
    """None params_dict produces a valid path without crashing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = setup_custom_logging(
            script_name="safe",
            params_dict=None,
            log_folder_path=tmpdir,
        )
        assert path.endswith(".log")


def test_timestamp_stdout_prefixes_each_line():
    import io
    import re
    from src.utils.log_utils import _TimestampedStream
    buf = io.StringIO()
    s = _TimestampedStream(buf)
    s.write("a\nb")
    s.write("c\n")
    lines = buf.getvalue().splitlines()
    assert len(lines) == 2
    assert all(re.match(r"\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\] ", l) for l in lines)
    assert lines[0].endswith("] a") and lines[1].endswith("] bc")
