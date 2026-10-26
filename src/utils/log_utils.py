# log_utils.py
import logging
import logging.config
import os
import sys
from datetime import datetime
from pathlib import Path


def setup_custom_logging(script_name='default', params_dict=None, log_folder_path ='../../logs'):
    """
    Setup logging with custom file name based on parameters

    Args:
        script_name (str): Name of the script
        params_dict (dict): Dictionary of parameters to include in filename
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(log_folder_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create parameter string
    param_str = ""
    if params_dict is not None:
        param_parts = []
        for key, value in params_dict.items():
            if value is not None:
                # For option parameter, take first 3 chars
                if key == 'option':
                    value = str(value)[:3]
                param_parts.append(f"{key[0]}{value}")
        param_str = "_".join(param_parts)

    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"{script_name}_{param_str}_{timestamp}.log"
    log_path = log_dir / log_filename

    return str(log_path)

class RelativePathFormatter(logging.Formatter):
    """Formatter that converts absolute paths to relative paths from project root"""

    def __init__(self, fmt=None, datefmt=None, project_root=None):
        super().__init__(fmt, datefmt)
        # Try to find project root (where .git or specific marker exists)
        if project_root is None:
            current = os.path.abspath(__file__)
            while current != os.path.dirname(current):
                if os.path.exists(os.path.join(current, '.git')) or \
                        os.path.exists(os.path.join(current, 'README.md')):
                    project_root = current
                    break
                current = os.path.dirname(current)
        self.project_root = project_root or os.getcwd()

    def format(self, record):
        # Convert pathname to relative path
        if hasattr(record, 'pathname'):
            try:
                record.pathname = os.path.relpath(record.pathname, self.project_root)
            except ValueError:
                # If relpath fails (e.g., different drives on Windows), keep absolute
                pass
        return super().format(record)

class _TimestampedStream:
    """Prefix every line written to ``stream`` with ``[YYYY-mm-dd HH:MM:SS]``."""

    def __init__(self, stream):
        self._stream = stream
        self._at_line_start = True

    def write(self, text):
        out = []
        for part in text.splitlines(keepends=True):
            if self._at_line_start:
                out.append(datetime.now().strftime("[%Y-%m-%d %H:%M:%S] "))
            out.append(part)
            self._at_line_start = part.endswith("\n")
        return self._stream.write("".join(out))

    def __getattr__(self, name):
        return getattr(self._stream, name)


def timestamp_stdout():
    """Timestamp each stdout line of a print-based script (the Hydra scripts log with asctime)."""
    if not isinstance(sys.stdout, _TimestampedStream):
        sys.stdout = _TimestampedStream(sys.stdout)
