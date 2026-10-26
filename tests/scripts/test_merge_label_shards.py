"""merge_label_shards.py: shards are concatenated in index order, a short shard is refused,
and the merged file is an exclusive create."""
import pickle
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "pipeline" / "merge_label_shards.py"


def _run(label, start, end):
    return subprocess.run([sys.executable, str(SCRIPT), "--label-path", str(label),
                           "--expect-start", str(start), "--expect-end", str(end)],
                          capture_output=True, text=True)


def _shard(tmp_path, start, end, n=None):
    path = tmp_path / f"label_x_shard{start}-{end}.pkl"
    path.write_bytes(pickle.dumps(list(range(start, start + (end - start if n is None else n)))))


def test_merge_in_order_then_refuse_overwrite(tmp_path):
    _shard(tmp_path, 2, 4)
    _shard(tmp_path, 0, 2)
    label = tmp_path / "label_x.pkl"

    assert _run(label, 0, 4).returncode == 0
    assert pickle.loads(label.read_bytes()) == [0, 1, 2, 3]
    assert "FileExistsError" in _run(label, 0, 4).stderr


def test_short_shard_is_refused(tmp_path):
    _shard(tmp_path, 0, 2, n=1)
    label = tmp_path / "label_x.pkl"
    assert _run(label, 0, 2).returncode == 2 and not label.exists()
