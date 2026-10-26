"""Tests for scripts/pipeline/merge_label_roles.py.

The two properties pinned down explicitly:

* **index integrity** -- the merged file is source order, trace-then-eval, and
  stays that way through pickling and re-reading;
* **inputs untouched** -- both source pickles are byte-identical after a run,
  including after a run that fails partway.
"""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.pipeline.merge_label_roles import file_md5, merge
from src.utils.label_roles import ROLE_EVAL, ROLE_TRAIN, instance_fingerprint, record_role


def make_record(*, trace=None, x=None, seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        "instance": SimpleNamespace(
            data={"adjacency": rng.integers(0, 2, (4, 4)), "det": rng.random(4)}
        ),
        "label": SimpleNamespace(forward_trace=trace, x=x, obj=float(seed)),
    }


def trace_record(seed):
    return make_record(trace={"steps": [seed]}, x=None, seed=seed)


def eval_record(seed):
    return make_record(trace=None, x=np.full(4, float(seed)), seed=seed)


def dump(path, records):
    with open(path, "wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


@pytest.fixture
def files(tmp_path):
    """45 trace + 5 eval records -- the 450/50 shape in miniature."""
    traces = [trace_record(i) for i in range(45)]
    evals = [eval_record(1000 + i) for i in range(5)]
    return SimpleNamespace(
        trace=dump(tmp_path / "trace.pkl", traces),
        eval=dump(tmp_path / "eval.pkl", evals),
        out=tmp_path / "merged.pkl",
        traces=traces,
        evals=evals,
    )


class TestHappyPath:
    def test_counts_and_boundary(self, files):
        manifest = merge(str(files.trace), str(files.eval), str(files.out), force=False, dry_run=False)
        assert manifest["n_trace"] == 45
        assert manifest["n_eval"] == 5
        assert manifest["n_total"] == 50
        assert manifest["boundary_index"] == 45

    def test_index_integrity_after_round_trip(self, files):
        """Every merged slot holds exactly the source record that belongs there."""
        merge(str(files.trace), str(files.eval), str(files.out), force=False, dry_run=False)
        with open(files.out, "rb") as f:
            merged = pickle.load(f)

        expected = [instance_fingerprint(r) for r in files.traces + files.evals]
        assert [instance_fingerprint(r) for r in merged] == expected

        # and the ordering is not merely a set match -- position by position
        for i, source in enumerate(files.traces):
            assert instance_fingerprint(merged[i]) == instance_fingerprint(source)
        for j, source in enumerate(files.evals):
            assert instance_fingerprint(merged[45 + j]) == instance_fingerprint(source)

    def test_roles_land_in_the_right_slots(self, files):
        merge(str(files.trace), str(files.eval), str(files.out), force=False, dry_run=False)
        with open(files.out, "rb") as f:
            merged = pickle.load(f)
        assert all(record_role(r) == ROLE_TRAIN for r in merged[:45])
        assert all(record_role(r) == ROLE_EVAL for r in merged[45:])

    def test_labels_survive_not_just_instances(self, files):
        merge(str(files.trace), str(files.eval), str(files.out), force=False, dry_run=False)
        with open(files.out, "rb") as f:
            merged = pickle.load(f)
        assert merged[0]["label"].forward_trace == {"steps": [0]}
        np.testing.assert_array_equal(merged[45]["label"].x, np.full(4, 1000.0))

    def test_manifest_written(self, files):
        merge(str(files.trace), str(files.eval), str(files.out), force=False, dry_run=False)
        manifest_path = files.out.with_suffix(files.out.suffix + ".manifest.json")
        assert manifest_path.exists()

    def test_no_temp_file_left_behind(self, files):
        merge(str(files.trace), str(files.eval), str(files.out), force=False, dry_run=False)
        assert not (files.out.with_suffix(files.out.suffix + ".tmp")).exists()


class TestInputsAreNeverModified:
    def test_md5_unchanged_on_success(self, files):
        before = (file_md5(files.trace), file_md5(files.eval))
        merge(str(files.trace), str(files.eval), str(files.out), force=False, dry_run=False)
        assert (file_md5(files.trace), file_md5(files.eval)) == before

    def test_md5_unchanged_on_failure(self, tmp_path):
        """A rejected merge must leave the inputs exactly as it found them."""
        bad_trace = dump(tmp_path / "bad.pkl", [trace_record(1), eval_record(2)])
        good_eval = dump(tmp_path / "eval.pkl", [eval_record(1000)])
        before = (file_md5(bad_trace), file_md5(good_eval))
        with pytest.raises(ValueError):
            merge(str(bad_trace), str(good_eval), str(tmp_path / "out.pkl"), force=False, dry_run=False)
        assert (file_md5(bad_trace), file_md5(good_eval)) == before

    def test_manifest_records_matching_in_and_out_md5(self, files):
        manifest = merge(str(files.trace), str(files.eval), str(files.out), force=False, dry_run=False)
        assert manifest["md5_trace_in"] == manifest["md5_trace_out"]
        assert manifest["md5_eval_in"] == manifest["md5_eval_out"]


class TestRefusals:
    def test_refuses_existing_output(self, files):
        files.out.write_bytes(b"do not clobber me")
        with pytest.raises(FileExistsError):
            merge(str(files.trace), str(files.eval), str(files.out), force=False, dry_run=False)
        assert files.out.read_bytes() == b"do not clobber me"

    def test_force_permits_overwrite(self, files):
        files.out.write_bytes(b"stale")
        merge(str(files.trace), str(files.eval), str(files.out), force=True, dry_run=False)
        with open(files.out, "rb") as f:
            assert len(pickle.load(f)) == 50

    def test_rejects_eval_record_in_trace_file(self, tmp_path):
        trace = dump(tmp_path / "t.pkl", [trace_record(1), eval_record(2)])
        ev = dump(tmp_path / "e.pkl", [eval_record(1000)])
        with pytest.raises(ValueError, match="expected every record to have role"):
            merge(str(trace), str(ev), str(tmp_path / "o.pkl"), force=False, dry_run=False)

    def test_rejects_trace_record_in_eval_file(self, tmp_path):
        trace = dump(tmp_path / "t.pkl", [trace_record(1)])
        ev = dump(tmp_path / "e.pkl", [eval_record(1000), trace_record(3)])
        with pytest.raises(ValueError, match="expected every record to have role"):
            merge(str(trace), str(ev), str(tmp_path / "o.pkl"), force=False, dry_run=False)

    def test_rejects_overlapping_instances(self, tmp_path):
        """The same instance in both files would put a test instance into training."""
        shared = trace_record(42)
        leaked = {"instance": shared["instance"], "label": SimpleNamespace(forward_trace=None, x=np.ones(4))}
        trace = dump(tmp_path / "t.pkl", [shared, trace_record(1)])
        ev = dump(tmp_path / "e.pkl", [leaked])
        with pytest.raises(ValueError, match="appear in BOTH files"):
            merge(str(trace), str(ev), str(tmp_path / "o.pkl"), force=False, dry_run=False)

    def test_rejects_duplicate_instances_within_a_file(self, tmp_path):
        dup = trace_record(5)
        trace = dump(tmp_path / "t.pkl", [dup, trace_record(6), dict(dup)])
        ev = dump(tmp_path / "e.pkl", [eval_record(1000)])
        with pytest.raises(ValueError, match="duplicate instances"):
            merge(str(trace), str(ev), str(tmp_path / "o.pkl"), force=False, dry_run=False)

    def test_rejects_empty_file(self, tmp_path):
        trace = dump(tmp_path / "t.pkl", [])
        ev = dump(tmp_path / "e.pkl", [eval_record(1000)])
        with pytest.raises(ValueError, match="empty"):
            merge(str(trace), str(ev), str(tmp_path / "o.pkl"), force=False, dry_run=False)

    def test_rejects_non_list_pickle(self, tmp_path):
        with open(tmp_path / "t.pkl", "wb") as f:
            pickle.dump({"not": "a list"}, f)
        ev = dump(tmp_path / "e.pkl", [eval_record(1000)])
        with pytest.raises(TypeError, match="expected a list of records"):
            merge(str(tmp_path / "t.pkl"), str(ev), str(tmp_path / "o.pkl"), force=False, dry_run=False)


class TestDryRun:
    def test_writes_nothing(self, files):
        manifest = merge(str(files.trace), str(files.eval), str(files.out), force=False, dry_run=True)
        assert manifest["dry_run"] is True
        assert manifest["n_total"] == 50
        assert not files.out.exists()

    def test_still_validates(self, tmp_path):
        trace = dump(tmp_path / "t.pkl", [trace_record(1), eval_record(2)])
        ev = dump(tmp_path / "e.pkl", [eval_record(1000)])
        with pytest.raises(ValueError, match="expected every record to have role"):
            merge(str(trace), str(ev), str(tmp_path / "o.pkl"), force=False, dry_run=True)
