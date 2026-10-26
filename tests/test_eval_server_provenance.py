"""save_results must record the MEASURED host and machine id, never a
caller-supplied string.

Motivation. `--server` used to overwrite `eval_server`, so the field held
whatever string the caller typed and could not be used to tell which host
produced a result. The invariant this file guards is the fix -- `save_results`
always records
`socket.gethostname()` (and, where available, `/etc/machine-id`) as the
measured host/machine id; a caller-supplied `server=` value lands only in
`eval_run_label`, which nothing downstream treats as a hostname.
"""
import os
import socket
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO_ROOT)

from scripts.eval.eval_regret import save_results  # noqa: E402


def test_save_results_records_measured_host_not_the_passed_string(monkeypatch, tmp_path):
    """`--server` must no longer displace the measurement; it lands in eval_run_label."""
    from scripts.eval import eval_regret
    # save_results asserts "eval_regret" is in its output path -- honour that guard
    # rather than defeating it, so this test exercises the real write path.
    regret_dir = tmp_path / "eval_regret"
    regret_dir.mkdir()
    monkeypatch.setattr(eval_regret, "REGRET_DIR", str(regret_dir))
    monkeypatch.setattr(eval_regret, "assert_within_base", lambda *_a, **_k: None)
    results = [{"instance_id": 0, "full_obj": 10.0, "reduced_obj": 9.0,
                "eval_on_xi": 11.0, "regret": 10.0}]
    summary = save_results(results, "selection", "small", "neurprise", k=1,
                           threads=40, server="iso-gk14-rerun")
    assert summary["eval_server"] == socket.gethostname()
    assert summary["eval_run_label"] == "iso-gk14-rerun"


def test_save_results_run_label_is_none_when_not_passed(monkeypatch, tmp_path):
    from scripts.eval import eval_regret
    regret_dir = tmp_path / "eval_regret"
    regret_dir.mkdir()
    monkeypatch.setattr(eval_regret, "REGRET_DIR", str(regret_dir))
    monkeypatch.setattr(eval_regret, "assert_within_base", lambda *_a, **_k: None)
    results = [{"instance_id": 0, "full_obj": 10.0, "reduced_obj": 9.0,
                "eval_on_xi": 11.0, "regret": 10.0}]
    summary = save_results(results, "selection", "small", "neurprise", k=1, threads=40)
    assert summary["eval_server"] == socket.gethostname()
    assert summary["eval_run_label"] is None


def test_save_results_records_a_measured_machine_id(monkeypatch, tmp_path):
    from scripts.eval import eval_regret
    regret_dir = tmp_path / "eval_regret"
    regret_dir.mkdir()
    monkeypatch.setattr(eval_regret, "REGRET_DIR", str(regret_dir))
    monkeypatch.setattr(eval_regret, "assert_within_base", lambda *_a, **_k: None)
    results = [{"instance_id": 0, "full_obj": 10.0, "reduced_obj": 9.0,
                "eval_on_xi": 11.0, "regret": 10.0}]
    summary = save_results(results, "selection", "small", "neurprise", k=1, threads=40)
    # equals what this machine reports, and there is no way to pass one in
    expected = None
    try:
        expected = open("/etc/machine-id").read().strip() or None
    except OSError:
        pass
    assert summary["eval_machine_id"] == expected
