"""The user entry point runs experiments without depending on paper-maintenance tools."""
import fnmatch
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def command_env(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").symlink_to(sys.executable)
    return {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "NEURPRISE_INPUT_ROOT": str(REPO), "NEURPRISE_RUN_ROOT": str(tmp_path / "runs")}


def test_eval_dry_run_needs_no_paper_comparison(command_env):
    # Reproduction must still generate model-selection and regret commands after tools move.
    result = subprocess.run(
        ["bash", str(REPO / "scripts/reproduce.sh"), "--dry-run", "eval", "--table", "1"],
        env=command_env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "04_05_eval_ml_2ro_e2e.py" in result.stdout
    assert "scripts/eval/eval_regret.py" in result.stdout
    generated = Path(command_env["NEURPRISE_RUN_ROOT"]) / "reproduce_eval.sh"
    # A failed experiment must propagate its exit status without a comparison stage.
    fake_bin = generated.parent / "failing python"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text("#!/bin/sh\nexit 19\n")
    fake_python.chmod(0o755)
    failed = subprocess.run(["bash", str(generated)], capture_output=True, text=True,
                            env={**command_env, "PATH": str(fake_bin) + os.pathsep + command_env["PATH"]})
    assert failed.returncode == 19
    assert not (REPO / "repro/compare.py").exists()
    assert not (REPO / "repro/cells.json").exists()


@pytest.mark.parametrize("mode", ["tables", "check"])
def test_author_modes_are_not_public_commands(command_env, mode):
    # Paper maintenance should not be exposed as an experiment-running mode.
    result = subprocess.run(["bash", str(REPO / "scripts/reproduce.sh"), mode],
                            env=command_env, capture_output=True, text=True)
    assert result.returncode == 2
    assert not Path(command_env["NEURPRISE_RUN_ROOT"]).exists()


def test_download_only_requests_reproduction_inputs(tmp_path, command_env):
    # A normal download must not fetch saved paper results or internal measurement logs.
    stub = tmp_path / "stub"
    stub.mkdir()
    (stub / "huggingface_hub.py").write_text(
        "import json, os\n"
        "def snapshot_download(repo, **kwargs):\n"
        "    with open(os.environ['CAPTURE'], 'w') as f: json.dump(kwargs, f)\n")
    capture = tmp_path / "download.json"
    env = {**command_env, "PYTHONPATH": str(stub), "CAPTURE": str(capture)}
    result = subprocess.run(["bash", str(REPO / "scripts/reproduce.sh"), "download"],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    patterns = json.loads(capture.read_text())["allow_patterns"]
    included = lambda name: any(fnmatch.fnmatch(name, pattern) for pattern in patterns)
    assert all(included(name) for name in ["checkpoints/main/model.pt", "data/selection/inst.pkl",
                                          "eval_regret/full_objs_cache.json"])
    assert not included("eval_regret/selection/SEL-20-50_neurprise_k1.json")
    assert not included("training_logs/encoding_ablation/CFLP-30-30-50_cv.json")
    assert not included("checkpoints/gk_rf/model.pkl")
    assert not included("selections/selection/prise.npz")


@pytest.mark.parametrize("selection", [["--kind", "baseline"], ["--kind", "random"], ["--kind", "neurprise"], ["--kind"]])
def test_eval_rejects_removed_method_option(command_env, selection):
    # Ambiguous selections must fail before any experiment commands are published.
    result = subprocess.run(
        ["bash", str(REPO / "scripts/reproduce.sh"), "--dry-run", "eval", "--table", "1", *selection],
        env=command_env, capture_output=True, text=True,
    )
    assert result.returncode == 2, result.stderr
    assert not (Path(command_env["NEURPRISE_RUN_ROOT"]) / "reproduce_eval.sh").exists()


def test_invalid_method_does_not_execute_an_old_script(command_env):
    # Invalid arguments must not fall through to commands left by an earlier invocation.
    run_root = Path(command_env["NEURPRISE_RUN_ROOT"])
    run_root.mkdir()
    previous = run_root / "reproduce_eval.sh"
    previous.write_text('touch "$NEURPRISE_RUN_ROOT/unexpected_execution"\n')
    result = subprocess.run(
        ["bash", str(REPO / "scripts/reproduce.sh"), "eval", "--table", "1", "--kind", "baseline"],
        env=command_env, capture_output=True, text=True,
    )
    assert result.returncode == 2, result.stderr
    assert not (run_root / "unexpected_execution").exists()


def test_release_recipes_only_run_neurprise():
    # Table 7's compression budget (baselines are scenario orders), Tables 15/16's NeurPRISE
    # scenario-count timing and Table 14's time-budgeted full MILP are the other kinds.
    recipes = json.loads((REPO / "repro/runs.json").read_text())["runs"]
    assert recipes and all(r["kind"] in ("neurprise", "runtime", "time_budget")
                           or r["id"] == "compression_budget"
                           for r in recipes)
    assert not (REPO / "src/baselines").exists() or not any((REPO / "src/baselines").rglob("*.py"))
    models = json.loads((REPO / "repro/checkpoints.json").read_text())["checkpoints"]
    assert models and all(m["path"].endswith(".pt") for m in models)


@pytest.mark.parametrize("args", [
    ["--table", "1", "--threads", "8"],
    ["--threads", "8", "--table", "1", "2"],
])
def test_local_threads_reach_both_evaluation_stages(command_env, args):
    result = subprocess.run(
        ["bash", str(REPO / "scripts/reproduce.sh"), "--dry-run", "eval", *args],
        env=command_env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    commands = [line for line in result.stdout.splitlines() if line.startswith("python ")]
    model_commands = [line for line in commands if "04_05_eval_ml_2ro_e2e.py" in line]
    regret_commands = [line for line in commands if "scripts/eval/eval_regret.py" in line]
    assert model_commands and regret_commands
    assert all("solver.threads=8 " in line for line in model_commands)
    assert all("--threads 8 " in line for line in regret_commands)


@pytest.mark.parametrize("args", [["--threads"], ["--threads", "0"],
                                  ["--threads", "-1"], ["--threads", "eight"]])
def test_invalid_threads_fail_before_generating_experiments(command_env, args):
    result = subprocess.run(
        ["bash", str(REPO / "scripts/reproduce.sh"), "eval", "--table", "1", *args],
        env=command_env, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "positive integer" in result.stderr
    assert not (Path(command_env["NEURPRISE_RUN_ROOT"]) / "reproduce_eval.sh").exists()


@pytest.mark.parametrize("tables", [["9"], ["999"], ["1", "9"]])
def test_unsupported_tables_fail_without_running_old_commands(command_env, tables):
    run_root = Path(command_env["NEURPRISE_RUN_ROOT"])
    run_root.mkdir()
    previous = run_root / "reproduce_eval.sh"
    old_text = 'touch "$NEURPRISE_RUN_ROOT/unexpected_execution"\n'
    previous.write_text(old_text)
    result = subprocess.run(
        ["bash", str(REPO / "scripts/reproduce.sh"), "eval", "--table", *tables, "--threads", "8"],
        env=command_env, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "No NeurPRISE evaluation recipes" in result.stderr
    assert "Available table groups:" in result.stderr
    assert "Experiments completed" not in result.stdout
    assert previous.read_text() == old_text
    assert not (run_root / "unexpected_execution").exists()


@pytest.mark.parametrize("args", [["--table"], ["--table", "--threads", "8"]])
def test_missing_table_number_does_not_run_all_experiments(command_env, args):
    result = subprocess.run(
        ["bash", str(REPO / "scripts/reproduce.sh"), "eval", *args],
        env=command_env, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "--table requires at least one table number" in result.stderr
    assert not (Path(command_env["NEURPRISE_RUN_ROOT"]) / "reproduce_eval.sh").exists()
