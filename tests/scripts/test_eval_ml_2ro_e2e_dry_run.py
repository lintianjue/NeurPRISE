import os
import subprocess
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "04_05_eval_ml_2ro_e2e.sh"


def _run_script(args, extra_env=None):
    env = os.environ.copy()
    env.update(extra_env or {})
    result = subprocess.run(
        ["bash", str(SCRIPT_PATH), *args],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result


# Motivation: when EVAL_SEED is set, eval.name should include _e<seed> and pass eval.seed override.
def test_ml_e2e_dry_run_includes_eval_seed_in_name_and_command():
    result = _run_script(
        [
            "vertex_cover",
            "20",
            "50",
            "2500",
            "one_shot_crossattn_edge_dim1_small",
            "score_weighted_loss_flat",
            "train_scale_norm",
            "cpu",
            "32",
            "40",
            "900",
            "uniform",
        ],
        {
            "INTERACTIVE": "0",
            "CONFIRM": "0",
            "DRY_RUN": "1",
            "K_LIST": "0.1",
            "EVAL_SEED": "1",
            "DISCORD_NOTIFY": "0",
        },
    )

    assert result.returncode == 0
    dry_run_lines = [line for line in result.stdout.splitlines() if line.startswith("DRY_RUN:")]
    assert len(dry_run_lines) == 1

    line = dry_run_lines[0]
    assert (
        "eval.name=ML_s42_e1_i2500_n20_s50_t-train_scale_norm_"
        "m-one_shot_crossattn_edge_dim1_small_l-score_weighted_loss_flat_test"
    ) in line
    assert "eval.seed=1" in line
    assert "eval.data_override.seed=" not in line


# Motivation: eval RNG seed and alt-data seed must be independently configurable in USE_ALT_DATA mode.
def test_ml_e2e_dry_run_supports_independent_eval_and_data_seed():
    result = _run_script(
        [
            "vertex_cover",
            "20",
            "50",
            "2500",
            "one_shot_crossattn_edge_dim1_small",
            "score_weighted_loss_flat",
            "train_scale_norm",
            "cpu",
            "32",
            "40",
            "900",
            "uniform",
        ],
        {
            "INTERACTIVE": "0",
            "CONFIRM": "0",
            "DRY_RUN": "1",
            "K_LIST": "0.1",
            "USE_ALT_DATA": "1",
            "EVAL_N_NODES": "50",
            "EVAL_N_SCENARIOS": "100",
            "EVAL_N_INSTANCES": "500",
            "EVAL_SEED": "2",
            "EVAL_DATA_SEED": "7",
            "DISCORD_NOTIFY": "0",
        },
    )

    assert result.returncode == 0
    dry_run_lines = [line for line in result.stdout.splitlines() if line.startswith("DRY_RUN:")]
    assert len(dry_run_lines) == 1

    line = dry_run_lines[0]
    assert (
        "eval.name=ML_s42_e2_tr-i2500_n20_s50_ev-i500_n50_s100_t-train_scale_norm_"
        "m-one_shot_crossattn_edge_dim1_small_l-score_weighted_loss_flat_test"
    ) in line
    assert "eval.seed=2" in line
    assert "eval.data_override.seed=7" in line
