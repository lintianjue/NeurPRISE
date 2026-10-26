"""Every automatic recipe in repro/runs.json builds a command, and says where its result lands."""
import sys
import json
import pytest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import repro.runs as runs  # noqa: E402


def test_every_auto_recipe_builds():
    recipes = runs.load()
    assert recipes
    for r in recipes:
        cmds = runs.build(r, threads=40, mip_gap=1e-4, time_limit=900)
        assert cmds and all(c["argv"][0] == "python" for c in cmds)
        assert runs.produces(r) == r["produces"]        # (M) rename an output in produces()
        if r["kind"] == "neurprise":
            assert any(a.startswith("eval.ckpt_path=") for a in cmds[0]["argv"])


def test_checkpoint_entry_selects_eval_config():
    got = runs.ckpt_overrides(runs.checkpoints()["main/VC-50-50"])
    assert got == ["--config-name", "eval_scale_norm", "model=det_source_sce_mean",
                   "losses=score_weighted_loss_flat"]
    assert (REPO / "configs" / "eval_scale_norm.yaml").is_file()


def test_checkpoints_are_named_once_and_recipes_resolve():
    ckpts = runs.checkpoints()
    paths = [c["path"] for c in ckpts.values()]
    assert len(paths) == len(set(paths))
    for key, c in ckpts.items():
        assert c["path"] == f"checkpoints/{key}.pt"
        if "train" in c:
            for part in ("model", "losses"):
                name = c["train"]["model" if part == "model" else "loss"]
                assert (REPO / "configs" / part / f"{name}.yaml").is_file(), (key, name)
    for r in runs.load():
        if r["kind"] == "neurprise":
            assert r["checkpoint"] in ckpts, r["id"]


@pytest.mark.parametrize("args", [["--table", "9"], ["--table", "1", "9"], ["--table"]])
def test_generator_rejects_unavailable_tables(monkeypatch, capsys, args):
    monkeypatch.setattr(sys, "argv", ["runs.py", *args, "--emit", "json"])
    with pytest.raises(SystemExit) as exc:
        runs.main()
    assert exc.value.code == 2
    assert capsys.readouterr().out == ""


def test_generator_still_builds_a_supported_table(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["runs.py", "--table", "1", "--threads", "8", "--emit", "json"])
    assert runs.main() == 0
    commands = json.loads(capsys.readouterr().out)
    expected = [command for recipe in runs.load([1])
                for command in runs.build(recipe, threads=8, mip_gap=1e-4, time_limit=900)]
    assert [c["argv"] for c in commands] == [c["argv"] for c in expected]
