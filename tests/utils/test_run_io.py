import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.utils.run_io import (
    INPUT_ROOT_VAR,
    REPO,
    RUN_ROOT_VAR,
    RunIOError,
    RunPaths,
    create_new,
    input_file,
    mutable_path,
    open_hydra_run,
    open_run,
    output_path,
    publish_new,
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# 1. existing file untouched
def test_publish_new_does_not_overwrite_existing_file(tmp_path):
    rp = open_run("b1", "r1", [], base=tmp_path / "runs", env={})
    final = rp.run_dir / "out.bin"
    final.write_bytes(b"original")
    before = sha256_bytes(final.read_bytes())

    with pytest.raises(FileExistsError):
        publish_new(rp, "out.bin", lambda fh: fh.write(b"new"))

    assert sha256_bytes(final.read_bytes()) == before


# 2. input/output same path
def test_output_path_and_open_run_reject_input_root_overlap(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    # output_path-level check: input root equals run_dir itself.
    rp = RunPaths(input_roots=(run_dir.resolve(),), run_dir=run_dir.resolve())
    with pytest.raises(RunIOError):
        output_path(rp, "x")

    # open_run-level check: base sits inside a declared input root.
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    base = input_root / "runs"
    with pytest.raises(RunIOError):
        open_run("b2", "r1", [input_root], base=base, env={})
    assert not (base / "b2" / "r1").exists()


# 3. symlink escape
def test_output_path_rejects_symlink_components(tmp_path):
    rp = open_run("b3", "r1", [], base=tmp_path / "runs", env={})
    real_sub = rp.run_dir / "real_sub"
    real_sub.mkdir()
    old_file = real_sub / "old_file.txt"
    old_file.write_bytes(b"old content")

    # symlinked directory component, resolves inside run_dir (so only the
    # symlink-component check, not the escape check, can catch it)
    sub_link = rp.run_dir / "sub"
    sub_link.symlink_to(real_sub, target_is_directory=True)
    with pytest.raises(RunIOError):
        output_path(rp, "sub/x")
    assert old_file.read_bytes() == b"old content"

    # target itself a symlink, also resolving inside run_dir
    direct_link = rp.run_dir / "link"
    direct_link.symlink_to(real_sub, target_is_directory=True)
    with pytest.raises(RunIOError):
        output_path(rp, "link")
    assert old_file.read_bytes() == b"old content"


# 4. .. escape
def test_output_path_rejects_dotdot_escape(tmp_path):
    rp = open_run("b4", "r1", [], base=tmp_path / "runs", env={})
    with pytest.raises(RunIOError):
        output_path(rp, "../x")


# 5. duplicate run dir
def test_open_run_rejects_duplicate_run_dir(tmp_path):
    base = tmp_path / "runs"
    # Pre-create the run dir WITHOUT a RUN.json so the mkdir exclusivity
    # itself (not RUN.json's own O_EXCL) is what must raise.
    (base / "b5" / "r1").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        open_run("b5", "r1", [], base=base, env={})


# 6. double publish
def test_publish_new_second_call_raises_and_content_unchanged(tmp_path):
    rp = open_run("b6", "r1", [], base=tmp_path / "runs", env={})
    final = publish_new(rp, "out.bin", lambda fh: fh.write(b"first"))
    before = sha256_bytes(final.read_bytes())

    with pytest.raises(FileExistsError):
        publish_new(rp, "out.bin", lambda fh: fh.write(b"second"))

    assert sha256_bytes(final.read_bytes()) == before
    assert final.read_bytes() == b"first"


# 7. exception leaves nothing
def test_publish_new_write_fn_exception_leaves_no_trace(tmp_path):
    rp = open_run("b7", "r1", [], base=tmp_path / "runs", env={})

    def boom(fh):
        fh.write(b"partial")
        raise ValueError("boom")

    with pytest.raises(ValueError):
        publish_new(rp, "out.bin", boom)

    assert not (rp.run_dir / "out.bin").exists()
    assert list(rp.run_dir.glob("*.partial.*")) == []


# 8. env bypass
def test_open_run_env_bypass_rejected_before_base_created(tmp_path):
    base = tmp_path / "runs"
    elsewhere = tmp_path / "elsewhere"
    with pytest.raises(RunIOError):
        open_run(
            "b8",
            "r1",
            [],
            base=base,
            env={"PROJECT_DIR": str(elsewhere)},
        )
    assert not base.exists()


# 9. RUN.json contents + exclusivity
def test_run_json_has_expected_keys_and_is_exclusive(tmp_path):
    rp = open_run("b9", "r1", [], base=tmp_path / "runs", env={})
    run_json = rp.run_dir / "RUN.json"
    data = json.loads(run_json.read_text(encoding="utf-8"))
    expected_keys = {
        "argv",
        "cwd",
        "commit",
        "sys_path_head",
        "input_roots",
        "run_dir",
        "env",
        "host",
        "created_at",
    }
    assert expected_keys.issubset(data.keys())

    with pytest.raises(FileExistsError):
        create_new(rp, "RUN.json", mode="w")


# 9b. RUN.json records the thread knobs from the env it was given, and versions
# only of libraries already imported (open_run must not import gurobipy/torch itself).
def test_run_json_records_thread_env_and_versions(tmp_path):
    env = {"OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "8"}
    rp = open_run("b9b", "r1", [], base=tmp_path / "runs", env=env)
    data = json.loads((rp.run_dir / "RUN.json").read_text(encoding="utf-8"))
    assert data["thread_env"]["OPENBLAS_NUM_THREADS"] == "4"
    assert data["thread_env"]["OMP_NUM_THREADS"] == "8"
    assert data["thread_env"]["MKL_NUM_THREADS"] is None
    assert data["cpu_count"] == os.cpu_count()
    assert set(data["versions"]) <= {"python", "numpy", "torch", "gurobipy"}
    assert ("gurobipy" in data["versions"]) == ("gurobipy" in sys.modules)


# 11. bad names
def test_open_run_rejects_bad_names(tmp_path):
    base = tmp_path / "runs"
    for batch, run_id in [("../a", "r1"), ("b", "a/b"), ("", "r1"), ("b", "")]:
        with pytest.raises(RunIOError):
            open_run(batch, run_id, [], base=base, env={})


# 12. create_new mode validation
def test_create_new_rejects_bad_modes(tmp_path):
    rp = open_run("b12", "r1", [], base=tmp_path / "runs", env={})
    with pytest.raises(ValueError):
        create_new(rp, "x.bin", mode="a")
    with pytest.raises(ValueError):
        create_new(rp, "x.bin", mode="r+")


# sanity: mutable_path returns the same checked path as output_path
def test_mutable_path_matches_output_path(tmp_path):
    rp = open_run("b13", "r1", [], base=tmp_path / "runs", env={})
    assert mutable_path(rp, "x.bin") == output_path(rp, "x.bin")


# Two contract behaviours the numbered cases above leave unguarded.
def test_output_path_rejects_the_run_dir_itself(tmp_path):
    """Mutation killed: dropping `target_real == run_dir_real` from output_path."""
    rp = open_run("b13", "r1", [], base=tmp_path / "runs", env={})
    with pytest.raises(RunIOError):
        output_path(rp, rp.run_dir)
    with pytest.raises(RunIOError):
        output_path(rp, ".")


def test_guard_env_checks_project_dir(tmp_path):
    """Mutation killed: removing PROJECT_DIR from guard_env's loop."""
    base = tmp_path / "runs"
    with pytest.raises(RunIOError):
        open_run("b14", "r1", [], base=base, env={"PROJECT_DIR": str(tmp_path / "elsewhere")})
    assert not base.exists()
    # A value inside the repo is accepted.
    open_run("b14", "r2", [], base=base, env={"PROJECT_DIR": str(REPO)})


# --- open_hydra_run / input_file ----------------------------------------------------

def test_open_hydra_run_uses_entry_stem_and_run_name(tmp_path):
    """(M) swap batch and run_id in open_hydra_run's open_run call."""
    in_root = tmp_path / "in"
    in_root.mkdir()
    env = {INPUT_ROOT_VAR: str(in_root), RUN_ROOT_VAR: str(tmp_path / "root")}
    cfg = OmegaConf.create({"run_name": "vc_1"})
    rp = open_hydra_run(cfg, "/x/scripts/03_train.py", env=env)
    assert rp.run_dir == Path(os.path.realpath(tmp_path / "root" / "03_train" / "vc_1"))
    assert (rp.run_dir / "RUN.json").is_file()
    assert rp.input_roots == (Path(os.path.realpath(in_root)),)


@pytest.mark.parametrize("var", [INPUT_ROOT_VAR, RUN_ROOT_VAR])
@pytest.mark.parametrize("missing_style", ["absent", "empty"])
def test_open_hydra_run_defaults_missing_variables(tmp_path, monkeypatch, var, missing_style):
    import src.utils.run_io as rio
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(rio, "REPO", repo)
    monkeypatch.setattr(rio, "DEFAULT_BASE", repo / "runs")
    env = {INPUT_ROOT_VAR: str(repo), RUN_ROOT_VAR: str(repo / "runs")}
    if missing_style == "absent":
        del env[var]
    else:
        env[var] = ""
    rp = open_hydra_run(OmegaConf.create({"run_name": "vc_1"}), "03_train.py", env=env)
    assert rp.input_roots == (repo,)
    assert rp.run_dir == repo / "runs/03_train/vc_1"


def test_run_json_records_the_two_launcher_variables(tmp_path):
    """(M) drop one of the two keys from the RUN.json env record."""
    in_root = tmp_path / "in"
    in_root.mkdir()
    env = {INPUT_ROOT_VAR: str(in_root), RUN_ROOT_VAR: str(tmp_path / "root")}
    cfg = OmegaConf.create({"run_name": "vc_1"})
    rp = open_hydra_run(cfg, "/x/scripts/03_train.py", env=env)
    data = json.loads((rp.run_dir / "RUN.json").read_text(encoding="utf-8"))
    assert data["env"][INPUT_ROOT_VAR] == env[INPUT_ROOT_VAR]
    assert data["env"][RUN_ROOT_VAR] == env[RUN_ROOT_VAR]


def test_input_file_accepts_inside_and_rejects_outside(tmp_path):
    """(M) drop the _is_equal_or_inside check from input_file."""
    root = tmp_path / "root"
    root.mkdir()
    rp = open_run("b15", "r1", [root], base=tmp_path / "runs", env={})

    inside = root / "a.pkl"
    inside.write_bytes(b"x")
    assert input_file(rp, inside) == Path(os.path.realpath(inside))

    outside = tmp_path / "elsewhere.pkl"
    outside.write_bytes(b"y")
    with pytest.raises(RunIOError):
        input_file(rp, outside)

    link = root / "link.pkl"
    link.symlink_to(outside)
    with pytest.raises(RunIOError):
        input_file(rp, link)

    subdir = root / "subdir"
    subdir.mkdir()
    with pytest.raises(RunIOError):
        input_file(rp, subdir)


# 24. NEURPRISE_RUN_ROOT is the default base when the caller passes none
def test_open_run_defaults_base_to_run_root_env(tmp_path):
    """(M) ignore RUN_ROOT_VAR in open_run: the run lands under the repo scratch default."""
    run_root = tmp_path / "rr"
    rp = open_run("b24", "r1", [], env={"NEURPRISE_RUN_ROOT": str(run_root)})
    assert rp.run_dir == (run_root / "b24" / "r1").resolve()
    assert (rp.run_dir / "RUN.json").exists()


def test_open_script_run_names_the_run_after_the_entry(tmp_path, monkeypatch):
    """(M) drop the file-stem batch or the input-root requirement in open_script_run."""
    import argparse
    from src.utils.run_io import add_run_args, open_script_run, RunIOError

    in_root = tmp_path / "in"
    in_root.mkdir()
    monkeypatch.setenv("NEURPRISE_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.delenv("NEURPRISE_INPUT_ROOT", raising=False)
    parser = argparse.ArgumentParser()
    add_run_args(parser)

    run = open_script_run(parser.parse_args(["--input-root", str(in_root), "--run-id", "r1"]),
                          "/x/scripts/mipgap_regret_sweep.py")
    assert run.run_dir == (tmp_path / "runs" / "mipgap_regret_sweep" / "r1").resolve()
    default_run = open_script_run(parser.parse_args(["--run-id", "default"]), "/x/e.py")
    assert default_run.input_roots == (REPO.resolve(),)


def test_default_data_is_readable_by_the_next_stage_without_overwriting(tmp_path, monkeypatch):
    import src.utils.run_io as rio
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(rio, "REPO", repo)
    base = repo / "runs"
    generation = rio.open_run("data", "one", [repo], base=base, env={})
    instances = rio.data_output_path(generation, "selection", "inst.pkl")
    with rio.create_new(rio.data_paths(generation), instances) as f:
        f.write(b"instances")
    labeling = rio.open_run("label", "two", [repo], base=base, env={})
    assert rio.input_file(labeling, repo / "data/selection/inst.pkl").read_bytes() == b"instances"
    labels = rio.data_output_path(labeling, "selection", "label.pkl")
    with rio.create_new(rio.data_paths(labeling), labels) as f:
        f.write(b"labels")
    training = rio.open_run("train", "three", [repo], base=base, env={})
    assert rio.input_file(training, labels).read_bytes() == b"labels"
    with rio.create_new(training, "model/best.pt") as f:
        f.write(b"model")
    assert (base / "train/three/model/best.pt").read_bytes() == b"model"
    with pytest.raises(FileExistsError, match="Data already exists"):
        rio.data_output_path(labeling, "selection", "inst.pkl")
    assert instances.read_bytes() == b"instances"
    with pytest.raises(RunIOError):
        rio.create_new(training, repo / "checkpoints/released.pt")


def test_default_data_rejects_symlinks_and_path_escape(tmp_path, monkeypatch):
    import src.utils.run_io as rio
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(rio, "REPO", repo)
    run = rio.open_run("data", "one", [repo], base=repo / "runs", env={})
    with pytest.raises(RunIOError):
        rio.data_output_path(run, "selection", "../escape.pkl")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (repo / "data").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(RunIOError):
        rio.data_output_path(run, "selection", "inst.pkl")
    assert not list(elsewhere.iterdir())


def test_repository_runs_cannot_alias_the_data_directory(tmp_path, monkeypatch):
    import src.utils.run_io as rio
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    monkeypatch.setattr(rio, "REPO", repo)
    (repo / "runs").symlink_to(repo / "data", target_is_directory=True)
    with pytest.raises(RunIOError, match="overlaps input root"):
        rio.open_run("train", "one", [repo], base=repo / "runs", env={})
    assert not (repo / "data/train").exists()
