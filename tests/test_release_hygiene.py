"""Tracked files carry no machine-specific paths or references to the private predecessors.

A hardcoded home path makes a script read or write someone else's tree; a reference
to the private repositories or the ICLR workspace points a reader at nothing.
"""
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

FORBIDDEN = re.compile(
    r"/home/\w+|~/workspace|Learn2RO|Neur2ROTest1|data_iclr|iclr_exec|eval_regret/iclr"
    r"|model/iclr|ICLR(?! 2022)|\biclr\b"
)
SELF = "tests/test_release_hygiene.py"


def test_no_home_paths_or_private_repo_references():
    files = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True,
                           check=True).stdout.split()
    hits = []
    for rel in files:
        if rel == SELF or not rel.endswith((".py", ".sh", ".yaml", ".yml", ".md", ".txt", ".toml")):
            continue
        for lineno, line in enumerate((REPO / rel).read_text(errors="replace").splitlines(), 1):
            if FORBIDDEN.search(line):
                hits.append(f"{rel}:{lineno}: {line.strip()[:120]}")
    assert not hits, "\n".join(hits)


def test_referenced_docs_exist():
    files = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True,
                           check=True).stdout.split()
    missing = set()
    for rel in files:
        if rel.endswith((".py", ".sh", ".yaml", ".md")):
            text = (REPO / rel).read_text(errors="replace")
            missing |= {m for m in re.findall(r"\bdoc/[\w./-]+\.md\b", text) if not (REPO / m).exists()}
    assert not missing, sorted(missing)


def test_pickled_classes_keep_their_module_paths():
    """The released instance and label pickles name these classes by module path;
    moving or renaming them makes every released data file unloadable."""
    import pickle

    from src.core.types import ForwardTrace, Instance, Labels

    for cls in (Instance, Labels, ForwardTrace):
        assert cls.__module__ == "src.core.types", cls
    blob = pickle.dumps(Instance(data={}))
    assert b"src.core.types" in blob and b"Instance" in blob
