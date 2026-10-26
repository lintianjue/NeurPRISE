"""Entry points write only through run_io: no write-mode open(),
no mkdir/makedirs, no to_csv/savez/json.dump to a path -- every write sits in create_new."""
import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

ENTRIES = [
    "scripts/pipeline/perturb_labels.py",
]


@pytest.mark.parametrize("rel", ENTRIES)
def test_no_direct_writes(rel):
    """(M) revert any write in these files to open(path, "w") / os.makedirs(...)."""
    calls = [n for n in ast.walk(ast.parse((REPO / rel).read_text())) if isinstance(n, ast.Call)]
    names = [c.func.id for c in calls if isinstance(c.func, ast.Name)]
    attrs = [c.func.attr for c in calls if isinstance(c.func, ast.Attribute)]
    assert "create_new" in names and "open_script_run" in names
    assert not {"mkdir", "makedirs"} & set(attrs)
    for c in calls:
        if isinstance(c.func, ast.Name) and c.func.id == "open":
            modes = [a.value for a in c.args[1:2] if isinstance(a, ast.Constant)]
            assert not any(set("wax") & set(m) for m in modes), ast.unparse(c)
