"""Harness tests for the regret evaluator (`scripts/eval/eval_regret.py`).

  the evaluator refuses to write outside its own project root;
  split-scoped artifacts: result cells stay split-suffixed;
  artifacts are strict RFC-8259 JSON (null + status, no Infinity/NaN)
      while legacy-encoded artifacts remain readable.

No Gurobi and no dataset are required: the solver-touching paths are tested
through their pure record/validation layer.
"""
import importlib.util
import json
import math
import os
import shutil
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
EVAL_REGRET_REL = os.path.join("scripts", "eval", "eval_regret.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_isolated_tree(root):
    """Copy the minimum tree an isolated reproduction run needs.

    Deliberately WITHOUT a ``.git`` or ``README.md`` marker, the state in which
    ``_find_project_root()`` degrades to ``src/``.
    """
    for rel in (EVAL_REGRET_REL,
                os.path.join("src", "__init__.py"),
                os.path.join("src", "utils", "__init__.py"),
                os.path.join("src", "utils", "regret_json.py"),
                # eval_regret's module-level imports.
                os.path.join("src", "utils", "run_io.py"),
                os.path.join("src", "eval", "__init__.py"),
                os.path.join("src", "eval", "regret.py")):
        dst = os.path.join(root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(REPO_ROOT, rel), dst)
    assert not os.path.exists(os.path.join(root, ".git"))
    assert not os.path.exists(os.path.join(root, "README.md"))
    return root


def _load_module(path, name):
    """Import a module by path, leaving sys.path / sys.modules as we found them.

    The evaluator inserts its own BASE on sys.path at import time; without this
    cleanup an isolated copy would shadow the source tree's ``src`` package for
    every later test in the session.
    """
    saved_path = list(sys.path)
    saved_src = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved_path
        for key in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            del sys.modules[key]
        sys.modules.update(saved_src)
    return mod


# ---------------------------------------------------------------------------
# Write guard and verifier exit codes
# ---------------------------------------------------------------------------

def test_write_outside_root_is_refused(tmp_path):
    root = _make_isolated_tree(str(tmp_path / "tree"))
    mod = _load_module(os.path.join(root, EVAL_REGRET_REL), "eval_regret_guard")
    outside = str(tmp_path / "not_the_tree" / "x.json")
    with pytest.raises(RuntimeError, match="refusing to write outside project root"):
        mod.assert_within_base(outside)
    # ...and a path inside the root is accepted.
    assert mod.assert_within_base(os.path.join(root, "eval_regret", "ok.json"))





class TestSplitScopedArtifacts:
    """Every result artifact is split-suffixed."""

    def test_save_results_places_the_split_in_the_filename(self, tmp_path):
        root = _make_isolated_tree(str(tmp_path / "tree"))
        mod = _load_module(os.path.join(root, EVAL_REGRET_REL), "eval_regret_split")
        mod.save_results(
            [{"instance_id": 0, "full_obj": 10.0, "reduced_obj": 9.0, "eval_on_xi": 10.5,
              "sel_size": 4, "regret": 5.0}],
            "selection", "small", "neurprise", k=4, extra="n250")
        assert os.path.exists(os.path.join(
            root, "eval_regret", "selection", "SEL-20-50_neurprise_k4_n250.json"))
        # The unsuffixed name must NOT have been produced.
        assert not os.path.exists(os.path.join(
            root, "eval_regret", "selection", "SEL-20-50_kmedoids_k4.json"))

# ---------------------------------------------------------------------------
# Strict JSON out, both encodings in
# ---------------------------------------------------------------------------

class TestStrictJson:

    def test_written_artifacts_contain_no_json_constants(self, tmp_path):
        from src.utils import regret_json as rj
        path = str(tmp_path / "cell.json")
        rj.dump_strict({"per_instance": [rj.annotate_record(
            {"regret": float("inf"), "eval_on_xi": float("inf"),
             "full_obj": 12.5, "reduced_obj": float("nan")})],
            "regret_mean": float("inf")}, path)
        raw = open(path).read()
        assert "Infinity" not in raw and "NaN" not in raw
        # Strict parsers must accept it.
        assert rj.is_strict_file(path)
        doc = json.loads(raw)
        rec = doc["per_instance"][0]
        assert rec["regret"] is None and rec["regret_status"] == "infeasible"
        assert rec["reduced_obj"] is None and rec["reduced_obj_status"] == "undefined"
        assert rec["full_obj"] == 12.5 and rec["full_obj_status"] == "finite"
        assert doc["regret_mean"] is None

    def test_reader_shim_handles_the_legacy_encoding(self, tmp_path):
        """Legacy files written with Infinity/NaN literals stay readable."""
        from src.utils import regret_json as rj
        legacy = str(tmp_path / "legacy.json")
        with open(legacy, "w") as f:
            f.write('{"per_instance": [{"regret": Infinity, "reduced_obj": NaN},'
                    ' {"regret": 0.1644, "reduced_obj": 1.0}]}')
        doc = rj.load_regret_json(legacy)
        vals = [rj.record_value(r, "regret") for r in doc["per_instance"]]
        assert math.isinf(vals[0]) and vals[1] == pytest.approx(0.1644)
        assert math.isnan(rj.record_value(doc["per_instance"][0], "reduced_obj"))
        assert not rj.is_strict_file(legacy)

    def test_round_trip_preserves_infeasibility(self, tmp_path):
        from src.utils import regret_json as rj
        rec = rj.annotate_record({"regret": float("inf")})
        assert math.isinf(rj.record_value(rec, "regret"))
        rec2 = rj.annotate_record({"regret": -float("inf")})
        assert rj.record_value(rec2, "regret") == -math.inf
        assert rec2["regret_status"] == "neg_infinite"

    def test_save_results_emits_strict_json(self, tmp_path):
        root = _make_isolated_tree(str(tmp_path / "tree"))
        mod = _load_module(os.path.join(root, EVAL_REGRET_REL), "eval_regret_strict")
        summary = mod.save_results(
            [{"instance_id": 0, "full_obj": 10.0, "reduced_obj": 9.0,
              "eval_on_xi": float("inf"), "sel_size": 1,
              "regret": float("inf")}],
            "selection", "small", "strictprobe", k=1)
        assert summary["n_infeasible"] == 1
        out = os.path.join(root, "eval_regret", "selection", "SEL-20-50_strictprobe_k1.json")
        raw = open(out).read()
        assert "Infinity" not in raw and "NaN" not in raw
        doc = json.loads(raw)
        assert doc["schema_version"] == "regret/1.1"
        assert doc["per_instance"][0]["regret_status"] == "infeasible"
