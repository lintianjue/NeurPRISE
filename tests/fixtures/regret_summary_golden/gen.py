"""Generate the golden fixtures for ``save_results``/``summarize``.

Run ONCE, before the summary statistics moved into
``src/eval/regret.py::summarize`` -- the fixtures freeze the pre-refactor
``save_results`` output byte-for-byte. Re-running the same procedure through
the refactored ``save_results`` must reproduce these files exactly; that
equality is what ``tests/eval/test_summarize_golden.py`` checks.

    PYTHONPATH=$PWD python tests/fixtures/regret_summary_golden/gen.py
    # or
    python -m tests.fixtures.regret_summary_golden.gen
"""
import importlib.util
import json
import os
import sys
import tempfile

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import scripts.eval.eval_regret as eval_regret  # noqa: E402

_VOLATILE_KEYS = ("eval_date", "eval_server", "eval_machine_id")
_FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_cases():
    """Load ``records.CASES`` by file path, not ``tests.fixtures...`` import.

    Another checkout on ``PYTHONPATH`` may ship a regular ``tests/__init__.py`` package; PEP 420 lets a regular
    package anywhere on ``sys.path`` win over this repo's namespace-package
    ``tests/`` dir, so ``import tests.fixtures...`` can silently resolve to the
    wrong tree. Loading by absolute path sidesteps the collision entirely.
    """
    path = os.path.join(_FIXTURE_DIR, "records.py")
    spec = importlib.util.spec_from_file_location("regret_summary_golden_records", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CASES


CASES = _load_cases()


def main():
    with tempfile.TemporaryDirectory(prefix="eval_regret_golden_", dir=eval_regret.BASE) as tmp:
        for case, rows in CASES.items():
            # One subdir per case: same (problem, scale, method, k) on every
            # case would collide on the same output filename.
            eval_regret.REGRET_DIR = os.path.join(tmp, case)
            eval_regret.RUN = None
            name = eval_regret._result_name("selection", "small", "neurprise", k=1)
            eval_regret.save_results(rows, "selection", "small", "neurprise", k=1)
            out_path = os.path.join(eval_regret.REGRET_DIR, "selection", name)
            with open(out_path) as f:
                obj = json.loads(f.read())
            for key in _VOLATILE_KEYS:
                obj.pop(key, None)
            fixture_path = os.path.join(_FIXTURE_DIR, f"{case}.json")
            with open(fixture_path, "w") as f:
                f.write(json.dumps(obj, indent=2, sort_keys=True))
            print(f"wrote {fixture_path}")


if __name__ == "__main__":
    main()
