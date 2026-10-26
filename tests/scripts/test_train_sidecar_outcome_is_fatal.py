"""The sidecar outcome must be a first-class exit condition.

Two entrances into an invisible-failure funnel, both in scripts/03_train.py:
  (a)  a broad `except Exception` that logs and continues
  (b)  `emit_loudly` returns a bool that is DISCARDED

Either way training exits 0 with no `.run.json`, and the cell is refused downstream only after
the full training budget has been spent.

The behavioural half is tested against the real `emit_loudly`. The structural half is an AST
assertion rather than a regex, because 03_train.py's emission block cannot be executed without a
full hydra config, a GPU and a fitted model -- and a lock that cannot be executed is still worth
more than no lock.
"""
import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRAIN = REPO_ROOT / "scripts/03_train.py"


@pytest.fixture(scope="module")
def tree():
    return ast.parse(TRAIN.read_text(encoding="utf-8"))


def _emit_calls(tree):
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "emit_loudly"]


def test_emit_loudly_result_is_not_discarded(tree):
    """(b): a bare `emit_loudly(...)` expression statement throws the answer away."""
    calls = _emit_calls(tree)
    assert calls, "emit_loudly is no longer called; this lock needs rewriting"
    for stmt in ast.walk(tree):
        if isinstance(stmt, ast.Expr) and stmt.value in calls:
            pytest.fail("emit_loudly's return value is discarded (bare expression statement)")


def test_a_sidecar_failure_reaches_a_raise(tree):
    """(a) + (b): the outcome must gate a raise, not merely a log line."""
    names = {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign)
             for t in n.targets if isinstance(t, ast.Name) and "sidecar" in t.id}
    assert names, "no sidecar-outcome variable is assigned"
    guarded = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and any(isinstance(x, ast.Name) and x.id in names for x in ast.walk(n.test))
        and any(isinstance(b, ast.Raise) for b in ast.walk(n))
    ]
    assert guarded, "no `if <sidecar outcome>: raise` -- a failed sidecar still exits 0"


def test_the_handler_records_the_failure_rather_than_only_logging(tree):
    """The handler may still catch -- a sidecar bug must not discard a finished model -- but it
    must leave a durable trace in the outcome variable."""
    for handler in (n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)):
        body = ast.dump(ast.Module(body=handler.body, type_ignores=[]))
        if "emit_loudly" in ast.dump(handler) or "sidecar" in body.lower():
            assert any(isinstance(b, ast.Assign) for b in ast.walk(handler)), (
                "the sidecar handler logs without recording the failure, so the run still exits 0")
            return
    pytest.fail("no sidecar exception handler found; this lock needs rewriting")


def _iter_nodes(stmts):
    for stmt in stmts:
        yield from ast.walk(stmt)


def _find_main(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    return None


def _containing_index(body, target):
    """Index into `body` (a list of top-level statements) of the statement that contains
    `target` somewhere in its subtree (or is `target` itself)."""
    for i, stmt in enumerate(body):
        for node in ast.walk(stmt):
            if node is target:
                return i
    return None


def test_the_raise_happens_after_ddp_teardown(tree):
    """Raising before destroy_process_group would leak a process group on every DDP failure.

    The teardown sits in a `finally:`, so a text-offset check ("does
    'destroy_process_group' appear before 'provenance sidecar was not written' in the
    source") does not prove the ordering: both appear inside the SAME function, and
    only control flow (not source order) decides what runs when. This asserts the actual
    control flow: (a) trainer.fit and emit_loudly are inside the Try whose finally calls
    destroy_process_group; (b) the sidecar-failure raise is a statement of main()'s own
    body, positioned after that Try in main()'s statement list -- i.e. it can only run once
    the try/finally (and therefore the teardown) has completed.
    """
    main_func = _find_main(tree)
    assert main_func is not None, "main() not found"

    try_node = None
    for node in ast.walk(main_func):
        if isinstance(node, ast.Try):
            if any(
                isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == "destroy_process_group"
                for stmt in node.finalbody for call in ast.walk(stmt)
            ):
                try_node = node
                break
    assert try_node is not None, "no try/finally with destroy_process_group in the finally block"

    assert any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "fit" and isinstance(n.func.value, ast.Name) and n.func.value.id == "trainer"
        for n in _iter_nodes(try_node.body)
    ), "trainer.fit(...) is not inside the Try body"

    assert any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "emit_loudly"
        for n in _iter_nodes(try_node.body)
    ), "emit_loudly(...) is not inside the Try body"

    try_index = _containing_index(main_func.body, try_node)
    assert try_index is not None, "the Try is not a (nested) part of main()'s own body"

    raise_node = None
    for n in ast.walk(main_func):
        if isinstance(n, ast.Raise) and any(
            isinstance(c, ast.Constant) and isinstance(c.value, str)
            and "provenance sidecar was not written" in c.value
            for c in ast.walk(n)
        ):
            raise_node = n
            break
    assert raise_node is not None, "no raise carrying the 'provenance sidecar was not written' message"

    raise_index = _containing_index(main_func.body, raise_node)
    assert raise_index is not None, "the sidecar raise is not a statement of main()'s own body"
    assert raise_index > try_index, "the sidecar raise precedes the try/finally that tears down DDP"


# ---- behavioural half: the real emit_loudly contract ----

def test_emit_loudly_returns_false_and_does_not_raise_on_a_bad_target(tmp_path):
    """The whole design rests on this: emit_loudly reports failure by RETURNING, never by
    raising, so the return value is the only signal there is."""
    from scripts.pipeline.write_run_sidecar import emit_loudly
    missing = tmp_path / "nonexistent" / "model.pt"
    ok = emit_loudly(missing, arm="exact", label_path=str(tmp_path / "labels.pkl"), extra={})
    assert ok is False, "emit_loudly did not report failure for a missing checkpoint"


def test_emit_loudly_returns_true_for_a_real_file(tmp_path):
    from scripts.pipeline.write_run_sidecar import emit_loudly
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"not a real checkpoint, but a real file")
    ok = emit_loudly(ckpt, arm="exact", label_path=str(tmp_path / "labels.pkl"), extra={"a": 1})
    assert ok is True
    assert (tmp_path / "model.pt.run.json").is_file(), "sidecar not written beside the checkpoint"
