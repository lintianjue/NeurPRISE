"""find_neurprise_eval_base must not pick among several matches by glob order."""
import os

import pytest

import scripts.eval.eval_regret as er

KEY = ("selection", "small")
E2E = "ML_s42_i2500_t-train_scale_norm_m-det_source_sce_mean_l-score_weighted{}_test/e2e_2ro"


def _mk(root, suffix):
    d = os.path.join(root, er.EVAL_DIRS[KEY], E2E.format(suffix))
    os.makedirs(d)
    return d


def test_single_match_is_returned(tmp_path, monkeypatch):
    monkeypatch.setattr(er, "INPUT_ROOT", str(tmp_path))
    d = _mk(str(tmp_path), "_loss_a")
    assert er.find_neurprise_eval_base(*KEY[:1], "x", scale=KEY[1]) == d


def test_two_matches_refuse_and_name_both(tmp_path, monkeypatch):
    monkeypatch.setattr(er, "INPUT_ROOT", str(tmp_path))
    a, b = _mk(str(tmp_path), "_loss_a"), _mk(str(tmp_path), "_loss_b")
    with pytest.raises(er.AmbiguousEvalBase) as exc:
        er.find_neurprise_eval_base(*KEY[:1], "x", scale=KEY[1])
    assert a in str(exc.value) and b in str(exc.value)


def test_eval_base_may_lie_in_a_declared_extra_root_only(tmp_path):
    """--eval-root adds a read root; a path in none of the roots is still refused.
    (M) accept any path / check only INPUT_ROOT."""
    inp, extra, other = (tmp_path / n for n in ("in", "e2e_run", "elsewhere"))
    for d in (inp, extra / "eval", other):
        d.mkdir(parents=True)
    assert er._assert_inside_any(str(extra / "eval"), [str(inp), str(extra)], "x") == \
        os.path.realpath(extra / "eval")
    with pytest.raises(RuntimeError):
        er._assert_inside_any(str(other), [str(inp), str(extra)], "x")
