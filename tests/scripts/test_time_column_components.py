"""The re-measured Time column must carry BOTH halves.

The paper's Time column is defined, in the generator that produced it, as
    sum-of-Gurobi-solves + selection time,   for every method.
Recording only the reduced solve would drop the selection half while still
printing a column of seconds -- indistinguishable from the published one, and
biased most against the method whose selection is most expensive (`gk`:
26 features per scenario, one deterministic MILP each).

Pinned here: the field names the table generator reads, the arithmetic, and the
refusal to emit a total that only some rows contributed to.
"""
import math

import pytest

from scripts.eval.eval_regret import _time_components

# The exact keys the table generator reads. Renaming one silently disconnects
# re-measured runs from the comparison they exist for.
CONTRACT_KEYS = {
    "selection_time_total_sec", "selection_time_per_instance_sec",
    "reduced_eval_time_total_sec", "reduced_eval_time_per_instance_sec",
}


def _rows(n=4, sel=0.5, red=2.0, gurobi=0.1, precomputed=False):
    """`red` is the WALL clock over build+solve; `gurobi` is model.Runtime.
    They are different spans and the published column used the first."""
    return [{"selection_sec": sel, "reduced_eval_wall_sec": red,
             "reduced_solve_sec": gurobi,
             "selection_is_precomputed": precomputed} for _ in range(n)]


def test_field_names_match_the_old_generator_contract():
    out = _time_components(_rows())
    assert CONTRACT_KEYS <= set(out)


def test_totals_and_the_column_value():
    out = _time_components(_rows(n=4, sel=0.5, red=2.0))
    assert out["selection_time_total_sec"] == pytest.approx(2.0)
    assert out["reduced_eval_time_total_sec"] == pytest.approx(8.0)
    assert out["selection_time_per_instance_sec"] == pytest.approx(0.5)
    # This is the quantity the paper prints.
    assert out["time_column_sec"] == pytest.approx(10.0)


def test_a_partial_sum_is_null_not_a_smaller_number():
    """The failure this prevents: one row without a timing makes the sum smaller,
    and a smaller number in a Time column reads as a faster method."""
    rows = _rows(n=4)
    rows[2].pop("selection_sec")
    out = _time_components(rows)
    assert out["selection_time_total_sec"] is None
    assert out["selection_time_per_instance_sec"] is None
    assert out["selection_time_n_recorded"] == 3
    # Half a definition is a different quantity, not a smaller one.
    assert out["time_column_sec"] is None
    # The half that IS complete still reports.
    assert out["reduced_eval_time_total_sec"] == pytest.approx(8.0)


def test_non_finite_timings_do_not_silently_join_the_sum():
    rows = _rows(n=3)
    rows[1]["reduced_eval_wall_sec"] = float("nan")
    out = _time_components(rows)
    assert out["reduced_eval_time_n_recorded"] == 2
    assert out["reduced_eval_time_total_sec"] is None
    assert out["time_column_sec"] is None


def test_precomputed_flag_needs_every_row_and_a_nonempty_run():
    assert _time_components(_rows(precomputed=True))["selection_is_precomputed"] is True
    assert _time_components(_rows(precomputed=False))["selection_is_precomputed"] is False
    mixed = _rows(n=3, precomputed=True)
    mixed[1]["selection_is_precomputed"] = False
    assert _time_components(mixed)["selection_is_precomputed"] is False
    # all([]) is True; an empty run must not be labelled precomputed.
    assert _time_components([])["selection_is_precomputed"] is False


def test_empty_run_emits_nulls_not_zeros():
    """A zero-second column would be a claim; there is nothing to claim."""
    out = _time_components([])
    assert out["selection_time_total_sec"] is None
    assert out["reduced_eval_time_total_sec"] is None
    assert out["time_column_sec"] is None
    assert out["selection_time_n_recorded"] == 0


def test_the_two_spans_are_separate_fields_and_the_column_uses_the_wall_one():
    """The 44x trap.

    `model.Runtime` excludes model construction; the published column timed wall
    clock over build+solve of each reduced instance. Measured on SEL-20-50 k=1:
    Gurobi Runtime summed to 0.093 s where the matched-span wall clock summed to
    1.87 s, against a published 4.7 s. Reporting Runtime under the published field
    name would let a "machine speed coefficient" absorb a 44x definitional gap.
    """
    out = _time_components(_rows(n=4, sel=0.5, red=2.0, gurobi=0.1))
    assert out["reduced_eval_time_total_sec"] == pytest.approx(8.0)      # wall
    assert out["reduced_solve_gurobi_total_sec"] == pytest.approx(0.4)   # Runtime
    # The printed column is selection + WALL, never selection + Runtime.
    assert out["time_column_sec"] == pytest.approx(10.0)
    assert out["time_column_sec"] != pytest.approx(2.4)


def test_a_missing_gurobi_time_does_not_null_the_published_column():
    """The two spans fail independently: losing Runtime must not blank the
    quantity the paper actually prints."""
    rows = _rows(n=3)
    rows[0].pop("reduced_solve_sec")
    out = _time_components(rows)
    assert out["reduced_solve_gurobi_total_sec"] is None
    assert out["reduced_solve_gurobi_n_recorded"] == 2
    assert out["time_column_sec"] is not None
