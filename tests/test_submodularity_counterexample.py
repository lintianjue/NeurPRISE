"""
Unit test: concrete counterexample showing v(R) is NOT submodular.

Problem: a miniature 2RO with two first-stage decisions x ∈ {a, b}
and three scenarios {s1, s2, s3}.

    f(x, s) = first-stage cost + worst-case recourse (combined for simplicity)

    Cost table:
              s1    s2    s3
        a      0    10     6
        b     10     0     6

    v(R) = min_{x ∈ {a,b}}  max_{s ∈ R} f(x, s)

We trace PRISE (greedy worst-case selection) on this instance and
show that the marginal gain of adding s2 *increases* when the
existing set grows from {s3} to {s3, s1}, violating submodularity.
"""
import itertools
import numpy as np
import pytest


# ── Problem definition ────────────────────────────────────────────
COSTS = {
    #       s1   s2   s3
    "a": [  0,  10,   6],
    "b": [ 10,   0,   6],
}
SCENARIOS = [0, 1, 2]           # indices for s1, s2, s3
DECISIONS = list(COSTS.keys())  # ["a", "b"]


def v(R: set) -> tuple:
    """
    Compute v(R) = min_x max_{s in R} f(x, s).

    Returns (value, optimal_decision).
    """
    assert len(R) > 0, "R must be nonempty"
    best_val = float("inf")
    best_x = None
    for x in DECISIONS:
        worst = max(COSTS[x][s] for s in R)
        if worst < best_val:
            best_val = worst
            best_x = x
    return best_val, best_x


def marginal_gain(S: set, e: int) -> float:
    """Δ(S, e) = v(S ∪ {e}) - v(S)."""
    return v(S | {e})[0] - v(S)[0]


# ── Tests ─────────────────────────────────────────────────────────

class TestVFunction:
    """Verify v(R) for every nonempty subset."""

    def test_singletons(self):
        assert v({0}) == (0, "a")   # v({s1}): a→0, b→10 → min=0
        assert v({1}) == (0, "b")   # v({s2}): a→10, b→0  → min=0
        assert v({2}) == (6, "a")   # v({s3}): a→6, b→6   → min=6 (tie)

    def test_pairs(self):
        assert v({0, 1}) == (10, "a")  # a→max(0,10)=10, b→max(10,0)=10 → 10
        assert v({0, 2}) == (6, "a")   # a→max(0,6)=6,  b→max(10,6)=10 → 6
        assert v({1, 2}) == (6, "b")   # a→max(10,6)=10, b→max(0,6)=6  → 6

    def test_full_set(self):
        assert v({0, 1, 2}) == (10, "a")  # a→10, b→10 → 10


class TestSubmodularityViolation:
    """Demonstrate that v(R) is NOT submodular."""

    def test_marginal_gain_increases(self):
        """
        Submodularity requires: S ⊆ T  ⟹  Δ(S, e) ≥ Δ(T, e).

        Here S = {s3}, T = {s1, s3}, e = s2:
          Δ({s3}, s2)       = v({s2,s3}) - v({s3})       = 6 - 6 = 0
          Δ({s1,s3}, s2)    = v({s1,s2,s3}) - v({s1,s3}) = 10 - 6 = 4

        Since 0 < 4, submodularity is VIOLATED.
        """
        S = {2}        # {s3}
        T = {0, 2}     # {s1, s3}
        e = 1          # s2

        delta_S = marginal_gain(S, e)
        delta_T = marginal_gain(T, e)

        assert delta_S == 0.0, f"Δ({{s3}}, s2) should be 0, got {delta_S}"
        assert delta_T == 4.0, f"Δ({{s1,s3}}, s2) should be 4, got {delta_T}"
        assert delta_S < delta_T, (
            f"Submodularity violated: Δ(S,e)={delta_S} < Δ(T,e)={delta_T} "
            f"with S⊆T"
        )


class TestPRISETrace:
    """Simulate PRISE greedy selection and show the non-submodular trace."""

    def test_prise_full_trace(self):
        """
        PRISE greedily picks the candidate maximising v(S_t ∪ {ξ_j}).

        Step 0 (S=∅): evaluate each scenario alone
            v({s1})=0, v({s2})=0, v({s3})=6  →  select s3 (score 6)
            gain = 6

        Step 1 (S={s3}): evaluate adding s1 or s2
            v({s3,s1})=6, v({s3,s2})=6  →  tie; select s1 (by index)
            gain = 6 - 6 = 0   ← scenario s2 has zero marginal gain here

        Step 2 (S={s3,s1}): evaluate adding s2
            v({s3,s1,s2})=10  →  select s2
            gain = 10 - 6 = 4  ← scenario s2 now has gain 4 (INCREASE)

        The gain of s2 jumped from 0 → 4, violating diminishing returns.
        """
        S = set()
        selected = []
        gains = []
        best_scores = []

        for t in range(3):
            # Evaluate each remaining candidate
            candidates = [j for j in SCENARIOS if j not in S]
            scores = {j: v(S | {j})[0] for j in candidates}

            # Select candidate with highest score (break ties by index)
            j_star = max(candidates, key=lambda j: (scores[j], -j))
            score = scores[j_star]

            # Compute marginal gain
            gain = score - (best_scores[-1] if best_scores else 0)

            selected.append(j_star)
            best_scores.append(score)
            gains.append(gain)
            S.add(j_star)

        # Verify PRISE selection order
        assert selected == [2, 0, 1], f"Expected [s3, s1, s2], got {selected}"

        # Verify best scores at each step
        assert best_scores == [6, 6, 10], f"Expected [6, 6, 10], got {best_scores}"

        # Verify gains
        assert gains == [6, 0, 4], f"Expected [6, 0, 4], got {gains}"

        # Key observation: gain of last step (4) > gain of second step (0)
        # This is the non-submodular behavior
        assert gains[2] > gains[1], (
            f"Expected non-diminishing: gain[2]={gains[2]} > gain[1]={gains[1]}"
        )


class TestMonotonicity:
    """v(R) is monotone even though it is not submodular."""

    def test_monotone_all_chains(self):
        """Adding any scenario never decreases v(R)."""
        for size in range(1, 4):
            for R in itertools.combinations(SCENARIOS, size):
                R_set = set(R)
                v_R = v(R_set)[0]
                for e in SCENARIOS:
                    if e not in R_set:
                        v_Re = v(R_set | {e})[0]
                        assert v_Re >= v_R, (
                            f"Monotonicity violated: v({R_set})={v_R} > "
                            f"v({R_set | {{e}}})={v_Re}"
                        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
