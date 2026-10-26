#!/usr/bin/env python3
"""Prove z = np.minimum(z, K*y) cannot underreport CFLP recourse infeasibility.

The projection only decreases z, so capacity only tightens.  Recourse
feasibility is monotone in capacity.  Therefore:
  feasible after projection  => feasible before   (safe)
  infeasible before          => infeasible after   (safe)
  feasible before, infeasible after => ghost capacity removed (correct)
  infeasible before, feasible after => IMPOSSIBLE
"""
import numpy as np
import pytest
import gurobipy as gp
from gurobipy import GRB


def solve_recourse(z, c, demand):
    """Solve CFLP recourse LP.  Returns dict with 'feasible' and 'obj'."""
    I, J = c.shape
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model(env=env)
    x = m.addVars(I, J, lb=0.0, vtype=GRB.CONTINUOUS, name="x")
    m.addConstrs(
        (gp.quicksum(x[i, j] for j in range(J)) <= float(z[i])
         for i in range(I)), name="cap")
    m.addConstrs(
        (gp.quicksum(x[i, j] for i in range(I)) >= float(demand[j])
         for j in range(J)), name="dem")
    m.setObjective(
        gp.quicksum(float(c[i, j]) * x[i, j]
                     for i in range(I) for j in range(J)), GRB.MINIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        return {"feasible": True, "obj": float(m.ObjVal)}
    return {"feasible": False, "obj": np.inf}


def project_z(z, K, y):
    return np.minimum(z, K * y)


def _base_data():
    """2 facilities (K=10 each), 3 customers, unit costs, demand total=9."""
    return (np.array([10.0, 10.0]),
            np.ones((2, 3)),
            np.array([4.0, 3.0, 2.0]))


# -- Test 1: projection is monotone-decreasing --

class TestProjectionOnlyDecreasesZ:
    """min(z, K*y) <= z elementwise for any y in {0,1}, z >= 0."""

    @pytest.mark.parametrize("seed", range(20))
    def test_random(self, seed):
        rng = np.random.default_rng(seed)
        I = rng.integers(2, 6)
        K = rng.uniform(1, 100, size=I)
        y = rng.integers(0, 2, size=I).astype(float)
        z = rng.uniform(0, 120, size=I)
        z_proj = project_z(z, K, y)
        assert np.all(z_proj <= z + 1e-15), "Projection must never increase z"
        assert np.all(z_proj <= K * y + 1e-15), "Projection must respect K*y"


# -- Test 2: feasible(proj) => feasible(raw)  [monotonicity] --

class TestFeasibleWithProjectionImpliesFeasibleWithout:
    """Extra capacity (raw z >= proj z) cannot break feasibility."""

    def test_slack_capacity(self):
        K, c, demand = _base_data()
        y = np.array([1.0, 1.0])
        z_raw = np.array([10.001, 10.002])
        z_proj = project_z(z_raw, K, y)
        assert np.allclose(z_proj, [10.0, 10.0])
        res_proj = solve_recourse(z_proj, c, demand)
        res_raw = solve_recourse(z_raw, c, demand)
        assert res_proj["feasible"], "cap=20 >= demand=9"
        assert res_raw["feasible"], "z_raw >= z_proj so must also be feasible"
        assert res_raw["obj"] <= res_proj["obj"] + 1e-6

    def test_tight_capacity(self):
        """Demand exactly equals projected capacity."""
        K = np.array([5.0, 4.0])
        c = np.ones((2, 2))
        demand = np.array([5.0, 4.0])  # total = sum(K) = 9
        y = np.array([1.0, 1.0])
        z_raw = np.array([5.003, 4.001])
        z_proj = project_z(z_raw, K, y)
        assert solve_recourse(z_proj, c, demand)["feasible"]
        assert solve_recourse(z_raw, c, demand)["feasible"]


# -- Test 3: ghost capacity (y=0, z>0) correctly removed --

class TestGhostCapacityCorrectlyRemoved:
    """y[i]=0 but z[i]>0 is a solver tolerance artifact.  Projection zeros it
    out, which may flip feasible->infeasible (correct behavior).
    """

    def test_large_ghost_flips_feasibility(self):
        """Fac 0 closed but z=5 (ghost).  Fac 1 open, z=4.  Total demand=9.
        Raw: 5+4=9 >= 9 feasible.  Projected: 0+4=4 < 9 infeasible."""
        K, c, demand = _base_data()
        y = np.array([0.0, 1.0])
        z_raw = np.array([5.0, 4.0])
        res_raw = solve_recourse(z_raw, c, demand)
        assert res_raw["feasible"], "Ghost capacity makes raw z feasible"
        z_proj = project_z(z_raw, K, y)
        assert z_proj[0] == 0.0
        res_proj = solve_recourse(z_proj, c, demand)
        assert not res_proj["feasible"], "Projection removes ghost => infeasible"

    def test_tiny_ghost_from_mip_tolerance(self):
        """y=0 everywhere, z=0.001 leak satisfies micro-demand without proj."""
        K = np.array([10.0, 10.0])
        c = np.ones((2, 2))
        demand = np.array([0.0005, 0.0004])
        y = np.array([0.0, 0.0])
        z_raw = np.array([0.001, 0.001])
        assert solve_recourse(z_raw, c, demand)["feasible"]
        z_proj = project_z(z_raw, K, y)
        assert np.allclose(z_proj, 0.0)
        assert not solve_recourse(z_proj, c, demand)["feasible"]


# -- Test 4: infeasible(raw) => infeasible(proj) --

class TestInfeasibilityPreservedAfterProjection:
    """If demand > capacity even before projection, still infeasible after."""

    def test_massive_demand(self):
        K, c, _ = _base_data()
        demand = np.array([10.0, 10.0, 10.0])  # 30 >> K.sum()=20
        y = np.array([1.0, 1.0])
        z_raw = np.array([10.0, 10.0])
        assert not solve_recourse(z_raw, c, demand)["feasible"]
        z_proj = project_z(z_raw, K, y)
        assert not solve_recourse(z_proj, c, demand)["feasible"]

    def test_partial_open(self):
        """One facility closed, open capacity < demand."""
        K = np.array([6.0, 6.0, 6.0])
        c = np.ones((3, 2))
        demand = np.array([8.0, 5.0])  # 13 > 12 open capacity
        y = np.array([1.0, 0.0, 1.0])
        z_raw = np.array([6.0, 0.0, 6.0])
        assert not solve_recourse(z_raw, c, demand)["feasible"]
        z_proj = project_z(z_raw, K, y)
        assert not solve_recourse(z_proj, c, demand)["feasible"]


# -- Test 5: tolerance at open facility (y=1, z=K+eps) --

class TestToleranceAtOpenFacility:
    """z[i]=K[i]+0.001 with y[i]=1.  Clamped to K[i]; negligible difference."""

    def test_clamp_preserves_feasibility(self):
        K, c, demand = _base_data()
        y = np.array([1.0, 1.0])
        z_raw = np.array([10.001, 10.001])
        z_proj = project_z(z_raw, K, y)
        assert np.allclose(z_proj, K)
        res_raw = solve_recourse(z_raw, c, demand)
        res_proj = solve_recourse(z_proj, c, demand)
        assert res_raw["feasible"] == res_proj["feasible"]
        assert abs(res_raw["obj"] - res_proj["obj"]) < 1e-4

    def test_tight_boundary(self):
        """Demand = K exactly; z=K+eps and z=K both feasible."""
        K = np.array([5.0, 4.0])
        c = np.ones((2, 2))
        demand = np.array([5.0, 4.0])
        y = np.array([1.0, 1.0])
        z_raw = np.array([5.001, 4.001])
        z_proj = project_z(z_raw, K, y)
        assert solve_recourse(z_proj, c, demand)["feasible"]
        assert solve_recourse(z_raw, c, demand)["feasible"]
