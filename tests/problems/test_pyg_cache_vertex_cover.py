# tests/test_pyg_cache_vertex_cover.py
import pytest
import numpy as np

from src.problems.vertex_cover import VertexCoverProblem


def test_to_pyg_data_is_cached_per_instance(monkeypatch):
    problem = VertexCoverProblem(n_nodes=6, n_scenarios=3, p_edge=10.0)
    inst = problem.sample()

    calls = {"n": 0}
    orig_build = problem._build_pyg_data

    def wrapped_build(instance):
        calls["n"] += 1
        return orig_build(instance)

    monkeypatch.setattr(problem, "_build_pyg_data", wrapped_build)

    g1 = problem.to_pyg_data(inst)
    g2 = problem.to_pyg_data(inst)

    # Builder should run only once due to cache.
    assert calls["n"] == 1

    # Cached object should be returned (same list identity).
    assert g1 is g2

    # Sanity: should have one Data per scenario
    assert len(g1) == problem.n_scenarios


def test_to_pyg_data_is_cached_per_instance(monkeypatch):
    problem = VertexCoverProblem(n_nodes=6, n_scenarios=3, p_edge=10.0)
    inst = problem.sample()

    calls = {"n": 0}
    orig_build = problem._build_pyg_data

    def wrapped_build(instance):
        calls["n"] += 1
        return orig_build(instance)

    monkeypatch.setattr(problem, "_build_pyg_data", wrapped_build)

    g1 = problem.to_pyg_data(inst)
    g2 = problem.to_pyg_data(inst)

    assert calls["n"] == 1
    assert g1 is g2
    assert len(g1) == problem.n_scenarios


def test_to_pyg_data_cache_is_not_shared_across_instances(monkeypatch):
    problem = VertexCoverProblem(n_nodes=6, n_scenarios=3, p_edge=10.0)
    inst_a = problem.sample()
    inst_b = problem.sample()

    calls = {"n": 0}
    orig_build = problem._build_pyg_data

    def wrapped_build(instance):
        calls["n"] += 1
        return orig_build(instance)

    monkeypatch.setattr(problem, "_build_pyg_data", wrapped_build)

    # First instance: builds once, then cached
    ga1 = problem.to_pyg_data(inst_a)
    ga2 = problem.to_pyg_data(inst_a)

    # Second instance: should build once independently, then cached
    gb1 = problem.to_pyg_data(inst_b)
    gb2 = problem.to_pyg_data(inst_b)

    assert calls["n"] == 2, "Expected one build per instance (2 instances => 2 builds)"

    assert ga1 is ga2
    assert gb1 is gb2

    # Optional sanity: caches shouldn't alias each other
    assert ga1 is not gb1