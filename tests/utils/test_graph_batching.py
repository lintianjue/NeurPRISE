import pytest
import torch
import numpy as np
from typing import List

from src.core.types import Instance
from src.problems.vertex_cover import VertexCoverProblem
from src.utils.graph_batching import build_batched_scenario_graph, ScenarioGraphBatch


# --- Helper to create distinct debug instances ---
def make_simple_instance(det_start: int, sce_start: int) -> Instance:
    """
    Creates a tiny VC instance with predictable values for tracing.
    Nodes: 0-1 (1 edge)
    Scenarios: 2
    """
    N, S = 2, 2
    # 1 edge: (0, 1)
    adj = np.array([[0, 1], [1, 0]], dtype=int)

    # Det costs: [det_start, det_start+1]
    det = np.array([det_start, det_start + 1], dtype=int)

    # Sce costs:
    # S0: [sce_start, sce_start]
    # S1: [sce_start+1, sce_start+1]
    sce = np.array([
        [sce_start, sce_start],
        [sce_start + 1, sce_start + 1]
    ], dtype=int)

    return Instance({
        "n_nodes": np.array([N]),
        "n_scenarios": np.array([S]),
        "n_min_cost": np.array([1]),
        "n_max_cost": np.array([100]),
        "p_edge": np.array([1.0]),
        "det": det,
        "sce": sce,
        "adjacency": adj,
    })


def test_build_batched_scenario_graph_structure():
    """
    Verifies the shape and structural integrity of the batched graph
    with the new 2RO feature set (9-dim nodes, 1-dim edges).
    """
    # 1. Setup
    problem = VertexCoverProblem(n_nodes=2, n_scenarios=2)
    inst1 = make_simple_instance(det_start=10, sce_start=5)
    inst2 = make_simple_instance(det_start=20, sce_start=15)
    instances = [inst1, inst2]
    B, S = 2, 2

    # 2. Batching
    batch = build_batched_scenario_graph(instances, problem, expected_S=S)

    # 3. Assertions
    assert isinstance(batch, ScenarioGraphBatch)
    assert batch.B == B
    assert batch.S == S

    # --- Node Features ---
    # Per Scenario: 2 vars (x) + 2 vars (y) + 1 var (t) + 1 cov + 2 compat + 1 budget = 9 nodes
    nodes_per_scen = 9
    total_nodes = B * S * nodes_per_scen

    assert batch.x.dim() == 2
    assert batch.x.shape[0] == total_nodes
    assert batch.x.shape[1] == 9, "Node features must be 9-dimensional"

    # --- Edge Attributes (Fixed Logic) ---
    # Per Scenario with N=2, 1 Edge:
    # 1. Cover (0-1): 4 vars (x0, y0, x1, y1) connected to 1 constraint.
    #    Bidirectional -> 4 * 2 = 8 edges.
    # 2. Compat (0, 1): 2 constraints. Each has 2 vars (x, y).
    #    Bidirectional -> 2 * (2 * 2) = 8 edges.
    # 3. Budget: 3 vars (y0, y1, t) connected to 1 constraint.
    #    Bidirectional -> 3 * 2 = 6 edges.
    # Total = 8 + 8 + 6 = 22 edges per scenario.

    edges_per_scen = 22
    total_edges = B * S * edges_per_scen

    assert batch.edge_index.shape == (2, total_edges)
    assert batch.edge_attr.dim() == 2
    assert batch.edge_attr.shape == (total_edges, 1), "Edge features must be 1-dimensional [val]"

    # --- Batch Indices ---
    assert batch.scen_batch.shape == (total_nodes,)
    assert batch.inst_batch.shape == (B * S,)

    # Check max indices
    assert batch.scen_batch.max().item() == (B * S) - 1
    assert batch.inst_batch.max().item() == B - 1


def test_batched_edge_attr_integrity():
    """
    Verifies that edge attributes are correctly preserved and concatenated.
    Specifically checks for the 'Budget' constraint weights which vary by instance.
    """
    problem = VertexCoverProblem(n_nodes=2, n_scenarios=2)
    # Inst 1 S0 y-costs: [5, 5]
    inst1 = make_simple_instance(det_start=10, sce_start=5)
    # Inst 2 S0 y-costs: [15, 15]
    inst2 = make_simple_instance(det_start=20, sce_start=15)

    batch = build_batched_scenario_graph([inst1, inst2], problem, expected_S=2)

    # We look for edges with weight 5.0 (from Inst 1) and 15.0 (from Inst 2)
    # These exist in the budget constraint connections.

    weights = batch.edge_attr[:, 0]

    has_wt_5 = (weights == 5.0).any()
    has_wt_15 = (weights == 15.0).any()

    assert has_wt_5, "Batch must contain edge weights from Instance 1"
    assert has_wt_15, "Batch must contain edge weights from Instance 2"

    # Edge weights must preserve per-instance costs.


def test_batched_node_feature_integrity():
    """
    Verifies that node features (specifically obj coeff and cost hints)
    are mapped to the correct instances in the batch.
    """
    problem = VertexCoverProblem(n_nodes=2, n_scenarios=2)
    inst1 = make_simple_instance(det_start=100, sce_start=5)
    inst2 = make_simple_instance(det_start=200, sce_start=5)

    batch = build_batched_scenario_graph([inst1, inst2], problem, expected_S=2)

    # Index 4 is obj_coeff.
    # Inst 1 x-nodes have cost 100. Inst 2 x-nodes have cost 200.
    obj_coeffs = batch.x[:, 4]

    assert (obj_coeffs == 100.0).any(), "Batch must contain Inst 1 det costs"
    assert (obj_coeffs == 200.0).any(), "Batch must contain Inst 2 det costs"

    # Verify that the scenario batch index aligns with these values
    # Find nodes belonging to Global Scenario 0 (Instance 0, Scenario 0)
    # and Global Scenario 2 (Instance 1, Scenario 0)

    nodes_s0 = (batch.scen_batch == 0)
    nodes_s2 = (batch.scen_batch == 2)

    # Check Inst 1 values
    assert (batch.x[nodes_s0, 4] == 100.0).any()
    # Check Inst 2 values
    assert (batch.x[nodes_s2, 4] == 200.0).any()


def test_empty_instance_list_raises_error():
    problem = VertexCoverProblem(n_nodes=2, n_scenarios=2)
    # The batcher expects >0 instances because it does torch.cat
    # Logic in build_batched_scenario_graph raises RuntimeError if no graphs produced

    # Usually passed as empty list
    with pytest.raises(RuntimeError):
        build_batched_scenario_graph([], problem, expected_S=2)


def test_mismatched_scenario_count_raises_error():
    """If problem generates 2 scenarios but we expect 5, should raise."""
    problem = VertexCoverProblem(n_nodes=2, n_scenarios=2)
    inst = make_simple_instance(10, 5)

    with pytest.raises(RuntimeError) as excinfo:
        build_batched_scenario_graph([inst], problem, expected_S=5)
    assert "Expected 5 scenarios, got 2" in str(excinfo.value)
