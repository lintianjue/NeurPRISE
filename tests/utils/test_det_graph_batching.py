import pytest
import torch
import numpy as np
from typing import List

from src.core.types import Instance, Labels
from src.problems.vertex_cover import VertexCoverProblem
from src.utils.graph_batching import build_batched_det_graph, DetGraphBatch


# --- Helper to create distinct debug instances ---
def make_vc_debug_instance(det_costs: List[int], adjacency: np.ndarray) -> Instance:
    """
    Creates a VC instance with specific deterministic costs and adjacency.
    """
    N = len(det_costs)
    S = 2  # Dummy scenario count
    det = np.array(det_costs, dtype=int)
    sce = np.zeros((S, N), dtype=int)  # Dummy sce

    return Instance({
        "n_nodes": np.array([N]),
        "n_scenarios": np.array([S]),
        "n_min_cost": np.array([1]),
        "n_max_cost": np.array([100]),
        "p_edge": np.array([1.0]),
        "det": det,
        "sce": sce,
        "adjacency": adjacency,
    })


def test_det_graph_batching_structure_and_values():
    """
    Verifies build_batched_det_graph correctly batches
    structural features (x, edge_index) and physics values (obj_coeff, edge_attr).
    """
    # ---------------------------------------------------------
    # 1. Setup: Create 2 distinct instances
    # ---------------------------------------------------------
    problem = VertexCoverProblem(n_nodes=2, n_scenarios=2)

    # Instance 0: N=2, Edge 0-1. Costs [10, 20]
    # Logic: -x0 - y0 - x1 - y1 <= -1
    adj0 = np.array([[0, 1], [1, 0]], dtype=int)
    inst0 = make_vc_debug_instance([10, 20], adj0)

    # Instance 1: N=2, NO Edges. Costs [55, 66]
    # Logic: Only compat constraints x+y<=1. No cover constraints.
    adj1 = np.array([[0, 0], [0, 0]], dtype=int)
    inst1 = make_vc_debug_instance([55, 66], adj1)

    instances = [inst0, inst1]

    # ---------------------------------------------------------
    # 2. Run Batching
    # ---------------------------------------------------------
    batch_out = build_batched_det_graph(instances, problem)

    assert isinstance(batch_out, DetGraphBatch)
    assert batch_out.B == 2

    # ---------------------------------------------------------
    # 3. Verify Node Features (x) and Objective Mapping
    # ---------------------------------------------------------
    # Expected Node Count per Instance (WITH t variable):
    # Inst 0:
    #   Vars: 2(x) + 2(y) + 1(t) = 5 vars.
    #   Constraints: 1(Edge) + 2(Compat) = 3 constraints.
    #   Total = 8 nodes.
    #
    # Inst 1:
    #   Vars: 2(x) + 2(y) + 1(t) = 5 vars.
    #   Constraints: 0(Edge) + 2(Compat) = 2 constraints.
    #   Total = 7 nodes.

    nodes_inst0 = 8
    nodes_inst1 = 7
    total_nodes = nodes_inst0 + nodes_inst1  # 15

    assert batch_out.x.shape[0] == total_nodes
    assert batch_out.batch.shape[0] == total_nodes

    # Verify Batch ID mapping
    # First 8 nodes should be Inst 0
    assert (batch_out.batch[:nodes_inst0] == 0).all()
    # Next 7 nodes should be Inst 1
    assert (batch_out.batch[nodes_inst0:] == 1).all()

    # Verify Objective Coefficients
    # In the 9-dim feature set, obj_coeff is at Index 4 (not 5).
    # Inst 0 (First 2 nodes are x0, x1): [10, 20]
    obj_coeffs = batch_out.x[:, 4]  # <--- FIXED INDEX (Was 5)

    assert obj_coeffs[0].item() == 10.0
    assert obj_coeffs[1].item() == 20.0

    # Inst 1 (Starts at offset 8. First 2 nodes x0, x1): [55, 66]
    inst1_start = nodes_inst0
    assert obj_coeffs[inst1_start].item() == 55.0
    assert obj_coeffs[inst1_start + 1].item() == 66.0

    # ---------------------------------------------------------
    # 4. Verify Edge Attributes (edge_attr)
    # ---------------------------------------------------------
    # Inst 0 (Has Edge): Expect -1 coeffs (Cover) and +1 coeffs (Compat)
    # Inst 1 (No Edge): Expect ONLY +1 coeffs (Compat)

    # Identify edges belonging to Inst 0 nodes vs Inst 1 nodes
    src_nodes = batch_out.edge_index[0]
    inst_ids_of_edges = batch_out.batch[src_nodes]

    mask_inst0 = (inst_ids_of_edges == 0)
    mask_inst1 = (inst_ids_of_edges == 1)

    edges_inst0 = batch_out.edge_attr[mask_inst0]
    edges_inst1 = batch_out.edge_attr[mask_inst1]

    # Inst 0 Check: Must have -1.0 (from cover constraint)
    vals0 = edges_inst0[:, 0]
    assert (vals0 == -1.0).any(), "Instance 0 must have cover constraints (-1.0)"
    assert (vals0 == 1.0).any(), "Instance 0 must have compat constraints (1.0)"

    # Inst 1 Check: Must NOT have -1.0 (No edges in adjacency)
    vals1 = edges_inst1[:, 0]
    assert (vals1 == 1.0).all(), "Instance 1 (no edges) should only have compat constraints (1.0)"
    assert not (vals1 == -1.0).any(), "Instance 1 should not have cover constraints"

    # ---------------------------------------------------------
    # 5. Verify Edge Index Offsets
    # ---------------------------------------------------------
    # Edges for Inst 1 must reference node indices starting from nodes_inst0 (8)
    edge_index_inst1 = batch_out.edge_index[:, mask_inst1]
    assert edge_index_inst1.min() >= nodes_inst0, "Instance 1 edges must use offset node indices"


def test_collate_creates_det_graph():
    # 1. Setup Problem
    N, S = 10, 5
    problem = VertexCoverProblem(n_nodes=N, n_scenarios=S)

    # 2. Create Dummy Batch (List of Dicts)
    batch_size = 4
    raw_batch = []
    for _ in range(batch_size):
        inst = problem.sample()
        # Dummy label: avoid solver dependency for a batching-only test.
        raw_batch.append({"instance": inst, "label": Labels()})

    # 3. Run Collate
    collated = problem.collate_fn(raw_batch)
    inst_batched = collated["instance"]

    # 4. Assertions
    # Check if 'det_graph' key exists
    assert "det_graph" in inst_batched.data, "collate_fn must populate 'det_graph'"

    det_data = inst_batched.data["det_graph"]
    required_keys = ["x", "edge_index", "edge_attr", "batch"]
    for k in required_keys:
        assert k in det_data, f"det_graph missing key: {k}"

    # 5. Check Dimensions
    # Nodes in det graph = (2N + 1 constraints + vars) roughly per instance
    # Just check it scales with B
    x = det_data["x"]
    batch_idx = det_data["batch"]

    assert isinstance(x, torch.Tensor)
    assert batch_idx.max().item() == batch_size - 1
    assert x.size(0) == batch_idx.size(0)

    # Check edge_attr is 2D (NEW requirement)
    edge_attr = det_data["edge_attr"]
    assert edge_attr.dim() == 2, f"edge_attr should be 2D, got {edge_attr.shape}"
    assert edge_attr.size(1) == problem._edge_attr_dim(), (
        f"edge_attr should have {problem._edge_attr_dim()} features, got {edge_attr.shape}"
    )

    print("✅ test_collate_creates_det_graph passed!")
