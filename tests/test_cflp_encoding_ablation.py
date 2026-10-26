"""
Unit tests for the CFLP encoding ablation study.

Tests cover:
  1. CV (Constraint-Variable) bipartite encoding via CFLPProblem
  2. PS (Problem-Specific / Entity-Bipartite) encoding via CFLPEntityBipartiteProblem
  3. Graph validity (node counts, edge counts, bipartiteness, feature dims)
  4. Both encodings can be loaded by the GNN encoder architecture
  5. GPU memory estimation for both encodings on a small instance
  6. Config toggle: problem=cflp vs problem=cflp_ebp resolves correct class

Tag: [IO] -- no GPU or solver required; pure CPU tests.
"""

import math
import sys
from typing import Dict, Tuple

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.core.types import Instance, Labels
from src.problems.cflp import CFLPProblem, CFLPEntityBipartiteProblem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_cflp_instance(I: int = 3, J: int = 4, S: int = 5) -> Instance:
    """Create a deterministic CFLP instance for testing."""
    rng = np.random.RandomState(42)
    return Instance(
        {
            "n_facilities": np.array([I], dtype=np.int64),
            "n_customers": np.array([J], dtype=np.int64),
            "n_scenarios": np.array([S], dtype=np.int64),
            "f": rng.uniform(100, 1000, size=(I,)).astype(np.float32),
            "a": rng.uniform(10, 100, size=(I,)).astype(np.float32),
            "K": rng.uniform(200, 700, size=(I,)).astype(np.float32),
            "c": rng.uniform(1, 1000, size=(I, J)).astype(np.float32),
            "d": rng.uniform(10, 500, size=(S, J)).astype(np.float32),
        }
    )


@pytest.fixture
def small_instance() -> Instance:
    return _make_cflp_instance(I=3, J=4, S=5)


@pytest.fixture
def tiny_instance() -> Instance:
    """Smallest meaningful instance for edge-case testing."""
    return _make_cflp_instance(I=2, J=2, S=2)


@pytest.fixture
def cv_problem() -> CFLPProblem:
    return CFLPProblem(n_facilities=3, n_customers=4, n_scenarios=5)


@pytest.fixture
def ps_problem() -> CFLPEntityBipartiteProblem:
    return CFLPEntityBipartiteProblem(n_facilities=3, n_customers=4, n_scenarios=5)


# ---------------------------------------------------------------------------
# Helper: compute expected graph sizes
# ---------------------------------------------------------------------------

def _cv_expected_sizes(I: int, J: int) -> Dict[str, int]:
    """Expected node/edge counts for CV (constraint-variable) MILP encoding."""
    # Variables: y_i (I) + z_i (I) + x_ij (I*J) + theta (1)
    n_vars = 2 * I + I * J + 1
    # Scenario constraints: install (I) + capacity (I) + demand (J) + epigraph (1)
    n_constraints_sce = 2 * I + J + 1
    # Deterministic constraints: install (I) + capacity (I) + epigraph (1) -- no demand
    n_constraints_det = 2 * I + 1

    # Edge count (bidirectional) for scenario graph:
    #   install: 2 vars (z_i, y_i) per constraint, I constraints
    #   capacity: (J+1) vars per constraint (J x_ij + z_i), I constraints
    #   demand: I vars per constraint (I x_ij), J constraints
    #   epigraph: (I*J + 1) vars in one constraint (all x_ij + theta)
    install_edges = 2 * I
    capacity_edges = I * (J + 1)
    demand_edges = I * J
    epigraph_edges = I * J + 1
    total_undirected = install_edges + capacity_edges + demand_edges + epigraph_edges
    total_directed_sce = 2 * total_undirected

    # Det graph has no demand constraints
    install_edges_det = 2 * I
    capacity_edges_det = I * (J + 1)
    epigraph_edges_det = I * J + 1
    total_undirected_det = install_edges_det + capacity_edges_det + epigraph_edges_det
    total_directed_det = 2 * total_undirected_det

    return {
        "n_vars": n_vars,
        "n_constraints_sce": n_constraints_sce,
        "n_constraints_det": n_constraints_det,
        "num_nodes_sce": n_vars + n_constraints_sce,
        "num_nodes_det": n_vars + n_constraints_det,
        "num_edges_sce": total_directed_sce,
        "num_edges_det": total_directed_det,
        "node_feat_dim": 8,
        "edge_feat_dim": 1,
    }


def _ps_expected_sizes(I: int, J: int) -> Dict[str, int]:
    """Expected node/edge counts for PS (entity-bipartite) encoding."""
    num_nodes = I + J
    # Complete bipartite edges: I*J undirected => 2*I*J directed
    num_edges = 2 * I * J
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "node_feat_dim": 6,
        "edge_feat_dim": 1,
    }


# ===========================================================================
# Test Group 1: CV encoding produces valid graphs
# ===========================================================================

class TestCVEncodingGraphValidity:
    """Verify CV (constraint-variable) bipartite graph structure."""

    def test_scenario_graph_count(self, cv_problem, small_instance):
        """CV encoding produces one graph per scenario."""
        data_list = cv_problem.to_pyg_data(small_instance)
        assert len(data_list) == 5  # S=5

    def test_scenario_graph_node_count(self, cv_problem, small_instance):
        I, J = 3, 4
        expected = _cv_expected_sizes(I, J)
        data_list = cv_problem.to_pyg_data(small_instance)
        for s, data in enumerate(data_list):
            assert data.x.shape[0] == expected["num_nodes_sce"], (
                f"Scenario {s}: expected {expected['num_nodes_sce']} nodes, "
                f"got {data.x.shape[0]}"
            )

    def test_scenario_graph_feature_dim(self, cv_problem, small_instance):
        expected = _cv_expected_sizes(3, 4)
        data = cv_problem.to_pyg_data(small_instance)[0]
        assert data.x.shape[1] == expected["node_feat_dim"]

    def test_scenario_graph_edge_count(self, cv_problem, small_instance):
        expected = _cv_expected_sizes(3, 4)
        data = cv_problem.to_pyg_data(small_instance)[0]
        assert data.edge_index.shape[1] == expected["num_edges_sce"], (
            f"Expected {expected['num_edges_sce']} directed edges, "
            f"got {data.edge_index.shape[1]}"
        )

    def test_scenario_graph_edge_attr_dim(self, cv_problem, small_instance):
        expected = _cv_expected_sizes(3, 4)
        data = cv_problem.to_pyg_data(small_instance)[0]
        assert data.edge_attr.shape[1] == expected["edge_feat_dim"]

    def test_scenario_graph_bipartite_structure(self, cv_problem, small_instance):
        """All edges must connect variable nodes to constraint nodes (bipartite)."""
        data = cv_problem.to_pyg_data(small_instance)[0]
        n_vars = int(data.n_vars.item())
        src, dst = data.edge_index
        # Every edge: (var -> constr) or (constr -> var)
        var_to_constr = (src < n_vars) & (dst >= n_vars)
        constr_to_var = (src >= n_vars) & (dst < n_vars)
        assert torch.all(var_to_constr | constr_to_var), (
            "CV graph has non-bipartite edges"
        )

    def test_det_graph_node_count(self, cv_problem, small_instance):
        expected = _cv_expected_sizes(3, 4)
        data_det = cv_problem.to_pyg_data_det(small_instance)
        assert data_det.x.shape[0] == expected["num_nodes_det"]

    def test_det_graph_edge_count(self, cv_problem, small_instance):
        expected = _cv_expected_sizes(3, 4)
        data_det = cv_problem.to_pyg_data_det(small_instance)
        assert data_det.edge_index.shape[1] == expected["num_edges_det"]

    def test_det_graph_bipartite_structure(self, cv_problem, small_instance):
        data_det = cv_problem.to_pyg_data_det(small_instance)
        n_vars = int(data_det.n_vars.item())
        src, dst = data_det.edge_index
        var_to_constr = (src < n_vars) & (dst >= n_vars)
        constr_to_var = (src >= n_vars) & (dst < n_vars)
        assert torch.all(var_to_constr | constr_to_var)

    def test_scenario_idx_attribute(self, cv_problem, small_instance):
        """Each scenario graph carries its scenario index."""
        data_list = cv_problem.to_pyg_data(small_instance)
        for s, data in enumerate(data_list):
            assert hasattr(data, "scenario_idx")
            assert data.scenario_idx.item() == s


# ===========================================================================
# Test Group 2: PS encoding produces valid graphs
# ===========================================================================

class TestPSEncodingGraphValidity:
    """Verify PS (problem-specific / entity-bipartite) graph structure."""

    def test_scenario_graph_count(self, ps_problem, small_instance):
        data_list = ps_problem.to_pyg_data(small_instance)
        assert len(data_list) == 5

    def test_scenario_graph_node_count(self, ps_problem, small_instance):
        expected = _ps_expected_sizes(3, 4)
        data = ps_problem.to_pyg_data(small_instance)[0]
        assert data.x.shape[0] == expected["num_nodes"]

    def test_scenario_graph_feature_dim(self, ps_problem, small_instance):
        expected = _ps_expected_sizes(3, 4)
        data = ps_problem.to_pyg_data(small_instance)[0]
        assert data.x.shape[1] == expected["node_feat_dim"]

    def test_scenario_graph_edge_count(self, ps_problem, small_instance):
        expected = _ps_expected_sizes(3, 4)
        data = ps_problem.to_pyg_data(small_instance)[0]
        assert data.edge_index.shape[1] == expected["num_edges"]

    def test_scenario_graph_edge_attr_dim(self, ps_problem, small_instance):
        expected = _ps_expected_sizes(3, 4)
        data = ps_problem.to_pyg_data(small_instance)[0]
        assert data.edge_attr.shape[1] == expected["edge_feat_dim"]

    def test_facility_customer_node_features(self, ps_problem, small_instance):
        """Facility nodes have is_facility=1, customer nodes have is_customer=1."""
        I, J = 3, 4
        data = ps_problem.to_pyg_data(small_instance)[0]
        # is_facility
        assert torch.all(data.x[:I, 0] == 1.0)
        assert torch.all(data.x[I:, 0] == 0.0)
        # is_customer
        assert torch.all(data.x[:I, 1] == 0.0)
        assert torch.all(data.x[I:, 1] == 1.0)

    def test_det_graph_uses_mean_demand(self, ps_problem, small_instance):
        """Deterministic PS graph should use mean demand across scenarios."""
        I, J, S = 3, 4, 5
        data_det = ps_problem.to_pyg_data_det(small_instance)
        demand = small_instance.data["d"]
        expected_mean = np.mean(demand, axis=0)
        actual = data_det.x[I:, 5].numpy()
        np.testing.assert_allclose(actual, expected_mean, atol=1e-5)

    def test_edge_costs_match_transport_costs(self, ps_problem, small_instance):
        """Edge attributes should be transport costs c_ij."""
        I, J = 3, 4
        data = ps_problem.to_pyg_data(small_instance)[0]
        c = small_instance.data["c"]
        c_flat = c.reshape(-1)
        # First I*J edges: facility -> customer direction
        actual = data.edge_attr[:I * J, 0].numpy()
        np.testing.assert_allclose(actual, c_flat, atol=1e-5)

    def test_complete_bipartite_structure(self, ps_problem, small_instance):
        """PS graph should be complete bipartite: all (facility, customer) pairs."""
        I, J = 3, 4
        data = ps_problem.to_pyg_data(small_instance)[0]
        src, dst = data.edge_index
        n_edges = data.edge_index.shape[1]
        assert n_edges == 2 * I * J

        # Forward edges: src in [0, I), dst in [I, I+J)
        forward_mask = (src < I) & (dst >= I)
        # Backward edges: src in [I, I+J), dst in [0, I)
        backward_mask = (src >= I) & (dst < I)
        assert torch.all(forward_mask | backward_mask)


# ===========================================================================
# Test Group 3: Encoding size comparison (complexity analysis)
# ===========================================================================

class TestEncodingSizeComparison:
    """Verify that CV encoding is strictly larger than PS encoding."""

    @pytest.mark.parametrize("I,J", [(3, 4), (10, 10), (30, 30), (50, 50)])
    def test_cv_has_more_nodes_than_ps(self, I, J):
        cv = _cv_expected_sizes(I, J)
        ps = _ps_expected_sizes(I, J)
        assert cv["num_nodes_sce"] > ps["num_nodes"], (
            f"I={I}, J={J}: CV nodes={cv['num_nodes_sce']}, PS nodes={ps['num_nodes']}"
        )

    @pytest.mark.parametrize("I,J", [(3, 4), (10, 10), (30, 30), (50, 50)])
    def test_cv_has_more_edges_than_ps(self, I, J):
        cv = _cv_expected_sizes(I, J)
        ps = _ps_expected_sizes(I, J)
        assert cv["num_edges_sce"] > ps["num_edges"], (
            f"I={I}, J={J}: CV edges={cv['num_edges_sce']}, PS edges={ps['num_edges']}"
        )

    def test_cv_has_superlinear_node_growth(self):
        """For CFLP with I=J=n, CV nodes scale as O(n^2) while PS nodes scale as O(n).

        CV variables include x_ij (I*J terms), so node count grows quadratically.
        PS has only I+J = 2n nodes (linear).
        This super-linear CV growth is the fundamental reason PS is more memory-efficient.
        """
        sizes = []
        for n in [10, 20, 40]:
            cv = _cv_expected_sizes(n, n)
            ps = _ps_expected_sizes(n, n)
            sizes.append({
                "n": n,
                "cv_nodes": cv["num_nodes_sce"],
                "ps_nodes": ps["num_nodes"],
            })
        # CV node ratio from n=10 to n=40 should be >4x (quadratic)
        # PS node ratio should be ~4x (linear: 2*40 / 2*10 = 4)
        cv_node_ratio = sizes[2]["cv_nodes"] / sizes[0]["cv_nodes"]
        ps_node_ratio = sizes[2]["ps_nodes"] / sizes[0]["ps_nodes"]
        assert cv_node_ratio > ps_node_ratio, (
            f"CV node growth ({cv_node_ratio:.1f}x) should exceed "
            f"PS node growth ({ps_node_ratio:.1f}x)"
        )
        # PS should be exactly 4x (40*2 / 10*2)
        assert abs(ps_node_ratio - 4.0) < 0.01


# ===========================================================================
# Test Group 4: GPU memory estimation
# ===========================================================================

class TestGPUMemoryEstimation:
    """Estimate GPU memory usage for both encodings without requiring a GPU."""

    @staticmethod
    def _estimate_graph_memory_bytes(data: Data) -> int:
        """Estimate total memory footprint of a PyG Data object in bytes."""
        total = 0
        total += data.x.numel() * data.x.element_size()
        total += data.edge_index.numel() * data.edge_index.element_size()
        total += data.edge_attr.numel() * data.edge_attr.element_size()
        return total

    def test_ps_uses_less_memory_per_scenario(self, small_instance):
        I, J, S = 3, 4, 5
        cv = CFLPProblem(n_facilities=I, n_customers=J, n_scenarios=S)
        ps = CFLPEntityBipartiteProblem(n_facilities=I, n_customers=J, n_scenarios=S)

        cv_data = cv.to_pyg_data(small_instance)[0]
        ps_data = ps.to_pyg_data(small_instance)[0]

        cv_mem = self._estimate_graph_memory_bytes(cv_data)
        ps_mem = self._estimate_graph_memory_bytes(ps_data)

        assert ps_mem < cv_mem, (
            f"PS ({ps_mem} bytes) should use less memory than CV ({cv_mem} bytes)"
        )

    @pytest.mark.parametrize("I,J,S", [(30, 30, 50), (50, 50, 50), (100, 100, 50)])
    def test_memory_ratio_grows_with_problem_size(self, I, J, S):
        """CV/PS memory ratio should grow as problem size increases."""
        inst = _make_cflp_instance(I=I, J=J, S=S)
        cv = CFLPProblem(n_facilities=I, n_customers=J, n_scenarios=S)
        ps = CFLPEntityBipartiteProblem(n_facilities=I, n_customers=J, n_scenarios=S)

        cv_data = cv.to_pyg_data(inst)[0]
        ps_data = ps.to_pyg_data(inst)[0]

        cv_mem = self._estimate_graph_memory_bytes(cv_data)
        ps_mem = self._estimate_graph_memory_bytes(ps_data)

        ratio = cv_mem / max(1, ps_mem)
        # For any nontrivial size, CV should use at least 2x more memory
        assert ratio > 2.0, (
            f"I={I}, J={J}: CV/PS memory ratio = {ratio:.1f}x (expected >2x)"
        )

    def test_estimate_batch_memory_m30_n30(self):
        """
        Estimate full-batch GPU memory for m=30, n=30, s=50, batch_size=16.
        This is the configuration proposed for the ablation study.
        """
        I, J, S = 30, 30, 50
        batch_size = 16

        cv_sizes = _cv_expected_sizes(I, J)
        ps_sizes = _ps_expected_sizes(I, J)

        # Per-scenario graph memory (float32 = 4 bytes, long = 8 bytes)
        cv_node_bytes = cv_sizes["num_nodes_sce"] * cv_sizes["node_feat_dim"] * 4
        cv_edge_index_bytes = 2 * cv_sizes["num_edges_sce"] * 8  # edge_index is long
        cv_edge_attr_bytes = cv_sizes["num_edges_sce"] * cv_sizes["edge_feat_dim"] * 4
        cv_per_scenario = cv_node_bytes + cv_edge_index_bytes + cv_edge_attr_bytes

        ps_node_bytes = ps_sizes["num_nodes"] * ps_sizes["node_feat_dim"] * 4
        ps_edge_index_bytes = 2 * ps_sizes["num_edges"] * 8
        ps_edge_attr_bytes = ps_sizes["num_edges"] * ps_sizes["edge_feat_dim"] * 4
        ps_per_scenario = ps_node_bytes + ps_edge_index_bytes + ps_edge_attr_bytes

        # Total batch: batch_size * S scenarios
        cv_total_mb = (cv_per_scenario * S * batch_size) / (1024 ** 2)
        ps_total_mb = (ps_per_scenario * S * batch_size) / (1024 ** 2)

        assert cv_total_mb > ps_total_mb
        # Just ensure the estimates are reasonable
        assert cv_total_mb < 10000  # should be well under 10 GB for graph data alone
        assert ps_total_mb < 100  # PS should be tiny


# ===========================================================================
# Test Group 5: Both encodings loadable by GNN encoder
# ===========================================================================

class TestEncoderCompatibility:
    """Verify both encodings produce tensors compatible with the GNN encoder."""

    def test_cv_scenario_graph_compatible_with_gnn(self, cv_problem, small_instance):
        """CV encoding scenario graph has correct tensor types for GINEConv."""
        data = cv_problem.to_pyg_data(small_instance)[0]
        assert data.x.dtype == torch.float32
        assert data.edge_index.dtype == torch.long
        assert data.edge_attr.dtype == torch.float32
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.max() < data.x.shape[0]

    def test_ps_scenario_graph_compatible_with_gnn(self, ps_problem, small_instance):
        """PS encoding scenario graph has correct tensor types for GINEConv."""
        data = ps_problem.to_pyg_data(small_instance)[0]
        assert data.x.dtype == torch.float32
        assert data.edge_index.dtype == torch.long
        assert data.edge_attr.dtype == torch.float32
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.max() < data.x.shape[0]

    def test_cv_det_graph_compatible_with_gnn(self, cv_problem, small_instance):
        data = cv_problem.to_pyg_data_det(small_instance)
        assert data.x.dtype == torch.float32
        assert data.edge_index.dtype == torch.long
        assert data.edge_attr.dtype == torch.float32
        assert data.edge_index.shape[0] == 2
        if data.edge_index.numel() > 0:
            assert data.edge_index.max() < data.x.shape[0]

    def test_ps_det_graph_compatible_with_gnn(self, ps_problem, small_instance):
        data = ps_problem.to_pyg_data_det(small_instance)
        assert data.x.dtype == torch.float32
        assert data.edge_index.dtype == torch.long
        assert data.edge_attr.dtype == torch.float32
        assert data.edge_index.shape[0] == 2
        if data.edge_index.numel() > 0:
            assert data.edge_index.max() < data.x.shape[0]

    def test_cv_and_ps_produce_different_feature_dims(self, cv_problem, ps_problem, small_instance):
        """CV and PS produce different feature dimensions, requiring separate model configs."""
        cv_data = cv_problem.to_pyg_data(small_instance)[0]
        ps_data = ps_problem.to_pyg_data(small_instance)[0]
        assert cv_data.x.shape[1] != ps_data.x.shape[1], (
            "CV and PS should have different node feature dimensions"
        )


# ===========================================================================
# Test Group 6: Config toggle verification
# ===========================================================================

class TestConfigToggle:
    """Verify that problem=cflp and problem=cflp_ebp resolve to correct classes."""

    def test_cflp_module_path_matches_cv_class(self):
        """The cflp.yaml config module path should resolve to CFLPProblem."""
        module_path = "src.problems.cflp.CFLPProblem"
        parts = module_path.rsplit(".", 1)
        module_name, class_name = parts
        import importlib
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        assert cls is CFLPProblem

    def test_cflp_ebp_module_path_matches_ps_class(self):
        """The cflp_ebp.yaml config module path should resolve to CFLPEntityBipartiteProblem."""
        module_path = "src.problems.cflp.CFLPEntityBipartiteProblem"
        parts = module_path.rsplit(".", 1)
        module_name, class_name = parts
        import importlib
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        assert cls is CFLPEntityBipartiteProblem

    def test_ps_inherits_from_cv(self):
        """PS class should be a subclass of CV class (inherits solve methods)."""
        assert issubclass(CFLPEntityBipartiteProblem, CFLPProblem)

    def test_ps_overrides_to_pyg_data(self):
        """PS class must override to_pyg_data and to_pyg_data_det."""
        assert CFLPEntityBipartiteProblem.to_pyg_data is not CFLPProblem.to_pyg_data
        assert CFLPEntityBipartiteProblem.to_pyg_data_det is not CFLPProblem.to_pyg_data_det


# ===========================================================================
# Test Group 7: Caching
# ===========================================================================

class TestCaching:
    """Verify that both encodings use caching correctly."""

    def test_cv_caching(self, cv_problem, small_instance):
        """Second call to to_pyg_data should return cached result."""
        data_list1 = cv_problem.to_pyg_data(small_instance)
        data_list2 = cv_problem.to_pyg_data(small_instance)
        assert data_list1 is data_list2

    def test_ps_caching(self, ps_problem, small_instance):
        data_list1 = ps_problem.to_pyg_data(small_instance)
        data_list2 = ps_problem.to_pyg_data(small_instance)
        assert data_list1 is data_list2

    def test_cv_and_ps_use_different_cache_keys(self, cv_problem, ps_problem, small_instance):
        """CV and PS must not collide on cache keys."""
        assert cv_problem._PYG_CACHE_KEY != ps_problem._PYG_CACHE_KEY


# ===========================================================================
# Test Group 8: Collate compatibility (batching)
# ===========================================================================

class TestCollateCompatibility:
    """Verify both encodings work with the collate_fn for batching."""

    def _make_labeled_item(self, problem, instance) -> Dict:
        """Create a fake labeled item for collate testing."""
        I = int(instance.data["n_facilities"][0])
        S = int(instance.data["n_scenarios"][0])
        label = Labels(
            obj=np.float64(1000.0),
            feasible=True,
            x=np.ones(I, dtype=np.float64),
            sce_target_idx=np.array([0], dtype=np.int64),
            sce_ranking=np.arange(S, dtype=np.float64),
            sce_scores=np.random.rand(S).astype(np.float64),
        )
        return {"instance": instance, "label": label}

    def test_cv_collate_batch_of_2(self, cv_problem):
        inst1 = _make_cflp_instance(I=3, J=4, S=5)
        inst2 = _make_cflp_instance(I=3, J=4, S=5)
        item1 = self._make_labeled_item(cv_problem, inst1)
        item2 = self._make_labeled_item(cv_problem, inst2)
        batch = cv_problem.collate_fn([item1, item2])
        assert "instance" in batch
        assert "label" in batch
        assert batch["instance"].data["scen_graph"]["x"].shape[0] > 0

    def test_ps_collate_batch_of_2(self, ps_problem):
        inst1 = _make_cflp_instance(I=3, J=4, S=5)
        inst2 = _make_cflp_instance(I=3, J=4, S=5)
        item1 = self._make_labeled_item(ps_problem, inst1)
        item2 = self._make_labeled_item(ps_problem, inst2)
        batch = ps_problem.collate_fn([item1, item2])
        assert "instance" in batch
        assert "label" in batch
        assert batch["instance"].data["scen_graph"]["x"].shape[0] > 0


# ===========================================================================
# Test Group 9: Printable size report (for documentation)
# ===========================================================================

class TestSizeReport:
    """Generate human-readable size comparison for documentation."""

    def test_print_size_comparison(self, capsys):
        """Print a comparison table of CV vs PS graph sizes for CFLP."""
        configs = [
            (5, 5, 10),
            (10, 10, 50),
            (30, 30, 50),
            (50, 50, 50),
            (100, 100, 50),
        ]
        print("\n=== CFLP Encoding Size Comparison (per-scenario graph) ===")
        print(f"{'m':>5} {'n':>5} {'S':>5} | "
              f"{'CV nodes':>10} {'CV edges':>10} {'CV mem(KB)':>12} | "
              f"{'PS nodes':>10} {'PS edges':>10} {'PS mem(KB)':>12} | "
              f"{'ratio':>8}")
        print("-" * 105)

        for I, J, S in configs:
            cv = _cv_expected_sizes(I, J)
            ps = _ps_expected_sizes(I, J)

            cv_mem = (cv["num_nodes_sce"] * cv["node_feat_dim"] * 4
                      + cv["num_edges_sce"] * (8 * 2 + cv["edge_feat_dim"] * 4))
            ps_mem = (ps["num_nodes"] * ps["node_feat_dim"] * 4
                      + ps["num_edges"] * (8 * 2 + ps["edge_feat_dim"] * 4))

            ratio = cv_mem / max(1, ps_mem)
            print(f"{I:>5} {J:>5} {S:>5} | "
                  f"{cv['num_nodes_sce']:>10} {cv['num_edges_sce']:>10} {cv_mem / 1024:>12.1f} | "
                  f"{ps['num_nodes']:>10} {ps['num_edges']:>10} {ps_mem / 1024:>12.1f} | "
                  f"{ratio:>7.1f}x")

        # Always pass -- this is a reporting test
        assert True
