import numpy as np
import pytest
import torch

from src.core.types import Instance
from src.problems.cflp import CFLPEntityBipartiteProblem


@pytest.fixture
def small_instance() -> Instance:
    f = np.array([1.0, 2.0], dtype=np.float32)
    a = np.array([3.0, 4.0], dtype=np.float32)
    K = np.array([10.0, 20.0], dtype=np.float32)
    c = np.array(
        [
            [5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0],
        ],
        dtype=np.float32,
    )
    demand = np.array(
        [
            [11.0, 12.0, 13.0],
            [14.0, 15.0, 16.0],
        ],
        dtype=np.float32,
    )

    return Instance(
        {
            "n_facilities": np.array([2], dtype=np.int64),
            "n_customers": np.array([3], dtype=np.int64),
            "n_scenarios": np.array([2], dtype=np.int64),
            "f": f,
            "a": a,
            "K": K,
            "c": c,
            "d": demand,
        }
    )


def test_entity_bipartite_scenario_graph_shapes_and_features(small_instance: Instance) -> None:
    # Motivation: verify entity-bipartite encoding uses I+J nodes and c_ij edges.
    problem = CFLPEntityBipartiteProblem(n_facilities=2, n_customers=3, n_scenarios=2)
    data_list = problem.to_pyg_data(small_instance)

    assert len(data_list) == 2
    data = data_list[0]

    I, J = 2, 3
    assert data.x.shape == (I + J, 6)
    assert data.edge_index.shape == (2, 2 * I * J)
    assert data.edge_attr.shape == (2 * I * J, 1)

    f = torch.tensor([1.0, 2.0], dtype=torch.float32)
    a = torch.tensor([3.0, 4.0], dtype=torch.float32)
    K = torch.tensor([10.0, 20.0], dtype=torch.float32)
    demand0 = torch.tensor([11.0, 12.0, 13.0], dtype=torch.float32)

    assert torch.allclose(data.x[:I, 0], torch.ones(I))
    assert torch.allclose(data.x[I:, 1], torch.ones(J))
    assert torch.allclose(data.x[:I, 2], f)
    assert torch.allclose(data.x[:I, 3], a)
    assert torch.allclose(data.x[:I, 4], K)
    assert torch.allclose(data.x[I:, 5], demand0)

    c_flat = torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=torch.float32)
    assert torch.allclose(data.edge_attr[: I * J, 0], c_flat)
    assert torch.allclose(data.edge_attr[I * J :, 0], c_flat)


def test_entity_bipartite_det_graph_uses_mean_demand(small_instance: Instance) -> None:
    # Motivation: deterministic graph should carry mean demand for customers.
    problem = CFLPEntityBipartiteProblem(n_facilities=2, n_customers=3, n_scenarios=2)
    det = problem.to_pyg_data_det(small_instance)

    expected = torch.tensor([12.5, 13.5, 14.5], dtype=torch.float32)
    assert torch.allclose(det.x[2:, 5], expected)
