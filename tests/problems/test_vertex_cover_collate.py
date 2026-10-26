import numpy as np
import torch

from src.core.types import Instance, Labels
from src.problems.vertex_cover import VertexCoverProblem


def _make_instance(det: np.ndarray, sce: np.ndarray) -> Instance:
    adjacency = np.array([[0, 1], [1, 0]], dtype=np.int64)
    return Instance(
        {
            "n_nodes": np.array([2], dtype=np.int64),
            "n_scenarios": np.array([2], dtype=np.int64),
            "det": det,
            "sce": sce,
            "adjacency": adjacency,
        }
    )


def test_collate_truncates_scenario_labels_to_expected_s() -> None:
    # Motivation: when label scenario fields are too long, truncate to expected S.
    problem = VertexCoverProblem(n_nodes=2, n_scenarios=2)

    inst0 = _make_instance(
        det=np.array([1.0, 2.0], dtype=np.float32),
        sce=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )
    inst1 = _make_instance(
        det=np.array([2.0, 3.0], dtype=np.float32),
        sce=np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
    )

    labels0 = Labels(
        sce_ranking=np.array([2, 1, 0], dtype=np.int64),
        sce_scores=np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )
    labels1 = Labels(
        sce_ranking=np.array([1, 0], dtype=np.int64),
        sce_scores=np.array([0.5, 0.6], dtype=np.float32),
    )

    batch = [
        {"instance": inst0, "label": labels0},
        {"instance": inst1, "label": labels1},
    ]

    out = problem.collate_fn(batch)
    sce_ranking = out["label"].sce_ranking
    sce_scores = out["label"].sce_scores

    assert sce_ranking.shape == (2, 2)
    assert sce_scores.shape == (2, 2)
    assert torch.equal(sce_ranking[0], torch.tensor([2, 1], dtype=torch.int64))
    assert torch.allclose(sce_scores[0], torch.tensor([0.1, 0.2], dtype=torch.float32), atol=1e-6)
