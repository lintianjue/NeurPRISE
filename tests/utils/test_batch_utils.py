import numpy as np
import pytest
import torch

from src.core.types import Labels, ForwardTraceBatch
from src.problems.cflp import CFLPProblem
from src.utils.batch_utils import _build_reduced_instance_from_np, to_device_labels


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_to_device_labels_moves_forward_trace_batch_to_cuda():
    # Arrange: ForwardTraceBatch has tensors but no .to(), so it must be handled via dataclass recursion
    trace = ForwardTraceBatch(
        K=2,
        S=5,
        selected_idx=torch.zeros((1, 3), dtype=torch.long, device="cpu"),
        candidate_scores=torch.zeros((1, 3, 5), dtype=torch.float32, device="cpu"),
        best_scores=torch.zeros((1, 3), dtype=torch.float32, device="cpu"),
        gains=torch.zeros((1, 3), dtype=torch.float32, device="cpu"),
        stop_step=torch.zeros((1,), dtype=torch.long, device="cpu"),
        step_mask=torch.ones((1, 3), dtype=torch.bool, device="cpu"),
        T=torch.tensor([3], dtype=torch.long, device="cpu"),
    )
    labels = Labels(
        sce_top_idx=torch.tensor([1, 3], dtype=torch.long, device="cpu"),
        forward_trace=trace,
    )

    # Act
    moved = to_device_labels(labels, torch.device("cuda"))

    # Assert
    assert moved is not None
    assert moved.sce_top_idx.device.type == "cuda"
    assert moved.forward_trace is not None
    assert moved.forward_trace.step_mask.device.type == "cuda"
    assert moved.forward_trace.selected_idx.device.type == "cuda"
    assert moved.forward_trace.candidate_scores.device.type == "cuda"


def test_get_scenarios_and_reduce_for_cflp_demand():
    """
    Motivation:
    - CFLP uses demand scenarios under key 'd'; ensure reducers honor that.
    """
    problem = CFLPProblem(n_facilities=1, n_customers=2, n_scenarios=2)
    data_np = {
        "d": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        "n_scenarios": np.array([2], dtype=int),
    }
    sce = problem.get_scenarios(data_np)
    assert np.allclose(sce, data_np["d"])

    reduced = _build_reduced_instance_from_np(problem, data_np, sce_idx=np.array([1], dtype=int))
    assert reduced.data["d"].shape == (1, 2)
    assert np.allclose(reduced.data["d"][0], np.array([3.0, 4.0], dtype=float))
    assert int(reduced.data["n_scenarios"][0]) == 1



import pytest
import torch
from src.utils.batch_utils import unpack_batch
from src.core.types import Instance


# ---------------------------------------------------------------------------
# TC-16: test_unpack_batch_all_input_formats
# ---------------------------------------------------------------------------
def test_unpack_batch_dict_format():
    """Dict batch {'instance': ..., 'label': ...} is unpacked correctly."""
    inst = Instance(data={"x": torch.ones(2, 3)})
    labels = Labels(obj=torch.tensor([1.0, 2.0]))
    batch = {"instance": inst, "label": labels}

    inst_out, labels_out = unpack_batch(batch, device=torch.device("cpu"))
    assert isinstance(inst_out, Instance)
    assert labels_out is not None


def test_unpack_batch_tuple_format():
    """Tuple batch (instance, labels) is unpacked correctly."""
    inst = Instance(data={"x": torch.ones(2, 3)})
    labels = Labels(obj=torch.tensor([1.0]))
    batch = (inst, labels)

    inst_out, labels_out = unpack_batch(batch, device=torch.device("cpu"))
    assert isinstance(inst_out, Instance)
    assert labels_out is not None


def test_unpack_batch_instance_only():
    """When batch is just an Instance, labels returns None."""
    inst = Instance(data={"x": torch.zeros(2, 3)})

    inst_out, labels_out = unpack_batch(inst, device=torch.device("cpu"))
    assert isinstance(inst_out, Instance)
    assert labels_out is None
