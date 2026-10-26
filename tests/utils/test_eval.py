# tests/test_eval.py
import numpy as np
import unittest
from unittest.mock import MagicMock, patch
import torch
from types import SimpleNamespace

# Import functions/classes under test
from src.utils.eval_helpers import evaluate_loader, _compute_k_eff
from src.utils.eval_utils import _compute_plateau_stop_step, _compute_step_mask


# --- 1. Heuristic Utils Tests ---
class TestHeuristicUtils(unittest.TestCase):

    def test_stop_step_label_is_0_indexed_when_plateau_triggers(self):
        best_scores = np.array([10.0, 10.0, 10.0])  # gains: 10, 0, 0
        stop_step = _compute_plateau_stop_step(best_scores, stop_M=2, eps_gain=0.0)
        self.assertEqual(stop_step, 2)

    def test_step_mask_stop_at_2(self):
        mask = _compute_step_mask(5, stop_step=2)
        self.assertEqual(mask.tolist(), [True, True, False, False, False])

    def test_step_mask_no_early_stop(self):
        mask = _compute_step_mask(3, stop_step=3)
        self.assertEqual(mask.tolist(), [True, True, True])

    def test_step_mask_stop_immediately(self):
        mask = _compute_step_mask(4, stop_step=0)
        self.assertEqual(mask.tolist(), [False, False, False, False])


# --- 2. Helper Mechanics Tests (K_eff) ---
class TestEvalHelpers(unittest.TestCase):

    def test_compute_k_eff_mechanics(self):
        S = 100

        # Case A: Explicit K
        self.assertEqual(_compute_k_eff(S, K=10), 10)
        self.assertEqual(_compute_k_eff(S, K=200), 100)  # Clamping to S

        # Case B: K Ratio
        self.assertEqual(_compute_k_eff(S, k_ratio=0.1), 10)
        self.assertEqual(_compute_k_eff(S, k_ratio=0.0001), 1)  # Min 1

        # Case C: Conflict (Both K and k_ratio provided)
        with self.assertRaises(ValueError):
            _compute_k_eff(S, K=10, k_ratio=0.1)

        # Case D: Invalid Inputs
        with self.assertRaises(ValueError):
            _compute_k_eff(S, K=-5)
        with self.assertRaises(ValueError):
            _compute_k_eff(S, k_ratio=1.5)


# --- 4. Evaluate Loader Tests (One-Shot vs Stepwise) ---
class TestEvaluateLoader(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.problem = MagicMock()

        # Mock Metrics Accumulators
        self.patcher_sel = patch("src.utils.eval_helpers.SelectionMetricsAccumulator")
        self.patcher_tf = patch("src.utils.eval_helpers.StepwiseTeacherMetricsAccumulator")
        self.patcher_unpack = patch("src.utils.eval_helpers.unpack_batch")

        self.MockSelAcc = self.patcher_sel.start()
        self.MockTFAcc = self.patcher_tf.start()
        self.mock_unpack = self.patcher_unpack.start()

        # Default mock returns
        self.MockSelAcc.return_value.compute.return_value = {"ndcg": 0.5}
        self.MockTFAcc.return_value.compute.return_value = {"acc": 0.5}

    def tearDown(self):
        self.patcher_sel.stop()
        self.patcher_tf.stop()
        self.patcher_unpack.stop()

    def _setup_batch(self, B=2, S=5):
        inst = torch.zeros(B, 1)
        labels = SimpleNamespace()
        labels.sce_ranking = torch.arange(S).expand(B, S)
        labels.sce_scores = torch.randn(B, S)
        labels.forward_trace = None

        dataloader = ["dummy_batch"]
        self.mock_unpack.return_value = (inst, labels)
        return dataloader

    def test_one_shot_extraction(self):
        """Test that One-Shot models (scores only) use topk(scores) for idx."""
        B, S, K = 2, 5, 2
        dataloader = self._setup_batch(B, S)

        model = MagicMock()
        # Scores favor last scenarios [4, 3]
        scores = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5],
                               [0.1, 0.2, 0.3, 0.4, 0.5]])

        # Model returns ONLY scores (no pick_idx_steps)
        model.return_value = {"scores": scores}

        result = evaluate_loader(model, dataloader, self.device, self.problem, K=K)
        expected_idx = torch.tensor([[4, 3], [4, 3]])

        self.assertTrue(torch.equal(result["idx"], expected_idx),
                        f"One-shot should use topk(scores). Got {result['idx']}")

    def test_stepwise_extraction_priority(self):
        """Test that Stepwise models prioritize pick_idx_steps over scores."""
        B, S, K = 2, 5, 2
        dataloader = self._setup_batch(B, S)

        model = MagicMock()
        # Scores favor [4, 3]
        scores = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5],
                               [0.1, 0.2, 0.3, 0.4, 0.5]])

        # Rollout explicitly picked [0, 1]
        pick_idx_steps = torch.tensor([[0, 1],
                                       [0, 1]])

        model.return_value = {
            "scores": scores,
            "pick_idx_steps": pick_idx_steps
        }

        result = evaluate_loader(model, dataloader, self.device, self.problem, K=K)

        self.assertTrue(torch.equal(result["idx"], pick_idx_steps),
                        f"Stepwise should use pick_idx_steps. Got {result['idx']}")

    def test_stepwise_slicing(self):
        """Test that pick_idx_steps is sliced if T > K."""
        B, S, K = 1, 5, 2
        dataloader = self._setup_batch(B, S)

        model = MagicMock()
        scores = torch.zeros(B, S)
        # Rollout ran 5 steps, we only want K=2
        pick_idx_steps = torch.tensor([[10, 11, 12, 13, 14]])

        model.return_value = {
            "scores": scores,
            "pick_idx_steps": pick_idx_steps
        }

        result = evaluate_loader(model, dataloader, self.device, self.problem, K=K)

        expected_idx = torch.tensor([[10, 11]])
        self.assertTrue(torch.equal(result["idx"], expected_idx))

    def test_k_eff_computation_integration(self):
        """Verify K vs k_ratio logic via the loader Integration."""
        B, S = 1, 10
        dataloader = self._setup_batch(B, S)
        model = MagicMock()
        scores = torch.rand(B, S)
        model.return_value = {"scores": scores}

        result = evaluate_loader(model, dataloader, self.device, self.problem, k_ratio=0.5)
        self.assertEqual(result["idx"].shape[1], 5, "k_ratio=0.5 with S=10 should result in K=5")

    def test_scenario_count_prefers_instance_over_labels_x(self):
        """Ensure scenario_count falls back to instance data when labels lack sce info."""
        B, S, N = 2, 100, 50
        inst = SimpleNamespace(data={"sce": torch.zeros(B, S, N)})
        labels = SimpleNamespace()
        labels.sce_ranking = None
        labels.sce_scores = None
        labels.forward_trace = None
        labels.x = torch.zeros(B, N)  # should not be used to infer S

        dataloader = ["dummy_batch"]
        self.mock_unpack.return_value = (inst, labels)

        model = MagicMock()
        scores = torch.rand(B, S)
        model.return_value = {"scores": scores}

        result = evaluate_loader(model, dataloader, self.device, self.problem, k_ratio=0.1)
        self.assertEqual(result["idx"].shape[1], 10, "S=100 with k_ratio=0.1 should yield K=10")


if __name__ == "__main__":
    unittest.main()
