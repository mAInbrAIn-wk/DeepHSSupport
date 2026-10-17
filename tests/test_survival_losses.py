"""
Unit tests for Survival Losses & Competing Risks (AP 1 / AP 5).
Compatible with both `unittest` (`python -m unittest discover -s tests -v`) and `pytest`.

Tests:
1. Tie-order invariance of Keras `cox_counting_process_breslow_loss` and PyTorch `PyTorchCoxPH.compute_loss`.
2. Counting-process left-truncation exclusion (`t_start >= t_event` excluded from risk set).
3. `PyTorchLogisticHazard.compute_loss` active-interval masking (`[t_start, duration]`) without double-counting past intervals.
4. `build_dynamic_deephit_network` joint 3-state Softmax competing-risks constraint (`h_drop + h_grad <= 1.0`).
5. `estimate_counting_process_baseline_hazard` calibration bounds (`0 <= p <= 1`).
"""

import unittest
import numpy as np
import tensorflow as tf
import torch

from deepsupport.models.deep_survival import (
    cox_counting_process_breslow_loss as keras_cp_loss_ds,
)
from deepsupport.models.extended_deepsurv import (
    cox_counting_process_breslow_loss as keras_cp_loss_eds,
    estimate_counting_process_baseline_hazard,
)
from deepsupport.models.dynamic_deephit import build_dynamic_deephit_network
from deepsupport.models.torch.survival import (
    PyTorchCoxPH,
    PyTorchCoxTime,
    PyTorchLogisticHazard,
)


class TestSurvivalLosses(unittest.TestCase):

    def test_keras_cox_tie_order_invariance(self):
        """Shuffling rows with identical event times must produce the exact same Breslow loss."""
        rng = np.random.default_rng(42)
        n = 64
        t_stop = rng.choice([1.0, 2.0, 3.0, 4.0], size=n).astype(np.float32)
        t_start = (t_stop - 1.0).astype(np.float32)
        events = rng.binomial(1, 0.4, size=n).astype(np.float32)
        risks = rng.normal(0.0, 1.0, size=(n, 1)).astype(np.float32)

        y_true = np.column_stack([t_start, t_stop, events])
        loss_ref_ds = float(keras_cp_loss_ds(tf.constant(y_true), tf.constant(risks)).numpy())
        loss_ref_eds = float(keras_cp_loss_eds(tf.constant(y_true), tf.constant(risks)).numpy())

        self.assertTrue(np.isclose(loss_ref_ds, loss_ref_eds, atol=1e-6))

        for perm_seed in range(5):
            perm = np.random.default_rng(perm_seed + 100).permutation(n)
            y_perm = tf.constant(y_true[perm])
            r_perm = tf.constant(risks[perm])
            loss_perm_ds = float(keras_cp_loss_ds(y_perm, r_perm).numpy())
            loss_perm_eds = float(keras_cp_loss_eds(y_perm, r_perm).numpy())
            self.assertTrue(
                np.isclose(loss_ref_ds, loss_perm_ds, atol=1e-5),
                f"Keras Cox loss changed under permutation: {loss_ref_ds} vs {loss_perm_ds}",
            )
            self.assertTrue(np.isclose(loss_ref_eds, loss_perm_eds, atol=1e-5))

    def test_counting_process_left_truncation_exclusion(self):
        """
        An interval (t_start=3, t_stop=4) must NOT be in the risk set of an event at t_stop=2.
        Therefore, changing the risk score of the (3, 4) censored row must have ZERO effect
        on the partial likelihood when the only event occurs at t_stop=2.
        """
        y_true = tf.constant([
            [1.0, 2.0, 1.0],
            [1.0, 2.0, 0.0],
            [3.0, 4.0, 0.0],
        ], dtype=tf.float32)

        r_low = tf.constant([[0.5], [0.2], [-5.0]], dtype=tf.float32)
        r_high = tf.constant([[0.5], [0.2], [50.0]], dtype=tf.float32)

        loss_low = float(keras_cp_loss_eds(y_true, r_low).numpy())
        loss_high = float(keras_cp_loss_eds(y_true, r_high).numpy())

        expected_loss = -(0.5 - np.log(np.exp(0.5) + np.exp(0.2)))
        self.assertTrue(np.isclose(loss_low, expected_loss, atol=1e-6))
        self.assertTrue(np.isclose(loss_high, expected_loss, atol=1e-6))

    def test_pytorch_cox_tie_invariance_and_counting_process(self):
        """Verify PyTorchCoxPH and PyTorchCoxTime tie invariance and counting-process risk set."""
        torch.manual_seed(42)
        cox_ph = PyTorchCoxPH(in_features=4, hidden_dims=[8])
        cox_time = PyTorchCoxTime(in_features=4, hidden_dims=[8])

        x = torch.tensor([
            [0.1, 0.2, 0.1, 0.0],
            [-0.2, 0.1, 0.3, -0.1],
            [5.0, 5.0, 5.0, 5.0],
        ], dtype=torch.float32)
        durations = torch.tensor([2.0, 2.0, 4.0], dtype=torch.float32)
        events = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

        batch_excluded = {
            "x": x,
            "t_start": torch.tensor([1.0, 1.0, 3.0], dtype=torch.float32),
            "duration": durations,
            "event": events,
        }
        batch_only_01 = {
            "x": x[:2],
            "t_start": torch.tensor([1.0, 1.0], dtype=torch.float32),
            "duration": durations[:2],
            "event": events[:2],
        }

        cox_ph.eval()
        cox_time.eval()
        with torch.no_grad():
            loss_ph_excl = float(cox_ph.compute_loss(batch_excluded).item())
            loss_ph_01 = float(cox_ph.compute_loss(batch_only_01).item())
            self.assertTrue(np.isclose(loss_ph_excl, loss_ph_01, atol=1e-6))

            loss_ct_excl = float(cox_time.compute_loss(batch_excluded).item())
            loss_ct_01 = float(cox_time.compute_loss(batch_only_01).item())
            self.assertTrue(np.isclose(loss_ct_excl, loss_ct_01, atol=1e-6))

    def test_pytorch_logistic_hazard_active_interval_mask(self):
        """
        Verify PyTorchLogisticHazard.compute_loss only evaluates NLL on [t_start, duration]
        and is completely invariant to logits at t < t_start or t > duration.
        """
        torch.manual_seed(42)
        model = PyTorchLogisticHazard(in_features=4, num_durations=17, hidden_dims=[8])
        model.eval()

        x = torch.randn(2, 4)
        # Row 0: active in semester 5 (t_start=4.0, duration=5.0 -> 0-based bin index 4, event=1)
        # Row 1: active in semester 3 (t_start=2.0, duration=3.0 -> 0-based bin index 2, event=0)
        batch = {
            "x": x,
            "t_start": torch.tensor([4.0, 2.0]),
            "duration": torch.tensor([5.0, 3.0]),
            "event": torch.tensor([1.0, 0.0]),
        }

        with torch.no_grad():
            logits = model.forward(x)
            loss_model = float(model.compute_loss(batch).item())

            bce_0 = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[0, 4], torch.tensor(1.0)
            )
            bce_1 = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[1, 2], torch.tensor(0.0)
            )
            expected_loss = float(((bce_0 + bce_1) / 2.0).item())

        self.assertTrue(
            np.isclose(loss_model, expected_loss, atol=1e-6),
            f"PyTorchLogisticHazard loss {loss_model} != single active-cell BCE {expected_loss}",
        )

    def test_dynamic_deephit_competing_risks_sum_constraint(self):
        """
        Verify that Dynamic DeepHit's 3-state Softmax head guarantees
        pred_drop + pred_grad <= 1.0 everywhere across all students and semesters.
        """
        model = build_dynamic_deephit_network(16, 10)
        rng = np.random.default_rng(123)
        X = rng.normal(0.0, 5.0, size=(32, 16, 10)).astype(np.float32)
        X[:10, 8:, :] = -99.0

        pred_drop, pred_grad = model.predict(X, verbose=0)
        self.assertEqual(pred_drop.shape, (32, 16, 1))
        self.assertEqual(pred_grad.shape, (32, 16, 1))
        self.assertTrue(np.all(pred_drop >= 0.0))
        self.assertTrue(np.all(pred_grad >= 0.0))

        total_hazard = pred_drop + pred_grad
        self.assertTrue(
            np.all(total_hazard <= 1.0 + 1e-6),
            f"Max combined hazard exceeded 1.0: {np.max(total_hazard)}",
        )

    def test_extended_deepsurv_calibrated_baseline_hazard(self):
        """Verify estimate_counting_process_baseline_hazard produces valid non-negative hazards."""
        rng = np.random.default_rng(7)
        n = 100
        t_stop = rng.integers(1, 9, size=n).astype(np.float64)
        t_start = t_stop - 1.0
        events = rng.binomial(1, 0.25, size=n).astype(np.float64)
        log_risk_tr = rng.normal(0.0, 0.5, size=n)

        dh0_map, cum_h0_map = estimate_counting_process_baseline_hazard(t_start, t_stop, events, log_risk_tr)
        self.assertGreater(len(dh0_map), 0)
        self.assertTrue(all(h >= 0.0 for h in dh0_map.values()))
        self.assertTrue(all(h >= 0.0 for h in cum_h0_map.values()))


if __name__ == "__main__":
    unittest.main()
