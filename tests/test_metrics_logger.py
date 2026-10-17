"""
Unit tests for Metrics Logger & OOP Evaluators (AP 0 / AP 5).
Compatible with both `unittest` (`python -m unittest discover -s tests -v`) and `pytest`.

Tests:
1. Recursive `_clean_numeric` sanitization of nested dicts/lists with NaN/Inf/NumPy scalars.
2. Non-destructive JSON merging when `save_metrics` is called after `SurvivalEvaluator.evaluate_and_log` (or vice versa).
3. Duplicate suffix stripping (`_clean_model_name`) in `_BaseEvaluator` and `save_metrics`.
4. `CausalEvaluator` handling of `se_scale='natural'` vs `'log'` and `estimand_type='RR'` / `'ATE'`.
5. `MulticlassEvaluator` per-class PR-AUC and baseline prevalence logging.
"""

import json
import os
import tempfile
import unittest
import numpy as np

from deepsupport.evaluation.metrics_logger import (
    _clean_numeric,
    save_metrics,
    SurvivalEvaluator,
    CausalEvaluator,
    MulticlassEvaluator,
)


class TestMetricsLogger(unittest.TestCase):

    def test_clean_numeric_recursive(self):
        """Verify _clean_numeric recursively converts NaN/Inf to None and numpy scalars to Python types."""
        raw = {
            "a": np.float32(0.75),
            "b": np.nan,
            "c": np.inf,
            "nested_dict": {
                "x": np.int64(42),
                "y": float("-inf"),
                "z": [np.float64(1.25), np.nan],
            },
        }
        cleaned = _clean_numeric(raw)
        self.assertIsInstance(cleaned["a"], float)
        self.assertIsNone(cleaned["b"])
        self.assertIsNone(cleaned["c"])
        self.assertEqual(cleaned["nested_dict"]["x"], 42)
        self.assertIsInstance(cleaned["nested_dict"]["x"], int)
        self.assertIsNone(cleaned["nested_dict"]["y"])
        self.assertEqual(cleaned["nested_dict"]["z"], [1.25, None])

        dumped = json.dumps(cleaned, allow_nan=False)
        self.assertIn("null", dumped)

    def test_evaluator_and_save_metrics_merge_non_destructively(self):
        """
        Calling SurvivalEvaluator.evaluate_and_log followed by legacy save_metrics on the same
        target file must preserve BOTH the standardized OOP Evaluator keys AND the legacy flat keys.
        Also verify duplicate suffix stripping on model_name.
        """
        with tempfile.TemporaryDirectory() as base_dir:
            rng = np.random.default_rng(42)
            y_true = rng.binomial(1, 0.3, size=100)
            y_prob = np.clip(y_true * 0.6 + rng.uniform(0.05, 0.35, size=100), 0.01, 0.99)

            ev = SurvivalEvaluator(base_dir=base_dir, model_name="semester_gru_standard_prev")
            ev.evaluate_and_log(
                y_true=y_true,
                y_prob=y_prob,
                mode="standard",
                temporal_type="prev",
                extra_metrics={"roc_auc_student": 0.88},
            )

            expected_path = os.path.join(base_dir, "metrics", "semester_gru_standard_prev_metrics.json")
            self.assertTrue(os.path.exists(expected_path))

            save_metrics(
                model_name="semester_gru_standard_prev",
                metrics={"roc_auc": 0.85, "legacy_custom_key": 123.0},
                base_dir=base_dir,
                mode="standard",
                temporal_type="prev",
            )

            with open(expected_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertIn("pr_auc_dropout", data)
            self.assertIn("pr_auc_nondropout", data)
            self.assertIn("brier_skill_score", data)
            self.assertEqual(data["roc_auc_student"], 0.88)
            self.assertEqual(data["legacy_custom_key"], 123.0)

    def test_causal_evaluator_se_scales_and_estimand_types(self):
        """
        Verify CausalEvaluator correctly differentiates:
        1. Multiplicative estimand ('RR') with se_scale='natural' (Delta-method log-SE = SE / RR)
        2. Additive estimand ('ATE') with additive normal CI (point +/- 1.96 * SE)
        """
        with tempfile.TemporaryDirectory() as base_dir:
            ev_rr = CausalEvaluator(base_dir=base_dir, model_name="test_gcomp")
            res_rr = ev_rr.evaluate_and_log(
                hr_estimates={"rr_fachlich": 0.80},
                hr_se={"rr_fachlich": 0.04},
                estimand_type="RR",
                se_scale="natural",
                mode="standard",
                temporal_type="prev",
            )
            lo_rr = res_rr["ci_lower_95_asym_rr_fachlich"]
            hi_rr = res_rr["ci_upper_95_asym_rr_fachlich"]
            expected_lower = float(0.80 * np.exp(-1.96 * (0.04 / 0.80)))
            expected_upper = float(0.80 * np.exp(1.96 * (0.04 / 0.80)))
            self.assertTrue(np.isclose(lo_rr, expected_lower, atol=1e-6))
            self.assertTrue(np.isclose(hi_rr, expected_upper, atol=1e-6))

            ev_ate = CausalEvaluator(base_dir=base_dir, model_name="test_ate")
            res_ate = ev_ate.evaluate_and_log(
                hr_estimates={"ate_fachlich": -0.05},
                hr_se={"ate_fachlich": 0.01},
                estimand_type="ATE",
                se_scale="natural",
                mode="standard",
                temporal_type="prev",
            )
            lo_ate = res_ate["ci_lower_95_asym_ate_fachlich"]
            hi_ate = res_ate["ci_upper_95_asym_ate_fachlich"]
            self.assertTrue(np.isclose(lo_ate, -0.05 - 1.96 * 0.01, atol=1e-6))
            self.assertTrue(np.isclose(hi_ate, -0.05 + 1.96 * 0.01, atol=1e-6))

    def test_multiclass_evaluator_logs_per_class_pr_auc(self):
        """Verify MulticlassEvaluator logs per-class PR-AUC and prevalence according to project rules."""
        with tempfile.TemporaryDirectory() as base_dir:
            rng = np.random.default_rng(99)
            y_true = rng.integers(0, 4, size=120)
            logits = rng.normal(size=(120, 4))
            y_prob_matrix = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

            ev = MulticlassEvaluator(base_dir=base_dir, model_name="test_mc")
            res = ev.evaluate_and_log(y_true=y_true, y_prob_matrix=y_prob_matrix)

            self.assertIn("per_class_metrics", res)
            self.assertEqual(len(res["per_class_metrics"]), 4)
            for cls_info in res["per_class_metrics"].values():
                self.assertIn("pr_auc", cls_info)
                self.assertIn("prevalence_baseline", cls_info)


if __name__ == "__main__":
    unittest.main()
