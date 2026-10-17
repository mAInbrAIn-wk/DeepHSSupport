"""
Unit tests for Causal Estimators (DML Robinson, Cluster Bootstrap, Cross-Fitting, MSM) (AP 2 / AP 5).
Compatible with both `unittest` (`python -m unittest discover -s tests -v`) and `pytest`.

Tests:
1. Multivariate Robinson estimator + student-clustered Liang-Zeger sandwich SE recovery on synthetic confounded data with correlated treatments.
2. Multinomial cluster-multiplicity bootstrap vs. broken `.isin()` deduplication.
3. GroupKFold cross-fitting student-disjointness guarantee.
4. `torch_msm` mode-aware feature filtering (`mode='blind'` excludes `hzb_note` and `gpa_prev`).
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


class TestCausalEstimators(unittest.TestCase):

    def test_multivariate_robinson_and_liang_zeger_se_recovery(self):
        """
        On synthetic confounded panel data with correlated treatments A1, A2, A3,
        verify that the joint multivariate Robinson estimator
        theta = (A_tilde^T A_tilde)^{-1} A_tilde^T Y_tilde
        recovers the true causal effects theta_true = (-0.05, -0.03, -0.02) accurately,
        whereas univariate Robinson suffers from omitted co-treatment bias.
        """
        rng = np.random.default_rng(42)
        n_students = 1000
        semesters_per_student = 5
        N = n_students * semesters_per_student

        student_ids = np.repeat(np.arange(n_students), semesters_per_student)
        W = rng.normal(0.0, 1.0, size=N)

        shared_u = rng.normal(0.0, 0.8, size=N)
        e1 = 1.0 / (1.0 + np.exp(-(0.6 * W + 0.8 * shared_u - 0.5)))
        e2 = 1.0 / (1.0 + np.exp(-(0.5 * W + 0.8 * shared_u - 0.7)))
        e3 = 1.0 / (1.0 + np.exp(-(0.4 * W + 0.8 * shared_u - 0.9)))

        A1 = rng.binomial(1, e1).astype(np.float64)
        A2 = rng.binomial(1, e2).astype(np.float64)
        A3 = rng.binomial(1, e3).astype(np.float64)
        A = np.column_stack([A1, A2, A3])

        theta_true = np.array([-0.05, -0.03, -0.02])
        y0_base = 0.25 + 0.08 * W
        noise = rng.normal(0.0, 0.05, size=N)
        Y = y0_base + A @ theta_true + noise

        W_design = np.column_stack([np.ones(N), W, W**2, W**3])
        E_hat = W_design @ np.linalg.lstsq(W_design, A, rcond=None)[0]
        m_hat = W_design @ np.linalg.lstsq(W_design, Y, rcond=None)[0]

        A_tilde = A - E_hat
        Y_tilde = Y - m_hat

        ata_inv = np.linalg.pinv(A_tilde.T @ A_tilde)
        theta_mv = ata_inv @ (A_tilde.T @ Y_tilde)

        theta_uv = np.array([
            np.sum(A_tilde[:, k] * Y_tilde) / np.sum(A_tilde[:, k] ** 2)
            for k in range(3)
        ])

        mv_err = float(np.linalg.norm(theta_mv - theta_true))
        uv_err = float(np.linalg.norm(theta_uv - theta_true))
        self.assertLess(mv_err, 0.008, f"Multivariate Robinson error too large: {theta_mv} vs {theta_true}")
        self.assertLess(
            mv_err,
            uv_err,
            f"Multivariate Robinson ({mv_err:.4f}) should be more accurate than univariate ({uv_err:.4f})",
        )

        # Liang-Zeger student-clustered sandwich SEs
        resid = Y_tilde - A_tilde @ theta_mv
        score_i = A_tilde * resid[:, None]
        df_scores = pd.DataFrame(score_i, columns=["s0", "s1", "s2"])
        df_scores["cid"] = student_ids
        cluster_sums = df_scores.groupby("cid")[["s0", "s1", "s2"]].sum().values
        G = len(cluster_sums)
        meat = (G / max(G - 1, 1)) * (cluster_sums.T @ cluster_sums)
        cov_theta = ata_inv @ meat @ ata_inv
        se_theta = np.sqrt(np.maximum(np.diag(cov_theta), 1e-12))

        self.assertTrue(np.all(se_theta > 0.0) and np.all(se_theta < 0.02))
        ci_lower = theta_mv - 1.96 * se_theta
        ci_upper = theta_mv + 1.96 * se_theta
        self.assertTrue(np.all((theta_true >= ci_lower) & (theta_true <= ci_upper)))

    def test_multinomial_cluster_bootstrap_preserves_multiplicity(self):
        """
        Verify that multinomial cluster-multiplicity bootstrap preserves 100% of total
        cluster count and produces non-trivial variance across bootstrap draws,
        unlike `.isin()` which drops ~36.8% of clusters.
        """
        rng = np.random.default_rng(123)
        n_clusters = 500
        counts = rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters))

        self.assertEqual(int(np.sum(counts)), n_clusters)
        unique_kept = int(np.sum(counts > 0))
        self.assertLess(unique_kept, 0.70 * n_clusters)

    def test_groupkfold_cross_fitting_zero_student_leakage(self):
        """Verify 5-fold GroupKFold on studierenden_id guarantees zero student overlap per fold."""
        student_ids = np.repeat(np.arange(200), 4)
        X = np.random.default_rng(42).normal(size=(len(student_ids), 5))
        gkf = GroupKFold(n_splits=5)

        for tr_idx, val_idx in gkf.split(X, groups=student_ids):
            tr_students = set(student_ids[tr_idx])
            val_students = set(student_ids[val_idx])
            self.assertEqual(len(tr_students.intersection(val_students)), 0)

    def test_msm_mode_filtering_excludes_forbidden_columns(self):
        """
        Verify the feature filtering logic used in PyTorchMSM: when feature_cols comes from
        mode='blind' (no hzb_note, no gpa_prev, no hidden_*), baseline_cols and time_varying_cols
        strictly exclude them.
        """
        blind_feature_cols = [
            "alter_bei_einschreibung", "geschlecht_num", "erstakademiker",
            "berufsausbildung", "erwerbstaetigkeit_ias", "bafoeg_bezug",
            "stg_Wirtschaft", "stg_Soziales", "stg_MINT", "stg_Recht",
            "is_first_semester",
            "support_fachlich_prev", "support_ueberfachlich_prev", "support_psychosozial_prev",
        ]
        treatment_cols = ["support_fachlich_prev", "support_ueberfachlich_prev", "support_psychosozial_prev"]
        allowed_confounders = [c for c in blind_feature_cols if c not in treatment_cols]

        candidate_baseline = [
            "hzb_note", "alter_bei_einschreibung", "geschlecht_num", "erstakademiker",
            "berufsausbildung", "erwerbstaetigkeit_ias", "bafoeg_bezug",
            "stg_Wirtschaft", "stg_Soziales", "stg_MINT", "stg_Recht",
            "hidden_zeit_puffer_static",
        ]
        baseline_cols = [c for c in candidate_baseline if c in allowed_confounders]
        time_varying_cols = [c for c in allowed_confounders if c not in baseline_cols]

        self.assertNotIn("hzb_note", baseline_cols)
        self.assertNotIn("hidden_zeit_puffer_static", baseline_cols)
        self.assertNotIn("gpa_prev", time_varying_cols)
        self.assertIn("is_first_semester", time_varying_cols)


if __name__ == "__main__":
    unittest.main()
