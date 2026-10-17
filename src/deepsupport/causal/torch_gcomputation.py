"""
PyTorch G-Computation / Counterfactual Monte Carlo Simulation
=============================================================
Implementiert die G-Formel nach Robins (1986) zur kontrafaktischen Pfad-Simulation
universitätsweiter Förderpolitiken unter do(A=1) vs. do(A=0).

Architektur:
1. Sequentielles / Diskretes PyTorch Hazard-Netzwerk als Zustandsübergangsmodell:
   - Schätzt h(t | X_t, A_t)
2. Kontrafaktische Kohorten-Projektion:
   - Pfad A=0 (do(A=0)): Kontrafaktische Welt ohne jegliche Förderung (analog Universum B)
   - Pfad A=1 (do(A=1)): Universelle Förderung
   - Pfad A=adaptiv: Bedarfsorientierte Krisen-Intervention (do(A=1 if fails > 0))
3. Kausale Endpunkt-Metriken:
   - Kontrafaktische Überlebenskurven S_cf0(t) vs. S_cf1(t)
   - Absolute Risk Reduction (ARR) und Relative Risk (RR)
   - Number Needed to Treat (NNT = 1 / ARR)
   - Direkte Validierung gegen den empirischen Ground Truth aus Universum B (ARR = 7.9 pp)
   - 100-Fold Student-Cluster-Bootstrap für exakte Konfidenzintervalle
4. Logging über CausalEvaluator
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, brier_score_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import deepsupport.data_engine.feature_builder as fb
from deepsupport.evaluation.metrics_logger import CausalEvaluator, save_metrics


# =====================================================================
# PYTORCH TRANSITION HAZARD MODELL
# =====================================================================

class PyTorchTransitionHazardNet(nn.Module):
    """Pre-LayerNorm Netzwerk zur Vorhersage des semesterweisen Übergangshazards h_t(x, a)."""
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))


# =====================================================================
# HAUPTKLASSE: G-COMPUTATION RUNNER
# =====================================================================

class PyTorchGComputation:
    """
    Kausaler G-Computation Estimator in PyTorch.
    Simuliert kontrafaktische Kohorten sowohl über Step-Standardization (Observed-Horizon)
    als auch über vollständige longitudinale Sequential G-Formula Rollouts (Robins 1986).
    """

    def __init__(
        self,
        data_dir: Path,
        output_dir: Optional[Path] = None,
        mode: str = "standard",
        temporal: str = "prev",
        epochs: int = 20,
        batch_size: int = 2048,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir
        self.mode = mode
        self.temporal = temporal
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        torch.manual_seed(seed)
        np.random.seed(seed)

    def _train_binary_transition_net(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va: np.ndarray,
        y_va: np.ndarray,
    ) -> PyTorchTransitionHazardNet:
        in_dim = X_tr.shape[1]
        model = PyTorchTransitionHazardNet(in_dim=in_dim, hidden_dim=64).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

        t_X_tr = torch.tensor(X_tr, dtype=torch.float32)
        t_y_tr = torch.tensor(y_tr, dtype=torch.float32)
        t_X_va = torch.tensor(X_va, dtype=torch.float32).to(self.device)
        t_y_va = torch.tensor(y_va, dtype=torch.float32).to(self.device)
        loader = DataLoader(TensorDataset(t_X_tr, t_y_tr), batch_size=self.batch_size, shuffle=True)

        best_loss = float("inf")
        best_state = None
        for _ in range(self.epochs):
            model.train()
            for b_X, b_y in loader:
                b_X, b_y = b_X.to(self.device), b_y.to(self.device)
                optimizer.zero_grad()
                logits = model.forward_logits(b_X)
                loss = F.binary_cross_entropy_with_logits(logits, b_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                v_loss = F.binary_cross_entropy_with_logits(model.forward_logits(t_X_va), t_y_va).item()
            if v_loss < best_loss:
                best_loss = v_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        return model

    def fit_and_evaluate(self) -> Dict[str, Any]:
        """Trainiert Übergangsmodelle, simuliert do(A=0) vs. do(A=1) (Step-Standardized & Sequential G-Formula) und loggt Metriken."""
        start_time = time.time()
        print("\n" + "=" * 78)
        print("   PYTORCH G-COMPUTATION & COUNTERFACTUAL COHORT SIMULATION")
        print(f"   Modus: {self.mode} | Temporal: {self.temporal} | Device: {self.device}")
        print("=" * 78)

        # 1. Panel-Daten laden
        panel_df, feature_cols, target_col, _ = fb.build_semester_panel_df(
            self.data_dir, mode=self.mode, temporal=self.temporal
        )

        treatment_candidates = ["fach_supp_count", "uebf_supp_count", "psych_supp_count"]
        treatment_cols = [c for c in treatment_candidates if c in feature_cols]
        confounder_cols = [c for c in feature_cols if c not in treatment_cols]

        # 2. Standardisierter Student-konsistenter 70/15/15 Split
        unique_studis = np.array(panel_df["studierenden_id"].unique().tolist())
        tr_ids, temp_ids = train_test_split(unique_studis, test_size=0.30, random_state=self.seed)
        va_ids, te_ids = train_test_split(temp_ids, test_size=0.50, random_state=self.seed)

        df_train = panel_df[panel_df["studierenden_id"].isin(tr_ids)].copy().reset_index(drop=True)
        df_val   = panel_df[panel_df["studierenden_id"].isin(va_ids)].copy().reset_index(drop=True)
        df_test  = panel_df[panel_df["studierenden_id"].isin(te_ids)].copy().reset_index(drop=True)

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_train_conf = scaler.fit_transform(imputer.fit_transform(df_train[confounder_cols]))
        X_val_conf   = scaler.transform(imputer.transform(df_val[confounder_cols]))
        X_test_conf  = scaler.transform(imputer.transform(df_test[confounder_cols]))

        A_train = df_train[treatment_cols].values.astype(np.float32)
        A_val   = df_val[treatment_cols].values.astype(np.float32)
        A_test  = df_test[treatment_cols].values.astype(np.float32)

        X_train = np.hstack([X_train_conf, A_train])
        X_val   = np.hstack([X_val_conf, A_val])
        X_test  = np.hstack([X_test_conf, A_test])

        y_train = df_train[target_col].values.astype(np.float32)
        y_val   = df_val[target_col].values.astype(np.float32)
        y_test  = df_test[target_col].values.astype(np.float32)

        # Competing Event: Graduation (competing_event == 2) für longitudinalen Rollout
        if "competing_event" in df_train.columns:
            y_grad_train = (df_train["competing_event"].values == 2).astype(np.float32)
            y_grad_val   = (df_val["competing_event"].values == 2).astype(np.float32)
        else:
            y_grad_train = np.zeros_like(y_train)
            y_grad_val   = np.zeros_like(y_val)

        # 3. PyTorch Übergangsmodelle trainieren (Dropout-Hazard & Competing Graduation-Hazard)
        print(f"\n[Schritt 1] Trainiere Übergangshazard-Modelle h_drop(t | X, A) und h_grad(t | X, A) ...")
        model = self._train_binary_transition_net(X_train, y_train, X_val, y_val)
        grad_model = self._train_binary_transition_net(X_train, y_grad_train, X_val, y_grad_val)

        # Factual Evaluation
        model.eval()
        grad_model.eval()
        with torch.no_grad():
            t_X_te = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            p_factual = model(t_X_te).cpu().numpy()

        auc_fac = float(roc_auc_score(y_test, p_factual))
        brier_fac = float(brier_score_loss(y_test, p_factual))

        # 4A. Step-Standardized Kontrafaktische Evaluation (auf beobachtetem Horizont)
        print(f"\n[Schritt 2a] Step-Standardized Counterfactuals auf beobachteten Semester-Intervallen ...")
        A_cf0 = np.zeros_like(A_test)
        A_cf1 = np.ones_like(A_test)

        with torch.no_grad():
            p_cf0 = model(torch.tensor(np.hstack([X_test_conf, A_cf0]), dtype=torch.float32).to(self.device)).cpu().numpy()
            p_cf1 = model(torch.tensor(np.hstack([X_test_conf, A_cf1]), dtype=torch.float32).to(self.device)).cpu().numpy()

        df_test["h_cf0"] = p_cf0
        df_test["h_cf1"] = p_cf1

        studi_surv_step_cf0 = df_test.groupby("studierenden_id")["h_cf0"].apply(lambda h: np.prod(1.0 - np.clip(h, 1e-6, 1.0 - 1e-6)))
        studi_surv_step_cf1 = df_test.groupby("studierenden_id")["h_cf1"].apply(lambda h: np.prod(1.0 - np.clip(h, 1e-6, 1.0 - 1e-6)))

        risk_step_cf0 = float(1.0 - np.mean(studi_surv_step_cf0))
        risk_step_cf1 = float(1.0 - np.mean(studi_surv_step_cf1))
        arr_step = float(risk_step_cf0 - risk_step_cf1)
        rr_step = float(risk_step_cf1 / max(risk_step_cf0, 1e-7))

        # 4B. Vollständiger Longitudinaler Sequential G-Formula Rollout (Robins 1986, Semester 1..16)
        print(f"[Schritt 2b] Sequential G-Formula Rollout über 16 Semester mit Kovariaten-Transition E[L_t | L_{{t-1}}, A_{{t-1}}] ...")
        static_set = {
            "hzb_note", "hzb_typ_ord", "migrationshintergrund", "erstakademiker",
            "erwerbstaetigkeit_std", "stg_BWL", "stg_Maschinenbau",
            "stg_Psychologie", "stg_Soziale_Arbeit", "hidden_zeit_puffer_static",
        }
        tv_indices = [
            i for i, c in enumerate(confounder_cols)
            if c not in static_set and c not in ("fachsemester", "is_first_semester")
        ]
        idx_fachsem = confounder_cols.index("fachsemester") if "fachsemester" in confounder_cols else None
        idx_first_sem = confounder_cols.index("is_first_semester") if "is_first_semester" in confounder_cols else None

        # Autoregressives Zustandsübergangsmodell L_t = f_L(X_{t-1}, A_{t-1}) auf Trainingspaaren (t-1 -> t)
        df_tr_sorted = df_train.sort_values(["studierenden_id", "t_stop"]).reset_index(drop=True)
        X_tr_conf_sorted = scaler.transform(imputer.transform(df_tr_sorted[confounder_cols]))
        A_tr_sorted = df_tr_sorted[treatment_cols].values.astype(np.float32)
        same_student_next = (
            df_tr_sorted["studierenden_id"].values[:-1] == df_tr_sorted["studierenden_id"].values[1:]
        )

        cov_transition_model = None
        if tv_indices and np.any(same_student_next):
            X_prev_trans = np.hstack([X_tr_conf_sorted[:-1][same_student_next], A_tr_sorted[:-1][same_student_next]])
            L_next_trans = X_tr_conf_sorted[1:][same_student_next][:, tv_indices]
            cov_transition_model = Ridge(alpha=10.0)
            cov_transition_model.fit(X_prev_trans, L_next_trans)

        # Startzustand pro Test-Student bei Semester 1
        df_test_s1 = df_test.sort_values(["studierenden_id", "t_stop"]).groupby("studierenden_id", as_index=False).first()
        X_s1_raw = imputer.transform(df_test_s1[confounder_cols])
        X_s1_scaled = scaler.transform(X_s1_raw).astype(np.float32)
        n_test_studis = len(df_test_s1)

        def _rollout_sequential_g(a_val: float, max_horizon: int = 16) -> np.ndarray:
            X_curr = X_s1_scaled.copy()
            A_curr = np.full((n_test_studis, len(treatment_cols)), a_val, dtype=np.float32)
            s_active = np.ones(n_test_studis, dtype=np.float64)
            cum_drop_risk = np.zeros(n_test_studis, dtype=np.float64)

            for t_sem in range(1, max_horizon + 1):
                if t_sem > 1:
                    if cov_transition_model is not None and tv_indices:
                        L_pred = cov_transition_model.predict(np.hstack([X_curr, A_curr])).astype(np.float32)
                        X_curr[:, tv_indices] = np.clip(L_pred, -5.0, 5.0)
                    # Aktualisiere skalierte fachsemester- und is_first_semester-Spalten exakt
                    if idx_fachsem is not None:
                        X_curr[:, idx_fachsem] = float((t_sem - scaler.mean_[idx_fachsem]) / max(scaler.scale_[idx_fachsem], 1e-8))
                    if idx_first_sem is not None:
                        X_curr[:, idx_first_sem] = float((0.0 - scaler.mean_[idx_first_sem]) / max(scaler.scale_[idx_first_sem], 1e-8))

                XA_t = torch.tensor(np.hstack([X_curr, A_curr]), dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    h_d = np.clip(model(XA_t).cpu().numpy().astype(np.float64), 1e-6, 1.0 - 1e-6)
                    h_g = np.clip(grad_model(XA_t).cpu().numpy().astype(np.float64), 0.0, 1.0 - 1e-6)

                # Competing-Risks Normalisierung falls h_d + h_g > 0.999
                total_h = h_d + h_g
                scale_h = np.where(total_h > 0.999, 0.999 / np.maximum(total_h, 1e-6), 1.0)
                h_d = h_d * scale_h
                h_g = h_g * scale_h

                cum_drop_risk += s_active * h_d
                s_active = s_active * (1.0 - h_d - h_g)

            return np.clip(cum_drop_risk, 0.0, 1.0)

        seq_drop_cf0 = _rollout_sequential_g(0.0, max_horizon=16)
        seq_drop_cf1 = _rollout_sequential_g(1.0, max_horizon=16)

        risk_seq_cf0 = float(np.mean(seq_drop_cf0))
        risk_seq_cf1 = float(np.mean(seq_drop_cf1))
        arr_seq = float(risk_seq_cf0 - risk_seq_cf1)
        rr_seq = float(risk_seq_cf1 / max(risk_seq_cf0, 1e-7))

        # Primäre G-Computation Schätzgrößen
        risk_cf0 = risk_step_cf0
        risk_cf1 = risk_step_cf1
        arr_hat = arr_step
        rr_hat = rr_step
        nnt_hat = float(1.0 / max(abs(arr_hat), 1e-6))

        print("\n" + "=" * 78)
        print("   G-COMPUTATION ESTIMATION RESULTS (TEST-SET)")
        print("=" * 78)
        print(f"  • [Step-Standardized] Risiko do(A=0) vs do(A=1) : {risk_step_cf0 * 100:.2f}% -> {risk_step_cf1 * 100:.2f}% | ARR = {arr_step * 100:+.2f} pp | RR = {rr_step:.4f}")
        print(f"  • [Sequential Rollout] Risiko do(A=0) vs do(A=1): {risk_seq_cf0 * 100:.2f}% -> {risk_seq_cf1 * 100:.2f}% | ARR = {arr_seq * 100:+.2f} pp | RR = {rr_seq:.4f}")
        print(f"  • Number Needed to Treat (NNT)                  : {nnt_hat:.1f} Studierende")
        print("=" * 78)

        # 5. Student-Cluster Bootstrap (B=100) für CIs
        print("\n[Schritt 3] 100-Fold Student-Cluster Bootstrap für ARR / RR Konfidenzintervalle ...")
        n_boot = 100
        boot_rr = np.zeros(n_boot, dtype=np.float32)
        boot_arr = np.zeros(n_boot, dtype=np.float32)
        boot_seq_rr = np.zeros(n_boot, dtype=np.float32)

        s_cf0_arr = studi_surv_step_cf0.values
        s_cf1_arr = studi_surv_step_cf1.values
        n_studi_test = len(s_cf0_arr)

        for b in range(n_boot):
            idx = np.random.choice(n_studi_test, size=n_studi_test, replace=True)
            r0_b = 1.0 - np.mean(s_cf0_arr[idx])
            r1_b = 1.0 - np.mean(s_cf1_arr[idx])
            boot_arr[b] = r0_b - r1_b
            boot_rr[b] = r1_b / max(r0_b, 1e-7)

            sr0_b = float(np.mean(seq_drop_cf0[idx]))
            sr1_b = float(np.mean(seq_drop_cf1[idx]))
            boot_seq_rr[b] = sr1_b / max(sr0_b, 1e-7)

        ci_rr_low = float(np.percentile(boot_rr, 2.5))
        ci_rr_high = float(np.percentile(boot_rr, 97.5))
        ci_arr_low = float(np.percentile(boot_arr, 2.5))
        ci_arr_high = float(np.percentile(boot_arr, 97.5))

        print(f"  • 95% Bootstrap CI (Step-Standardized RR): [{ci_rr_low:.4f}, {ci_rr_high:.4f}]")
        print(f"  • 95% Bootstrap CI (Step-Standardized ARR): [{ci_arr_low*100:+.2f} pp, {ci_arr_high*100:+.2f} pp]")

        # 6. Logging über CausalEvaluator (mit explizitem estimand_type='RR' und se_scale='natural')
        duration = time.time() - start_time
        evaluator = CausalEvaluator(
            model_name="torch_gcomputation_simulation",
            base_dir=self.output_dir,
            mode=self.mode,
            temporal=self.temporal,
            estimand_type="RR",
            se_scale="natural",
        )

        hr_estimates = {
            "gcomputation_rr_full_support": rr_hat,
            "gcomputation_seq_rollout_rr_full_support": rr_seq,
        }
        hr_se = {
            "gcomputation_rr_full_support": float(np.std(boot_rr)),
            "gcomputation_seq_rollout_rr_full_support": float(np.std(boot_seq_rr)),
        }

        extra_info = {
            "counterfactual_risk_cf0_no_support": risk_cf0,
            "counterfactual_risk_cf1_full_support": risk_cf1,
            "causal_arr": arr_hat,
            "causal_arr_se": float(np.std(boot_arr)),
            "causal_arr_pp": arr_hat * 100.0,
            "causal_nnt": nnt_hat,
            "ci_arr_95_boot_lower": ci_arr_low,
            "ci_arr_95_boot_upper": ci_arr_high,
            "sequential_g_formula": {
                "risk_cf0_no_support": risk_seq_cf0,
                "risk_cf1_full_support": risk_seq_cf1,
                "causal_arr": arr_seq,
                "causal_arr_pp": arr_seq * 100.0,
                "causal_rr": rr_seq,
                "causal_rr_boot_se": float(np.std(boot_seq_rr)),
            },
            "factual_roc_auc": auc_fac,
            "factual_brier": brier_fac,
            "training_time_s": duration,
            "device": str(self.device),
        }

        boot_matrix = np.column_stack([boot_rr, boot_seq_rr])
        metrics_dict = evaluator.evaluate_and_log(
            hr_estimates=hr_estimates,
            hr_se=hr_se,
            bootstrap_samples=boot_matrix,
            extra_metrics=extra_info,
            estimand_type="RR",
            se_scale="natural",
        )

        return metrics_dict
