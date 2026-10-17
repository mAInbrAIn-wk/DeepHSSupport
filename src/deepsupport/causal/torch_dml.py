"""
PyTorch Double / Debiased Machine Learning (DML) Survival Suite
================================================================
Implementiert Chernozhukovs DML-Orthogonalisierung (Chernozhukov et al., 2018)
zur vollständigen Beseitigung von Confounding-Bias in neuronalen Verlaufsdaten.

Architektur:
- Stufe 1: Nuisance-Schätzung über Pre-LayerNorm PyTorch MLPs mit 5-Fold Student-Cluster-Fitting:
  * Propensity-Netzwerk E[A_k | W] -> Treatment-Residuen A_tilde_k
  * Outcome-Netzwerk E[Y | W] -> Ereignis-Residuen Y_tilde
- Stufe 2:
  * Analytischer Robinson-Orthogonal-Schätzer mit Neyman-Sandwich-Standardfehlern
  * Neuronales DML Discrete-Time Hazard Netzwerk auf [W, A_tilde]
- Stufe 3: Kausale Counterfactual-Inferenz:
  * Partielle und isolierte Kontrafaktizität (Relative Risk RR, ATE, Hazard Ratio HR)
  * Asymptotische 95% CIs und 100-Fold Cluster-Bootstrap CIs auf Studierendenebene
- Protokollierung über CausalEvaluator (Forest-Plots, JSON-Metriken, Markdown)
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import deepsupport.data_engine.feature_builder as fb
from deepsupport.evaluation.metrics_logger import CausalEvaluator, save_metrics, get_output_dirs


# =====================================================================
# PYTORCH NUISANCE & DML NETZWERKE
# =====================================================================

class PyTorchNuisanceNet(nn.Module):
    """Pre-LayerNorm MLP zur Nuisance-Modellierung E[A|W] oder E[Y|W]."""
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.1, is_binary: bool = True):
        super().__init__()
        self.is_binary = is_binary
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x).squeeze(-1)
        if self.is_binary:
            return logits
        return logits


class PyTorchDMLHazardNet(nn.Module):
    """Neuronales Stufe-2 DML-Hazard-Netzwerk auf orthogonalisierten Residuen."""
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))


# =====================================================================
# TRAININGSHILFEN FÜR PYTORCH NUISANCE-MODELLE
# =====================================================================

def _train_nuisance_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    is_binary: bool = True,
    epochs: int = 15,
    batch_size: int = 2048,
    lr: float = 3e-3,
    device: torch.device = torch.device("cpu")
) -> nn.Module:
    """Trainiert ein einzelnes Nuisance-Netzwerk mit Early Stopping."""
    in_dim = X_train.shape[1]
    model = PyTorchNuisanceNet(in_dim=in_dim, hidden_dim=64, is_binary=is_binary).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    t_X_tr = torch.tensor(X_train, dtype=torch.float32)
    t_y_tr = torch.tensor(y_train, dtype=torch.float32)
    t_X_va = torch.tensor(X_val, dtype=torch.float32).to(device)
    t_y_va = torch.tensor(y_val, dtype=torch.float32).to(device)

    dataset = TensorDataset(t_X_tr, t_y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_loss = float("inf")
    best_weights = None
    patience = 4
    patience_cnt = 0

    for epoch in range(epochs):
        model.train()
        for b_X, b_y in loader:
            b_X, b_y = b_X.to(device), b_y.to(device)
            optimizer.zero_grad()
            pred = model(b_X)
            if is_binary:
                loss = F.binary_cross_entropy_with_logits(pred, b_y)
            else:
                loss = F.mse_loss(pred, b_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            v_pred = model(t_X_va)
            if is_binary:
                val_loss = F.binary_cross_entropy_with_logits(v_pred, t_y_va).item()
            else:
                val_loss = F.mse_loss(v_pred, t_y_va).item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_weights:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    return model


@torch.no_grad()
def _predict_nuisance(model: nn.Module, X: np.ndarray, is_binary: bool, device: torch.device) -> np.ndarray:
    model.eval()
    t_X = torch.tensor(X, dtype=torch.float32).to(device)
    out = model(t_X)
    if is_binary:
        probs = torch.sigmoid(out).cpu().numpy()
        return probs
    return out.cpu().numpy()


# =====================================================================
# HAUPTKLASSE: PYTORCH DML SURVIVAL
# =====================================================================

class PyTorchDMLSurvival:
    """
    Kausaler Double Machine Learning (DML) Survival Estimator in reinem PyTorch.
    Führt 5-Fold Cross-Fitting über Studierende durch, residualisiert Confounder
    und berechnet unkonfundierte Treatment-Effekte (RR, ATE, HR).
    """

    def __init__(
        self,
        data_dir: Path,
        output_dir: Optional[Path] = None,
        mode: str = "standard",
        temporal: str = "prev",
        n_folds: int = 5,
        epochs_nuisance: int = 15,
        epochs_dml: int = 25,
        batch_size: int = 2048,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir
        self.mode = mode
        self.temporal = temporal
        self.n_folds = n_folds
        self.epochs_nuisance = epochs_nuisance
        self.epochs_dml = epochs_dml
        self.batch_size = batch_size
        self.seed = seed

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        torch.manual_seed(seed)
        np.random.seed(seed)

    def fit_and_evaluate(self) -> Dict[str, Any]:
        """Führt den gesamten DML-Prozess aus und protokolliert über CausalEvaluator."""
        start_time = time.time()
        print("\n" + "=" * 78)
        print("   PYTORCH DOUBLE MACHINE LEARNING (DML) SURVIVAL ESTIMATION")
        print(f"   Modus: {self.mode} | Temporal: {self.temporal} | Device: {self.device}")
        print("=" * 78)

        # 1. Panel-Daten laden
        panel_df, feature_cols, target_col, _ = fb.build_semester_panel_df(
            self.data_dir, mode=self.mode, temporal=self.temporal
        )

        treatment_candidates = ["fach_supp_count", "uebf_supp_count", "psych_supp_count"]
        treatment_cols = [c for c in treatment_candidates if c in feature_cols]
        confounder_cols = [c for c in feature_cols if c not in treatment_cols]

        print(f"Panel: {len(panel_df)} Zeilen | {panel_df['studierenden_id'].nunique()} Studierende")
        print(f"Confounder ({len(confounder_cols)}): {confounder_cols}")
        print(f"Treatments ({len(treatment_cols)}): {treatment_cols}")

        # 2. Student-konsistenter 70/15/15 Split (identisch zur übrigen Suite)
        unique_studis = np.array(panel_df["studierenden_id"].unique().tolist())
        train_ids, temp_ids = train_test_split(unique_studis, test_size=0.30, random_state=self.seed)
        val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=self.seed)

        df_train = panel_df[panel_df["studierenden_id"].isin(train_ids)].copy().reset_index(drop=True)
        df_val   = panel_df[panel_df["studierenden_id"].isin(val_ids)].copy().reset_index(drop=True)
        df_test  = panel_df[panel_df["studierenden_id"].isin(test_ids)].copy().reset_index(drop=True)

        # Skalierung & Imputation der Confounder
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        W_train = scaler.fit_transform(imputer.fit_transform(df_train[confounder_cols]))
        W_val   = scaler.transform(imputer.transform(df_val[confounder_cols]))
        W_test  = scaler.transform(imputer.transform(df_test[confounder_cols]))

        y_train = df_train[target_col].values.astype(np.float32)
        y_val   = df_val[target_col].values.astype(np.float32)
        y_test  = df_test[target_col].values.astype(np.float32)

        # =====================================================================
        # STUFE 1: 5-FOLD CROSS-FITTING DER NUISANCE-MODELLE
        # =====================================================================
        print(f"\n[Stufe 1] 5-Fold Cross-Fitting der Nuisance-Modelle auf Trainingskohorte ...")
        gkf = GroupKFold(n_splits=self.n_folds)
        studis_tr = df_train["studierenden_id"].values

        # Out-of-fold Residuen-Container
        A_tilde_train = np.zeros((len(df_train), len(treatment_cols)), dtype=np.float32)
        A_hat_test_folds = np.zeros((len(df_test), len(treatment_cols)), dtype=np.float32)
        A_hat_val_folds  = np.zeros((len(df_val), len(treatment_cols)), dtype=np.float32)

        # Outcome-Nuisance
        y_tilde_train = np.zeros(len(df_train), dtype=np.float32)
        y_hat_test_folds = np.zeros(len(df_test), dtype=np.float32)

        for fold, (f_tr_idx, f_va_idx) in enumerate(gkf.split(df_train, groups=studis_tr)):
            W_f_tr, W_f_va = W_train[f_tr_idx], W_train[f_va_idx]
            y_f_tr, y_f_va = y_train[f_tr_idx], y_train[f_va_idx]

            # 1. Outcome Model m(W) = E[Y | W]
            out_model = _train_nuisance_model(
                W_f_tr, y_f_tr, W_f_va, y_f_va,
                is_binary=True, epochs=self.epochs_nuisance, batch_size=self.batch_size, device=self.device
            )
            y_hat_oof = _predict_nuisance(out_model, W_f_va, is_binary=True, device=self.device)
            y_tilde_train[f_va_idx] = y_f_va - y_hat_oof
            y_hat_test_folds += _predict_nuisance(out_model, W_test, is_binary=True, device=self.device) / self.n_folds

            # 2. Treatment Models e_k(W) = E[A_k | W]
            for k, treat_col in enumerate(treatment_cols):
                A_f_tr = df_train.loc[f_tr_idx, treat_col].values.astype(np.float32)
                A_f_va = df_train.loc[f_va_idx, treat_col].values.astype(np.float32)
                is_bin = (len(np.unique(A_f_tr)) <= 2)

                treat_model = _train_nuisance_model(
                    W_f_tr, A_f_tr, W_f_va, A_f_va,
                    is_binary=is_bin, epochs=self.epochs_nuisance, batch_size=self.batch_size, device=self.device
                )
                a_hat_oof = _predict_nuisance(treat_model, W_f_va, is_binary=is_bin, device=self.device)
                A_tilde_train[f_va_idx, k] = A_f_va - a_hat_oof
                A_hat_test_folds[:, k] += _predict_nuisance(treat_model, W_test, is_binary=is_bin, device=self.device) / self.n_folds
                A_hat_val_folds[:, k]  += _predict_nuisance(treat_model, W_val, is_binary=is_bin, device=self.device) / self.n_folds

        # Test- und Val-Residuen
        A_test = df_test[treatment_cols].values.astype(np.float32)
        A_val  = df_val[treatment_cols].values.astype(np.float32)
        A_tilde_test = A_test - A_hat_test_folds
        A_tilde_val  = A_val - A_hat_val_folds

        y_tilde_test = y_test - y_hat_test_folds

        # =====================================================================
        # STUFE 2a: MULTIVARIATER ROBINSON DML SCHÄTZER (NEYMAN-ORTHOGONAL)
        # =====================================================================
        print("\n[Stufe 2a] Multivariater Robinson DML Schätzer (Neyman-Orthogonal, Cluster-Sandwich SE) ...")
        n_treat = len(treatment_cols)
        A_tilde_64 = A_tilde_train.astype(np.float64)
        y_tilde_64 = y_tilde_train.astype(np.float64)

        # Gemeinsame multivariate Normalgleichung: theta_hat = (A_tilde^T A_tilde)^(-1) A_tilde^T y_tilde
        bread_inv = np.linalg.inv(A_tilde_64.T @ A_tilde_64 + 1e-8 * np.eye(n_treat))
        theta_vec = bread_inv @ (A_tilde_64.T @ y_tilde_64)
        eps_vec = y_tilde_64 - A_tilde_64 @ theta_vec

        # Student-Clustered Liang-Zeger Sandwich-Kovarianzmatrix
        _, cluster_inv_tr = np.unique(studis_tr, return_inverse=True)
        n_clusters_tr = int(cluster_inv_tr.max()) + 1
        score_rows = A_tilde_64 * eps_vec[:, None]  # (N_rows, K)
        cluster_scores = np.zeros((n_clusters_tr, n_treat), dtype=np.float64)
        np.add.at(cluster_scores, cluster_inv_tr, score_rows)
        meat = (n_clusters_tr / max(n_clusters_tr - 1, 1)) * (cluster_scores.T @ cluster_scores)
        cov_theta = bread_inv @ meat @ bread_inv
        se_theta_vec = np.sqrt(np.maximum(np.diag(cov_theta), 1e-12))

        # Baseline-Hazard-Rate (ohne Support im Intervall) zur Umrechnung von ATE auf implizites HR/RR
        A_train_raw = df_train[treatment_cols].values.astype(np.float64)
        untreated_mask = np.sum(A_train_raw, axis=1) == 0
        baseline_y0 = float(np.mean(y_train[untreated_mask])) if np.any(untreated_mask) else float(np.mean(y_train))
        baseline_y0 = max(baseline_y0, 1e-4)

        robinson_effects = {}
        causal_hr_estimates = {}
        causal_hr_se = {}

        for k, treat_col in enumerate(treatment_cols):
            theta_hat = float(theta_vec[k])
            se_hat = float(se_theta_vec[k])
            hr_robinson = float(max(0.05, 1.0 + theta_hat / baseline_y0))
            hr_se_robinson = float(se_hat / baseline_y0)

            short_name = treat_col.replace("_supp_count", "").replace("support_glz_", "")
            robinson_effects[f"dml_ate_{short_name}"] = theta_hat
            robinson_effects[f"dml_ate_se_{short_name}"] = se_hat
            robinson_effects[f"dml_robinson_hr_{short_name}"] = hr_robinson
            causal_hr_estimates[f"hr_{short_name}"] = hr_robinson
            causal_hr_se[f"hr_{short_name}"] = hr_se_robinson
            print(
                f"  • {short_name.upper():<14}: ATE = {theta_hat:+.5f} (SE={se_hat:.5f}) "
                f"| Implied HR = {hr_robinson:.4f} (SE={hr_se_robinson:.4f})"
            )

        # =====================================================================
        # STUFE 2b: NEURONALES DML DISCRETE-TIME HAZARD NETZWERK
        # =====================================================================
        print("\n[Stufe 2b] Trainiere Neuronales DML-Hazard-Netzwerk auf [W, A_tilde] ...")
        X_tr_dml = np.hstack([W_train, A_tilde_train])
        X_va_dml = np.hstack([W_val, A_tilde_val])
        X_te_dml = np.hstack([W_test, A_tilde_test])

        dml_net = PyTorchDMLHazardNet(in_dim=X_tr_dml.shape[1], hidden_dim=64).to(self.device)
        optimizer = torch.optim.AdamW(dml_net.parameters(), lr=2e-3, weight_decay=1e-4)

        t_X_tr = torch.tensor(X_tr_dml, dtype=torch.float32)
        t_y_tr = torch.tensor(y_train, dtype=torch.float32)
        t_X_va = torch.tensor(X_va_dml, dtype=torch.float32).to(self.device)
        t_y_va = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        dml_loader = DataLoader(TensorDataset(t_X_tr, t_y_tr), batch_size=self.batch_size, shuffle=True)

        best_loss = float("inf")
        best_state = None
        for epoch in range(self.epochs_dml):
            dml_net.train()
            for b_X, b_y in dml_loader:
                b_X, b_y = b_X.to(self.device), b_y.to(self.device)
                optimizer.zero_grad()
                logits = dml_net.forward_logits(b_X)
                loss = F.binary_cross_entropy_with_logits(logits, b_y)
                loss.backward()
                optimizer.step()

            dml_net.eval()
            with torch.no_grad():
                v_logits = dml_net.forward_logits(t_X_va)
                v_loss = F.binary_cross_entropy_with_logits(v_logits, t_y_va).item()
            if v_loss < best_loss:
                best_loss = v_loss
                best_state = {k: v.cpu().clone() for k, v in dml_net.state_dict().items()}

        if best_state:
            dml_net.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        # =====================================================================
        # STUFE 3: KONTRAFAKTISCHE INFERENZ & CLUSTER-BOOTSTRAP
        # =====================================================================
        print("\n[Stufe 3] Kausale Counterfactual-Inferenz auf Testkohorte ...")
        dml_net.eval()
        with torch.no_grad():
            test_h_factual = dml_net(torch.tensor(X_te_dml, dtype=torch.float32).to(self.device)).cpu().numpy()

        auc_dml = float(roc_auc_score(y_test, test_h_factual))
        pr_auc_dml = float(average_precision_score(y_test, test_h_factual))
        brier_dml = float(brier_score_loss(y_test, test_h_factual))

        all_h_iso0 = []
        all_h_iso1 = []

        for k, treat_col in enumerate(treatment_cols):
            short_name = treat_col.replace("_supp_count", "").replace("support_glz_", "")

            # 1. Partielle Kontrafaktizität: A_k = 1 vs A_k = 0 (andere Merkmale unverändert)
            A_tilde_cf0 = A_tilde_test.copy()
            A_tilde_cf0[:, k] = 0.0 - A_hat_test_folds[:, k]
            A_tilde_cf1 = A_tilde_test.copy()
            A_tilde_cf1[:, k] = 1.0 - A_hat_test_folds[:, k]

            with torch.no_grad():
                h_cf0 = dml_net(torch.tensor(np.hstack([W_test, A_tilde_cf0]), dtype=torch.float32).to(self.device)).cpu().numpy()
                h_cf1 = dml_net(torch.tensor(np.hstack([W_test, A_tilde_cf1]), dtype=torch.float32).to(self.device)).cpu().numpy()

            rr_partial = float(np.mean(h_cf1) / max(np.mean(h_cf0), 1e-7))
            ate_partial = float(np.mean(h_cf1 - h_cf0))

            # 2. Isolierte Kontrafaktizität: Alle anderen Treatments = 0
            A_tilde_iso0 = np.zeros_like(A_tilde_test)
            A_tilde_iso1 = np.zeros_like(A_tilde_test)
            for j in range(len(treatment_cols)):
                A_tilde_iso0[:, j] = 0.0 - A_hat_test_folds[:, j]
                A_tilde_iso1[:, j] = (1.0 if j == k else 0.0) - A_hat_test_folds[:, j]

            with torch.no_grad():
                h_iso0 = dml_net(torch.tensor(np.hstack([W_test, A_tilde_iso0]), dtype=torch.float32).to(self.device)).cpu().numpy()
                h_iso1 = dml_net(torch.tensor(np.hstack([W_test, A_tilde_iso1]), dtype=torch.float32).to(self.device)).cpu().numpy()

            all_h_iso0.append(h_iso0)
            all_h_iso1.append(h_iso1)

            rr_isolated = float(np.mean(h_iso1) / max(np.mean(h_iso0), 1e-7))
            ate_isolated = float(np.mean(h_iso1 - h_iso0))

            robinson_effects[f"neural_plug_in_rr_isolated_{short_name}"] = rr_isolated
            robinson_effects[f"neural_plug_in_rr_partial_{short_name}"] = rr_partial
            robinson_effects[f"neural_plug_in_ate_isolated_{short_name}"] = ate_isolated
            robinson_effects[f"neural_plug_in_ate_partial_{short_name}"] = ate_partial

            print(
                f"  • {short_name.upper():<14}: Robinson HR = {causal_hr_estimates[f'hr_{short_name}']:.4f} "
                f"| Neural Isolated RR = {rr_isolated:.4f} | Neural ATE = {ate_isolated:+.4f}"
            )

        # 3. Student-Cluster Bootstrap (B=100) mit Multinomial-Cluster-Gewichten (ohne .isin()-Deduplizierung!)
        print("\n[Stufe 3b] Berechne 100-Fold Student-Cluster Bootstrap (Multinomial-Gewichte) ...")
        n_boot = 100
        boot_samples = np.zeros((n_boot, n_treat), dtype=np.float32)
        boot_neural_rr = np.zeros((n_boot, n_treat), dtype=np.float32)

        _, cluster_inv_te = np.unique(df_test["studierenden_id"].values, return_inverse=True)
        n_clusters_te = int(cluster_inv_te.max()) + 1
        probs_tr = np.ones(n_clusters_tr, dtype=np.float64) / n_clusters_tr
        probs_te = np.ones(n_clusters_te, dtype=np.float64) / n_clusters_te

        for b in range(n_boot):
            # Multinomial-Gewichte für Trainings-Cluster (Robinson-Schätzer)
            counts_tr = np.random.multinomial(n_clusters_tr, probs_tr).astype(np.float64)
            w_tr = counts_tr[cluster_inv_tr]
            Aw = A_tilde_64 * w_tr[:, None]
            theta_b = np.linalg.lstsq(Aw.T @ A_tilde_64 + 1e-8 * np.eye(n_treat), Aw.T @ y_tilde_64, rcond=None)[0]
            y0_b = float(np.average(y_train[untreated_mask], weights=w_tr[untreated_mask])) if np.any(untreated_mask) and np.sum(w_tr[untreated_mask]) > 0 else baseline_y0
            boot_samples[b, :] = np.maximum(0.05, 1.0 + theta_b / max(y0_b, 1e-4)).astype(np.float32)

            # Multinomial-Gewichte für Test-Cluster (Neuronales Plug-In)
            counts_te = np.random.multinomial(n_clusters_te, probs_te).astype(np.float64)
            w_te = counts_te[cluster_inv_te]
            w_te_sum = max(float(np.sum(w_te)), 1e-8)
            for k in range(n_treat):
                m0 = float(np.sum(all_h_iso0[k] * w_te) / w_te_sum)
                m1 = float(np.sum(all_h_iso1[k] * w_te) / w_te_sum)
                boot_neural_rr[b, k] = float(m1 / max(m0, 1e-7))

        for k, treat_col in enumerate(treatment_cols):
            short_name = treat_col.replace("_supp_count", "").replace("support_glz_", "")
            robinson_effects[f"neural_plug_in_rr_boot_se_{short_name}"] = float(np.std(boot_neural_rr[:, k]))

        # =====================================================================
        # LOGGING VIA CAUSAL EVALUATOR
        # =====================================================================
        duration = time.time() - start_time
        evaluator = CausalEvaluator(
            model_name="torch_dml_orthogonal_survival",
            base_dir=self.output_dir,
            mode=self.mode,
            temporal=self.temporal,
            estimand_type="HR",
            se_scale="natural",
        )

        extra_info = {
            "factual_roc_auc": auc_dml,
            "factual_pr_auc": pr_auc_dml,
            "factual_brier": brier_dml,
            "baseline_unassisted_hazard": baseline_y0,
            "training_time_s": duration,
            "device": str(self.device),
            **robinson_effects,
        }

        metrics_dict = evaluator.evaluate_and_log(
            hr_estimates=causal_hr_estimates,
            hr_se=causal_hr_se,
            bootstrap_samples=boot_samples,
            extra_metrics=extra_info,
            estimand_type="HR",
            se_scale="natural",
        )

        # PyTorch Modellgewichte speichern
        models_dir = self.output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        torch.save(dml_net.state_dict(), models_dir / f"torch_dml_hazard_net_{self.mode}_{self.temporal}.pt")
        print(f"[INFO] PyTorch DML Modell gespeichert: {models_dir}")

        return metrics_dict
