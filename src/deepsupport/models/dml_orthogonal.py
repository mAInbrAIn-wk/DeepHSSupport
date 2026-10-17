"""
Double / Debiased Machine Learning (DML) Orthogonalized Survival Model
========================================================================
Implementiert Chernozhukovs DML-Orthogonalisierung zur vollständigen Elimination
des reaktiven Confounding-Bias in neuronalen Survival-Panels.

Unterstützt über feature_builder.py:
- Alle Modi (standard, gradeblind, blind, oracle, realistic)
- Temporale Modi (temporal='prev' oder 'cum')

Stufe 1: Propensity / Treatment-Erwartung E[A_t | W_t] -> Treatment-Residuum A_tilde_t
Stufe 2: Neuronales DML Hazard-Modell auf orthogonalisierten Residuen A_tilde_t
Stufe 3: Kausale Counterfactual-Inferenz (RR & ATE)
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# Import metrics_logger and feature_builder
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deepsupport.evaluation.metrics_logger import save_metrics, save_keras_model, plot_learning_curve, plot_roc_curve, plot_pr_curve, CausalEvaluator
import deepsupport.data_engine.feature_builder as fb


def train_dml_orthogonal_survival(data_dir: Path = Path('src/output_dl'),
                                  temporal: str = 'prev',
                                  mode: str = 'standard',
                                  epochs: int = 50,
                                  batch_size: int = 2048,
                                  output_dir: Path = None):
    if output_dir is None:
        out_env = os.environ.get("OUTPUT_DIR")
        output_dir = Path(out_env) if out_env else Path(data_dir)
    else:
        output_dir = Path(output_dir)

    print("\n" + "=" * 74)
    print(f"   DOUBLE MACHINE LEARNING (DML) ORTHOGONALIZED SURVIVAL (temporal={temporal}, mode={mode})")
    print("=" * 74)

    panel_df, feature_cols, target_col, _ = fb.build_semester_panel_df(
        data_dir, mode=mode, temporal=temporal
    )

    treatment_candidates = ['fach_supp_count', 'uebf_supp_count', 'psych_supp_count']
    treatment_cols = [c for c in treatment_candidates if c in feature_cols]
    confounder_cols = [c for c in feature_cols if c not in treatment_cols]

    print(f"Panel: {len(panel_df)} Zeilen | Confounder: {len(confounder_cols)} | Treatments: {treatment_cols}")

    # 3-Way Student Group Split
    unique_studis = np.array(panel_df['studierenden_id'].unique().tolist())
    train_ids, temp_ids = train_test_split(unique_studis, test_size=0.30, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=42)

    train_panel = panel_df[panel_df['studierenden_id'].isin(train_ids)].copy().reset_index(drop=True)
    val_panel   = panel_df[panel_df['studierenden_id'].isin(val_ids)].copy().reset_index(drop=True)
    test_panel  = panel_df[panel_df['studierenden_id'].isin(test_ids)].copy().reset_index(drop=True)

    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='median')

    W_train = scaler.fit_transform(imputer.fit_transform(train_panel[confounder_cols]))
    W_val   = scaler.transform(imputer.transform(val_panel[confounder_cols]))
    W_test  = scaler.transform(imputer.transform(test_panel[confounder_cols]))

    y_train = train_panel[target_col].values.astype(float)
    y_val   = val_panel[target_col].values.astype(float)
    y_test  = test_panel[target_col].values.astype(float)

    # -------------------------------------------------------------------------
    # STUFE 1: 5-FOLD GROUPKFOLD CROSS-FITTING (TREATMENTS & OUTCOME)
    # -------------------------------------------------------------------------
    print("\n[Stufe 1] 5-Fold GroupKFold Cross-Fitting für E[A_t | W_t] und E[Y_t | W_t] ...")
    n_folds = 5
    gkf = GroupKFold(n_splits=n_folds)
    studis_tr = train_panel['studierenden_id'].values

    n_treat = len(treatment_cols)
    A_tilde_train = np.zeros((len(train_panel), n_treat), dtype=np.float64)
    A_hat_val     = np.zeros((len(val_panel), n_treat), dtype=np.float64)
    A_hat_test    = np.zeros((len(test_panel), n_treat), dtype=np.float64)
    y_tilde_train = np.zeros(len(train_panel), dtype=np.float64)

    for f_tr_idx, f_va_idx in gkf.split(train_panel, groups=studis_tr):
        W_f_tr, W_f_va = W_train[f_tr_idx], W_train[f_va_idx]
        y_f_tr, y_f_va = y_train[f_tr_idx], y_train[f_va_idx]

        out_reg = Ridge(alpha=1.0)
        out_reg.fit(W_f_tr, y_f_tr)
        y_tilde_train[f_va_idx] = y_f_va - out_reg.predict(W_f_va)

        for k, supp_col in enumerate(treatment_cols):
            A_f_tr = train_panel.loc[f_tr_idx, supp_col].values.astype(float)
            A_f_va = train_panel.loc[f_va_idx, supp_col].values.astype(float)

            reg_model = Ridge(alpha=1.0)
            reg_model.fit(W_f_tr, A_f_tr)

            A_tilde_train[f_va_idx, k] = A_f_va - reg_model.predict(W_f_va)
            A_hat_val[:, k]  += reg_model.predict(W_val) / n_folds
            A_hat_test[:, k] += reg_model.predict(W_test) / n_folds

    A_val_raw  = val_panel[treatment_cols].values.astype(float)
    A_test_raw = test_panel[treatment_cols].values.astype(float)
    A_tilde_val  = A_val_raw - A_hat_val
    A_tilde_test = A_test_raw - A_hat_test

    # Multivariater Neyman-Orthogonaler Robinson-Schätzer mit Cluster-Sandwich-SE
    bread_inv = np.linalg.inv(A_tilde_train.T @ A_tilde_train + 1e-8 * np.eye(n_treat))
    theta_vec = bread_inv @ (A_tilde_train.T @ y_tilde_train)
    eps_vec = y_tilde_train - A_tilde_train @ theta_vec

    _, cluster_inv_tr = np.unique(studis_tr, return_inverse=True)
    n_clusters_tr = int(cluster_inv_tr.max()) + 1
    cluster_scores = np.zeros((n_clusters_tr, n_treat), dtype=np.float64)
    np.add.at(cluster_scores, cluster_inv_tr, A_tilde_train * eps_vec[:, None])
    meat = (n_clusters_tr / max(n_clusters_tr - 1, 1)) * (cluster_scores.T @ cluster_scores)
    se_theta_vec = np.sqrt(np.maximum(np.diag(bread_inv @ meat @ bread_inv), 1e-12))

    A_train_raw = train_panel[treatment_cols].values.astype(float)
    untreated_mask = np.sum(A_train_raw, axis=1) == 0
    baseline_y0 = max(float(np.mean(y_train[untreated_mask])) if np.any(untreated_mask) else float(np.mean(y_train)), 1e-4)

    # -------------------------------------------------------------------------
    # STUFE 2: NEURONALES DML DISCRETE-TIME HAZARD MODELL
    # -------------------------------------------------------------------------
    print("\n[Stufe 2] Trainiere Orthogonalisiertes DML-Hazard-Netzwerk ...")
    X_train_dml = np.hstack([W_train, A_tilde_train])
    X_val_dml   = np.hstack([W_val, A_tilde_val])
    X_test_dml  = np.hstack([W_test, A_tilde_test])

    tf.random.set_seed(42)
    dml_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_dml.shape[1],)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    dml_model.compile(optimizer=tf.keras.optimizers.Adam(0.003), loss='binary_crossentropy', metrics=['AUC'])
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    history = dml_model.fit(
        X_train_dml, y_train,
        validation_data=(X_val_dml, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=[es], verbose=0
    )

    # -------------------------------------------------------------------------
    # STUFE 3: KAUSALE INFERENZ & COUNTERFACTUAL SIMULATION
    # -------------------------------------------------------------------------
    print("\n[Stufe 3] Kausale Counterfactual-Inferenz (Robinson & Neuronale Counterfactuals) ...")
    test_h_factual = dml_model.predict(X_test_dml, verbose=0).flatten()
    auc_dml = float(roc_auc_score(y_test, test_h_factual))
    pr_auc_dml = float(average_precision_score(y_test, test_h_factual))
    brier_dml = float(brier_score_loss(y_test, test_h_factual))

    causal_results = {}
    hr_estimates = {}
    hr_se = {}
    name_map = {'fach': 'fachlich', 'uebf': 'ueberfachlich', 'psych': 'psychosozial'}

    for idx_treat, supp_col in enumerate(treatment_cols):
        # 1. Partielle Kontrafaktizität: Setze dieses Treatment A_k = 0 vs. 1
        A_tilde_cf0 = A_tilde_test.copy()
        A_tilde_cf0[:, idx_treat] = 0.0 - A_hat_test[:, idx_treat]
        X_test_cf0 = np.hstack([W_test, A_tilde_cf0])
        h_cf0 = dml_model.predict(X_test_cf0, verbose=0).flatten()

        A_tilde_cf1 = A_tilde_test.copy()
        A_tilde_cf1[:, idx_treat] = 1.0 - A_hat_test[:, idx_treat]
        X_test_cf1 = np.hstack([W_test, A_tilde_cf1])
        h_cf1 = dml_model.predict(X_test_cf1, verbose=0).flatten()

        rr_partial = float(np.mean(h_cf1) / (np.mean(h_cf0) + 1e-7))
        ate_partial = float(np.mean(h_cf1 - h_cf0))

        # 2. Isolierte Kontrafaktizität: Alle anderen Treatments A_j = 0
        A_tilde_iso0 = np.zeros_like(A_tilde_test)
        A_tilde_iso1 = np.zeros_like(A_tilde_test)
        for j in range(len(treatment_cols)):
            A_tilde_iso0[:, j] = 0.0 - A_hat_test[:, j]
            A_tilde_iso1[:, j] = (1.0 if j == idx_treat else 0.0) - A_hat_test[:, j]

        h_iso0 = dml_model.predict(np.hstack([W_test, A_tilde_iso0]), verbose=0).flatten()
        h_iso1 = dml_model.predict(np.hstack([W_test, A_tilde_iso1]), verbose=0).flatten()

        rr_isolated = float(np.mean(h_iso1) / (np.mean(h_iso0) + 1e-7))
        ate_isolated = float(np.mean(h_iso1 - h_iso0))

        theta_k = float(theta_vec[idx_treat])
        se_k = float(se_theta_vec[idx_treat])
        robinson_hr = float(max(0.05, 1.0 + theta_k / baseline_y0))
        robinson_hr_se = float(se_k / baseline_y0)

        short_name = supp_col.replace('_supp_count', '').replace('support_glz_', '')
        full_name = name_map.get(short_name, short_name)
        hr_estimates[f"hr_{full_name}"] = robinson_hr
        hr_se[f"hr_{full_name}"] = robinson_hr_se

        causal_results[short_name] = {
            "robinson": {"ate": theta_k, "ate_se": se_k, "implied_hr": robinson_hr, "implied_hr_se": robinson_hr_se},
            "partial": {"mean_rr": rr_partial, "median_rr": rr_partial, "ate": ate_partial},
            "isolated": {"mean_rr": rr_isolated, "median_rr": rr_isolated, "ate": ate_isolated}
        }
        print(f"  • {short_name.upper():<14}: Robinson HR = {robinson_hr:.4f} (ATE={theta_k:+.4f}), Partial RR = {rr_partial:.4f}, Isolated RR = {rr_isolated:.4f}")

    print("\n" + "=" * 74)
    print("   ERGEBNISSE DOUBLE MACHINE LEARNING SURVIVAL (TEST-SET)")
    print("=" * 74)
    print(f"  • ROC-AUC (Factual Hazard): {auc_dml:.4f}")
    print(f"  • PR-AUC                  : {pr_auc_dml:.4f}")
    print(f"  • Brier Score             : {brier_dml:.4f}")
    print("=" * 74)

    # Logging
    base_dir = output_dir
    model_name = f"dml_orthogonal_survival_{temporal}_{mode}" if (temporal != 'prev' or mode != 'standard') else "dml_orthogonal_survival"

    metrics_dict = {
        "model_type": model_name,
        "temporal": temporal,
        "mode": mode,
        "ROC-AUC_Panel": auc_dml,
        "PR-AUC_Panel": pr_auc_dml,
        "Brier_Score": brier_dml,
        **causal_results
    }

    evaluator = CausalEvaluator(
        base_dir=base_dir,
        model_name=model_name,
        temporal=temporal,
        mode=mode,
        estimand_type="HR",
        se_scale="natural",
    )
    evaluator.evaluate_and_log(
        hr_estimates=hr_estimates,
        hr_se=hr_se,
        extra_metrics=metrics_dict,
        keras_model=dml_model,
        history=history.history,
        mode=mode,
        temporal_type=temporal,
        estimand_type="HR",
        se_scale="natural",
    )
    save_metrics(model_name, metrics_dict, base_dir)
    save_keras_model(dml_model, model_name, base_dir)

    print(f"[OK] DML-Modell und Kausal-Inferenz erfolgreich gespeichert unter {base_dir}.")
    return dml_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Double Machine Learning Survival Model")
    parser.add_argument('--data_dir', type=str, default='src/output_dl')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--temporal', type=str, default='prev', choices=['prev', 'cum'])
    parser.add_argument('--mode', type=str, default='standard')
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()

    train_dml_orthogonal_survival(
        Path(args.data_dir),
        temporal=args.temporal,
        mode=args.mode,
        epochs=args.epochs,
        output_dir=Path(args.output_dir) if args.output_dir else None
    )
