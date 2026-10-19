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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# Import metrics_logger and feature_builder
sys.path.insert(0, str(Path(__file__).parent))
from metrics_logger import save_metrics, save_keras_model, plot_learning_curve, plot_roc_curve, plot_pr_curve
import feature_builder as fb


def train_dml_orthogonal_survival(data_dir: Path = Path('src/output_dl'),
                                  temporal: str = 'prev',
                                  mode: str = 'standard',
                                  epochs: int = 50,
                                  batch_size: int = 2048):
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

    train_panel = panel_df[panel_df['studierenden_id'].isin(train_ids)].copy()
    val_panel   = panel_df[panel_df['studierenden_id'].isin(val_ids)].copy()
    test_panel  = panel_df[panel_df['studierenden_id'].isin(test_ids)].copy()

    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='median')

    W_train = scaler.fit_transform(imputer.fit_transform(train_panel[confounder_cols]))
    W_val   = scaler.transform(imputer.transform(val_panel[confounder_cols]))
    W_test  = scaler.transform(imputer.transform(test_panel[confounder_cols]))

    # -------------------------------------------------------------------------
    # STUFE 1: TREATMENT CONDITIONAL EXPECTATION & RESIDUALS
    # -------------------------------------------------------------------------
    print("\n[Stufe 1] Schätze Treatment-Modelle E[A_t | W_t] (Ridge Regression) ...")
    residuals_train, residuals_val, residuals_test, a_hat_test_list = [], [], [], []

    for supp_col in treatment_cols:
        y_treat_train = train_panel[supp_col].values.astype(float)
        y_treat_val   = val_panel[supp_col].values.astype(float)
        y_treat_test  = test_panel[supp_col].values.astype(float)

        reg_model = Ridge(alpha=1.0)
        reg_model.fit(W_train, y_treat_train)

        a_hat_train = reg_model.predict(W_train)
        a_hat_val   = reg_model.predict(W_val)
        a_hat_test  = reg_model.predict(W_test)

        residuals_train.append(y_treat_train - a_hat_train)
        residuals_val.append(y_treat_val - a_hat_val)
        residuals_test.append(y_treat_test - a_hat_test)
        a_hat_test_list.append(a_hat_test)

    A_tilde_train = np.column_stack(residuals_train)
    A_tilde_val   = np.column_stack(residuals_val)
    A_tilde_test  = np.column_stack(residuals_test)
    A_hat_test    = np.column_stack(a_hat_test_list)

    # -------------------------------------------------------------------------
    # STUFE 2: NEURONALES DML DISCRETE-TIME HAZARD MODELL
    # -------------------------------------------------------------------------
    print("\n[Stufe 2] Trainiere Orthogonalisiertes DML-Hazard-Netzwerk ...")
    X_train_dml = np.hstack([W_train, A_tilde_train])
    X_val_dml   = np.hstack([W_val, A_tilde_val])
    X_test_dml  = np.hstack([W_test, A_tilde_test])

    y_train = train_panel[target_col].values
    y_val   = val_panel[target_col].values
    y_test  = test_panel[target_col].values

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
    print("\n[Stufe 3] Kausale Counterfactual-Inferenz (Isolierte und Partielle Support-Effekte) ...")
    test_h_factual = dml_model.predict(X_test_dml, verbose=0).flatten()
    auc_dml = float(roc_auc_score(y_test, test_h_factual))
    pr_auc_dml = float(average_precision_score(y_test, test_h_factual))
    brier_dml = float(brier_score_loss(y_test, test_h_factual))

    causal_results = {}
    for idx_treat, supp_col in enumerate(treatment_cols):
        # 1. Partielle Kontrafaktizität: Setze dieses Treatment A_k = 0 vs. Realwert
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

        short_name = supp_col.replace('_supp_count', '').replace('support_glz_', '')
        causal_results[short_name] = {
            "partial": {"mean_rr": rr_partial, "median_rr": rr_partial, "ate": ate_partial},
            "isolated": {"mean_rr": rr_isolated, "median_rr": rr_isolated, "ate": ate_isolated}
        }
        print(f"  • {short_name.upper():<14}: Partial RR = {rr_partial:.4f}, Isolated RR = {rr_isolated:.4f}, ATE = {ate_partial:+.4f}")

    print("\n" + "=" * 74)
    print("   ERGEBNISSE DOUBLE MACHINE LEARNING SURVIVAL (TEST-SET)")
    print("=" * 74)
    print(f"  • ROC-AUC (Factual Hazard): {auc_dml:.4f}")
    print(f"  • PR-AUC                  : {pr_auc_dml:.4f}")
    print(f"  • Brier Score             : {brier_dml:.4f}")
    print("=" * 74)

    # Logging
    base_dir = data_dir
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
    save_metrics(model_name, metrics_dict, base_dir)
    save_keras_model(dml_model, model_name, base_dir)
    plot_learning_curve(history.history, model_name, base_dir, metric_name='AUC')
    plot_roc_curve(y_test, test_h_factual, model_name, base_dir)
    plot_pr_curve(y_test, test_h_factual, model_name, base_dir)

    print(f"[OK] DML-Modell und Kausal-Inferenz erfolgreich gespeichert unter {base_dir}.")
    return dml_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Double Machine Learning Survival Model")
    parser.add_argument('--data_dir', type=str, default='src/output_dl')
    parser.add_argument('--temporal', type=str, default='prev', choices=['prev', 'cum'])
    parser.add_argument('--mode', type=str, default='standard')
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()

    train_dml_orthogonal_survival(Path(args.data_dir), temporal=args.temporal, mode=args.mode, epochs=args.epochs)
