"""
Extended Neural Survival Models (Time-Varying & Delta Panel Edition)
====================================================================
Trainiert Extended DeepSurv (Neuronales Cox-Modell mit Breslow-Loss)
und Extended Discrete-Time Logistic Hazard auf dem Person-Semester-Panel.

Unterstützt über feature_builder.py:
- Alle Modi: standard, gradeblind, blind, oracle, realistic
- Temporale Varianten: temporal='prev' (Vorsemester/Delta) oder temporal='cum' (Gesamthistorie)
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LayerNormalization

# Import metrics_logger and feature_builder
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deepsupport.evaluation.metrics_logger import save_metrics, save_keras_model, plot_learning_curve, plot_roc_curve, plot_pr_curve, SurvivalEvaluator
import deepsupport.data_engine.feature_builder as fb


def cox_counting_process_breslow_loss(y_true, y_pred):
    """
    Keras Loss für Counting-Process Partial Likelihood bei intervallzensierten Person-Semestern.
    y_true[:, 0]: t_start (falls 3 Spalten)
    y_true[:, 1]: t_stop  (bzw. y_true[:, 0] falls 2 Spalten)
    y_true[:, 2]: event   (bzw. y_true[:, 1] falls 2 Spalten)
    y_pred[:, 0]: log-risk (r_i)
    """
    y_true = tf.cast(y_true, tf.float32)
    risk = tf.cast(y_pred[:, 0], tf.float32)

    if y_true.shape[-1] == 3:
        t_start = y_true[:, 0]
        t_stop = y_true[:, 1]
        event = y_true[:, 2]
    else:
        t_stop = y_true[:, 0]
        event = y_true[:, 1]
        t_start = tf.zeros_like(t_stop)

    t_stop_i = tf.expand_dims(t_stop, 1)    # (B, 1)
    t_start_j = tf.expand_dims(t_start, 0)  # (1, B)
    t_stop_j = tf.expand_dims(t_stop, 0)    # (1, B)

    # Exaktes Counting-Process Risikoset R(t_i) = {j : t_start_j < t_stop_i <= t_stop_j}
    in_risk_set = tf.logical_and(t_start_j < t_stop_i, t_stop_j >= t_stop_i)

    shape_bb = tf.shape(in_risk_set)
    risk_matrix = tf.broadcast_to(tf.expand_dims(risk, 0), shape_bb)
    neg_inf_matrix = tf.fill(shape_bb, tf.constant(-1e9, dtype=tf.float32))
    masked_risks = tf.where(in_risk_set, risk_matrix, neg_inf_matrix)

    log_denom = tf.reduce_logsumexp(masked_risks, axis=1)

    uncensored_loss = -tf.reduce_sum((risk - log_denom) * event)
    num_events = tf.reduce_sum(event) + 1e-7
    return uncensored_loss / num_events


breslow_cox_loss = cox_counting_process_breslow_loss


def estimate_counting_process_baseline_hazard(t_start, t_stop, event, risk_scores):
    """
    Schätzt die diskreten Breslow Baseline-Hazard-Inkremente dh_0(t) und die
    kumulative Baseline-Hazard H_0(t) unter Berücksichtigung von (t_start, t_stop].
    """
    t_start = np.asarray(t_start, dtype=np.float64)
    t_stop = np.asarray(t_stop, dtype=np.float64)
    event = np.asarray(event, dtype=np.float64)
    exp_risk = np.exp(np.clip(np.asarray(risk_scores, dtype=np.float64), -20.0, 20.0))

    unique_times = np.sort(np.unique(t_stop))
    dh0_map = {}
    cum_h0_map = {}
    cum_val = 0.0

    for t in unique_times:
        at_risk = (t_start < t) & (t_stop >= t)
        d_t = np.sum((t_stop == t) & (event == 1.0))
        denom = np.sum(exp_risk[at_risk])
        dh = float(d_t / (denom + 1e-7)) if denom > 0 else 0.0
        cum_val += dh
        dh0_map[float(t)] = dh
        cum_h0_map[float(t)] = cum_val

    return dh0_map, cum_h0_map


def train_extended_deep_survival(data_dir: Path = Path('src/output_dl'),
                                temporal: str = 'prev',
                                mode: str = 'standard',
                                epochs_ds: int = 150,
                                epochs_lh: int = 60):
    print("\n" + "=" * 74)
    print(f"   EXTENDED NEURAL SURVIVAL MODELS (DeepSurv & Logistic Hazard | temporal={temporal}, mode={mode})")
    print("=" * 74)

    panel_df, feature_cols, target_col, _ = fb.build_semester_panel_df(
        data_dir, mode=mode, temporal=temporal
    )

    # Identifiziere Spaltentypen
    cat_candidates = ['stg_name', 'erstakademiker', 'hzb_typ', 'migrationshintergrund']
    cat_cols = [c for c in cat_candidates if c in feature_cols]
    
    treatment_candidates = ['fach_supp_count', 'uebf_supp_count', 'psych_supp_count']
    treatment_cols = [c for c in treatment_candidates if c in feature_cols]

    num_cols = [c for c in feature_cols if c not in cat_cols and c not in treatment_cols]

    print(f"Panel: {len(panel_df)} Zeilen | Features: {len(feature_cols)} (Num: {len(num_cols)}, Cat: {len(cat_cols)}, Treat: {len(treatment_cols)})")

    # 3-Way Student Group Split (kein Student-Leakage zwischen Train, Val und Test)
    unique_studis = np.array(panel_df['studierenden_id'].unique().tolist())
    train_ids, temp_ids = train_test_split(unique_studis, test_size=0.30, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=42)

    train_panel = panel_df[panel_df['studierenden_id'].isin(train_ids)].copy()
    val_panel = panel_df[panel_df['studierenden_id'].isin(val_ids)].copy()
    test_panel = panel_df[panel_df['studierenden_id'].isin(test_ids)].copy()

    transformers = []
    if num_cols:
        transformers.append(('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols))
    if cat_cols:
        transformers.append(('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))]), cat_cols))
    if treatment_cols:
        transformers.append(('treatments', 'passthrough', treatment_cols))

    preprocessor = ColumnTransformer(transformers)

    X_train = preprocessor.fit_transform(train_panel[feature_cols])
    X_val = preprocessor.transform(val_panel[feature_cols])
    X_test = preprocessor.transform(test_panel[feature_cols])

    y_train_surv = np.column_stack([
        train_panel['t_start'].values,
        train_panel['t_stop'].values,
        train_panel[target_col].values,
    ])
    y_val_surv = np.column_stack([
        val_panel['t_start'].values,
        val_panel['t_stop'].values,
        val_panel[target_col].values,
    ])
    y_test_surv = np.column_stack([
        test_panel['t_start'].values,
        test_panel['t_stop'].values,
        test_panel[target_col].values,
    ])

    input_dim = X_train.shape[1]

    # -------------------------------------------------------------------------
    # 1. EXTENDED DEEPSURV (NEURONALES COX MODELL)
    # -------------------------------------------------------------------------
    print("\n[1/2] Trainiere Extended DeepSurv ...")
    tf.random.set_seed(42)

    deepsurv = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        LayerNormalization(),
        Dropout(0.15),
        Dense(64, activation='relu'),
        LayerNormalization(),
        Dropout(0.15),
        Dense(32, activation='relu'),
        LayerNormalization(),
        Dropout(0.15),
        Dense(16, activation='relu'),
        LayerNormalization(),
        Dense(1, activation='linear', use_bias=False)
    ])

    deepsurv.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=cox_counting_process_breslow_loss)
    es_ds = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    history_ds = deepsurv.fit(
        X_train, y_train_surv,
        validation_data=(X_val, y_val_surv),
        epochs=epochs_ds, batch_size=min(4096, len(X_train)),
        callbacks=[es_ds], verbose=0
    )

    train_risk = deepsurv.predict(X_train, verbose=0).flatten()
    test_risk = deepsurv.predict(X_test, verbose=0).flatten()

    # Kalibrierte Intervall-Ausfallwahrscheinlichkeiten via Breslow Baseline Hazard
    dh0_map, _ = estimate_counting_process_baseline_hazard(
        train_panel['t_start'].values,
        train_panel['t_stop'].values,
        train_panel[target_col].values,
        train_risk,
    )
    dh0_test = np.array([dh0_map.get(float(t), 0.0) for t in test_panel['t_stop'].values], dtype=np.float64)
    calibrated_prob_ds = 1.0 - np.exp(-dh0_test * np.exp(np.clip(test_risk, -20.0, 20.0)))
    calibrated_prob_ds = np.clip(calibrated_prob_ds, 0.0, 1.0)

    # -------------------------------------------------------------------------
    # 2. EXTENDED DISCRETE-TIME HAZARD (LOGISTIC HAZARD)
    # -------------------------------------------------------------------------
    print("[2/2] Trainiere Extended Logistic Hazard ...")
    tf.random.set_seed(42)

    dtl_hazard = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])

    dtl_hazard.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss='binary_crossentropy', metrics=['AUC'])
    es_lh = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    history_lh = dtl_hazard.fit(
        X_train, train_panel[target_col].values,
        validation_data=(X_val, val_panel[target_col].values),
        epochs=epochs_lh, batch_size=2048,
        callbacks=[es_lh], verbose=0
    )

    test_h_pred = dtl_hazard.predict(X_test, verbose=0).flatten()

    # -------------------------------------------------------------------------
    # BEWERTUNG
    # -------------------------------------------------------------------------
    auc_deepsurv = float(roc_auc_score(test_panel[target_col], calibrated_prob_ds))
    auc_dtl = float(roc_auc_score(test_panel[target_col], test_h_pred))
    brier_ds = float(brier_score_loss(test_panel[target_col], calibrated_prob_ds))
    brier_dtl = float(brier_score_loss(test_panel[target_col], test_h_pred))
    pr_auc_ds = float(average_precision_score(test_panel[target_col], calibrated_prob_ds))
    pr_auc_dtl = float(average_precision_score(test_panel[target_col], test_h_pred))

    print("\n" + "=" * 74)
    print("   ERGEBNISSE NEURONALE SURVIVAL-MODELLE (TEST-SET)")
    print("=" * 74)
    print(f"{'Modell-Typ':<35} | {'ROC-AUC':<10} | {'PR-AUC':<10} | {'Brier Score':<12}")
    print("-" * 74)
    print(f"{'Extended DeepSurv':<35} | {auc_deepsurv:<10.4f} | {pr_auc_ds:<10.4f} | {brier_ds:<12.4f}")
    print(f"{'Extended Logistic Hazard':<35} | {auc_dtl:<10.4f} | {pr_auc_dtl:<10.4f} | {brier_dtl:<12.4f}")
    print("=" * 74)

    # -------------------------------------------------------------------------
    # METRICS LOGGING & MODEL SAVING
    # -------------------------------------------------------------------------
    base_dir = data_dir
    ds_name = f"extended_deepsurv_{temporal}_{mode}"
    lh_name = f"extended_logistic_hazard_{temporal}_{mode}"

    metrics_ds = {
        "model_type": ds_name,
        "temporal": temporal,
        "mode": mode,
        "ROC-AUC_Panel": auc_deepsurv,
        "PR-AUC_Panel": pr_auc_ds,
        "Brier_Score": brier_ds,
    }
    evaluator_ds = SurvivalEvaluator(base_dir=base_dir, model_name="extended_deepsurv", mode=mode, temporal=temporal)
    evaluator_ds.evaluate_and_log(
        y_true=test_panel[target_col].values,
        y_prob=calibrated_prob_ds,
        t_stop=test_panel['t_stop'].values,
        model=deepsurv,
        mode=mode,
        temporal_type=temporal,
        history=history_ds.history if hasattr(history_ds, 'history') else history_ds,
        extra_metrics=metrics_ds
    )

    # Abwärtskompatible Standard-Modellnamen
    save_metrics(f"extended_deepsurv_{temporal}", metrics_ds, base_dir, mode=mode, temporal_type=temporal)
    save_keras_model(deepsurv, f"extended_deepsurv_{temporal}", base_dir)
    if temporal == 'prev' and mode == 'standard':
        save_metrics("extended_deepsurv_delta", metrics_ds, base_dir, mode=mode, temporal_type=temporal)
        save_keras_model(deepsurv, "extended_deepsurv_delta", base_dir)

    metrics_dtl = {
        "model_type": lh_name,
        "temporal": temporal,
        "mode": mode,
        "ROC-AUC_Panel": auc_dtl,
        "PR-AUC_Panel": pr_auc_dtl,
        "Brier_Score": brier_dtl
    }
    evaluator_lh = SurvivalEvaluator(base_dir=base_dir, model_name="extended_logistic_hazard", mode=mode, temporal=temporal)
    evaluator_lh.evaluate_and_log(
        y_true=test_panel[target_col].values,
        y_prob=test_h_pred,
        t_stop=test_panel['t_stop'].values,
        model=dtl_hazard,
        mode=mode,
        temporal_type=temporal,
        history=history_lh.history if hasattr(history_lh, 'history') else history_lh,
        extra_metrics=metrics_dtl
    )

    save_metrics(f"extended_logistic_hazard_{temporal}", metrics_dtl, base_dir)
    save_keras_model(dtl_hazard, f"extended_logistic_hazard_{temporal}", base_dir)
    if temporal == 'prev' and mode == 'standard':
        save_metrics("extended_logistic_hazard_delta", metrics_dtl, base_dir)
        save_keras_model(dtl_hazard, "extended_logistic_hazard_delta", base_dir)

    print(f"\n[OK] Training und Logging für {ds_name} und {lh_name} abgeschlossen.")
    return deepsurv, dtl_hazard


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extended Neural Survival Models")
    parser.add_argument('--data_dir', type=str, default='src/output_dl')
    parser.add_argument('--temporal', type=str, default='prev', choices=['prev', 'cum'])
    parser.add_argument('--mode', type=str, default='standard')
    parser.add_argument('--epochs_ds', type=int, default=100)
    parser.add_argument('--epochs_lh', type=int, default=40)
    args = parser.parse_args()

    train_extended_deep_survival(
        data_dir=Path(args.data_dir),
        temporal=args.temporal,
        mode=args.mode,
        epochs_ds=args.epochs_ds,
        epochs_lh=args.epochs_lh
    )
