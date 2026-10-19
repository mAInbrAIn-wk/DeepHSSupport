"""
Deep Survival Analysis (Landmark DeepSurv & Logistic Hazard)
============================================================
Trainiert DeepSurv (Breslow Partial Likelihood) und Discrete-Time Logistic Hazard
auf Landmark-Features (Semester 1–2) aus `feature_builder.py`.

Features:
- Breslow Non-Parametric Baseline Hazard H_0(t)
- Harrell's Concordance Index (C-Index)
- Discrete-Time Logistic Hazard Modell
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Import metrics_logger and feature_builder
sys.path.insert(0, str(Path(__file__).parent))
from metrics_logger import save_metrics, plot_learning_curve, save_keras_model, plot_roc_curve, plot_pr_curve
import feature_builder as fb


def breslow_cox_partial_loss(y_true, y_pred):
    """Entrauschte Cox Partial Log-Likelihood Loss Function mit Breslow Tie-Korrektur."""
    time = y_true[:, 0]
    event = y_true[:, 1]
    risk = y_pred[:, 0]

    sort_idx = tf.argsort(time, direction='DESCENDING')
    time_sorted = tf.gather(time, sort_idx)
    event_sorted = tf.gather(event, sort_idx)
    risk_sorted = tf.gather(risk, sort_idx)

    exp_risk = tf.exp(risk_sorted)
    cum_exp_risk = tf.cumsum(exp_risk)
    log_risk = risk_sorted - tf.math.log(cum_exp_risk + 1e-7)

    uncensored_loss = -tf.reduce_sum(log_risk * event_sorted)
    num_events = tf.reduce_sum(event_sorted) + 1e-7
    return uncensored_loss / num_events


def fast_c_index(time, event, risk_scores):
    """Vektorisierter C-Index."""
    order = np.argsort(time)
    time = time[order]
    event = event[order]
    risk_scores = risk_scores[order]

    event_mask = event == 1
    if not np.any(event_mask):
        return 0.5

    concordant, permissible = 0.0, 0.0
    for i in np.where(event_mask)[0]:
        greater = time > time[i]
        if np.any(greater):
            permissible += np.sum(greater)
            diff = risk_scores[i] - risk_scores[greater]
            concordant += np.sum(diff > 0) + 0.5 * np.sum(diff == 0)

    return float(concordant / permissible) if permissible > 0 else 0.5


def estimate_cumulative_baseline_hazard(time, event, risk_scores):
    """Schätzt die kumulative Baseline-Hazard-Funktion H_0(t) nach Breslow."""
    order = np.argsort(time)
    time_sorted = time[order]
    event_sorted = event[order]
    risk_sorted = risk_scores[order]

    unique_times = np.unique(time_sorted[event_sorted == 1])
    h0 = []
    exp_risk = np.exp(risk_sorted)

    for t in unique_times:
        at_risk = time_sorted >= t
        d_j = np.sum((time_sorted == t) & (event_sorted == 1))
        denom = np.sum(exp_risk[at_risk])
        h0.append(d_j / (denom + 1e-7))

    cum_h0 = np.cumsum(h0)
    return pd.DataFrame({'time': unique_times, 'cum_h0': cum_h0})


def predict_survival_function(baseline_hazard_df, risk_score):
    """Berechnet die individuelle Überlebensfunktion S(t|x) = exp(-H_0(t) * exp(risk))."""
    times = baseline_hazard_df['time'].values
    cum_h0 = baseline_hazard_df['cum_h0'].values
    surv_prob = np.exp(-cum_h0 * np.exp(risk_score))
    return pd.DataFrame({'time': times, 'survival_prob': surv_prob})


def train_deep_survival(data_dir: Path = Path('src/output_dl'),
                         mode: str = 'standard',
                         epochs_ds: int = 150,
                         epochs_lh: int = 50):
    print("\n" + "=" * 74)
    print(f"   LANDMARK DEEP SURVIVAL ANALYSIS (DeepSurv & Logistic Hazard | mode={mode})")
    print("=" * 74)

    df_lm, feature_cols, target_col, _ = fb.build_landmark_dataset(
        data_dir, t0=2, mode=mode, target='survival', target_type='survival'
    )

    y_surv = np.column_stack([df_lm['studiendauer_semester'].values, df_lm['is_dropout'].values])
    y_event = df_lm['is_dropout'].values

    print(f"Dataset: {len(df_lm)} Studierende, {len(feature_cols)} Landmark-Features")

    # 3-Way Split
    idx = np.arange(len(df_lm))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=42, stratify=y_event)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42, stratify=y_event[temp_idx])

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_train = scaler.fit_transform(imputer.fit_transform(df_lm.loc[train_idx, feature_cols]))
    X_val   = scaler.transform(imputer.transform(df_lm.loc[val_idx, feature_cols]))
    X_test  = scaler.transform(imputer.transform(df_lm.loc[test_idx, feature_cols]))

    input_dim = X_train.shape[1]

    # 1. Landmark DeepSurv
    print("\n[1/2] Trainiere Landmark DeepSurv ...")
    tf.random.set_seed(42)
    deepsurv = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        LayerNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        LayerNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        LayerNormalization(),
        Dense(1, activation='linear', use_bias=False)
    ])
    deepsurv.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=breslow_cox_partial_loss)
    es_ds = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

    hist_ds = deepsurv.fit(
        X_train, y_surv[train_idx],
        validation_data=(X_val, y_surv[val_idx]),
        epochs=epochs_ds, batch_size=min(4096, len(X_train)),
        callbacks=[es_ds], verbose=0
    )

    test_risk = deepsurv.predict(X_test, verbose=0).flatten()
    c_idx_ds = fast_c_index(y_surv[test_idx, 0], y_surv[test_idx, 1], test_risk)
    auc_ds = float(roc_auc_score(y_event[test_idx], test_risk))

    # 2. Landmark Logistic Hazard
    print("[2/2] Trainiere Landmark Logistic Hazard ...")
    tf.random.set_seed(42)
    lh = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        LayerNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        LayerNormalization(),
        Dense(1, activation='sigmoid')
    ])
    lh.compile(optimizer=tf.keras.optimizers.Adam(0.003), loss='binary_crossentropy', metrics=['AUC'])
    es_lh = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    hist_lh = lh.fit(
        X_train, y_event[train_idx],
        validation_data=(X_val, y_event[val_idx]),
        epochs=epochs_lh, batch_size=256,
        callbacks=[es_lh], verbose=0
    )

    test_p_lh = lh.predict(X_test, verbose=0).flatten()
    auc_lh = float(roc_auc_score(y_event[test_idx], test_p_lh))
    pr_auc_lh = float(average_precision_score(y_event[test_idx], test_p_lh))
    brier_lh = float(brier_score_loss(y_event[test_idx], test_p_lh))
    c_idx_lh = fast_c_index(y_surv[test_idx, 0], y_surv[test_idx, 1], test_p_lh)

    print("\n" + "=" * 74)
    print("   ERGEBNISSE LANDMARK DEEP SURVIVAL (TEST-SET)")
    print("=" * 74)
    print(f"  • DeepSurv C-Index          : {c_idx_ds:.4f}")
    print(f"  • DeepSurv ROC-AUC          : {auc_ds:.4f}")
    print(f"  • Logistic Hazard ROC-AUC   : {auc_lh:.4f}")
    print(f"  • Logistic Hazard PR-AUC    : {pr_auc_lh:.4f}")
    print(f"  • Logistic Hazard C-Index   : {c_idx_lh:.4f}")
    print(f"  • Logistic Hazard Brier     : {brier_lh:.4f}")
    print("=" * 74)

    # Logging
    base_dir = data_dir
    model_name_ds = f"deep_survival_{mode}" if mode != 'standard' else "deep_survival"
    model_name_lh = f"logistic_hazard_landmark_{mode}" if mode != 'standard' else "logistic_hazard_landmark"

    metrics_ds = {
        "C-Index": c_idx_ds,
        "ROC-AUC": auc_ds
    }
    save_metrics(model_name_ds, metrics_ds, base_dir)
    save_keras_model(deepsurv, model_name_ds, base_dir)
    plot_learning_curve(hist_ds.history, model_name_ds, base_dir, metric_name='loss')
    plot_roc_curve(y_event[test_idx], test_risk, model_name_ds, base_dir)

    metrics_lh = {
        "ROC-AUC": auc_lh,
        "PR-AUC": pr_auc_lh,
        "C-Index": c_idx_lh,
        "Brier_Score": brier_lh
    }
    save_metrics(model_name_lh, metrics_lh, base_dir)
    save_keras_model(lh, model_name_lh, base_dir)
    plot_learning_curve(hist_lh.history, model_name_lh, base_dir, metric_name='AUC')
    plot_roc_curve(y_event[test_idx], test_p_lh, model_name_lh, base_dir)
    plot_pr_curve(y_event[test_idx], test_p_lh, model_name_lh, base_dir)

    print(f"[OK] Landmark Survival Modelle gespeichert unter {base_dir}.")
    return deepsurv, lh


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Landmark Deep Survival")
    parser.add_argument('--data_dir', type=str, default='src/output_dl')
    parser.add_argument('--mode', type=str, default='standard')
    parser.add_argument('--epochs_ds', type=int, default=100)
    parser.add_argument('--epochs_lh', type=int, default=40)
    args = parser.parse_args()

    train_deep_survival(Path(args.data_dir), mode=args.mode, epochs_ds=args.epochs_ds, epochs_lh=args.epochs_lh)
