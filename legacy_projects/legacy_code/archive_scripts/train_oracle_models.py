"""
Oracle Models (Theoretical Predictability Upper Bound)
======================================================
Trainiert Orakel-Modelle (DeepSurv & Logistic Hazard) mit direktem Zugriff
auf die latenten Simulationsvariablen (hidden_motivation_prev, hidden_soziale_integration_prev,
hidden_erwartete_note_prev) aus `feature_builder.py` (mode='oracle'),
um den theoretischen maximalen ROC-AUC Lift zu bestimmen.
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
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input

# Import metrics_logger, extended_deep_survival, and feature_builder
sys.path.insert(0, str(Path(__file__).parent))
from extended_deep_survival import breslow_cox_loss
from metrics_logger import save_metrics, save_keras_model
import feature_builder as fb


def build_logistic_hazard(input_dim: int) -> Sequential:
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss='binary_crossentropy')
    return model


def build_deepsurv(input_dim: int) -> Model:
    inputs = Input(shape=(input_dim,))
    x = Dense(32, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    x = BatchNormalization()(x)
    risk = Dense(1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=risk)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=breslow_cox_loss)
    return model


def train_oracle_models(data_dir: Path = Path('src/output_dl'),
                        temporal: str = 'prev',
                        epochs_ds: int = 50,
                        epochs_lh: int = 25):
    print("\n" + "=" * 74)
    print(f"   ORACLE MODELS BENCHMARK (THEORETICAL MAXIMUM LIFT | temporal={temporal})")
    print("=" * 74)

    # 1. Base Panel (Observable features)
    panel_base, cols_base, target_col, _ = fb.build_semester_panel_df(data_dir, mode='standard', temporal=temporal)

    # 2. Oracle Panel (Observable + Hidden features)
    panel_oracle, cols_oracle, _, _ = fb.build_semester_panel_df(data_dir, mode='oracle', temporal=temporal)

    print(f"Features: Base = {len(cols_base)}, Oracle = {len(cols_oracle)}")

    # 3-Way Student Split
    unique_studis = np.array(panel_base['studierenden_id'].unique().tolist())
    train_ids, test_ids = train_test_split(unique_studis, test_size=0.20, random_state=42)

    tr_mask = panel_base['studierenden_id'].isin(train_ids)
    te_mask = panel_base['studierenden_id'].isin(test_ids)

    scaler_base = StandardScaler()
    scaler_oracle = StandardScaler()
    imputer = SimpleImputer(strategy='median')

    X_tr_base = scaler_base.fit_transform(imputer.fit_transform(panel_base.loc[tr_mask, cols_base]))
    X_te_base = scaler_base.transform(imputer.transform(panel_base.loc[te_mask, cols_base]))

    X_tr_ora = scaler_oracle.fit_transform(imputer.fit_transform(panel_oracle.loc[tr_mask, cols_oracle]))
    X_te_ora = scaler_oracle.transform(imputer.transform(panel_oracle.loc[te_mask, cols_oracle]))

    y_tr_surv = np.column_stack([panel_base.loc[tr_mask, 't_stop'].values, panel_base.loc[tr_mask, target_col].values])
    y_te_surv = np.column_stack([panel_base.loc[te_mask, 't_stop'].values, panel_base.loc[te_mask, target_col].values])

    y_tr_event = panel_base.loc[tr_mask, target_col].values
    y_te_event = panel_base.loc[te_mask, target_col].values

    # Train Baseline Models
    print("\n[1/4] Trainiere Baseline Logistic Hazard...")
    tf.random.set_seed(42)
    lh_base = build_logistic_hazard(X_tr_base.shape[1])
    lh_base.fit(X_tr_base, y_tr_event, epochs=epochs_lh, batch_size=2048, verbose=0)
    p_lh_base = lh_base.predict(X_te_base, verbose=0).flatten()

    print("[2/4] Trainiere Oracle Logistic Hazard...")
    lh_oracle = build_logistic_hazard(X_tr_ora.shape[1])
    lh_oracle.fit(X_tr_ora, y_tr_event, epochs=epochs_lh, batch_size=2048, verbose=0)
    p_lh_oracle = lh_oracle.predict(X_te_ora, verbose=0).flatten()

    print("[3/4] Trainiere Baseline DeepSurv...")
    ds_base = build_deepsurv(X_tr_base.shape[1])
    ds_base.fit(X_tr_base, y_tr_surv, epochs=epochs_ds, batch_size=min(4096, len(X_tr_base)), verbose=0)
    p_ds_base = ds_base.predict(X_te_base, verbose=0).flatten()

    print("[4/4] Trainiere Oracle DeepSurv...")
    ds_oracle = build_deepsurv(X_tr_ora.shape[1])
    ds_oracle.fit(X_tr_ora, y_tr_surv, epochs=epochs_ds, batch_size=min(4096, len(X_tr_ora)), verbose=0)
    p_ds_oracle = ds_oracle.predict(X_te_ora, verbose=0).flatten()

    # Evaluation
    auc_lh_base = float(roc_auc_score(y_te_event, p_lh_base))
    auc_lh_ora = float(roc_auc_score(y_te_event, p_lh_oracle))

    auc_ds_base = float(roc_auc_score(y_te_event, p_ds_base))
    auc_ds_ora = float(roc_auc_score(y_te_event, p_ds_oracle))

    lift_lh = auc_lh_ora - auc_lh_base
    lift_ds = auc_ds_ora - auc_ds_base

    print("\n" + "=" * 74)
    print("   ERGEBNISSE ORACLE LIFT EVALUATION (TEST-SET)")
    print("=" * 74)
    print(f"  • Logistic Hazard Base ROC-AUC   : {auc_lh_base:.4f}")
    print(f"  • Logistic Hazard Oracle ROC-AUC : {auc_lh_ora:.4f}  (Lift: {lift_lh:+.4f})")
    print(f"  • DeepSurv Base ROC-AUC          : {auc_ds_base:.4f}")
    print(f"  • DeepSurv Oracle ROC-AUC        : {auc_ds_ora:.4f}  (Lift: {lift_ds:+.4f})")
    print("=" * 74)

    # Logging
    base_dir = data_dir
    metrics_dict = {
        "ROC-AUC_Baseline_LogisticHazard": auc_lh_base,
        "ROC-AUC_Oracle_LogisticHazard": auc_lh_ora,
        "ROC-AUC_Lift_LogisticHazard": lift_lh,
        "ROC-AUC_Baseline_DeepSurv": auc_ds_base,
        "ROC-AUC_Oracle_DeepSurv": auc_ds_ora,
        "ROC-AUC_Lift_DeepSurv": lift_ds
    }
    save_metrics("oracle_lift", metrics_dict, base_dir)
    save_keras_model(lh_oracle, "oracle_logistic_hazard", base_dir)
    save_keras_model(ds_oracle, "oracle_deepsurv", base_dir)

    print(f"[OK] Oracle-Modelle und Lift-Metriken erfolgreich gespeichert.")
    return metrics_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Oracle Lift Benchmark")
    parser.add_argument('--data_dir', type=str, default='src/output_dl')
    parser.add_argument('--temporal', type=str, default='prev', choices=['prev', 'cum'])
    args = parser.parse_args()

    train_oracle_models(Path(args.data_dir), temporal=args.temporal)
