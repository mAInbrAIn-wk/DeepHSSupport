"""
Erwerb-Blind / DSGVO Realistic Model Training
=============================================
Trainiert Modelle ohne hochsensible oder geschützte Merkmale
(keine Erwerbstätigkeit, kein Migrationshintergrund, kein Psychosozial-Support, keine Noten bei Bedarf).
Vergleicht Full Baseline vs. Realistic DSGVO-Konforme Modelle.
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# Import metrics_logger and feature_builder
sys.path.insert(0, str(Path(__file__).parent))
from metrics_logger import save_metrics, save_keras_model
import feature_builder as fb


def build_hazard_model(input_dim: int) -> Sequential:
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss='binary_crossentropy', metrics=['AUC'])
    return model


def train_erwerb_blind_models(data_dir: Path = Path('src/output_dl'),
                              temporal: str = 'prev',
                              epochs: int = 30):
    print("\n" + "=" * 74)
    print(f"   DSGVO / REALISTIC FEATURE-BLIND SURVIVAL (temporal={temporal})")
    print("=" * 74)

    # 1. Full Standard Panel
    panel_full, cols_full, target_col, _ = fb.build_semester_panel_df(data_dir, mode='standard', temporal=temporal)

    # 2. Realistic Panel (DSGVO-compliant: No Erwerb, No Migrationshintergrund, No Psych Support)
    panel_real, cols_real, _, _ = fb.build_semester_panel_df(data_dir, mode='realistic', temporal=temporal)

    print(f"Features: Full = {len(cols_full)}, Realistic/DSGVO = {len(cols_real)}")

    unique_studis = np.array(panel_full['studierenden_id'].unique().tolist())
    train_ids, test_ids = train_test_split(unique_studis, test_size=0.20, random_state=42)

    tr_mask = panel_full['studierenden_id'].isin(train_ids)
    te_mask = panel_full['studierenden_id'].isin(test_ids)

    scaler_full = StandardScaler()
    scaler_real = StandardScaler()
    imputer = SimpleImputer(strategy='median')

    X_tr_full = scaler_full.fit_transform(imputer.fit_transform(panel_full.loc[tr_mask, cols_full]))
    X_te_full = scaler_full.transform(imputer.transform(panel_full.loc[te_mask, cols_full]))

    X_tr_real = scaler_real.fit_transform(imputer.fit_transform(panel_real.loc[tr_mask, cols_real]))
    X_te_real = scaler_real.transform(imputer.transform(panel_real.loc[te_mask, cols_real]))

    y_tr = panel_full.loc[tr_mask, target_col].values
    y_te = panel_full.loc[te_mask, target_col].values

    # Train Full Model
    print("\n[1/2] Trainiere Full Baseline Hazard Model...")
    tf.random.set_seed(42)
    m_full = build_hazard_model(X_tr_full.shape[1])
    m_full.fit(X_tr_full, y_tr, epochs=epochs, batch_size=2048, verbose=0)
    p_full = m_full.predict(X_te_full, verbose=0).flatten()

    # Train Realistic Model
    print("[2/2] Trainiere Realistic DSGVO Hazard Model...")
    tf.random.set_seed(42)
    m_real = build_hazard_model(X_tr_real.shape[1])
    m_real.fit(X_tr_real, y_tr, epochs=epochs, batch_size=2048, verbose=0)
    p_real = m_real.predict(X_te_real, verbose=0).flatten()

    auc_full = float(roc_auc_score(y_te, p_full))
    pr_full = float(average_precision_score(y_te, p_full))

    auc_real = float(roc_auc_score(y_te, p_real))
    pr_real = float(average_precision_score(y_te, p_real))

    drop_auc = auc_full - auc_real

    print("\n" + "=" * 74)
    print("   ERGEBNISSE DSGVO-REALISTIC SURVIVAL (TEST-SET)")
    print("=" * 74)
    print(f"  • Full Model ROC-AUC      : {auc_full:.4f}")
    print(f"  • Realistic Model ROC-AUC : {auc_real:.4f}  (Verlust: {-drop_auc:+.4f})")
    print("=" * 74)

    # Logging
    base_dir = data_dir
    metrics_dict = {
        "ROC-AUC_Full": auc_full,
        "PR-AUC_Full": pr_full,
        "ROC-AUC_Realistic": auc_real,
        "PR-AUC_Realistic": pr_real,
        "ROC-AUC_Drop": drop_auc
    }
    save_metrics("erwerb_blind_models", metrics_dict, base_dir)
    save_keras_model(m_real, "realistic_logistic_hazard", base_dir)

    print(f"[OK] DSGVO-Modelle erfolgreich gespeichert unter {base_dir}.")
    return metrics_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DSGVO Realistic Survival Training")
    parser.add_argument('--data_dir', type=str, default='src/output_dl')
    parser.add_argument('--temporal', type=str, default='prev', choices=['prev', 'cum'])
    args = parser.parse_args()

    train_erwerb_blind_models(Path(args.data_dir), temporal=args.temporal)
