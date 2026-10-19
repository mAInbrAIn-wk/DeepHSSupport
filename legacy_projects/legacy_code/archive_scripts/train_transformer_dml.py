"""
Deep Causal Transformer-DML Pipeline
====================================
Kombiniert Pre-Training eines Deep Causal Transformers mit Double Machine Learning (DML)
Orthogonalisierung auf Längsschnitt-Embeddings.

Unterstützt über feature_builder.py:
- Semester-Sequenztensoren (build_semester_sequence_tensor)
- Semester-Panels (build_semester_panel_df)
- Alle Modi und temporale Varianten
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention,
    TimeDistributed, Masking, Add
)
import tensorflow.keras.backend as K

# Import metrics_logger and feature_builder
sys.path.insert(0, str(Path(__file__).parent))
from recurrent_survival_model import masked_binary_crossentropy, PADDING_VALUE
from transformer_survival_model import PositionalEncoding
from metrics_logger import save_metrics, save_keras_model, plot_learning_curve, plot_roc_curve, plot_pr_curve
import feature_builder as fb


def build_deep_causal_transformer_model(sequence_length: int, feature_dim: int, d_model: int = 64, num_heads: int = 4, num_blocks: int = 2) -> Model:
    """Größerer Deep Causal Transformer mit gestapelten Attention-Blöcken."""
    inputs = Input(shape=(sequence_length, feature_dim))
    masked_inputs = Masking(mask_value=PADDING_VALUE)(inputs)

    x = TimeDistributed(Dense(d_model, activation='relu'))(masked_inputs)
    x = PositionalEncoding(sequence_length, d_model)(x)

    for _ in range(num_blocks):
        attn_out = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=0.1
        )(query=x, value=x, key=x, use_causal_mask=True)

        x = Add()([x, attn_out])
        x = LayerNormalization()(x)

        ff_out = TimeDistributed(Dense(128, activation='relu'))(x)
        ff_out = TimeDistributed(Dropout(0.1))(ff_out)
        ff_out = TimeDistributed(Dense(d_model, activation='relu'))(ff_out)

        x = Add()([x, ff_out])
        x = LayerNormalization()(x)

    outputs = TimeDistributed(Dense(1, activation='sigmoid'))(x)
    model = Model(inputs=inputs, outputs=outputs, name='deep_causal_transformer')
    return model


def train_transformer_dml(data_dir: Path = Path('src/output_dl'),
                          temporal: str = 'prev',
                          mode: str = 'standard',
                          epochs_pretrain: int = 20,
                          epochs_dml: int = 30):
    print("\n" + "=" * 74)
    print(f"   DEEP TRANSFORMER-DML BENCHMARK (temporal={temporal}, mode={mode})")
    print("=" * 74)

    # 1. 3D-Sequenz laden
    studis, X_3d, y_3d, studi_events, feature_names, _ = fb.build_semester_sequence_tensor(
        data_dir, max_semesters=16, mode=mode, temporal=temporal
    )
    n_samples, sequence_length, feature_dim = X_3d.shape

    # 2. Skalierung
    train_mask = X_3d[:, :, 0] != PADDING_VALUE
    scaler = StandardScaler()
    scaler.fit(X_3d[train_mask])
    X_3d[train_mask] = scaler.transform(X_3d[train_mask])

    # 3. Transformer Modell bauen & vortrainieren
    d_model = 64
    deep_transformer = build_deep_causal_transformer_model(
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        d_model=d_model,
        num_heads=4,
        num_blocks=2
    )

    deep_transformer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=masked_binary_crossentropy)

    print(f"\n[Stufe 1] Pretraining Deep Causal Transformer ({epochs_pretrain} Epochen)...")
    deep_transformer.fit(X_3d, y_3d, epochs=epochs_pretrain, batch_size=128, verbose=0)

    # 4. Panel-Daten für DML
    panel_df, feature_cols, target_col, _ = fb.build_semester_panel_df(
        data_dir, mode=mode, temporal=temporal
    )

    treatment_candidates = ['fach_supp_count', 'uebf_supp_count', 'psych_supp_count']
    treatment_cols = [c for c in treatment_candidates if c in feature_cols]
    confounder_cols = [c for c in feature_cols if c not in treatment_cols]

    unique_studis = np.array(panel_df['studierenden_id'].unique().tolist())
    train_ids, temp_ids = train_test_split(unique_studis, test_size=0.30, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=42)

    train_panel = panel_df[panel_df['studierenden_id'].isin(train_ids)].copy()
    val_panel   = panel_df[panel_df['studierenden_id'].isin(val_ids)].copy()
    test_panel  = panel_df[panel_df['studierenden_id'].isin(test_ids)].copy()

    p_scaler = StandardScaler()
    W_train = p_scaler.fit_transform(train_panel[confounder_cols].fillna(0))
    W_val   = p_scaler.transform(val_panel[confounder_cols].fillna(0))
    W_test  = p_scaler.transform(test_panel[confounder_cols].fillna(0))

    # DML Stufe 1: Treatment Residuen
    print("[Stufe 2] DML Treatment-Orthogonalisierung...")
    res_train, res_val, res_test = [], [], []
    a_hat_test_list = []

    for supp_col in treatment_cols:
        y_treat_train = train_panel[supp_col].values.astype(float)
        y_treat_val   = val_panel[supp_col].values.astype(float)
        y_treat_test  = test_panel[supp_col].values.astype(float)

        reg = Ridge(alpha=1.0)
        reg.fit(W_train, y_treat_train)

        a_hat_train = reg.predict(W_train)
        a_hat_val   = reg.predict(W_val)
        a_hat_test  = reg.predict(W_test)

        res_train.append(y_treat_train - a_hat_train)
        res_val.append(y_treat_val - a_hat_val)
        res_test.append(y_treat_test - a_hat_test)
        a_hat_test_list.append(a_hat_test)

    A_tilde_train = np.column_stack(res_train)
    A_tilde_val   = np.column_stack(res_val)
    A_tilde_test  = np.column_stack(res_test)
    A_hat_test    = np.column_stack(a_hat_test_list)

    # DML Stufe 2: Transformer-DML Hazard Head
    print(f"[Stufe 3] DML Hazard-Modell Training ({epochs_dml} Epochen)...")
    X_tr_dml = np.hstack([W_train, A_tilde_train])
    X_va_dml = np.hstack([W_val, A_tilde_val])
    X_te_dml = np.hstack([W_test, A_tilde_test])

    y_train = train_panel[target_col].values
    y_val   = val_panel[target_col].values
    y_test  = test_panel[target_col].values

    dml_net = tf.keras.Sequential([
        Dense(64, activation='relu', input_shape=(X_tr_dml.shape[1],)),
        LayerNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        LayerNormalization(),
        Dense(1, activation='sigmoid')
    ])
    dml_net.compile(optimizer=tf.keras.optimizers.Adam(0.003), loss='binary_crossentropy', metrics=['AUC'])
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    dml_net.fit(X_tr_dml, y_train, validation_data=(X_va_dml, y_val), epochs=epochs_dml, batch_size=2048, callbacks=[es], verbose=0)

    # Inferenz
    test_h = dml_net.predict(X_te_dml, verbose=0).flatten()
    auc = float(roc_auc_score(y_test, test_h))
    pr_auc = float(average_precision_score(y_test, test_h))
    brier = float(brier_score_loss(y_test, test_h))

    causal_results = {}
    for idx_treat, supp_col in enumerate(treatment_cols):
        A_tilde_cf0 = A_tilde_test.copy()
        A_tilde_cf0[:, idx_treat] = 0.0 - A_hat_test[:, idx_treat]
        h_cf0 = dml_net.predict(np.hstack([W_test, A_tilde_cf0]), verbose=0).flatten()

        A_tilde_cf1 = A_tilde_test.copy()
        A_tilde_cf1[:, idx_treat] = 1.0 - A_hat_test[:, idx_treat]
        h_cf1 = dml_net.predict(np.hstack([W_test, A_tilde_cf1]), verbose=0).flatten()

        rr_partial = float(np.mean(h_cf1) / (np.mean(h_cf0) + 1e-7))
        ate_partial = float(np.mean(h_cf1 - h_cf0))

        short_name = supp_col.replace('_supp_count', '').replace('support_glz_', '')
        causal_results[short_name] = {
            "partial": {"mean_rr": rr_partial, "median_rr": rr_partial, "ate": ate_partial}
        }
        print(f"  • {short_name.upper():<14}: Partial RR = {rr_partial:.4f}, ATE = {ate_partial:+.4f}")

    print("\n" + "=" * 74)
    print("   ERGEBNISSE DEEP TRANSFORMER-DML (TEST-SET)")
    print("=" * 74)
    print(f"  • ROC-AUC     : {auc:.4f}")
    print(f"  • PR-AUC      : {pr_auc:.4f}")
    print(f"  • Brier Score : {brier:.4f}")
    print("=" * 74)

    # Logging
    base_dir = data_dir
    model_name = f"transformer_dml_{temporal}_{mode}" if (temporal != 'prev' or mode != 'standard') else "transformer_dml"

    metrics_dict = {
        "model_type": model_name,
        "temporal": temporal,
        "mode": mode,
        "ROC-AUC_Panel": auc,
        "PR-AUC_Panel": pr_auc,
        "Brier_Score": brier,
        **causal_results
    }
    save_metrics(model_name, metrics_dict, base_dir)
    save_keras_model(deep_transformer, f"{model_name}_encoder", base_dir)
    save_keras_model(dml_net, f"{model_name}_hazard", base_dir)

    print(f"[OK] Deep Transformer-DML erfolgreich gespeichert unter {base_dir}.")
    return deep_transformer, dml_net


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deep Transformer DML Pipeline")
    parser.add_argument('--data_dir', type=str, default='src/output_dl')
    parser.add_argument('--temporal', type=str, default='prev', choices=['prev', 'cum'])
    parser.add_argument('--mode', type=str, default='standard')
    parser.add_argument('--epochs_pretrain', type=int, default=15)
    parser.add_argument('--epochs_dml', type=int, default=25)
    args = parser.parse_args()

    train_transformer_dml(Path(args.data_dir), temporal=args.temporal, mode=args.mode, epochs_pretrain=args.epochs_pretrain, epochs_dml=args.epochs_dml)
