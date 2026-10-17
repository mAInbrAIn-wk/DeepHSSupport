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
from typing import Tuple
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GroupKFold
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
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deepsupport.models.semester_gru import masked_binary_crossentropy, PADDING_VALUE
from deepsupport.models.semester_transformer import PositionalEncoding
from deepsupport.evaluation.metrics_logger import save_metrics, save_keras_model, plot_learning_curve, plot_roc_curve, plot_pr_curve, CausalEvaluator
import deepsupport.data_engine.feature_builder as fb


def build_deep_causal_transformer_model(sequence_length: int, feature_dim: int, d_model: int = 64, num_heads: int = 4, num_blocks: int = 2) -> Tuple[Model, Model]:
    """Größerer Deep Causal Transformer mit gestapelten Attention-Blöcken sowie explizitem Embedding-Extraktor."""
    inputs = Input(shape=(sequence_length, feature_dim))
    masked_inputs = Masking(mask_value=PADDING_VALUE)(inputs)

    x = TimeDistributed(Dense(d_model, activation='relu'))(masked_inputs)
    x = PositionalEncoding(sequence_length, d_model)(x)

    for block_idx in range(num_blocks):
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
        x = LayerNormalization(name=f"transformer_emb_ln_{block_idx}")(x)

    outputs = TimeDistributed(Dense(1, activation='sigmoid'), name='hazard_out')(x)
    model = Model(inputs=inputs, outputs=outputs, name='deep_causal_transformer')
    embedding_model = Model(inputs=inputs, outputs=x, name='deep_causal_transformer_embeddings')
    return model, embedding_model


def train_transformer_dml(data_dir: Path = Path('src/output_dl'),
                          temporal: str = 'prev',
                          mode: str = 'standard',
                          epochs_pretrain: int = 20,
                          epochs_dml: int = 30):
    print("\n" + "=" * 74)
    print(f"   DEEP TRANSFORMER-DML BENCHMARK (temporal={temporal}, mode={mode})")
    print("=" * 74)

    # 1. 3D-Sequenz und 2D-Panel laden
    studis, X_3d, y_3d, studi_events, feature_names, _ = fb.build_semester_sequence_tensor(
        data_dir, max_semesters=16, mode=mode, temporal=temporal
    )
    panel_df, feature_cols, target_col, _ = fb.build_semester_panel_df(
        data_dir, mode=mode, temporal=temporal
    )
    n_samples, sequence_length, feature_dim = X_3d.shape

    # 2. Strikter 70/15/15 Student-Split VOR jeglicher Skalierung und VOR dem Transformer-Pretraining!
    unique_studis = np.array(panel_df['studierenden_id'].unique().tolist())
    train_ids, temp_ids = train_test_split(unique_studis, test_size=0.30, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=42)

    train_id_set, val_id_set, test_id_set = set(train_ids), set(val_ids), set(test_ids)
    seq_tr_mask = np.array([s in train_id_set for s in studis], dtype=bool)
    seq_va_mask = np.array([s in val_id_set for s in studis], dtype=bool)
    seq_te_mask = np.array([s in test_id_set for s in studis], dtype=bool)

    scaled_3d, _ = fb.scale_3d_sequences(
        {
            "train": X_3d[seq_tr_mask],
            "val": X_3d[seq_va_mask],
            "test": X_3d[seq_te_mask],
            "all": X_3d,
        },
        fit_key="train",
    )
    X_3d_train = scaled_3d["train"]
    X_3d_val   = scaled_3d["val"]
    X_3d_all   = scaled_3d["all"]
    y_3d_train = y_3d[seq_tr_mask]
    y_3d_val   = y_3d[seq_va_mask]

    # 3. Transformer Modell bauen & ausschließlich auf Train-Studierenden vortrainieren
    tf.random.set_seed(42)
    d_model = 64
    deep_transformer, embedding_extractor = build_deep_causal_transformer_model(
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        d_model=d_model,
        num_heads=4,
        num_blocks=2
    )

    deep_transformer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=masked_binary_crossentropy)
    es_pre = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print(f"\n[Stufe 1] Pretraining Deep Causal Transformer nur auf Train-Split ({epochs_pretrain} Epochen)...")
    deep_transformer.fit(
        X_3d_train, y_3d_train,
        validation_data=(X_3d_val, y_3d_val),
        epochs=epochs_pretrain,
        batch_size=128,
        callbacks=[es_pre],
        verbose=0
    )

    # 4. 64-dimensionale kausale Sequenz-Embeddings extrahieren und auf (studierenden_id, t_stop) mappen
    emb_3d_all = embedding_extractor.predict(X_3d_all, batch_size=256, verbose=0)  # (N_studis, 16, 64)
    studi_to_seq_idx = {str(s): idx for idx, s in enumerate(studis)}

    def _extract_panel_embeddings(df_sub: pd.DataFrame) -> np.ndarray:
        s_indices = np.array([studi_to_seq_idx.get(str(s), 0) for s in df_sub['studierenden_id'].values], dtype=np.int64)
        t_indices = np.clip(df_sub['t_stop'].values.astype(np.int64) - 1, 0, sequence_length - 1)
        return emb_3d_all[s_indices, t_indices, :]

    treatment_candidates = ['fach_supp_count', 'uebf_supp_count', 'psych_supp_count']
    treatment_cols = [c for c in treatment_candidates if c in feature_cols]
    confounder_cols = [c for c in feature_cols if c not in treatment_cols]

    train_panel = panel_df[panel_df['studierenden_id'].isin(train_ids)].copy().reset_index(drop=True)
    val_panel   = panel_df[panel_df['studierenden_id'].isin(val_ids)].copy().reset_index(drop=True)
    test_panel  = panel_df[panel_df['studierenden_id'].isin(test_ids)].copy().reset_index(drop=True)

    Z_emb_train = _extract_panel_embeddings(train_panel)
    Z_emb_val   = _extract_panel_embeddings(val_panel)
    Z_emb_test  = _extract_panel_embeddings(test_panel)

    p_scaler = StandardScaler()
    W_tab_train = p_scaler.fit_transform(train_panel[confounder_cols].fillna(0))
    W_tab_val   = p_scaler.transform(val_panel[confounder_cols].fillna(0))
    W_tab_test  = p_scaler.transform(test_panel[confounder_cols].fillna(0))

    emb_scaler = StandardScaler()
    Z_emb_train_sc = emb_scaler.fit_transform(Z_emb_train)
    Z_emb_val_sc   = emb_scaler.transform(Z_emb_val)
    Z_emb_test_sc  = emb_scaler.transform(Z_emb_test)

    # Vereinigte Confounder-Repräsentation [W_tabular, Z_transformer_embedding]
    W_train = np.hstack([W_tab_train, Z_emb_train_sc])
    W_val   = np.hstack([W_tab_val, Z_emb_val_sc])
    W_test  = np.hstack([W_tab_test, Z_emb_test_sc])

    y_train = train_panel[target_col].values.astype(float)
    y_val   = val_panel[target_col].values.astype(float)
    y_test  = test_panel[target_col].values.astype(float)

    # 5. DML Stufe 1: 5-Fold GroupKFold Cross-Fitting auf [W_tabular, Z_emb]
    print("[Stufe 2] 5-Fold GroupKFold DML Orthogonalisierung auf Transformer-Embeddings...")
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

        out_reg = Ridge(alpha=10.0)
        out_reg.fit(W_f_tr, y_f_tr)
        y_tilde_train[f_va_idx] = y_f_va - out_reg.predict(W_f_va)

        for k, supp_col in enumerate(treatment_cols):
            A_f_tr = train_panel.loc[f_tr_idx, supp_col].values.astype(float)
            A_f_va = train_panel.loc[f_va_idx, supp_col].values.astype(float)

            reg = Ridge(alpha=10.0)
            reg.fit(W_f_tr, A_f_tr)

            A_tilde_train[f_va_idx, k] = A_f_va - reg.predict(W_f_va)
            A_hat_val[:, k]  += reg.predict(W_val) / n_folds
            A_hat_test[:, k] += reg.predict(W_test) / n_folds

    A_tilde_val  = val_panel[treatment_cols].values.astype(float) - A_hat_val
    A_tilde_test = test_panel[treatment_cols].values.astype(float) - A_hat_test

    # Multivariater Robinson-Schätzer mit Cluster-Sandwich-SE
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

    # DML Stufe 2: Transformer-DML Hazard Head
    print(f"[Stufe 3] DML Hazard-Modell Training ({epochs_dml} Epochen)...")
    X_tr_dml = np.hstack([W_train, A_tilde_train])
    X_va_dml = np.hstack([W_val, A_tilde_val])
    X_te_dml = np.hstack([W_test, A_tilde_test])

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
    hr_estimates = {}
    hr_se = {}
    name_map = {'fach': 'fachlich', 'uebf': 'ueberfachlich', 'psych': 'psychosozial'}

    for idx_treat, supp_col in enumerate(treatment_cols):
        A_tilde_cf0 = A_tilde_test.copy()
        A_tilde_cf0[:, idx_treat] = 0.0 - A_hat_test[:, idx_treat]
        h_cf0 = dml_net.predict(np.hstack([W_test, A_tilde_cf0]), verbose=0).flatten()

        A_tilde_cf1 = A_tilde_test.copy()
        A_tilde_cf1[:, idx_treat] = 1.0 - A_hat_test[:, idx_treat]
        h_cf1 = dml_net.predict(np.hstack([W_test, A_tilde_cf1]), verbose=0).flatten()

        rr_partial = float(np.mean(h_cf1) / (np.mean(h_cf0) + 1e-7))
        ate_partial = float(np.mean(h_cf1 - h_cf0))

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
            "partial": {"mean_rr": rr_partial, "median_rr": rr_partial, "ate": ate_partial}
        }
        print(f"  • {short_name.upper():<14}: Robinson HR = {robinson_hr:.4f} (ATE={theta_k:+.4f}), Partial RR = {rr_partial:.4f}")

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
        keras_model=dml_net,
        mode=mode,
        temporal_type=temporal,
        estimand_type="HR",
        se_scale="natural",
    )
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
