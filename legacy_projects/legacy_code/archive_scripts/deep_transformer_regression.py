"""
Deep Transformer Regression & Survival Models (Enlarged Capacity + Dual Causal/Masked Architectures)
=======================================================================================================
Implementiert 4 hochentwickelte Transformer-Modelle auf Basis von `feature_builder.py`:
1. Deep Semester-Transformer Regressor (d_model=128, 8 Heads, 3 Blöcke, Attention Pooling)
2. Deep Exam-Transformer Regressor (d_model=128, 8 Heads, 3 Blöcke, Attention Pooling)
3. Deep Exam-Transformer Causal Survival (use_causal_mask=True, TimeDistributed hazard)
4. Deep Exam-Transformer Masked Survival (Attention Pooling, Static Event)
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, Masking, Layer, TimeDistributed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, average_precision_score, brier_score_loss

# Import metrics_logger and feature_builder
sys.path.insert(0, str(Path(__file__).parent))
from metrics_logger import save_metrics, save_keras_model, plot_learning_curve
import feature_builder as fb

PADDING_VALUE = -99.0


class AttentionPooling(Layer):
    """Gelerntes Attention-Weighted Pooling über die Zeitschritte T."""
    def __init__(self, d_model=128, **kwargs):
        super(AttentionPooling, self).__init__(**kwargs)
        self.d_model = d_model
        self.score_dense = Dense(1, activation='tanh')

    def call(self, inputs):
        scores = self.score_dense(inputs)
        is_padded = tf.reduce_all(tf.equal(inputs, PADDING_VALUE), axis=-1, keepdims=True)
        padding_mask = tf.cast(is_padded, tf.float32) * -1e9
        scores = scores + padding_mask
        weights = tf.nn.softmax(scores, axis=1)
        pooled = tf.reduce_sum(inputs * weights, axis=1)
        return pooled


def masked_binary_crossentropy(y_true, y_pred):
    """Masked Binary Crossentropy für sequenzielle Hazard-Vorhersagen."""
    mask = tf.not_equal(y_true, PADDING_VALUE)
    mask = tf.cast(mask, tf.float32)
    y_true_safe = tf.where(tf.equal(y_true, PADDING_VALUE), tf.zeros_like(y_true), y_true)
    bce = tf.keras.losses.binary_crossentropy(y_true_safe, y_pred)
    bce = tf.expand_dims(bce, axis=-1) if len(bce.shape) < len(mask.shape) else bce
    masked_loss = bce * mask
    return tf.reduce_sum(masked_loss) / (tf.reduce_sum(mask) + 1e-7)


def build_deep_transformer_backbone(input_shape, d_model=128, num_heads=8, num_blocks=3, dropout_rate=0.2):
    inputs = Input(shape=input_shape)
    x = Dense(d_model, activation='relu')(inputs)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)

    for _ in range(num_blocks):
        attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout_rate)(x, x)
        x = Add()([x, attn_out])
        x = LayerNormalization()(x)

        ffn = Dense(d_model * 2, activation='relu')(x)
        ffn = Dropout(dropout_rate)(ffn)
        ffn = Dense(d_model)(ffn)
        x = Add()([x, ffn])
        x = LayerNormalization()(x)

    pooled = AttentionPooling(d_model=d_model)(x)
    head = Dense(64, activation='relu')(pooled)
    head = LayerNormalization()(head)
    head = Dropout(dropout_rate)(head)

    return inputs, head


def build_causal_transformer_survival_model(input_shape, d_model=128, num_heads=8, num_blocks=3, dropout_rate=0.2):
    inputs = Input(shape=input_shape)
    x = Masking(mask_value=PADDING_VALUE)(inputs)
    x = Dense(d_model, activation='relu')(x)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)

    for _ in range(num_blocks):
        attn_out = MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout_rate
        )(x, x, use_causal_mask=True)
        x = Add()([x, attn_out])
        x = LayerNormalization()(x)

        ffn = Dense(d_model * 2, activation='relu')(x)
        ffn = Dropout(dropout_rate)(ffn)
        ffn = Dense(d_model)(ffn)
        x = Add()([x, ffn])
        x = LayerNormalization()(x)

    time_dense = TimeDistributed(Dense(32, activation='relu'))(x)
    time_dense = TimeDistributed(LayerNormalization())(time_dense)
    outputs = TimeDistributed(Dense(1, activation='sigmoid'))(time_dense)

    model = Model(inputs=inputs, outputs=outputs, name="Causal_Exam_Transformer_Survival")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=masked_binary_crossentropy)
    return model


def train_deep_transformer_models(data_dir: Path = Path('src/output_dl'),
                                  temporal: str = 'prev',
                                  mode: str = 'standard',
                                  epochs: int = 30,
                                  batch_size: int = 128):
    print("\n" + "=" * 74)
    print(f"   DEEP TRANSFORMER SUITE (REGRESSION & SURVIVAL | temporal={temporal}, mode={mode})")
    print("=" * 74)

    # 1. Deep Semester Transformer Regressor
    print("\n[1/3] Deep Semester-Transformer Regressor...")
    studis_s, X_sem, _, _, _, _ = fb.build_semester_sequence_tensor(data_dir, max_semesters=16, mode=mode, temporal=temporal, target_type='gpa')
    df_abschluesse, _ = fb._load_raw_data(data_dir)
    note_dict = df_abschluesse.set_index('studierenden_id')['abschlussnote'].to_dict()

    y_s = np.array([note_dict.get(s, np.nan) for s in studis_s])
    valid_mask_s = ~np.isnan(y_s)
    X_sem_clean = X_sem[valid_mask_s]
    y_sem_clean = y_s[valid_mask_s]

    idx_s = np.arange(len(X_sem_clean))
    tr_s, te_s = train_test_split(idx_s, test_size=0.20, random_state=42)
    tr_s, va_s = train_test_split(tr_s, test_size=0.20, random_state=42)

    scaler_s = StandardScaler()
    v_mask_tr = X_sem_clean[tr_s, :, 0] != PADDING_VALUE
    scaler_s.fit(X_sem_clean[tr_s][v_mask_tr])

    for subset in [tr_s, va_s, te_s]:
        vm = X_sem_clean[subset, :, 0] != PADDING_VALUE
        X_sem_clean[subset][vm] = scaler_s.transform(X_sem_clean[subset][vm])

    inp_s, head_s = build_deep_transformer_backbone(X_sem_clean.shape[1:])
    out_s = Dense(1, activation='linear')(head_s)
    m_sem_reg = Model(inputs=inp_s, outputs=out_s)
    m_sem_reg.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    m_sem_reg.fit(X_sem_clean[tr_s], y_sem_clean[tr_s], validation_data=(X_sem_clean[va_s], y_sem_clean[va_s]), epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)
    preds_s = m_sem_reg.predict(X_sem_clean[te_s], verbose=0).flatten()
    r2_s = float(r2_score(y_sem_clean[te_s], preds_s))
    rmse_s = float(np.sqrt(mean_squared_error(y_sem_clean[te_s], preds_s)))
    print(f"  -> Deep Semester-Transformer Regressor: R2={r2_s:.4f}, RMSE={rmse_s:.4f}")

    # 2. Deep Exam Transformer Regressor
    print("\n[2/3] Deep Exam-Transformer Regressor...")
    studis_e, X_ex, _, _, _, _ = fb.build_exam_sequence_tensor(data_dir, max_exams=40, mode=mode, temporal=temporal, target_type='grade')
    y_e = np.array([note_dict.get(s, np.nan) for s in studis_e])
    valid_mask_e = ~np.isnan(y_e)
    X_ex_clean = X_ex[valid_mask_e]
    y_ex_clean = y_e[valid_mask_e]

    idx_e = np.arange(len(X_ex_clean))
    tr_e, te_e = train_test_split(idx_e, test_size=0.20, random_state=42)
    tr_e, va_e = train_test_split(tr_e, test_size=0.20, random_state=42)

    scaler_e = StandardScaler()
    v_mask_e = X_ex_clean[tr_e, :, 0] != PADDING_VALUE
    scaler_e.fit(X_ex_clean[tr_e][v_mask_e])

    for subset in [tr_e, va_e, te_e]:
        vm = X_ex_clean[subset, :, 0] != PADDING_VALUE
        X_ex_clean[subset][vm] = scaler_e.transform(X_ex_clean[subset][vm])

    inp_e, head_e = build_deep_transformer_backbone(X_ex_clean.shape[1:])
    out_e = Dense(1, activation='linear')(head_e)
    m_ex_reg = Model(inputs=inp_e, outputs=out_e)
    m_ex_reg.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])

    m_ex_reg.fit(X_ex_clean[tr_e], y_ex_clean[tr_e], validation_data=(X_ex_clean[va_e], y_ex_clean[va_e]), epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)
    preds_e = m_ex_reg.predict(X_ex_clean[te_e], verbose=0).flatten()
    r2_e = float(r2_score(y_ex_clean[te_e], preds_e))
    rmse_e = float(np.sqrt(mean_squared_error(y_ex_clean[te_e], preds_e)))
    print(f"  -> Deep Exam-Transformer Regressor: R2={r2_e:.4f}, RMSE={rmse_e:.4f}")

    # 3. Deep Exam Causal Transformer Survival
    print("\n[3/3] Deep Exam-Transformer Causal Survival...")
    studis_sv, X_sv, y_sv, st_events, _, _ = fb.build_exam_sequence_tensor(data_dir, max_exams=40, mode=mode, temporal=temporal)
    idx_sv = np.arange(len(X_sv))
    tr_sv, te_sv = train_test_split(idx_sv, test_size=0.20, random_state=42, stratify=st_events)
    tr_sv, va_sv = train_test_split(tr_sv, test_size=0.20, random_state=42, stratify=st_events[tr_sv])

    scaler_sv = StandardScaler()
    v_mask_sv = X_sv[tr_sv, :, 0] != PADDING_VALUE
    scaler_sv.fit(X_sv[tr_sv][v_mask_sv])

    for subset in [tr_sv, va_sv, te_sv]:
        vm = X_sv[subset, :, 0] != PADDING_VALUE
        X_sv[subset][vm] = scaler_sv.transform(X_sv[subset][vm])

    m_sv = build_causal_transformer_survival_model(X_sv.shape[1:])
    m_sv.fit(X_sv[tr_sv], y_sv[tr_sv], validation_data=(X_sv[va_sv], y_sv[va_sv]), epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)

    test_preds_sv = m_sv.predict(X_sv[te_sv], verbose=0)
    te_mask_sv = X_sv[te_sv, :, 0] != PADDING_VALUE
    y_test_flat = y_sv[te_sv][te_mask_sv].flatten()
    preds_flat = test_preds_sv[te_mask_sv].flatten()

    auc_sv = float(roc_auc_score(y_test_flat, preds_flat))
    pr_auc_sv = float(average_precision_score(y_test_flat, preds_flat))
    print(f"  -> Deep Exam Causal Transformer Survival: ROC-AUC={auc_sv:.4f}, PR-AUC={pr_auc_sv:.4f}")

    # Logging
    base_dir = data_dir
    metrics_dict = {
        "Deep_Semester_Transformer_R2": r2_s,
        "Deep_Semester_Transformer_RMSE": rmse_s,
        "Deep_Exam_Transformer_R2": r2_e,
        "Deep_Exam_Transformer_RMSE": rmse_e,
        "Deep_Exam_Survival_ROC_AUC": auc_sv,
        "Deep_Exam_Survival_PR_AUC": pr_auc_sv
    }
    bench_name = f"deep_transformer_benchmark_{temporal}_{mode}" if (temporal != 'prev' or mode != 'standard') else "deep_transformer_benchmark"
    save_metrics(bench_name, metrics_dict, base_dir)
    save_keras_model(m_sem_reg, f"deep_semester_transformer_regressor_{mode}" if mode != 'standard' else "deep_semester_transformer_regressor", base_dir)
    save_keras_model(m_ex_reg, f"deep_exam_transformer_regressor_{mode}" if mode != 'standard' else "deep_exam_transformer_regressor", base_dir)
    save_keras_model(m_sv, f"deep_exam_transformer_survival_{mode}" if mode != 'standard' else "deep_exam_transformer_survival", base_dir)

    print(f"\n[OK] Deep Transformer Suite erfolgreich abgeschlossen.")
    return metrics_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deep Transformer Suite")
    parser.add_argument('--data_dir', type=str, default='src/output_dl')
    parser.add_argument('--temporal', type=str, default='prev', choices=['prev', 'cum'])
    parser.add_argument('--mode', type=str, default='standard')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    train_deep_transformer_models(Path(args.data_dir), temporal=args.temporal, mode=args.mode, epochs=args.epochs, batch_size=args.batch_size)
