"""
Deep Transformer Regression & Survival Models (V4.2 Modernized Architecture)
=============================================================================
Modernisierte Transformer-Suite zur Beseitigung von Overfitting und Rechenzeitflaschenhaelsen:
1. Deep Semester-Transformer Regressor (d_model=64, 4 Heads, 2 Bloecke, Sin/Cos Encoding, L2-Reg)
2. Deep Exam-Transformer Regressor (d_model=64, 4 Heads, 2 Bloecke, Sin/Cos Encoding, L2-Reg)
3. Deep Exam-Transformer Causal Survival (Kausales Masking, TimeDistributed Hazard, optional Focal Loss)

Features & Sideprojects:
- Beseitigung des historischen Overfittings (Reduktion d_model 128 -> 64, 8 -> 4 Heads)
- Sin/Cos Positional Encoding auf allen Sequenzpfaden
- Sideproject A (Regularisierungs-Benchmark): --reg_type in ['hybrid', 'l2', 'dropout', 'elasticnet', 'none']
- Sideproject B (Asymmetrischer Loss): --loss_type in ['bce', 'focal'] mit --gamma und --alpha
- Standardisierte OOP-Evaluatoren: RegressionEvaluator & SurvivalEvaluator
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, 
    Add, Masking, Layer, TimeDistributed
)
from tensorflow.keras.regularizers import l2, l1_l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    roc_auc_score, average_precision_score, brier_score_loss
)

# Pfadaufloesung fuer deepsupport Package
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SRC_DIR = ROOT_DIR if ROOT_DIR.name == 'src' else ROOT_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deepsupport.evaluation.metrics_logger import (
    RegressionEvaluator, SurvivalEvaluator, save_keras_model, get_output_dirs
)
import deepsupport.data_engine.feature_builder as fb

PADDING_VALUE = -99.0


@tf.keras.utils.register_keras_serializable()
class SinCosPositionalEncoding(Layer):
    """Sinusoidales Positional Encoding gemaess Vaswani et al."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        d_model = tf.shape(inputs)[2]
        
        positions = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        
        angle_rates = 1.0 / tf.pow(10000.0, (2.0 * (i // 2.0)) / tf.cast(d_model, tf.float32))
        angle_rads = positions * angle_rates
        
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        
        pos_encoding = tf.reshape(tf.stack([sines, cosines], axis=-1), [seq_len, d_model])
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return inputs + tf.cast(pos_encoding, inputs.dtype)


@tf.keras.utils.register_keras_serializable()
class AttentionPooling(Layer):
    """Gelerntes Attention-Weighted Pooling ueber die Zeitschritte T mit echter Keras-Maskenunterstuetzung."""
    def __init__(self, d_model: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.supports_masking = True
        self.score_dense = Dense(1, activation='tanh')

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, mask=None):
        scores = self.score_dense(inputs)
        if mask is not None:
            valid_mask = tf.cast(mask, tf.float32)[:, :, tf.newaxis]
            padding_mask = (1.0 - valid_mask) * -1e9
        else:
            is_padded = tf.reduce_all(tf.equal(inputs, PADDING_VALUE), axis=-1, keepdims=True)
            padding_mask = tf.cast(is_padded, tf.float32) * -1e9
        scores = scores + padding_mask
        weights = tf.nn.softmax(scores, axis=1)
        pooled = tf.reduce_sum(inputs * weights, axis=1)
        return pooled


def masked_binary_crossentropy(y_true, y_pred):
    """Masked Binary Crossentropy fuer sequenzielle Hazard-Vorhersagen."""
    mask = tf.not_equal(y_true, PADDING_VALUE)
    mask = tf.cast(mask, tf.float32)
    y_true_safe = tf.where(tf.equal(y_true, PADDING_VALUE), tf.zeros_like(y_true), y_true)
    bce = tf.keras.losses.binary_crossentropy(y_true_safe, y_pred)
    bce = tf.expand_dims(bce, axis=-1) if len(bce.shape) < len(mask.shape) else bce
    masked_loss = bce * mask
    return tf.reduce_sum(masked_loss) / (tf.reduce_sum(mask) + 1e-7)


def masked_focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """
    Masked Focal Loss (Lin et al., 2017) fuer unausgeglichene Dropout-Zeitserien.
    Fokussiert den Gradienten auf die harte Minderheitenklasse (Dropouts).
    """
    def loss(y_true, y_pred):
        mask = tf.not_equal(y_true, PADDING_VALUE)
        mask = tf.cast(mask, tf.float32)
        y_true_safe = tf.where(tf.equal(y_true, PADDING_VALUE), tf.zeros_like(y_true), y_true)
        
        eps = tf.keras.backend.epsilon()
        p = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        
        p_t = tf.where(tf.equal(y_true_safe, 1.0), p, 1.0 - p)
        alpha_t = tf.where(tf.equal(y_true_safe, 1.0), alpha, 1.0 - alpha)
        
        focal_weight = alpha_t * tf.pow(1.0 - p_t, gamma)
        bce = -tf.math.log(p_t)
        loss_val = focal_weight * bce
        
        loss_val = tf.expand_dims(loss_val, axis=-1) if len(loss_val.shape) < len(mask.shape) else loss_val
        masked_loss = loss_val * mask
        return tf.reduce_sum(masked_loss) / (tf.reduce_sum(mask) + 1e-7)
    return loss


def _get_regularizer(reg_type: str, l2_val: float = 1e-4):
    """Sideproject A: Flexible Regularisierungs-Strategie."""
    if reg_type == 'l2':
        return l2(l2_val), 0.0
    elif reg_type == 'dropout':
        return None, 0.20
    elif reg_type == 'elasticnet':
        return l1_l2(l1=1e-5, l2=l2_val), 0.0
    elif reg_type == 'hybrid':
        return l2(l2_val), 0.15
    elif reg_type == 'none':
        return None, 0.0
    else:
        return l2(l2_val), 0.15


def build_modern_transformer_backbone(
    input_shape: Tuple[int, int],
    d_model: int = 64,
    num_heads: int = 4,
    num_blocks: int = 2,
    reg_type: str = 'hybrid',
    l2_val: float = 1e-4
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Kompakter, regularisierter Transformer-Encoder fuer Tabellensequenzen.
    """
    kernel_reg, dropout_rate = _get_regularizer(reg_type, l2_val)

    inputs = Input(shape=input_shape)
    x = Dense(d_model, activation='relu', kernel_regularizer=kernel_reg)(inputs)
    x = SinCosPositionalEncoding()(x)
    x = LayerNormalization()(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)

    for _ in range(num_blocks):
        attn_out = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )(x, x)
        x = Add()([x, attn_out])
        x = LayerNormalization()(x)

        ffn = Dense(d_model * 2, activation='relu', kernel_regularizer=kernel_reg)(x)
        if dropout_rate > 0:
            ffn = Dropout(dropout_rate)(ffn)
        ffn = Dense(d_model, kernel_regularizer=kernel_reg)(ffn)
        x = Add()([x, ffn])
        x = LayerNormalization()(x)

    pooled = AttentionPooling(d_model=d_model)(x)
    head = Dense(32, activation='relu', kernel_regularizer=kernel_reg)(pooled)
    head = LayerNormalization()(head)
    if dropout_rate > 0:
        head = Dropout(dropout_rate)(head)

    return inputs, head


def build_modern_causal_transformer_survival(
    input_shape: Tuple[int, int],
    d_model: int = 64,
    num_heads: int = 4,
    num_blocks: int = 2,
    reg_type: str = 'hybrid',
    l2_val: float = 1e-4,
    loss_type: str = 'bce',
    gamma: float = 2.0,
    alpha: float = 0.25,
    lr: float = 0.002
) -> Model:
    """
    Kausales Sequenzmodell fuer Exam-Hazard mit optionalem Focal Loss.
    """
    kernel_reg, dropout_rate = _get_regularizer(reg_type, l2_val)

    inputs = Input(shape=input_shape)
    x = Masking(mask_value=PADDING_VALUE)(inputs)
    x = Dense(d_model, activation='relu', kernel_regularizer=kernel_reg)(x)
    x = SinCosPositionalEncoding()(x)
    x = LayerNormalization()(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)

    for _ in range(num_blocks):
        attn_out = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )(x, x, use_causal_mask=True)
        x = Add()([x, attn_out])
        x = LayerNormalization()(x)

        ffn = Dense(d_model * 2, activation='relu', kernel_regularizer=kernel_reg)(x)
        if dropout_rate > 0:
            ffn = Dropout(dropout_rate)(ffn)
        ffn = Dense(d_model, kernel_regularizer=kernel_reg)(ffn)
        x = Add()([x, ffn])
        x = LayerNormalization()(x)

    time_dense = TimeDistributed(Dense(32, activation='relu', kernel_regularizer=kernel_reg))(x)
    time_dense = TimeDistributed(LayerNormalization())(time_dense)
    outputs = TimeDistributed(Dense(1, activation='sigmoid'))(time_dense)

    loss_fn = masked_focal_loss(gamma=gamma, alpha=alpha) if loss_type == 'focal' else masked_binary_crossentropy
    model = Model(inputs=inputs, outputs=outputs, name=f"Causal_Exam_Transformer_{loss_type}")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss_fn)
    return model


def train_deep_transformer_models(
    data_dir: Path = Path('src/output_dl'),
    output_dir: Optional[Path] = None,
    temporal: str = 'prev',
    mode: str = 'gradeblind',
    d_model: int = 64,
    num_heads: int = 4,
    num_blocks: int = 2,
    reg_type: str = 'hybrid',
    l2_val: float = 1e-4,
    loss_type: str = 'bce',
    gamma: float = 2.0,
    alpha: float = 0.25,
    epochs: int = 25,
    batch_size: int = 128
) -> Dict[str, Any]:
    """
    Orchestrierung der 3 modernisierten Deep Transformer Architekturen.
    """
    data_dir = Path(data_dir)
    target_out = Path(output_dir) if output_dir else data_dir
    metrics_dir, plots_dir, models_dir = get_output_dirs(target_out)

    print("\n" + "=" * 78)
    print(f"   MODERN DEEP TRANSFORMER SUITE (d={d_model}, h={num_heads}, b={num_blocks})")
    print(f"   Reg-Type: {reg_type} (L2={l2_val}) | Loss-Type: {loss_type} | Mode: {mode} | Temporal: {temporal}")
    print("=" * 78)

    # -------------------------------------------------------------
    # 1. Deep Semester-Transformer Regressor (GPA / Abschlussnote)
    # -------------------------------------------------------------
    print("\n[1/3] Deep Semester-Transformer Regressor (GPA)...")
    t0_sem = time.time()
    studis_s, X_sem, _, _, _, _ = fb.build_semester_sequence_tensor(
        data_dir, max_semesters=16, mode=mode, temporal=temporal, target_type='gpa'
    )
    df_abschluesse, _ = fb._load_raw_data(data_dir)
    note_dict = df_abschluesse.set_index('studierenden_id')['abschlussnote'].to_dict()

    y_s = np.array([note_dict.get(s, np.nan) for s in studis_s])
    valid_mask_s = ~np.isnan(y_s)
    X_sem_clean = X_sem[valid_mask_s]
    y_sem_clean = y_s[valid_mask_s]
    studis_s_clean = studis_s[valid_mask_s]

    tr_ids_s, va_ids_s, te_ids_s = fb.split_student_ids(studis_s_clean, seed=42)
    tr_s = np.where(np.isin(studis_s_clean, tr_ids_s))[0]
    va_s = np.where(np.isin(studis_s_clean, va_ids_s))[0]
    te_s = np.where(np.isin(studis_s_clean, te_ids_s))[0]

    scaled_s, _ = fb.scale_3d_sequences(
        {"train": X_sem_clean[tr_s], "val": X_sem_clean[va_s], "test": X_sem_clean[te_s]},
        fit_key="train",
        padding_value=PADDING_VALUE,
    )
    X_sem_tr, X_sem_va, X_sem_te = scaled_s["train"], scaled_s["val"], scaled_s["test"]

    inp_s, head_s = build_modern_transformer_backbone(
        X_sem_clean.shape[1:], d_model=d_model, num_heads=num_heads,
        num_blocks=num_blocks, reg_type=reg_type, l2_val=l2_val
    )
    out_s = Dense(1, activation='linear', kernel_regularizer=_get_regularizer(reg_type, l2_val)[0])(head_s)
    m_sem_reg = Model(inputs=inp_s, outputs=out_s, name="Deep_Semester_Transformer_Regressor")
    m_sem_reg.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])
    callbacks_sem = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0)
    ]

    hist_s = m_sem_reg.fit(
        X_sem_tr, y_sem_clean[tr_s],
        validation_data=(X_sem_va, y_sem_clean[va_s]),
        epochs=epochs, batch_size=batch_size, callbacks=callbacks_sem, verbose=0
    )
    preds_s = m_sem_reg.predict(X_sem_te, verbose=0).flatten()
    fit_time_sem = time.time() - t0_sem

    ev_sem = RegressionEvaluator(base_dir=target_out, model_name='deep_semester_transformer_regressor', mode=mode, temporal=temporal)
    metrics_sem = ev_sem.evaluate_and_log(
        y_true=y_sem_clean[te_s], y_pred=preds_s, history=hist_s.history,
        model=m_sem_reg, fit_time_s=fit_time_sem
    )

    # -------------------------------------------------------------
    # 2. Deep Exam-Transformer Regressor (Pruefungsebene)
    # -------------------------------------------------------------
    print("\n[2/3] Deep Exam-Transformer Regressor (Exam History)...")
    t0_ex = time.time()
    studis_e, X_ex, _, _, _, _ = fb.build_exam_sequence_tensor(
        data_dir, max_exams=40, mode=mode, temporal=temporal, target_type='grade'
    )
    y_e = np.array([note_dict.get(s, np.nan) for s in studis_e])
    valid_mask_e = ~np.isnan(y_e)
    X_ex_clean = X_ex[valid_mask_e]
    y_ex_clean = y_e[valid_mask_e]
    studis_e_clean = studis_e[valid_mask_e]

    tr_ids_e, va_ids_e, te_ids_e = fb.split_student_ids(studis_e_clean, seed=42)
    tr_e = np.where(np.isin(studis_e_clean, tr_ids_e))[0]
    va_e = np.where(np.isin(studis_e_clean, va_ids_e))[0]
    te_e = np.where(np.isin(studis_e_clean, te_ids_e))[0]

    scaled_e, _ = fb.scale_3d_sequences(
        {"train": X_ex_clean[tr_e], "val": X_ex_clean[va_e], "test": X_ex_clean[te_e]},
        fit_key="train",
        padding_value=PADDING_VALUE,
    )
    X_ex_tr, X_ex_va, X_ex_te = scaled_e["train"], scaled_e["val"], scaled_e["test"]

    inp_e, head_e = build_modern_transformer_backbone(
        X_ex_clean.shape[1:], d_model=d_model, num_heads=num_heads,
        num_blocks=num_blocks, reg_type=reg_type, l2_val=l2_val
    )
    out_e = Dense(1, activation='linear', kernel_regularizer=_get_regularizer(reg_type, l2_val)[0])(head_e)
    m_ex_reg = Model(inputs=inp_e, outputs=out_e, name="Deep_Exam_Transformer_Regressor")
    m_ex_reg.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])

    callbacks_ex = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0)
    ]

    hist_e = m_ex_reg.fit(
        X_ex_tr, y_ex_clean[tr_e],
        validation_data=(X_ex_va, y_ex_clean[va_e]),
        epochs=epochs, batch_size=batch_size, callbacks=callbacks_ex, verbose=0
    )
    preds_e = m_ex_reg.predict(X_ex_te, verbose=0).flatten()
    fit_time_ex = time.time() - t0_ex

    ev_ex = RegressionEvaluator(base_dir=target_out, model_name='deep_exam_transformer_regressor', mode=mode, temporal=temporal)
    metrics_ex = ev_ex.evaluate_and_log(
        y_true=y_ex_clean[te_e], y_pred=preds_e, history=hist_e.history,
        model=m_ex_reg, fit_time_s=fit_time_ex
    )

    # -------------------------------------------------------------
    # 3. Deep Exam Causal Transformer Survival (Hazard & Dropout)
    # -------------------------------------------------------------
    print(f"\n[3/3] Deep Exam Causal Transformer Survival (Loss={loss_type})...")
    t0_sv = time.time()
    studis_sv, X_sv, y_sv, st_events, _, _ = fb.build_exam_sequence_tensor(
        data_dir, max_exams=40, mode=mode, temporal=temporal
    )
    tr_ids_sv, va_ids_sv, te_ids_sv = fb.split_student_ids(studis_sv, seed=42, stratify=(st_events == 1).astype(int))
    tr_sv = np.where(np.isin(studis_sv, tr_ids_sv))[0]
    va_sv = np.where(np.isin(studis_sv, va_ids_sv))[0]
    te_sv = np.where(np.isin(studis_sv, te_ids_sv))[0]

    te_mask_sv = X_sv[te_sv, :, 0] != PADDING_VALUE
    scaled_sv, _ = fb.scale_3d_sequences(
        {"train": X_sv[tr_sv], "val": X_sv[va_sv], "test": X_sv[te_sv]},
        fit_key="train",
        padding_value=PADDING_VALUE,
    )
    X_sv_tr, X_sv_va, X_sv_te = scaled_sv["train"], scaled_sv["val"], scaled_sv["test"]

    m_sv = build_modern_causal_transformer_survival(
        X_sv.shape[1:], d_model=d_model, num_heads=num_heads,
        num_blocks=num_blocks, reg_type=reg_type, l2_val=l2_val,
        loss_type=loss_type, gamma=gamma, alpha=alpha
    )
    callbacks_sv = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0)
    ]
    hist_sv = m_sv.fit(
        X_sv_tr, y_sv[tr_sv],
        validation_data=(X_sv_va, y_sv[va_sv]),
        epochs=epochs, batch_size=batch_size, callbacks=callbacks_sv, verbose=0
    )
    fit_time_sv = time.time() - t0_sv

    test_preds_sv = m_sv.predict(X_sv_te, verbose=0)
    y_test_flat = y_sv[te_sv][te_mask_sv].flatten()
    preds_flat = test_preds_sv[te_mask_sv].flatten()

    ev_sv = SurvivalEvaluator(base_dir=target_out, model_name=f"deep_exam_transformer_survival_{loss_type}", mode=mode, temporal=temporal)
    metrics_sv = ev_sv.evaluate_and_log(
        y_true=y_test_flat, y_prob=preds_flat, history=hist_sv.history,
        model=m_sv, fit_time_s=fit_time_sv
    )

    # Zusammenfassung
    overall_metrics = {
        "deep_semester_regressor": metrics_sem,
        "deep_exam_regressor": metrics_ex,
        "deep_exam_survival": metrics_sv,
        "config": {
            "d_model": d_model,
            "num_heads": num_heads,
            "num_blocks": num_blocks,
            "reg_type": reg_type,
            "l2_val": l2_val,
            "loss_type": loss_type,
            "gamma": gamma,
            "alpha": alpha,
            "mode": mode,
            "temporal": temporal
        }
    }
    summary_path = metrics_dir / f"deep_transformer_suite_summary_{mode}_{temporal}_{loss_type}_{reg_type}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(overall_metrics, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 78)
    print(f"   [OK] DEEP TRANSFORMER SUITE ERFOLGREICH BEENDET")
    print(f"   Semester R2 : {metrics_sem.get('r2_score', metrics_sem.get('r2', 0.0)):.4f} (RMSE: {metrics_sem.get('rmse', 0.0):.4f})")
    print(f"   Exam R2     : {metrics_ex.get('r2_score', metrics_ex.get('r2', 0.0)):.4f} (RMSE: {metrics_ex.get('rmse', 0.0):.4f})")
    print(f"   Survival AUC: {metrics_sv.get('roc_auc', 0.0):.4f} (PR-AUC: {metrics_sv.get('pr_auc_dropout', 0.0):.4f})")
    print("=" * 78)

    return overall_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Modern Deep Transformer Suite (V4.2)")
    parser.add_argument('--data_dir', type=str, default='data_v4_grid/S01_baseline/universe_A')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--temporal', type=str, default='prev', choices=['prev', 'cum'])
    parser.add_argument('--mode', type=str, default='gradeblind', choices=['standard', 'gradeblind', 'blind', 'oracle', 'realistic'])
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--reg_type', type=str, default='hybrid', choices=['hybrid', 'l2', 'dropout', 'elasticnet', 'none'])
    parser.add_argument('--l2_val', type=float, default=1e-4)
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'])
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    train_deep_transformer_models(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        temporal=args.temporal,
        mode=args.mode,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        reg_type=args.reg_type,
        l2_val=args.l2_val,
        loss_type=args.loss_type,
        gamma=args.gamma,
        alpha=args.alpha,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
