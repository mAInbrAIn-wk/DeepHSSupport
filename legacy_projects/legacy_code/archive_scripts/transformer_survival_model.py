"""
Causal Transformer Survival Analysis (Semester Attention Edition)
=================================================================
Verwendet einen Keras Causal Transformer Encoder mit MultiHeadAttention
und Kausal-Maskierung (use_causal_mask=True), um Future Leakage mathematisch auszuschließen.

Unterstützt über feature_builder.py:
- Alle Modi (standard, gradeblind, blind, oracle, realistic)
- Temporale Modi (temporal='prev' oder 'cum')
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
from metrics_logger import save_metrics, save_keras_model, plot_learning_curve, plot_roc_curve, plot_pr_curve
import feature_builder as fb


class PositionalEncoding(tf.keras.layers.Layer):
    """Sinusoidales Positional Encoding für temporäre Reihenfolgen."""
    def __init__(self, sequence_length: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.sequence_length = sequence_length
        self.d_model = d_model

        position = np.arange(sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe = np.zeros((sequence_length, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = tf.cast(pe[np.newaxis, :, :], dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pe[:, :tf.shape(inputs)[1], :]


def build_causal_transformer_survival_model(sequence_length: int, feature_dim: int, d_model: int = 32, num_heads: int = 4) -> Model:
    inputs = Input(shape=(sequence_length, feature_dim))
    masked_inputs = Masking(mask_value=PADDING_VALUE)(inputs)

    x = TimeDistributed(Dense(d_model))(masked_inputs)
    x = PositionalEncoding(sequence_length, d_model)(x)

    attn_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=0.1
    )(query=x, value=x, key=x, use_causal_mask=True)

    x = Add()([x, attn_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    ffn = TimeDistributed(Dense(64, activation='relu'))(x)
    ffn = TimeDistributed(Dense(d_model))(ffn)
    ffn = Dropout(0.1)(ffn)

    x = Add()([x, ffn])
    x = LayerNormalization(epsilon=1e-6)(x)

    outputs = TimeDistributed(Dense(1, activation='sigmoid'))(x)

    model = Model(inputs=inputs, outputs=outputs, name="Causal_Transformer_Survival")
    model.compile(optimizer=tf.keras.optimizers.Adam(0.003), loss=masked_binary_crossentropy)
    return model


def train_transformer_survival(data_dir: Path = Path('src/output_dl'),
                               max_semesters: int = 16,
                               temporal: str = 'prev',
                               mode: str = 'standard',
                               epochs: int = 35,
                               batch_size: int = 128):
    print("\n" + "=" * 74)
    print(f"   CAUSAL SEMESTER TRANSFORMER SURVIVAL (temporal={temporal}, mode={mode})")
    print("=" * 74)

    studis, X_seq, y_seq, studi_events, feature_names, _ = fb.build_semester_sequence_tensor(
        data_dir, max_semesters=max_semesters, mode=mode, temporal=temporal
    )

    n_samples, n_timesteps, n_features = X_seq.shape
    print(f"Dataset: Shape={X_seq.shape} ({n_features} Features pro Semester)")

    # 3-Way Split
    idx = np.arange(n_samples)
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=42, stratify=studi_events)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42, stratify=studi_events[temp_idx])

    X_train, y_train = X_seq[train_idx].copy(), y_seq[train_idx].copy()
    X_val, y_val = X_seq[val_idx].copy(), y_seq[val_idx].copy()
    X_test, y_test = X_seq[test_idx].copy(), y_seq[test_idx].copy()

    # Skalierung
    train_mask = X_train[:, :, 0] != PADDING_VALUE
    val_mask = X_val[:, :, 0] != PADDING_VALUE
    test_mask = X_test[:, :, 0] != PADDING_VALUE

    scaler = StandardScaler()
    scaler.fit(X_train[train_mask])

    X_train[train_mask] = scaler.transform(X_train[train_mask])
    X_val[val_mask] = scaler.transform(X_val[val_mask])
    X_test[test_mask] = scaler.transform(X_test[test_mask])

    # Modell
    tf.random.set_seed(42)
    model = build_causal_transformer_survival_model(n_timesteps, n_features, d_model=32, num_heads=4)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print(f"\nTrainiere Transformer ({epochs} Epochen, Batch-Size {batch_size})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=[es], verbose=0
    )

    # Evaluation
    test_preds = model.predict(X_test, verbose=0)

    y_test_flat = y_test[test_mask].flatten()
    preds_flat = test_preds[test_mask].flatten()

    auc = float(roc_auc_score(y_test_flat, preds_flat))
    pr_auc = float(average_precision_score(y_test_flat, preds_flat))
    brier = float(brier_score_loss(y_test_flat, preds_flat))

    test_student_events = studi_events[test_idx]
    surv_probs = np.ones(len(test_idx))
    for i in range(len(test_idx)):
        s_len = np.sum(test_mask[i])
        if s_len > 0:
            h_t = np.clip(test_preds[i, :s_len, 0], 1e-6, 1.0 - 1e-6)
            surv_probs[i] = np.prod(1.0 - h_t)
    student_auc = float(roc_auc_score(test_student_events, 1.0 - surv_probs))

    print("\n" + "=" * 74)
    print("   ERGEBNISSE CAUSAL SEMESTER TRANSFORMER SURVIVAL (TEST-SET)")
    print("=" * 74)
    print(f"  • Semester ROC-AUC     : {auc:.4f}")
    print(f"  • Semester PR-AUC      : {pr_auc:.4f}")
    print(f"  • Brier Score          : {brier:.4f}")
    print(f"  • Studierenden ROC-AUC : {student_auc:.4f}")
    print("=" * 74)

    # Logging
    base_dir = data_dir
    model_name = f"transformer_survival_{temporal}"
    if mode != 'standard':
        model_name += f"_{mode}"

    metrics_dict = {
        "model_type": model_name,
        "temporal": temporal,
        "mode": mode,
        "ROC-AUC_Semester": auc,
        "PR-AUC_Semester": pr_auc,
        "Brier_Score": brier,
        "ROC-AUC_Student": student_auc
    }
    save_metrics(model_name, metrics_dict, base_dir)
    save_keras_model(model, model_name, base_dir)
    plot_learning_curve(history.history, model_name, base_dir, metric_name='loss')
    plot_roc_curve(y_test_flat, preds_flat, model_name, base_dir)
    plot_pr_curve(y_test_flat, preds_flat, model_name, base_dir)

    if temporal == 'prev' and mode == 'standard':
        save_metrics("transformer_survival", metrics_dict, base_dir)
        save_keras_model(model, "transformer_survival", base_dir)

    print(f"[OK] Training und Logging für {model_name} abgeschlossen.")
    return model, scaler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Causal Semester Transformer Survival")
    parser.add_argument('--data_dir', type=str, default='src/output_dl')
    parser.add_argument('--temporal', type=str, default='prev', choices=['prev', 'cum'])
    parser.add_argument('--mode', type=str, default='standard')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    train_transformer_survival(
        data_dir=Path(args.data_dir),
        temporal=args.temporal,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
