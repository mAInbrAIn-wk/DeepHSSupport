"""
Exam-Level Causal Transformer Survival Model (DTL Hazard auf Prüfungsebene)
=============================================================================
Transformer-Modell mit Kausalem Attention-Masking auf Ebene der einzelnen Prüfungen.
Verfolgt die Trajektorie von Prüfung zu Prüfung und berechnet die bedingte Hazard Rate h(k).

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
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, TimeDistributed, Masking
import tensorflow.keras.backend as K

# Import metrics_logger and feature_builder
sys.path.insert(0, str(Path(__file__).parent))
from recurrent_survival_model import masked_binary_crossentropy, PADDING_VALUE
from metrics_logger import save_metrics, save_keras_model, plot_learning_curve, plot_roc_curve, plot_pr_curve
import feature_builder as fb


def build_exam_causal_transformer(max_exams: int, num_features: int, d_model: int = 64, num_heads: int = 4) -> Model:
    inputs = Input(shape=(max_exams, num_features))

    x = Dense(d_model)(inputs)
    x = LayerNormalization()(x)

    causal_mask = tf.linalg.band_part(tf.ones((max_exams, max_exams)), -1, 0)
    causal_mask = tf.cast(causal_mask, tf.bool)

    # Block 1
    attn_out1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(
        x, x, attention_mask=causal_mask
    )
    x1 = LayerNormalization()(x + Dropout(0.1)(attn_out1))
    ff_out1 = Dense(d_model, activation='relu')(x1)
    x1 = LayerNormalization()(x1 + Dropout(0.1)(ff_out1))

    # Block 2
    attn_out2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(
        x1, x1, attention_mask=causal_mask
    )
    x2 = LayerNormalization()(x1 + Dropout(0.1)(attn_out2))
    ff_out2 = Dense(d_model, activation='relu')(x2)
    x2 = LayerNormalization()(x2 + Dropout(0.1)(ff_out2))

    time_dense = TimeDistributed(Dense(32, activation='relu'))(x2)
    time_dense = TimeDistributed(LayerNormalization())(time_dense)
    outputs = TimeDistributed(Dense(1, activation='sigmoid'))(time_dense)

    model = Model(inputs=inputs, outputs=outputs, name="exam_causal_transformer")
    model.compile(optimizer=tf.keras.optimizers.Adam(0.003), loss=masked_binary_crossentropy)
    return model


def train_transformer_exam_survival(data_dir: Path = Path('src/output_dl'),
                                   max_exams: int = 40,
                                   temporal: str = 'prev',
                                   mode: str = 'standard',
                                   epochs: int = 25,
                                   batch_size: int = 256):
    print("\n" + "=" * 74)
    print(f"   CAUSAL EXAM TRANSFORMER SURVIVAL (temporal={temporal}, mode={mode})")
    print("=" * 74)

    studis, X_seq, y_seq, studi_events, feature_names, _ = fb.build_exam_sequence_tensor(
        data_dir, max_exams=max_exams, mode=mode, temporal=temporal
    )

    n_samples, n_timesteps, n_features = X_seq.shape
    print(f"Dataset: Shape={X_seq.shape} ({n_features} Features pro Prüfung)")

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
    model = build_exam_causal_transformer(n_timesteps, n_features, d_model=64, num_heads=4)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

    print(f"\nTrainiere Exam-Transformer ({epochs} Epochen, Batch-Size {batch_size})...")
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
            h_k = np.clip(test_preds[i, :s_len, 0], 1e-6, 1.0 - 1e-6)
            surv_probs[i] = np.prod(1.0 - h_k)
    student_auc = float(roc_auc_score(test_student_events, 1.0 - surv_probs))

    print("\n" + "=" * 74)
    print("   ERGEBNISSE CAUSAL EXAM TRANSFORMER SURVIVAL (TEST-SET)")
    print("=" * 74)
    print(f"  • Prüfungs ROC-AUC     : {auc:.4f}")
    print(f"  • Prüfungs PR-AUC      : {pr_auc:.4f}")
    print(f"  • Brier Score          : {brier:.4f}")
    print(f"  • Studierenden ROC-AUC : {student_auc:.4f}")
    print("=" * 74)

    # Logging
    base_dir = data_dir
    model_name = f"transformer_exam_survival_{temporal}"
    if mode != 'standard':
        model_name += f"_{mode}"

    metrics_dict = {
        "model_type": model_name,
        "temporal": temporal,
        "mode": mode,
        "ROC-AUC_Exam": auc,
        "PR-AUC_Exam": pr_auc,
        "Brier_Score": brier,
        "ROC-AUC_Student": student_auc
    }
    save_metrics(model_name, metrics_dict, base_dir)
    save_keras_model(model, model_name, base_dir)
    plot_learning_curve(history.history, model_name, base_dir, metric_name='loss')
    plot_roc_curve(y_test_flat, preds_flat, model_name, base_dir)
    plot_pr_curve(y_test_flat, preds_flat, model_name, base_dir)

    if temporal == 'prev' and mode == 'standard':
        save_metrics("transformer_exam_survival", metrics_dict, base_dir)
        save_keras_model(model, "transformer_exam_survival", base_dir)

    print(f"[OK] Training und Logging für {model_name} abgeschlossen.")
    return model, scaler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Exam-Level Causal Transformer Survival")
    parser.add_argument('--data_dir', type=str, default='src/output_dl')
    parser.add_argument('--temporal', type=str, default='prev', choices=['prev', 'cum'])
    parser.add_argument('--mode', type=str, default='standard')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    train_transformer_exam_survival(
        data_dir=Path(args.data_dir),
        temporal=args.temporal,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
