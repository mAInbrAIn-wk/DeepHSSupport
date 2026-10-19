"""
Recurrent Exam-Level Survival Analysis (Keras GRU Sequenz auf Prüfungsebene)
===========================================================================
Erstellt und trainiert ein 3D-Sequenz-GRU-Modell (N, max_exams, n_features),
bei dem jeder Zeitschritt eine EINZELNE PRÜFUNG darstellt.

Konsolidiert die bisherigen Skripte (recurrent_exam_survival, _v2, _delta) in ein
einheitliches, parametrisiertes Skript mit voller feature_builder.py Anbindung.

Unterstützt:
- Alle Modi (standard, gradeblind, blind, oracle, realistic)
- Temporale Modi: temporal='prev' (Vor-Prüfungs-Werte) oder temporal='cum' (Gesamthistorie vor Prüfung k)
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Masking, GRU, TimeDistributed, LayerNormalization

# Import metrics_logger and feature_builder
sys.path.insert(0, str(Path(__file__).parent))
from recurrent_survival_model import masked_binary_crossentropy, PADDING_VALUE
from metrics_logger import save_metrics, save_keras_model, plot_learning_curve, plot_roc_curve, plot_pr_curve
import feature_builder as fb


def train_recurrent_exam_survival_model(data_dir: Path = Path('src/output_dl'),
                                       max_exams: int = 40,
                                       temporal: str = 'prev',
                                       mode: str = 'standard',
                                       epochs: int = 25,
                                       batch_size: int = 128):
    print("\n" + "=" * 74)
    print(f"   RECURRENT EXAM-LEVEL SURVIVAL GRU (temporal={temporal}, mode={mode})")
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

    # Keras GRU Modell
    tf.random.set_seed(42)
    model = Sequential([
        Masking(mask_value=PADDING_VALUE, input_shape=(n_timesteps, n_features)),
        GRU(64, return_sequences=True, dropout=0.2),
        LayerNormalization(),
        GRU(32, return_sequences=True, dropout=0.2),
        LayerNormalization(),
        TimeDistributed(Dense(16, activation='relu')),
        Dropout(0.2),
        TimeDistributed(Dense(1, activation='sigmoid'))
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003), loss=masked_binary_crossentropy)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

    print(f"\nTrainiere Exam-GRU ({epochs} Epochen, Batch-Size {batch_size})...")
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

    # Aggregiertes Studierenden-Risiko
    test_student_events = studi_events[test_idx]
    surv_probs = np.ones(len(test_idx))
    for i in range(len(test_idx)):
        s_len = np.sum(test_mask[i])
        if s_len > 0:
            h_k = np.clip(test_preds[i, :s_len, 0], 1e-6, 1.0 - 1e-6)
            surv_probs[i] = np.prod(1.0 - h_k)
    pred_student_risk = 1.0 - surv_probs
    student_auc = float(roc_auc_score(test_student_events, pred_student_risk))

    print("\n" + "=" * 74)
    print("   ERGEBNISSE RECURRENT EXAM SURVIVAL GRU (TEST-SET)")
    print("=" * 74)
    print(f"  • Prüfungs-Ebene ROC-AUC : {auc:.4f}")
    print(f"  • Prüfungs-Ebene PR-AUC  : {pr_auc:.4f}")
    print(f"  • Brier Score            : {brier:.4f}")
    print(f"  • Studierenden ROC-AUC   : {student_auc:.4f}")
    print("=" * 74)

    # Logging
    base_dir = data_dir
    model_name = f"recurrent_exam_survival_{temporal}"
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

    # Abwärtskompatible Dateinamen
    if temporal == 'prev' and mode == 'standard':
        save_metrics("recurrent_exam_survival_delta", metrics_dict, base_dir)
        save_metrics("recurrent_exam_survival", metrics_dict, base_dir)
        save_keras_model(model, "recurrent_exam_survival_delta", base_dir)
        save_keras_model(model, "recurrent_exam_survival", base_dir)

    print(f"[OK] Training und Logging für {model_name} abgeschlossen.")
    return model, scaler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Recurrent Exam-Level Survival GRU")
    parser.add_argument('--data_dir', type=str, default='src/output_dl')
    parser.add_argument('--temporal', type=str, default='prev', choices=['prev', 'cum'])
    parser.add_argument('--mode', type=str, default='standard')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    train_recurrent_exam_survival_model(
        data_dir=Path(args.data_dir),
        temporal=args.temporal,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
