"""
Autoregressive Next-Exam Prediction Model (Dual-Head Multi-Task Edition)
========================================================================
Prognostiziert die Note und Bestehenswahrscheinlichkeit der NÄCHSTEN Prüfung (k+1)
auf Basis der bisherigen Prüfungshistorie (1..k) und des Kontexts der nächsten Prüfung.

Architektur:
- Sequentieller GRU / Causal Attention Encoder über Prüfungen 1..k
- Late Fusion mit statischen Demographien + nächstem Prüfungskontext (Versuch, Schwierigkeit, Support)
- Dual-Head Multi-Task Output:
  1. Head 1 (Regression): Note der nächsten Prüfung Y_{k+1} in [1.0, 5.0] (MSE-Loss)
  2. Head 2 (Klassifikation): Bestehenswahrscheinlichkeit P(bestanden_{k+1}) (BCE-Loss)
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, average_precision_score, brier_score_loss

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Masking, GRU, LayerNormalization, Concatenate

# Import metrics_logger and feature_builder
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deepsupport.evaluation.metrics_logger import save_metrics, save_keras_model, plot_learning_curve, plot_parity_plot, DualHeadEvaluator
import deepsupport.data_engine.feature_builder as fb

PADDING_VALUE = -99.0


def build_dual_head_next_exam_model(seq_timesteps: int, seq_features: int, context_features: int) -> Model:
    """Baut das Dual-Head Late-Fusion Next-Exam Modell."""
    # Input 1: Prüfungshistorie 1..k
    seq_input = Input(shape=(seq_timesteps, seq_features), name='exam_history')
    masked_seq = Masking(mask_value=PADDING_VALUE)(seq_input)
    gru_out = GRU(64, return_sequences=False, dropout=0.2)(masked_seq)
    gru_out = LayerNormalization()(gru_out)

    # Input 2: Kontext der nächsten Prüfung (k+1) + Demographie
    ctx_input = Input(shape=(context_features,), name='next_exam_context')
    ctx_dense = Dense(32, activation='relu')(ctx_input)
    ctx_dense = LayerNormalization()(ctx_dense)

    # Fusion
    merged = Concatenate()([gru_out, ctx_dense])
    shared = Dense(64, activation='relu')(merged)
    shared = LayerNormalization()(shared)
    shared = Dropout(0.2)(shared)

    # Head 1: Noten-Regression (k+1)
    h_grade = Dense(32, activation='relu')(shared)
    out_grade = Dense(1, activation='linear', name='out_grade')(h_grade)

    # Head 2: Bestehens-Klassifikation (k+1)
    h_pass = Dense(32, activation='relu')(shared)
    out_pass = Dense(1, activation='sigmoid', name='out_pass')(h_pass)

    model = Model(inputs=[seq_input, ctx_input], outputs=[out_grade, out_pass], name='autoregressive_next_exam_dual_head')
    return model


def prepare_next_exam_dataset(data_dir: Path, max_history_len: int = 30, mode: str = 'standard'):
    """Delegiert an die zentralisierte Feature-Engine (fb.prepare_next_exam_dataset)."""
    return fb.prepare_next_exam_dataset(data_dir=data_dir, max_history_len=max_history_len, mode=mode)


def train_autoregressive_next_exam(data_dir: Path = Path('src/output_dl'),
                                  output_dir: Optional[Path] = None,
                                  mode: str = 'standard',
                                  epochs: int = 25,
                                  batch_size: int = 256):
    print("\n" + "=" * 74)
    print(f"   AUTOREGRESSIVE NEXT-EXAM PREDICTION (DUAL-HEAD MULTI-TASK | mode={mode})")
    print("=" * 74)

    X_hist, X_ctx, y_grade, y_pass, student_ids = fb.prepare_next_exam_dataset(data_dir, mode=mode)

    # 3-Way Group-Consistent Split on student IDs via feature_builder
    tr_students, va_students, te_students = fb.split_student_ids(student_ids, seed=42)

    tr_idx = np.where(np.isin(student_ids, tr_students))[0]
    va_idx = np.where(np.isin(student_ids, va_students))[0]
    te_idx = np.where(np.isin(student_ids, te_students))[0]

    # Zentralisierte 3D-Sequenz-Skalierung (verhindert den NumPy Fancy-Indexing Copy-Bug!)
    scaled_hist, _ = fb.scale_3d_sequences(
        {"train": X_hist[tr_idx], "val": X_hist[va_idx], "test": X_hist[te_idx]},
        fit_key="train",
        padding_value=PADDING_VALUE,
    )
    X_hist_tr, X_hist_va, X_hist_te = scaled_hist["train"], scaled_hist["val"], scaled_hist["test"]

    scaler_ctx = StandardScaler()
    X_ctx_tr = scaler_ctx.fit_transform(X_ctx[tr_idx]).astype(np.float32)
    X_ctx_va = scaler_ctx.transform(X_ctx[va_idx]).astype(np.float32)
    X_ctx_te = scaler_ctx.transform(X_ctx[te_idx]).astype(np.float32)

    # Modell
    tf.random.set_seed(42)
    model = build_dual_head_next_exam_model(
        seq_timesteps=X_hist.shape[1],
        seq_features=X_hist.shape[2],
        context_features=X_ctx.shape[1]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=['mse', 'binary_crossentropy'],
        loss_weights=[1.0, 0.8],
        metrics=[['mae'], ['accuracy']]
    )

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    lr_sched = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=0)

    # 1/K_i Sample-Gewichtung pro Studierendem (verhindert Dominanz von Langzeitstudierenden)
    unique_tr, counts_tr = np.unique(student_ids[tr_idx], return_counts=True)
    count_map_tr = dict(zip(unique_tr, counts_tr))
    sw_tr = np.array([1.0 / max(1, count_map_tr[s]) for s in student_ids[tr_idx]], dtype=np.float32)
    sw_tr *= len(sw_tr) / np.sum(sw_tr)

    print(f"\nTrainiere Dual-Head Next-Exam Modell ({epochs} Epochen, 1/K_i gewichtet)...")
    history = model.fit(
        [X_hist_tr, X_ctx_tr],
        [y_grade[tr_idx], y_pass[tr_idx]],
        sample_weight=sw_tr,
        validation_data=(
            [X_hist_va, X_ctx_va],
            [y_grade[va_idx], y_pass[va_idx]]
        ),
        epochs=epochs, batch_size=batch_size,
        callbacks=[es, lr_sched], verbose=0
    )

    # Evaluation
    preds_grade, preds_pass = model.predict(
        [X_hist_te, X_ctx_te],
        verbose=0
    )
    preds_grade = preds_grade.flatten()
    preds_pass = preds_pass.flatten()

    y_te_grade = y_grade[te_idx]
    y_te_pass = y_pass[te_idx]

    # Metriken Grade
    mse = float(mean_squared_error(y_te_grade, preds_grade))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_te_grade, preds_grade))
    r2 = float(r2_score(y_te_grade, preds_grade))

    # Metriken Pass
    auc_pass = float(roc_auc_score(y_te_pass, preds_pass))
    pr_pass = float(average_precision_score(y_te_pass, preds_pass))
    brier_pass = float(brier_score_loss(y_te_pass, preds_pass))

    print("\n" + "=" * 74)
    print("   ERGEBNISSE NEXT-EXAM DUAL-HEAD PREDICTION (TEST-SET)")
    print("=" * 74)
    print(f"  • Note (k+1) R2 Score        : {r2:.4f}")
    print(f"  • Note (k+1) RMSE            : {rmse:.4f}")
    print(f"  • Note (k+1) MAE             : {mae:.4f}")
    print(f"  • Bestanden (k+1) ROC-AUC    : {auc_pass:.4f}")
    print(f"  • Bestanden (k+1) PR-AUC     : {pr_pass:.4f}")
    print(f"  • Bestanden (k+1) Brier Score: {brier_pass:.4f}")
    print("=" * 74)

    # Logging
    base_dir = Path(output_dir) if output_dir is not None else data_dir
    model_name = "autoregressive_next_exam_dual_head" if mode == 'standard' else f"autoregressive_next_exam_dual_head_{mode}"

    metrics_dict = {
        "Next_Exam_Grade_R2": r2,
        "Next_Exam_Grade_RMSE": rmse,
        "Next_Exam_Grade_MAE": mae,
        "Next_Exam_Pass_ROC_AUC": auc_pass,
        "Next_Exam_Pass_PR_AUC": pr_pass,
        "Next_Exam_Pass_Brier_Score": brier_pass
    }
    
    evaluator = DualHeadEvaluator(base_dir=base_dir, model_name=model_name, mode=mode)
    evaluator.evaluate_and_log(
        y_grade_true=y_te_grade, y_grade_pred=preds_grade,
        y_pass_true=y_te_pass, y_pass_prob=preds_pass,
        history=history.history, model=model, mode=mode
    )

    print(f"[OK] Autoregressive Next-Exam Modell erfolgreich gespeichert.")
    return metrics_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Autoregressive Next-Exam Prediction")
    parser.add_argument('--data_dir', type=str, default='src/output_dl')
    parser.add_argument('--mode', type=str, default='standard')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    train_autoregressive_next_exam(Path(args.data_dir), mode=args.mode, epochs=args.epochs, batch_size=args.batch_size)
