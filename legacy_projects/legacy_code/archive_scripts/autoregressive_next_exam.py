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
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, average_precision_score, brier_score_loss

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Masking, GRU, LayerNormalization, Concatenate

# Import metrics_logger and feature_builder
sys.path.insert(0, str(Path(__file__).parent))
from metrics_logger import save_metrics, save_keras_model, plot_learning_curve, plot_parity_plot
import feature_builder as fb

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


def prepare_next_exam_dataset(data_dir: Path, max_history_len: int = 30):
    """Erstellt Paare von (Historie 1..k, Kontext k+1 -> Note k+1, Bestanden k+1)."""
    df_ab, df_pr = fb._load_raw_data(data_dir)

    df_pr = df_pr.sort_values(['studierenden_id', 'pruefung_id']).reset_index(drop=True)
    df_pr['bestanden_int'] = df_pr['bestanden'].astype(int)
    df_pr['note_clean'] = df_pr['note'].fillna(5.0)

    # Demographie
    df_ab['hzb_ord'] = df_ab['hzb_typ'].map(fb.HZB_ORDINAL_MAP).fillna(3.0)
    demo_dict = df_ab.set_index('studierenden_id')[['hzb_note', 'hzb_ord', 'erwerbstaetigkeit_std', 'erstakademiker']].to_dict('index')

    hist_feats = ['versuch', 'schwierigkeit', 'cp', 'bestanden_int', 'note_clean',
                  'support_vorher_fachlich', 'support_vorher_ueberfachlich', 'support_vorher_psychosozial', 'support_glz_fachlich', 'support_glz_ueberfachlich', 'support_glz_psychosozial', 'fachsemester']

    X_hist_list, X_ctx_list, y_grade_list, y_pass_list, student_id_list = [], [], [], [], []

    grouped = df_pr.groupby('studierenden_id')
    print(f"Erstelle Autoregressive Next-Exam Samples für {len(grouped)} Studierende ...")

    for s_id, group in grouped:
        demo = demo_dict.get(s_id, {'hzb_note': 2.5, 'hzb_ord': 3.0, 'erwerbstaetigkeit_std': 0.0, 'erstakademiker': 0})
        demo_vec = [float(demo['hzb_note']), float(demo['hzb_ord']), float(demo['erwerbstaetigkeit_std']), float(bool(demo['erstakademiker']))]

        records = group[hist_feats].values
        n_exams = len(records)

        # Erzeuge Samples für jeden Schritt k >= 1 (mindestens 1 Prüfung Historie)
        for k in range(1, min(n_exams, 35)):
            history = records[:k]
            next_exam = records[k]

            # Padded Sequence
            pad_seq = np.full((max_history_len, len(hist_feats)), PADDING_VALUE, dtype=np.float32)
            hist_len = min(len(history), max_history_len)
            pad_seq[:hist_len] = history[-hist_len:]

            # Kontext der nächsten Prüfung: Versuch, Schwierigkeit, CP, Supports + Demographie
            ctx = [next_exam[0], next_exam[1], next_exam[2], next_exam[5], next_exam[6], next_exam[7], next_exam[8], next_exam[9], next_exam[10], next_exam[11]] + demo_vec

            X_hist_list.append(pad_seq)
            X_ctx_list.append(ctx)
            y_grade_list.append(next_exam[4]) # note_clean
            y_pass_list.append(next_exam[3])  # bestanden_int
            student_id_list.append(s_id)

    X_hist = np.array(X_hist_list, dtype=np.float32)
    X_ctx = np.array(X_ctx_list, dtype=np.float32)
    y_grade = np.array(y_grade_list, dtype=np.float32)
    y_pass = np.array(y_pass_list, dtype=np.float32)
    student_ids = np.array(student_id_list)

    print(f"Gesamtanzahl Next-Exam Samples: {len(X_hist):,}")
    return X_hist, X_ctx, y_grade, y_pass, student_ids


def train_autoregressive_next_exam(data_dir: Path = Path('src/output_dl'),
                                  epochs: int = 25,
                                  batch_size: int = 256):
    print("\n" + "=" * 74)
    print("   AUTOREGRESSIVE NEXT-EXAM PREDICTION (DUAL-HEAD MULTI-TASK)")
    print("=" * 74)

    X_hist, X_ctx, y_grade, y_pass, student_ids = prepare_next_exam_dataset(data_dir)

    # 3-Way Group-Consistent Split on student IDs
    unique_students = np.unique(student_ids)
    tr_students, temp_students = train_test_split(unique_students, test_size=0.30, random_state=42)
    va_students, te_students = train_test_split(temp_students, test_size=0.50, random_state=42)

    tr_idx = np.where(np.isin(student_ids, tr_students))[0]
    va_idx = np.where(np.isin(student_ids, va_students))[0]
    te_idx = np.where(np.isin(student_ids, te_students))[0]

    # Skalierung
    v_mask_tr = X_hist[tr_idx, :, 0] != PADDING_VALUE
    scaler_seq = StandardScaler()
    scaler_seq.fit(X_hist[tr_idx][v_mask_tr])

    scaler_ctx = StandardScaler()
    scaler_ctx.fit(X_ctx[tr_idx])

    for split in [tr_idx, va_idx, te_idx]:
        vm = X_hist[split, :, 0] != PADDING_VALUE
        X_hist[split][vm] = scaler_seq.transform(X_hist[split][vm])
        X_ctx[split] = scaler_ctx.transform(X_ctx[split])

    # Modell
    tf.random.set_seed(42)
    model = build_dual_head_next_exam_model(
        seq_timesteps=X_hist.shape[1],
        seq_features=X_hist.shape[2],
        context_features=X_ctx.shape[1]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        loss={'out_grade': 'mse', 'out_pass': 'binary_crossentropy'},
        loss_weights={'out_grade': 1.0, 'out_pass': 0.8},
        metrics={'out_grade': ['mae'], 'out_pass': ['accuracy']}
    )

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

    print(f"\nTrainiere Dual-Head Next-Exam Modell ({epochs} Epochen)...")
    history = model.fit(
        {'exam_history': X_hist[tr_idx], 'next_exam_context': X_ctx[tr_idx]},
        {'out_grade': y_grade[tr_idx], 'out_pass': y_pass[tr_idx]},
        validation_data=(
            {'exam_history': X_hist[va_idx], 'next_exam_context': X_ctx[va_idx]},
            {'out_grade': y_grade[va_idx], 'out_pass': y_pass[va_idx]}
        ),
        epochs=epochs, batch_size=batch_size,
        callbacks=[es], verbose=0
    )

    # Evaluation
    preds_grade, preds_pass = model.predict(
        {'exam_history': X_hist[te_idx], 'next_exam_context': X_ctx[te_idx]},
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
    base_dir = data_dir
    model_name = "autoregressive_next_exam_dual_head"

    metrics_dict = {
        "Next_Exam_Grade_R2": r2,
        "Next_Exam_Grade_RMSE": rmse,
        "Next_Exam_Grade_MAE": mae,
        "Next_Exam_Pass_ROC_AUC": auc_pass,
        "Next_Exam_Pass_PR_AUC": pr_pass,
        "Next_Exam_Pass_Brier_Score": brier_pass
    }
    save_metrics(model_name, metrics_dict, base_dir)
    save_keras_model(model, model_name, base_dir)
    plot_learning_curve(history.history, model_name, base_dir, metric_name='loss')
    plot_parity_plot(y_te_grade, preds_grade, f"{model_name}_grade", base_dir)

    print(f"[OK] Autoregressive Next-Exam Modell erfolgreich gespeichert.")
    return metrics_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Autoregressive Next-Exam Prediction")
    parser.add_argument('--data_dir', type=str, default='src/output_dl')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    train_autoregressive_next_exam(Path(args.data_dir), epochs=args.epochs, batch_size=args.batch_size)
