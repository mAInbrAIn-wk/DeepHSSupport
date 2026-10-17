"""
Dynamic DeepHit Competing Risks Model (Semester Level)
======================================================
Modelliert zeitveränderliche konkurrierende Risiken (Dropout vs. Abschluss)
über ein Shared-GRU Backbone mit zwei separaten TimeDistributed Output-Heads.

Unterstützt über feature_builder.py:
- Alle Modi: standard, gradeblind, blind, oracle, realistic
- Temporale Modi: temporal='prev' (Vorsemester/Delta) oder temporal='cum' (Gesamthistorie)
- Dual-Target Competing Risks: y_seq[:, :, 0] = Dropout, y_seq[:, :, 1] = Abschluss
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
from tensorflow.keras.layers import Input, Dense, Dropout, Masking, GRU, LayerNormalization, TimeDistributed

# Import metrics_logger and feature_builder
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deepsupport.models.semester_gru import masked_binary_crossentropy, PADDING_VALUE
from deepsupport.evaluation.metrics_logger import save_metrics, save_keras_model, plot_learning_curve, plot_roc_curve, plot_pr_curve, SurvivalEvaluator
import deepsupport.data_engine.feature_builder as fb


def build_dynamic_deephit_network(n_timesteps: int, n_features: int) -> Model:
    """
    Erstellt ein Dual-Head GRU Competing-Risks Modell mit konsistenter 3-Zustands-Softmax-Parametrisierung
    pro Zeitschritt (0: Aktiv/Weiterstudium, 1: Dropout, 2: Abschluss), sodass
    h_drop(t) + h_grad(t) <= 1.0 für alle t strikt garantiert ist.
    """
    inputs = Input(shape=(n_timesteps, n_features), name='input_sequence')
    masked = Masking(mask_value=PADDING_VALUE)(inputs)

    # Gemeinsames Shared Feature Extraction Backbone
    x = GRU(64, return_sequences=True, dropout=0.2, name='shared_gru_1')(masked)
    x = LayerNormalization(name='ln_1')(x)
    x = GRU(32, return_sequences=True, dropout=0.2, name='shared_gru_2')(x)
    x = LayerNormalization(name='ln_2')(x)

    # Gemeinsame Competing-Risks Repräsentation + 3-Wege Softmax (Aktiv, Dropout, Abschluss)
    h_joint = TimeDistributed(Dense(16, activation='relu'), name='joint_cr_dense')(x)
    h_joint = Dropout(0.2)(h_joint)
    probs_3 = TimeDistributed(Dense(3, activation='softmax'), name='competing_risks_softmax')(h_joint)

    # Feste lineare Selektoren auf Zustand 1 (Dropout) und Zustand 2 (Abschluss),
    # vermeidet Lambda-Layer beim .keras-Speichern und garantiert h_drop + h_grad <= 1
    sel_drop = Dense(
        1,
        use_bias=False,
        trainable=False,
        kernel_initializer=tf.constant_initializer([[0.0], [1.0], [0.0]]),
        name='sel_dropout',
    )
    sel_grad = Dense(
        1,
        use_bias=False,
        trainable=False,
        kernel_initializer=tf.constant_initializer([[0.0], [0.0], [1.0]]),
        name='sel_grad',
    )
    out_dropout = TimeDistributed(sel_drop, name='out_dropout')(probs_3)
    out_grad = TimeDistributed(sel_grad, name='out_grad')(probs_3)

    model = Model(inputs=inputs, outputs=[out_dropout, out_grad], name='dynamic_deephit_competing_risks')
    return model


def train_dynamic_deephit_model(data_dir: Path = Path('src/output_dl'),
                                max_semesters: int = 16,
                                temporal: str = 'prev',
                                mode: str = 'standard',
                                epochs: int = 35,
                                batch_size: int = 128):
    print("\n" + "=" * 74)
    print(f"   DYNAMIC DEEPHIT COMPETING RISKS MODEL (temporal={temporal}, mode={mode})")
    print("=" * 74)

    studis, X_seq, y_seq, studi_events, feature_names, _ = fb.build_semester_sequence_tensor(
        data_dir, max_semesters=max_semesters, mode=mode, temporal=temporal, target_type='competing_risks'
    )

    n_samples, n_timesteps, n_features = X_seq.shape
    print(f"Dataset: Shape={X_seq.shape} ({n_features} Features, Competing Risks: Dropout & Grad)")

    # 3-Way Split
    idx = np.arange(n_samples)
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=42, stratify=(studi_events == 1).astype(int))
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42, stratify=(studi_events[temp_idx] == 1).astype(int))

    y_train = y_seq[train_idx].copy()
    y_val = y_seq[val_idx].copy()
    y_test = y_seq[test_idx].copy()

    # Zentralisierte 3D-Skalierung über feature_builder
    scaled_splits, scaler = fb.scale_3d_sequences(
        {"train": X_seq[train_idx], "val": X_seq[val_idx], "test": X_seq[test_idx]},
        fit_key="train",
    )
    X_train, X_val, X_test = scaled_splits["train"], scaled_splits["val"], scaled_splits["test"]
    test_mask = X_test[:, :, 0] != PADDING_VALUE

    # Targets separieren
    y_train_drop = y_train[:, :, 0:1]
    y_train_grad = y_train[:, :, 1:2]

    y_val_drop = y_val[:, :, 0:1]
    y_val_grad = y_val[:, :, 1:2]

    y_test_drop = y_test[:, :, 0:1]
    y_test_grad = y_test[:, :, 1:2]

    # Modell kompilieren
    tf.random.set_seed(42)
    model = build_dynamic_deephit_network(n_timesteps, n_features)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss={
            'out_dropout': masked_binary_crossentropy,
            'out_grad': masked_binary_crossentropy
        },
        loss_weights={'out_dropout': 1.0, 'out_grad': 0.8}
    )

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print(f"\nTrainiere Dynamic DeepHit ({epochs} Epochen, Batch-Size {batch_size})...")
    history = model.fit(
        X_train, {'out_dropout': y_train_drop, 'out_grad': y_train_grad},
        validation_data=(X_val, {'out_dropout': y_val_drop, 'out_grad': y_val_grad}),
        epochs=epochs, batch_size=batch_size,
        callbacks=[es], verbose=0
    )

    # Evaluation
    pred_drop, pred_grad = model.predict(X_test, verbose=0)

    test_mask_1d = test_mask.flatten()
    y_drop_flat = y_test_drop.flatten()[test_mask_1d]
    pred_drop_flat = pred_drop.flatten()[test_mask_1d]

    y_grad_flat = y_test_grad.flatten()[test_mask_1d]
    pred_grad_flat = pred_grad.flatten()[test_mask_1d]

    auc_drop = float(roc_auc_score(y_drop_flat, pred_drop_flat))
    pr_auc_drop = float(average_precision_score(y_drop_flat, pred_drop_flat))
    brier_drop = float(brier_score_loss(y_drop_flat, pred_drop_flat))

    auc_grad = float(roc_auc_score(y_grad_flat, pred_grad_flat))
    pr_auc_grad = float(average_precision_score(y_grad_flat, pred_grad_flat))

    print("\n" + "=" * 74)
    print("   ERGEBNISSE DYNAMIC DEEPHIT COMPETING RISKS (TEST-SET)")
    print("=" * 74)
    print(f"  • Dropout Hazard ROC-AUC  : {auc_drop:.4f}")
    print(f"  • Dropout Hazard PR-AUC   : {pr_auc_drop:.4f}")
    print(f"  • Dropout Brier Score     : {brier_drop:.4f}")
    print(f"  • Abschluss Hazard ROC-AUC: {auc_grad:.4f}")
    print(f"  • Abschluss Hazard PR-AUC : {pr_auc_grad:.4f}")
    print("=" * 74)

    # Logging
    base_dir = data_dir
    model_name = f"dynamic_deephit_{temporal}"
    if mode != 'standard':
        model_name += f"_{mode}"

    metrics_dict = {
        "model_type": model_name,
        "temporal": temporal,
        "mode": mode,
        "ROC-AUC_Dropout": auc_drop,
        "PR-AUC_Dropout": pr_auc_drop,
        "Brier_Score_Dropout": brier_drop,
        "ROC-AUC_Graduation": auc_grad,
        "PR-AUC_Graduation": pr_auc_grad
    }
    
    evaluator = SurvivalEvaluator(base_dir=base_dir, model_name=model_name, temporal=temporal, mode=mode)
    evaluator.evaluate_and_log(
        y_true=y_drop_flat,
        y_prob=pred_drop_flat,
        model=model,
        history=history.history if hasattr(history, 'history') else history,
        extra_metrics=metrics_dict
    )
    save_metrics(model_name, metrics_dict, base_dir)

    if temporal == 'prev' and mode == 'standard':
        save_metrics("dynamic_deephit_delta", metrics_dict, base_dir)
        save_metrics("dynamic_deephit", metrics_dict, base_dir)
        save_keras_model(model, "dynamic_deephit_delta", base_dir)

    print(f"[OK] Training und Logging für {model_name} abgeschlossen.")
    return model, scaler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dynamic DeepHit Competing Risks Model")
    parser.add_argument('--data_dir', type=str, default='src/output_dl')
    parser.add_argument('--temporal', type=str, default='prev', choices=['prev', 'cum'])
    parser.add_argument('--mode', type=str, default='standard')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    train_dynamic_deephit_model(
        data_dir=Path(args.data_dir),
        temporal=args.temporal,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
