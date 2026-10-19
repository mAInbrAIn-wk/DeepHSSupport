"""
Zeitreihen-Analyse: Variante 1 (Semester-basierte Zeitreihe)
===========================================================
Modelliert und prognostiziert den Semester-Notenverlauf mit recurrenten Netzen (LSTM/GRU).

Unterstützt über feature_builder.py:
- Vektorisierte Tensor-Erstellung (build_semester_sequence_tensor mit target_type='gpa')
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Masking, LSTM, GRU, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Import metrics_logger and feature_builder
sys.path.insert(0, str(Path(__file__).parent))
from metrics_logger import save_metrics, save_keras_model, plot_learning_curve, plot_parity_plot
import feature_builder as fb

PADDING_VALUE = -99.0


def masked_mse_loss(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, PADDING_VALUE), tf.float32)
    diff = (y_true - y_pred) * mask
    return tf.reduce_sum(tf.square(diff)) / (tf.reduce_sum(mask) + 1e-7)


def train_timeseries_semester(data_dir: Path = Path('src/output_dl'),
                              max_semesters: int = 16,
                              temporal: str = 'prev',
                              mode: str = 'standard',
                              epochs: int = 35,
                              batch_size: int = 128):
    print("\n" + "=" * 74)
    print(f"   SEMESTER-LEVEL LSTM TIMESERIES GPA REGRESSION (temporal={temporal}, mode={mode})")
    print("=" * 74)

    studis, X_seq, y_seq, studi_events, feature_names, _ = fb.build_semester_sequence_tensor(
        data_dir, max_semesters=max_semesters, mode=mode, temporal=temporal, target_type='gpa'
    )

    n_samples, n_timesteps, n_features = X_seq.shape
    print(f"Dataset: Shape={X_seq.shape} ({n_features} Features pro Semester)")

    # 3-Way Split
    idx = np.arange(n_samples)
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42)

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

    # Keras LSTM Modell
    tf.random.set_seed(42)
    model = Sequential([
        Masking(mask_value=PADDING_VALUE, input_shape=(n_timesteps, n_features)),
        LSTM(64, return_sequences=True, dropout=0.2),
        LayerNormalization(),
        LSTM(32, return_sequences=True, dropout=0.2),
        LayerNormalization(),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003), loss=masked_mse_loss, metrics=['mae'])
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print(f"\nTrainiere Semester-LSTM ({epochs} Epochen, Batch-Size {batch_size})...")
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

    mse = float(mean_squared_error(y_test_flat, preds_flat))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test_flat, preds_flat))
    r2 = float(r2_score(y_test_flat, preds_flat))

    print("\n" + "=" * 74)
    print("   ERGEBNISSE SEMESTER TIMESERIES LSTM (TEST-SET)")
    print("=" * 74)
    print(f"  • R2 Score : {r2:.4f}")
    print(f"  • RMSE     : {rmse:.4f}")
    print(f"  • MAE      : {mae:.4f}")
    print("=" * 74)

    # Logging
    base_dir = data_dir
    model_name = f"timeseries_semester_lstm_{temporal}_{mode}" if (temporal != 'prev' or mode != 'standard') else "timeseries_semester_lstm"

    metrics_dict = {
        "model_type": model_name,
        "temporal": temporal,
        "mode": mode,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MSE": mse
    }
    save_metrics(model_name, metrics_dict, base_dir)
    save_keras_model(model, model_name, base_dir)
    plot_learning_curve(history.history, model_name, base_dir, metric_name='loss')
    plot_parity_plot(y_test_flat, preds_flat, model_name, base_dir)

    print(f"[OK] Training und Logging für {model_name} abgeschlossen.")
    return model, scaler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Semester Timeseries LSTM")
    parser.add_argument('--data_dir', type=str, default='src/output_dl')
    parser.add_argument('--temporal', type=str, default='prev', choices=['prev', 'cum'])
    parser.add_argument('--mode', type=str, default='standard')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    train_timeseries_semester(
        data_dir=Path(args.data_dir),
        temporal=args.temporal,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
