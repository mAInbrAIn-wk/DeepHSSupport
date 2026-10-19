"""
Zeitreihen-Analyse: Semester-Transformer Regressor (Abschlussnoten-Vorhersage)
=============================================================================
Transformer-Architektur auf Semester-Ebene zur Vorhersage der Abschlussnote / GPA-Trajektorie.
Verwendet Multi-Head Self-Attention, Masking und LayerNormalization.

Unterstützt über feature_builder.py:
- Vektorisierte Tensor-Erstellung (build_semester_sequence_tensor)
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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Masking
from tensorflow.keras.callbacks import EarlyStopping

# Import metrics_logger and feature_builder
sys.path.insert(0, str(Path(__file__).parent))
from metrics_logger import save_metrics, save_keras_model, plot_learning_curve, plot_parity_plot
from timeseries_semester import PADDING_VALUE
import feature_builder as fb


def build_semester_transformer(max_semesters: int, num_features: int, d_model: int = 64, num_heads: int = 4) -> Model:
    inputs = Input(shape=(max_semesters, num_features))
    masked_inputs = Masking(mask_value=PADDING_VALUE)(inputs)

    x = Dense(d_model)(masked_inputs)
    x = LayerNormalization()(x)

    attn_out1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
    x1 = LayerNormalization()(x + Dropout(0.1)(attn_out1))
    ff_out1 = Dense(d_model, activation='relu')(x1)
    x1 = LayerNormalization()(x1 + Dropout(0.1)(ff_out1))

    attn_out2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x1, x1)
    x2 = LayerNormalization()(x1 + Dropout(0.1)(attn_out2))
    ff_out2 = Dense(d_model, activation='relu')(x2)
    x2 = LayerNormalization()(x2 + Dropout(0.1)(ff_out2))

    pooled = GlobalAveragePooling1D()(x2)

    dense_out = Dense(32, activation='relu')(pooled)
    dense_out = LayerNormalization()(dense_out)
    dense_out = Dropout(0.2)(dense_out)
    outputs = Dense(1, activation='linear')(dense_out)

    model = Model(inputs=inputs, outputs=outputs, name="semester_transformer_regressor")
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])
    return model


def train_timeseries_semester_transformer(data_dir: Path = Path('src/output_dl'),
                                         max_semesters: int = 16,
                                         temporal: str = 'prev',
                                         mode: str = 'standard',
                                         epochs: int = 35,
                                         batch_size: int = 128):
    print("\n" + "=" * 74)
    print(f"   SEMESTER TRANSFORMER REGRESSOR (temporal={temporal}, mode={mode})")
    print("=" * 74)

    # Lade Absolventen mit bekannter Abschlussnote
    studis, X_seq, y_seq, studi_events, feature_names, _ = fb.build_semester_sequence_tensor(
        data_dir, max_semesters=max_semesters, mode=mode, temporal=temporal, target_type='gpa'
    )

    # Abschlussnoten-Zielwert pro Student
    df_abschluesse, _ = fb._load_raw_data(data_dir)
    note_dict = df_abschluesse.set_index('studierenden_id')['abschlussnote'].to_dict()

    y_student = np.array([note_dict.get(s, np.nan) for s in studis])
    valid_grad_mask = ~np.isnan(y_student)

    X_grad = X_seq[valid_grad_mask]
    y_grad = y_student[valid_grad_mask]

    n_samples, n_timesteps, n_features = X_grad.shape
    print(f"Dataset: {n_samples} Absolventen, Shape={X_grad.shape}")

    # 3-Way Split
    idx = np.arange(n_samples)
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42)

    X_train, y_train = X_grad[train_idx].copy(), y_grad[train_idx].copy()
    X_val, y_val = X_grad[val_idx].copy(), y_grad[val_idx].copy()
    X_test, y_test = X_grad[test_idx].copy(), y_grad[test_idx].copy()

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
    model = build_semester_transformer(n_timesteps, n_features, d_model=64, num_heads=4)
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print(f"\nTrainiere Semester-Transformer ({epochs} Epochen, Batch-Size {batch_size})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=[es], verbose=0
    )

    # Evaluation
    test_preds = model.predict(X_test, verbose=0).flatten()

    mse = float(mean_squared_error(y_test, test_preds))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, test_preds))
    r2 = float(r2_score(y_test, test_preds))

    print("\n" + "=" * 74)
    print("   ERGEBNISSE SEMESTER TRANSFORMER REGRESSION (TEST-SET)")
    print("=" * 74)
    print(f"  • R2 Score : {r2:.4f}")
    print(f"  • RMSE     : {rmse:.4f}")
    print(f"  • MAE      : {mae:.4f}")
    print("=" * 74)

    # Logging
    base_dir = data_dir
    model_name = f"timeseries_semester_transformer_{temporal}_{mode}" if (temporal != 'prev' or mode != 'standard') else "timeseries_semester_transformer"

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
    plot_parity_plot(y_test, test_preds, model_name, base_dir)

    print(f"[OK] Training und Logging für {model_name} abgeschlossen.")
    return model, scaler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Semester Timeseries Transformer")
    parser.add_argument('--data_dir', type=str, default='src/output_dl')
    parser.add_argument('--temporal', type=str, default='prev', choices=['prev', 'cum'])
    parser.add_argument('--mode', type=str, default='standard')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    train_timeseries_semester_transformer(
        data_dir=Path(args.data_dir),
        temporal=args.temporal,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
