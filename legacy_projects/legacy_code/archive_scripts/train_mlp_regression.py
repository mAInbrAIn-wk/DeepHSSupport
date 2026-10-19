"""
Training Script: Abschlussnoten-Vorhersage (Regression)
======================================================
Vorhersage der Abschlussnote ('abschlussnote') auf Basis von Landmark-Features
aus `feature_builder.py`.

Modelle:
1. Linear Regression (Ridge Baseline)
2. Support Vector Regression (SVR)
3. Random Forest Regressor
4. Keras Multi-Layer Perceptron (MLP Regressor)
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Import metrics_logger and feature_builder
sys.path.insert(0, str(Path(__file__).parent))
from metrics_logger import save_metrics, plot_learning_curve, save_keras_model
import feature_builder as fb


def build_and_train_mlp_regressor(input_dim: int, X_train, y_train, X_val, y_val, epochs: int = 100, batch_size: int = 64):
    tf.random.set_seed(42)
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        LayerNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        LayerNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        LayerNormalization(),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=[es], verbose=0
    )
    return model, history


def run_regression_training(data_dir: Path = Path('src/output_dl'),
                            graduates_only: bool = True,
                            mode: str = 'standard',
                            epochs: int = 80,
                            batch_size: int = 64):
    print("\n" + "=" * 74)
    print(f"   LANDMARK ABSCHLUSSNOTEN-REGRESSION (graduates_only={graduates_only}, mode={mode})")
    print("=" * 74)

    df_lm, feature_cols, target_col, _ = fb.build_landmark_dataset(
        data_dir, t0=2, mode=mode, target='abschlussnote', target_type='continuous', graduates_only=graduates_only
    )

    y = df_lm['abschlussnote'].values
    print(f"Dataset geladen: {len(df_lm)} Studierende, {len(feature_cols)} Landmark-Features")

    # 3-Way Split
    X_train_raw, X_temp, y_train, y_temp = train_test_split(df_lm[feature_cols], y, test_size=0.30, random_state=42)
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_train = scaler.fit_transform(imputer.fit_transform(X_train_raw))
    X_val = scaler.transform(imputer.transform(X_val_raw))
    X_test = scaler.transform(imputer.transform(X_test_raw))

    input_dim = X_train.shape[1]

    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "SVR (RBF)": SVR(C=1.0, epsilon=0.1)
    }

    results = {}

    for name, model in models.items():
        print(f"\nTrainiere {name} ...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        results[name] = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
        print(f"  -> {name}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    # 4. Keras MLP Regressor
    print("\nTrainiere Keras MLP Regressor ...")
    mlp_model, mlp_history = build_and_train_mlp_regressor(input_dim, X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)

    mlp_preds = mlp_model.predict(X_test, verbose=0).flatten()
    mse_mlp = float(mean_squared_error(y_test, mlp_preds))
    rmse_mlp = float(np.sqrt(mse_mlp))
    mae_mlp = float(mean_absolute_error(y_test, mlp_preds))
    r2_mlp = float(r2_score(y_test, mlp_preds))

    results["Keras MLP Regressor"] = {"mse": mse_mlp, "rmse": rmse_mlp, "mae": mae_mlp, "r2": r2_mlp}
    print(f"  -> Keras MLP Regressor: R2={r2_mlp:.4f}, RMSE={rmse_mlp:.4f}, MAE={mae_mlp:.4f}")

    # Logging
    base_dir = data_dir
    model_name = f"mlp_regression_{mode}" if mode != 'standard' else "mlp_regression"
    save_metrics(model_name, results, base_dir)
    save_keras_model(mlp_model, model_name, base_dir)
    plot_learning_curve(mlp_history.history, model_name, base_dir, metric_name='loss')

    print("\n" + "=" * 74)
    print(f"[OK] Regression erfolgreich trainiert und unter {data_dir} geloggt.")
    print("=" * 74)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Landmark GPA Regression")
    parser.add_argument('--data_dir', type=str, default='src/output_dl')
    parser.add_argument('--graduates_only', action='store_true', default=True)
    parser.add_argument('--mode', type=str, default='standard')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    run_regression_training(Path(args.data_dir), graduates_only=args.graduates_only, mode=args.mode, epochs=args.epochs)
