"""
Training Script: Status-Vorhersage mit 3-Wege-Split (Train/Val/Test) & Lernkurven
==================================================================================
Vorhersage des Studienabschluss-Status ('status' oder binäres 'is_dropout')
auf Basis von Landmark-Features (Semester 1–2) aus `feature_builder.py`.

Modelle:
1. Naive Bayes Classifier
2. Random Forest Classifier
3. Support Vector Machine (SVC)
4. Keras Multi-Layer Perceptron (MLP)
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, brier_score_loss
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Import metrics_logger and feature_builder
sys.path.insert(0, str(Path(__file__).parent))
from metrics_logger import save_metrics, plot_roc_curve, plot_pr_curve, plot_learning_curve, save_keras_model, plot_confusion_matrix
import feature_builder as fb


def build_and_train_mlp(input_dim: int, num_classes: int, X_train, y_train, X_val, y_val, epochs: int = 100, batch_size: int = 64):
    tf.random.set_seed(42)
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        LayerNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        LayerNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        LayerNormalization(),
        Dense(1 if num_classes == 2 else num_classes, activation='sigmoid' if num_classes == 2 else 'softmax')
    ])

    loss = 'binary_crossentropy' if num_classes == 2 else 'sparse_categorical_crossentropy'
    metrics = ['accuracy']
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=loss, metrics=metrics)

    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=[es], verbose=0
    )
    return model, history


def run_baseline_training(data_dir: Path = Path('src/output_dl'),
                          binary_target: bool = True,
                          mode: str = 'standard',
                          epochs: int = 80,
                          batch_size: int = 64):
    print("\n" + "=" * 74)
    print(f"   LANDMARK CLASSIFICATION BASELINES (binary={binary_target}, mode={mode})")
    print("=" * 74)

    target_name = 'dropout' if binary_target else 'status'
    target_type = 'binary' if binary_target else 'multiclass'

    df_lm, feature_cols, target_col, _ = fb.build_landmark_dataset(
        data_dir, t0=2, mode=mode, target=target_name, target_type=target_type
    )

    if binary_target:
        y = df_lm['is_dropout'].values
        class_names = ['absolviert', 'abgebrochen']
        num_classes = 2
    else:
        le = LabelEncoder()
        y = le.fit_transform(df_lm['status'].astype(str))
        class_names = [str(c) for c in le.classes_]
        num_classes = len(class_names)

    print(f"Dataset geladen: {len(df_lm)} Studierende, {len(feature_cols)} Landmark-Features")

    # 3-Way Split
    X_train_raw, X_temp, y_train, y_temp = train_test_split(df_lm[feature_cols], y, test_size=0.30, random_state=42, stratify=y)
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    # Skalierung
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(imputer.fit_transform(X_train_raw))
    X_val = scaler.transform(imputer.transform(X_val_raw))
    X_test = scaler.transform(imputer.transform(X_test_raw))

    input_dim = X_train.shape[1]

    models = {
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "SVM (RBF)": SVC(probability=True, random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"\nTrainiere {name} ...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

        if binary_target and hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = float(roc_auc_score(y_test, y_proba))
            pr_auc = float(average_precision_score(y_test, y_proba))
            brier = float(brier_score_loss(y_test, y_proba))
        else:
            roc_auc, pr_auc, brier = 0.0, 0.0, 0.0

        results[name] = {
            "accuracy": acc, "f1_weighted": f1,
            "roc_auc": roc_auc, "pr_auc": pr_auc, "brier": brier
        }
        print(f"  -> {name}: Acc={acc:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")

    # 4. Keras MLP
    print("\nTrainiere Keras MLP ...")
    mlp_model, mlp_history = build_and_train_mlp(input_dim, num_classes, X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)

    if binary_target:
        mlp_probs = mlp_model.predict(X_test, verbose=0).flatten()
        mlp_pred = (mlp_probs >= 0.5).astype(int)
        roc_auc_mlp = float(roc_auc_score(y_test, mlp_probs))
        pr_auc_mlp = float(average_precision_score(y_test, mlp_probs))
        brier_mlp = float(brier_score_loss(y_test, mlp_probs))
    else:
        mlp_probs = mlp_model.predict(X_test, verbose=0)
        mlp_pred = np.argmax(mlp_probs, axis=1)
        roc_auc_mlp, pr_auc_mlp, brier_mlp = 0.0, 0.0, 0.0

    acc_mlp = float(accuracy_score(y_test, mlp_pred))
    f1_mlp = float(f1_score(y_test, mlp_pred, average='weighted', zero_division=0))

    results["Keras MLP"] = {
        "accuracy": acc_mlp, "f1_weighted": f1_mlp,
        "roc_auc": roc_auc_mlp, "pr_auc": pr_auc_mlp, "brier": brier_mlp
    }
    print(f"  -> Keras MLP: Acc={acc_mlp:.4f}, F1={f1_mlp:.4f}, ROC-AUC={roc_auc_mlp:.4f}, PR-AUC={pr_auc_mlp:.4f}")

    # Metrics Logging
    base_dir = data_dir
    model_name = f"mlp_baseline_{mode}" if mode != 'standard' else "mlp_baseline"
    save_metrics(model_name, results, base_dir)
    save_keras_model(mlp_model, model_name, base_dir)
    plot_learning_curve(mlp_history.history, model_name, base_dir, metric_name='accuracy')

    if binary_target:
        plot_roc_curve(y_test, mlp_probs, model_name, base_dir)
        plot_pr_curve(y_test, mlp_probs, model_name, base_dir)

    print("\n" + "=" * 74)
    print(f"[OK] Baselines erfolgreich trainiert und unter {data_dir} geloggt.")
    print("=" * 74)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Landmark Classification Baselines")
    parser.add_argument('--data_dir', type=str, default='src/output_dl')
    parser.add_argument('--binary', action='store_true', default=True)
    parser.add_argument('--mode', type=str, default='standard')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    run_baseline_training(Path(args.data_dir), binary_target=args.binary, mode=args.mode, epochs=args.epochs)
