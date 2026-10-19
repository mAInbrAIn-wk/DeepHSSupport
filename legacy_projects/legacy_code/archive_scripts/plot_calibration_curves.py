"""
Kalibrierungsanalyse & Reliability Diagrams
============================================
Erstellt Kalibrierungskurven (Reliability Diagrams) und Brier Score Analysen
für die probabilistischen Survival-Modelle unter Klassen-Imbalance.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

import tensorflow as tf

import feature_builder as fb
from metrics_logger import save_metrics
import tensorflow.keras.backend as K

PADDING_VALUE = -99.0

@tf.keras.utils.register_keras_serializable()
def masked_binary_crossentropy(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, PADDING_VALUE), tf.float32)
    y_true_clean = tf.maximum(y_true, 0.0)
    bce = K.binary_crossentropy(y_true_clean, y_pred)
    return tf.reduce_sum(bce * mask) / (tf.reduce_sum(mask) + 1e-7)


def main(data_dir=None):
    print("\n==========================================================================")
    print("   KALIBRIERUNGSANALYSE & RELIABILITY DIAGRAMME")
    print("==========================================================================")
    
    if data_dir is not None:
        data_dir = Path(data_dir)
    elif os.environ.get('DATA_DIR'):
        data_dir = Path(os.environ['DATA_DIR'])
    else:
        data_dir = Path('../output_dl') if Path('../output_dl').exists() else Path('output_dl')
    plots_dir = data_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    panel_df, f_cols, target_col, _ = fb.build_semester_panel_df(data_dir, mode='standard', temporal='prev')
    
    studis = panel_df['studierenden_id'].unique()
    train_ids, test_ids = train_test_split(studis, test_size=0.20, random_state=42)
    
    train_panel = panel_df[panel_df['studierenden_id'].isin(train_ids)].copy()
    test_panel  = panel_df[panel_df['studierenden_id'].isin(test_ids)].copy()
    
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='median')
    X_train = scaler.fit_transform(imputer.fit_transform(train_panel[f_cols]))
    X_test = scaler.transform(imputer.transform(test_panel[f_cols]))
    y_test = test_panel['event'].values
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfekt kalibriert")
    
    metrics = {}
    
    # 1. Extended Logistic Hazard
    for name in ["extended_logistic_hazard_prev.keras", "extended_logistic_hazard_cum.keras", "extended_logistic_hazard_delta.keras", "extended_logistic_hazard.keras"]:
        lh_model_path = data_dir / "models" / name
        if lh_model_path.exists():
            try:
                lh_model = tf.keras.models.load_model(lh_model_path, custom_objects={'masked_binary_crossentropy': masked_binary_crossentropy})
                if lh_model.input_shape[-1] == X_test.shape[1]:
                    y_prob_lh = lh_model.predict(X_test, verbose=0).flatten()
                    prob_true_lh, prob_pred_lh = calibration_curve(y_test, y_prob_lh, n_bins=10)
                    brier_lh = brier_score_loss(y_test, y_prob_lh)
                    plt.plot(prob_pred_lh, prob_true_lh, "s-", label=f"Logistic Hazard ({name.replace('.keras', '')}, Brier={brier_lh:.4f})")
                    print(f"Logistic Hazard Brier Score ({name}): {brier_lh:.4f}")
                    metrics[f"Brier_{name.replace('.keras', '')}"] = float(brier_lh)
                    break
            except Exception as e:
                print(f"Fehler bei {name}: {e}")

    # 2. Dynamic DeepHit (Dropout / Event 1)
    for name in ["dynamic_deephit_prev.keras", "dynamic_deephit_cum.keras", "dynamic_deephit_delta.keras", "dynamic_deephit.keras"]:
        dh_model_path = data_dir / "models" / name
        if dh_model_path.exists():
            try:
                dh_model = tf.keras.models.load_model(dh_model_path)
                studis_dh, X_seq_dh, y_seq_dh, _, _, _ = fb.build_semester_sequence_tensor(data_dir, mode='standard', temporal='prev', target_type='competing_risks')
                test_idx = np.where(np.isin(studis_dh, test_ids))[0]
                X_test_dh = X_seq_dh[test_idx]
                y_test_dh = y_seq_dh[test_idx, :, 0].flatten()
                valid_m = (y_test_dh != PADDING_VALUE)
                dh_preds = dh_model.predict(X_test_dh, verbose=0)
                if isinstance(dh_preds, list):
                    y_prob_dh = dh_preds[0].flatten()
                else:
                    y_prob_dh = dh_preds[:, :, 0].flatten()
                y_prob_dh_v = y_prob_dh[valid_m]
                y_true_dh_v = y_test_dh[valid_m]
                prob_true_dh, prob_pred_dh = calibration_curve(y_true_dh_v, y_prob_dh_v, n_bins=10)
                brier_dh = brier_score_loss(y_true_dh_v, y_prob_dh_v)
                plt.plot(prob_pred_dh, prob_true_dh, "o-", label=f"DeepHit ({name.replace('.keras', '')}, Brier={brier_dh:.4f})")
                print(f"DeepHit Brier Score ({name}): {brier_dh:.4f}")
                metrics[f"Brier_{name.replace('.keras', '')}"] = float(brier_dh)
                break
            except Exception as e:
                print(f"DeepHit Kalibrierungskurve übersprungen: {e}")

    plt.xlabel("Vorhergesagte Abbruchwahrscheinlichkeit")
    plt.ylabel("Tatsächlicher Anteil Abbrüche")
    plt.title("Reliability Diagram (Kalibrierungskurve auf Testdaten)")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    
    out_plot = plots_dir / "calibration_curves.png"
    plt.savefig(out_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Kalibrierungsdiagramm erfolgreich gespeichert unter: {out_plot}")
    save_metrics("calibration_analysis", metrics, data_dir)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    args = parser.parse_args()
    main(data_dir=args.data_dir)
