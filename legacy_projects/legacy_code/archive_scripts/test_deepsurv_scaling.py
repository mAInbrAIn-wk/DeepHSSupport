"""
DeepSurv Scaling Benchmark (V4.1)
=================================
Vergleicht systematisch:
  1. DeepSurv mit Large-Batch (32.768, 100 Epochen)
  2. DeepSurv mit Full-Batch (len(X_train) ~250.000, 100 Epochen)
  3. Logistic Hazard (Referenz)
Misst exakte Laufzeiten, RAM-Bedarf, ROC-AUC, PR-AUC und C-Index.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import feature_builder as fb

def breslow_cox_loss(y_true, y_pred):
    time = y_true[:, 0]
    event = y_true[:, 1]
    risk = y_pred[:, 0]

    sort_idx = tf.argsort(time, direction='DESCENDING')
    risk_sorted = tf.gather(risk, sort_idx)
    event_sorted = tf.gather(event, sort_idx)

    cum_exp_risk = tf.cumsum(tf.exp(risk_sorted))
    log_risk = risk_sorted - tf.math.log(cum_exp_risk + 1e-7)

    uncensored_loss = -tf.reduce_sum(log_risk * event_sorted)
    num_events = tf.reduce_sum(event_sorted) + 1e-7
    return uncensored_loss / num_events

def run_deepsurv_test(data_dir: Path):
    print("=" * 80)
    print("   DEEPSURV BATCH- & EPOCHEN-SKALIERUNGSTEST (V4.1)")
    print(f"   Datenverzeichnis: {data_dir.resolve()}")
    print("=" * 80)

    panel_df, feature_cols, target_col, _ = fb.build_semester_panel_df(
        data_dir, mode='standard', temporal='prev'
    )

    cat_cols = [c for c in ['stg_name', 'erstakademiker', 'hzb_typ', 'migrationshintergrund'] if c in feature_cols]
    treat_cols = [c for c in ['fach_supp_count', 'uebf_supp_count', 'psych_supp_count'] if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols and c not in treat_cols]

    unique_studis = np.array(panel_df['studierenden_id'].unique().tolist())
    train_ids, test_ids = train_test_split(unique_studis, test_size=0.20, random_state=42)

    tr_panel = panel_df[panel_df['studierenden_id'].isin(train_ids)].copy()
    te_panel = panel_df[panel_df['studierenden_id'].isin(test_ids)].copy()

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), num_cols),
        ('treat', 'passthrough', treat_cols)
    ])

    X_train = preprocessor.fit_transform(tr_panel[num_cols + treat_cols])
    X_test = preprocessor.transform(te_panel[num_cols + treat_cols])

    y_train_surv = np.column_stack([tr_panel['t_stop'].values, tr_panel[target_col].values])
    y_test_event = te_panel[target_col].values

    n_train = len(X_train)
    print(f"Train-Panel: {n_train} Zeilen | Test-Panel: {len(X_test)} Zeilen | Features: {X_train.shape[1]}")

    experiments = [
        ("DeepSurv (Batch 32k, 100 Epochen)", 32768, 100, 0.002, "deepsurv"),
        ("DeepSurv (Full-Batch, 100 Epochen)", n_train, 100, 0.01, "deepsurv"),
        ("Logistic Hazard (Batch 2048, 40 Epochen)", 2048, 40, 0.001, "loghazard")
    ]

    results = []

    for name, b_size, epochs, lr, m_type in experiments:
        print(f"\n>>> STARTE: {name} (Batch: {b_size}, Epochen: {epochs}, LR: {lr}) ...")
        tf.random.set_seed(42)

        inp = tf.keras.Input(shape=(X_train.shape[1],))
        x = tf.keras.layers.Dense(64, activation='relu')(inp)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)

        if m_type == "deepsurv":
            out = tf.keras.layers.Dense(1, use_bias=False)(x)
            model = tf.keras.Model(inp, out)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=breslow_cox_loss)
            target_fit = y_train_surv
        else:
            out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            model = tf.keras.Model(inp, out)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy')
            target_fit = tr_panel[target_col].values

        t0 = time.time()
        model.fit(X_train, target_fit, epochs=epochs, batch_size=b_size, verbose=0)
        elapsed = time.time() - t0

        preds = model.predict(X_test, batch_size=4096, verbose=0).flatten()
        auc = float(roc_auc_score(y_test_event, preds))
        pr_auc = float(average_precision_score(y_test_event, preds))
        brier = float(brier_score_loss(y_test_event, tf.sigmoid(preds).numpy() if m_type == "deepsurv" else preds))

        print(f"   [ERGEBNIS] Dauer: {elapsed:.2f}s ({elapsed/60:.2f} Min.) | ROC-AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f} | Brier: {brier:.4f}")

        results.append({
            "Setup": name,
            "Batch_Size": b_size,
            "Epochs": epochs,
            "Laufzeit_s": round(elapsed, 2),
            "ROC_AUC": round(auc, 4),
            "PR_AUC": round(pr_auc, 4),
            "Brier_Score": round(brier, 4)
        })

    # Ausgabe Tabelle
    print("\n" + "=" * 80)
    print("   SYNOPTISCHE ERGEBNISTABELLE")
    print("=" * 80)
    print(f"{'Setup':<42} | {'Dauer (s)':<10} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Brier':<8}")
    print("-" * 84)
    for r in results:
        print(f"{r['Setup']:<42} | {r['Laufzeit_s']:<10} | {r['ROC_AUC']:<8} | {r['PR_AUC']:<8} | {r['Brier_Score']:<8}")

    out_file = data_dir / 'metrics' / 'deepsurv_scaling_test_results.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Ergebnisse gespeichert unter: {out_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='output_dl')
    args = parser.parse_args()
    run_deepsurv_test(Path(args.data_dir))
