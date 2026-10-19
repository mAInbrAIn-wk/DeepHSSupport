"""
Counterfactual Relative Risk Analysis für Recurrent GRU v2 (Exam Level)
========================================================================
Berechnet das Relative Risiko (RR) für jeden der drei Support-Typen separat
auf dem bestehenden Prüfungs-Sequenzmodell (recurrent_exam_survival_v2.keras).
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from recurrent_exam_survival_v2 import build_recurrent_exam_dataset_v2, masked_binary_crossentropy, PADDING_VALUE
from metrics_logger import save_metrics

def main():
    print("\n==========================================================================")
    print("   COUNTERFACTUAL INFERENCE: RECURRENT GRU V2 (EXAM LEVEL - DUAL STRAND)")
    print("==========================================================================")
    
    possible_dirs = [Path("src/output_dl"), Path("output_dl"), Path("../src/output_dl"), Path("../output_dl")]
    data_dir = None
    for p in possible_dirs:
        if (p / "models" / "recurrent_exam_survival_v2.keras").exists() or (p / "agg_abschluesse.csv").exists():
            data_dir = p
            break
    if data_dir is None:
        data_dir = Path("output_dl")
        
    model_path = data_dir / 'models' / 'recurrent_exam_survival_v2.keras'
    if not model_path.exists():
        model_path = data_dir / 'recurrent_exam_survival_v2.keras'
        
    if not model_path.exists():
        print(f"Modell nicht gefunden in: {data_dir}")
        return
        
    print("Lade Datensatz & Modell...")
    studis, X_seq, y_seq, studi_events = build_recurrent_exam_dataset_v2(data_dir)
    N, K_max, F = X_seq.shape
    
    train_idx, temp_idx, _, y_temp_event = train_test_split(
        np.arange(N), studi_events, test_size=0.30, random_state=42, stratify=studi_events
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp_event, test_size=0.50, random_state=42, stratify=y_temp_event
    )
    
    X_train = X_seq[train_idx].copy()
    X_test = X_seq[test_idx].copy()
    
    scaler = StandardScaler()
    valid_mask_train = (X_train[:, :, 0] != PADDING_VALUE)
    scaler.fit(X_train[valid_mask_train])
    
    model = tf.keras.models.load_model(model_path, custom_objects={'masked_binary_crossentropy': masked_binary_crossentropy})
    valid_mask_test = (X_test[:, :, 0] != PADDING_VALUE)
    
    # Letzter Zeitschritt-Index pro Test-Student
    last_step_indices = []
    for i in range(len(test_idx)):
        steps = np.where(X_test[i, :, 0] != PADDING_VALUE)[0]
        last_step_indices.append(steps[-1] if len(steps) > 0 else -1)
    
    ALL_SUPP_IDXS = [3, 4, 5, 6, 7, 8]
    
    metrics_all = {}
    
    for (idx_vor, idx_glz), prefix, label in [
        ((3, 4), 'fach',  'Fachlicher Support'),
        ((5, 6), 'uebf',  'Überfachlicher Support'),
        ((7, 8), 'psych', 'Psychosozialer Support'),
    ]:
        # 1. PARTIELL (≙ A vs C/D/E): Ziel-Support 0 vs. beobachtet, andere beobachtet
        X_c_p = X_test.copy()
        X_t_p = X_test.copy()
        X_c_p[valid_mask_test, idx_vor] = 0.0
        X_c_p[valid_mask_test, idx_glz] = 0.0
        
        X_c_p[valid_mask_test] = scaler.transform(X_c_p[valid_mask_test])
        X_t_p[valid_mask_test] = scaler.transform(X_t_p[valid_mask_test])
        
        preds_c_p = model.predict(X_c_p, verbose=0)
        preds_t_p = model.predict(X_t_p, verbose=0)
        
        r_c_p = np.array([preds_c_p[i, ls, 0] if ls >= 0 else 0.0 for i, ls in enumerate(last_step_indices)])
        r_t_p = np.array([preds_t_p[i, ls, 0] if ls >= 0 else 0.0 for i, ls in enumerate(last_step_indices)])
        rrs_p = r_t_p / np.clip(r_c_p, 1e-7, 1.0)
        
        mean_rr_p   = float(np.mean(rrs_p))
        median_rr_p = float(np.median(rrs_p))
        q05_p       = float(np.quantile(rrs_p, 0.05))
        q95_p       = float(np.quantile(rrs_p, 0.95))
        
        # 2. ISOLIERT REALISTISCH (≙ B vs F/G/H): Alle 0 vs. nur Ziel beobachtet, andere 0
        X_c_i = X_test.copy()
        X_t_i = X_test.copy()
        for idx in ALL_SUPP_IDXS:
            X_c_i[valid_mask_test, idx] = 0.0
            X_t_i[valid_mask_test, idx] = 0.0
        X_t_i[valid_mask_test, idx_vor] = X_test[valid_mask_test, idx_vor]
        X_t_i[valid_mask_test, idx_glz] = X_test[valid_mask_test, idx_glz]
        
        X_c_i[valid_mask_test] = scaler.transform(X_c_i[valid_mask_test])
        X_t_i[valid_mask_test] = scaler.transform(X_t_i[valid_mask_test])
        
        preds_c_i = model.predict(X_c_i, verbose=0)
        preds_t_i = model.predict(X_t_i, verbose=0)
        
        r_c_i = np.array([preds_c_i[i, ls, 0] if ls >= 0 else 0.0 for i, ls in enumerate(last_step_indices)])
        r_t_i = np.array([preds_t_i[i, ls, 0] if ls >= 0 else 0.0 for i, ls in enumerate(last_step_indices)])
        rrs_i = r_t_i / np.clip(r_c_i, 1e-7, 1.0)
        
        mean_rr_i   = float(np.mean(rrs_i))
        median_rr_i = float(np.median(rrs_i))
        q05_i       = float(np.quantile(rrs_i, 0.05))
        q95_i       = float(np.quantile(rrs_i, 0.95))
        
        print(f"\n--- {label} ({prefix}) ---")
        print(f"  PARTIELL:           Mean RR = {mean_rr_p:.4f}, Median RR = {median_rr_p:.4f} [{q05_p:.4f}, {q95_p:.4f}]")
        print(f"  ISOLIERT (realist): Mean RR = {mean_rr_i:.4f}, Median RR = {median_rr_i:.4f} [{q05_i:.4f}, {q95_i:.4f}]")
        
        metrics_all[f"{prefix}_partial"] = {"mean_rr": mean_rr_p, "median_rr": median_rr_p, "q05": q05_p, "q95": q95_p}
        metrics_all[f"{prefix}_isolated"] = {"mean_rr": mean_rr_i, "median_rr": median_rr_i, "q05": q05_i, "q95": q95_i}
        
        # Abwärtskompatible Keys
        metrics_all[f"Mean_RR_{prefix}"]   = mean_rr_p
        metrics_all[f"Median_RR_{prefix}"] = median_rr_p
        metrics_all[f"Q05_RR_{prefix}"]    = q05_p
        metrics_all[f"Q95_RR_{prefix}"]    = q95_p

    print("\n" + "=" * 74)
    save_metrics("counterfactual_rnn_delta", metrics_all, data_dir)
    print("Counterfactual Analysis für Recurrent GRU v2 abgeschlossen.")

if __name__ == '__main__':
    main()
