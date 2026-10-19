"""
Counterfactual Relative Risk Analysis für Dynamic DeepHit Delta
================================================================
Berechnet das Relative Risiko (RR) für jeden der drei semester-lokalen Support-Typen
auf dem neu trainierten Dynamic DeepHit Delta Competing Risks Modell.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dynamic_deephit_delta_model import build_competing_risks_dataset_delta
from recurrent_survival_model import masked_binary_crossentropy, PADDING_VALUE
from metrics_logger import save_metrics

def main(data_dir=None):
    print("\n==========================================================================")
    print("   COUNTERFACTUAL RELATIVE RISK ANALYSIS (DYNAMIC DEEPHIT DELTA)")
    print("==========================================================================")
    
    if data_dir is not None:
        data_dir = Path(data_dir)
    elif os.environ.get('DATA_DIR'):
        data_dir = Path(os.environ['DATA_DIR'])
    else:
        data_dir = Path('output_dl') if (Path('output_dl/models/dynamic_deephit_delta.keras').exists() or Path('output_dl/agg_abschluesse.csv').exists()) else Path('../output_dl')
    
    model_path = data_dir / 'models' / 'dynamic_deephit_delta.keras'
    if not model_path.exists():
        model_path = data_dir / 'dynamic_deephit_delta.keras'
    if not model_path.exists():
        model_path = Path('../output_dl/models/dynamic_deephit_delta.keras')
    if not model_path.exists():
        model_path = Path('output_dl/dynamic_deephit_delta.keras')
        
    if not model_path.exists():
        print(f"Modell nicht gefunden: {model_path}")
        return
        
    print("Lade Datensatz & DeepHit Delta Modell...")
    studis, X_seq, y_dropout, y_grad, studi_events = build_competing_risks_dataset_delta(data_dir, max_semesters=16)
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
    
    metrics_all = {}
    
    for feature_idx, prefix, supp_name in [(4, 'fach', 'Fachlicher Support'), (5, 'uebf', 'Überfachlicher Support'), (6, 'psych', 'Psychosozialer Support')]:
        # 1. PARTIELL (≙ A vs C/D/E): Ziel-Support 0 vs. beobachtet, andere beobachtet
        X_c_part = X_test.copy()
        X_t_part = X_test.copy()
        X_c_part[valid_mask_test, feature_idx] = 0.0
        
        X_c_part[valid_mask_test] = scaler.transform(X_c_part[valid_mask_test])
        X_t_part[valid_mask_test] = scaler.transform(X_t_part[valid_mask_test])
        
        p0_p = model.predict(X_c_part, verbose=0)[0].flatten()[valid_mask_test.flatten()]
        p1_p = model.predict(X_t_part, verbose=0)[0].flatten()[valid_mask_test.flatten()]
        rrs_p = p1_p / np.clip(p0_p, 1e-7, 1.0)
        
        mean_rr_p   = float(np.mean(rrs_p))
        median_rr_p = float(np.median(rrs_p))
        q05_p       = float(np.quantile(rrs_p, 0.05))
        q95_p       = float(np.quantile(rrs_p, 0.95))
        
        # 2. ISOLIERT REALISTISCH (≙ B vs F/G/H): Alle 0 vs. nur Ziel beobachtet, andere 0
        X_c_iso = X_test.copy()
        X_t_iso = X_test.copy()
        X_c_iso[valid_mask_test, 4] = 0.0
        X_c_iso[valid_mask_test, 5] = 0.0
        X_c_iso[valid_mask_test, 6] = 0.0
        
        X_t_iso[valid_mask_test, 4] = 0.0
        X_t_iso[valid_mask_test, 5] = 0.0
        X_t_iso[valid_mask_test, 6] = 0.0
        X_t_iso[valid_mask_test, feature_idx] = X_test[valid_mask_test, feature_idx] # beobachtete Dosis
        
        X_c_iso[valid_mask_test] = scaler.transform(X_c_iso[valid_mask_test])
        X_t_iso[valid_mask_test] = scaler.transform(X_t_iso[valid_mask_test])
        
        p0_i = model.predict(X_c_iso, verbose=0)[0].flatten()[valid_mask_test.flatten()]
        p1_i = model.predict(X_t_iso, verbose=0)[0].flatten()[valid_mask_test.flatten()]
        rrs_i = p1_i / np.clip(p0_i, 1e-7, 1.0)
        
        mean_rr_i   = float(np.mean(rrs_i))
        median_rr_i = float(np.median(rrs_i))
        q05_i       = float(np.quantile(rrs_i, 0.05))
        q95_i       = float(np.quantile(rrs_i, 0.95))
        
        print(f"\n--- Support-Typ: {supp_name} ({prefix}) ---")
        print(f"  PARTIELL:           Mean RR = {mean_rr_p:.4f}, Median RR = {median_rr_p:.4f} [{q05_p:.4f}, {q95_p:.4f}]")
        print(f"  ISOLIERT (realist): Mean RR = {mean_rr_i:.4f}, Median RR = {median_rr_i:.4f} [{q05_i:.4f}, {q95_i:.4f}]")
        
        metrics_all[f"{prefix}_partial"] = {"mean_rr": mean_rr_p, "median_rr": median_rr_p, "q05": q05_p, "q95": q95_p}
        metrics_all[f"{prefix}_isolated"] = {"mean_rr": mean_rr_i, "median_rr": median_rr_i, "q05": q05_i, "q95": q95_i}
        
        # Abwärtskompatible Keys
        metrics_all[f"Mean_RR_{prefix}"]   = mean_rr_p
        metrics_all[f"Median_RR_{prefix}"] = median_rr_p
        metrics_all[f"Q05_RR_{prefix}"]    = q05_p
        metrics_all[f"Q95_RR_{prefix}"]    = q95_p

    save_metrics("counterfactual_rr_deephit_delta", metrics_all, data_dir)
    print("\nCounterfactual Analysis für DeepHit Delta abgeschlossen.")

if __name__ == '__main__':
    main()
