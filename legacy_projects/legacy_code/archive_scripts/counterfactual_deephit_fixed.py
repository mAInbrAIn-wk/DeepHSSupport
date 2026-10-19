"""
Counterfactual Relative Risk Analysis für Dynamic DeepHit (Semester Level - Existing Model)
=============================================================================================
Berechnet das Relative Risiko (RR) für jeden der drei Support-Typen separat
auf dem bestehenden DeepHit Modell (dynamic_deephit_competing.keras).
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import feature_builder as fb
from recurrent_survival_model import masked_binary_crossentropy, PADDING_VALUE
from metrics_logger import save_metrics

def main(data_dir=None):
    print("\n==========================================================================")
    print("   COUNTERFACTUAL INFERENCE: DYNAMIC DEEPHIT (EXISTING MODEL)")
    print("==========================================================================")
    
    if data_dir is None:
        data_dir = Path('../output_dl') if Path('../output_dl').exists() else Path('output_dl')
    else:
        data_dir = Path(data_dir)
        
    possible_names = [
        'dynamic_deephit_prev.keras',
        'dynamic_deephit_competing.keras',
        'dynamic_deephit_model.keras',
        'dynamic_deephit_delta.keras'
    ]
    model_path = None
    for name in possible_names:
        cand = data_dir / 'models' / name
        if cand.exists():
            model_path = cand
            break
            
    if model_path is None:
        print(f"Modell nicht gefunden in {data_dir / 'models'}")
        return
        
    print("Lade Datensatz & DeepHit Modell...")
    studis, X_seq, y_seq, studi_events, feature_names, _ = fb.build_semester_sequence_tensor(
        data_dir, max_semesters=16, mode='standard', temporal='prev', target_type='competing_risks'
    )
    y_dropout = y_seq[:, :, 0:1]
    y_grad = y_seq[:, :, 1:2]
    N, K_max, F = X_seq.shape
    
    train_idx, temp_idx = train_test_split(
        np.arange(N), test_size=0.30, random_state=42, stratify=(studi_events == 1).astype(int)
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=42, stratify=(studi_events[temp_idx] == 1).astype(int)
    )
    
    X_train = X_seq[train_idx].copy()
    X_test = X_seq[test_idx].copy()
    
    scaler = StandardScaler()
    valid_mask_train = (X_train[:, :, 0] != PADDING_VALUE)
    scaler.fit(X_train[valid_mask_train])
    
    model = tf.keras.models.load_model(model_path, custom_objects={'masked_binary_crossentropy': masked_binary_crossentropy})
    valid_mask_test = (X_test[:, :, 0] != PADDING_VALUE)
    
    # Control: Support = 0 für alle 3 Typen
    X_control = X_test.copy()
    X_control[valid_mask_test, 3] = 0.0 # fach
    X_control[valid_mask_test, 4] = 0.0 # uebf
    X_control[valid_mask_test, 5] = 0.0 # psych
    X_control[valid_mask_test] = scaler.transform(X_control[valid_mask_test])
    
    # DeepHit outputs [y_dropout, y_grad]
    preds_control_dropout = model.predict(X_control, verbose=0)[0].flatten()
    preds_control_dropout = np.clip(preds_control_dropout[valid_mask_test.flatten()], 1e-7, 1.0)
    
    metrics_all = {}
    
    for feature_idx, supp_name in [(3, 'fach'), (4, 'uebf'), (5, 'psych')]:
        X_treated = X_test.copy()
        X_treated[valid_mask_test, 3] = 0.0
        X_treated[valid_mask_test, 4] = 0.0
        X_treated[valid_mask_test, 5] = 0.0
        X_treated[valid_mask_test, feature_idx] = 1.0
        X_treated[valid_mask_test] = scaler.transform(X_treated[valid_mask_test])
        
        preds_treated_dropout = model.predict(X_treated, verbose=0)[0].flatten()
        preds_treated_dropout = preds_treated_dropout[valid_mask_test.flatten()]
        
        rrs = preds_treated_dropout / preds_control_dropout
        
        mean_rr = float(np.mean(rrs))
        median_rr = float(np.median(rrs))
        q05 = float(np.quantile(rrs, 0.05))
        q95 = float(np.quantile(rrs, 0.95))
        
        print(f"\n--- Support-Typ: {supp_name.upper()} ---")
        print(f"  Mean Relative Risk (RR)  : {mean_rr:.4f}")
        print(f"  Median Relative Risk (RR): {median_rr:.4f}")
        print(f"  5%-95% KI                : [{q05:.4f}, {q95:.4f}]")
        
        metrics_all[f"Mean_RR_{supp_name}"] = mean_rr
        metrics_all[f"Median_RR_{supp_name}"] = median_rr
        metrics_all[f"Q05_RR_{supp_name}"] = q05
        metrics_all[f"Q95_RR_{supp_name}"] = q95

    save_metrics("counterfactual_deephit_fixed", metrics_all, data_dir)
    print("\nCounterfactual Analysis für DeepHit (Existing) abgeschlossen.")

if __name__ == '__main__':
    main()
