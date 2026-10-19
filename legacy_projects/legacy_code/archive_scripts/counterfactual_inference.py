"""
Counterfactual Inference Wrapper für Deep Survival Modelle
========================================================
Lädt ein trainiertes Sequenzmodell (z.B. recurrent_exam_survival.keras),
erzeugt künstlich behandelte (Support=1) und unbehandelte (Support=0) 
Kopien der Testdaten und berechnet die kausale Hazard Ratio (HR).
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import der Dataset-Bau-Funktion und Loss
from recurrent_exam_survival import build_recurrent_exam_dataset, masked_binary_crossentropy, PADDING_VALUE

def run_counterfactual_inference():
    print("Starte Counterfactual Inference...")
    
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / 'output_dl'
    models_dir = data_dir / 'models'
    
    model_path = models_dir / 'recurrent_exam_survival.keras'
    if not model_path.exists():
        print(f"Modell nicht gefunden: {model_path}")
        return
        
    print(f"Lade Keras Modell: {model_path.name}")
    # custom_objects nötig wegen der masked_binary_crossentropy
    model = tf.keras.models.load_model(model_path, custom_objects={'masked_binary_crossentropy': masked_binary_crossentropy})
    
    # 1. Daten laden und identischen Split rekonstruieren
    studis, X_seq, y_seq, studi_events = build_recurrent_exam_dataset(data_dir)
    N, K_max, F = X_seq.shape
    
    train_idx, temp_idx, _, y_temp_event = train_test_split(
        np.arange(N), studi_events, test_size=0.30, random_state=42, stratify=studi_events
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp_event, test_size=0.50, random_state=42, stratify=y_temp_event
    )
    
    X_train = X_seq[train_idx].copy()
    X_test = X_seq[test_idx].copy()
    y_test = y_seq[test_idx]
    
    # Feature 3 = fach_supp_cum
    FACH_SUPP_IDX = 3
    
    # 2. Counterfactual Datensätze erstellen (VOR Skalierung)
    X_test_control = X_test.copy()
    X_test_treated = X_test.copy()
    
    # Wir setzen fach_supp_cum (Feature 3) auf 0 (Control) bzw. 1 (Treated) für alle GÜLTIGEN Schritte
    valid_mask_test = (X_test[:, :, 0] != PADDING_VALUE)
    
    # ACHTUNG: Nur gültige Zeitschritte überschreiben, PADDING_VALUE muss PADDING_VALUE bleiben!
    X_test_control[valid_mask_test, FACH_SUPP_IDX] = 0.0
    X_test_treated[valid_mask_test, FACH_SUPP_IDX] = 1.0
    
    # 3. Scaler fitten und transformieren
    scaler = StandardScaler()
    valid_mask_train = (X_train[:, :, 0] != PADDING_VALUE)
    scaler.fit(X_train[valid_mask_train])
    
    # Standard-Testdaten skalieren
    X_test_scaled = X_test.copy()
    X_test_scaled[valid_mask_test] = scaler.transform(X_test[valid_mask_test])
    
    # Counterfactuals skalieren
    X_test_control[valid_mask_test] = scaler.transform(X_test_control[valid_mask_test])
    X_test_treated[valid_mask_test] = scaler.transform(X_test_treated[valid_mask_test])
    
    # 4. Inferenz durchführen
    print("Führe Modellvorhersagen durch (Inferenz)...")
    preds_baseline = model.predict(X_test_scaled, verbose=0)
    preds_control = model.predict(X_test_control, verbose=0)
    preds_treated = model.predict(X_test_treated, verbose=0)
    
    # 5. Counterfactual Hazard Ratios berechnen
    # Wir filtern die Padded-Werte heraus
    preds_c_flat = preds_control.flatten()[valid_mask_test.flatten()]
    preds_t_flat = preds_treated.flatten()[valid_mask_test.flatten()]
    
    # Vermeide Division durch 0
    preds_c_flat = np.clip(preds_c_flat, 1e-7, 1.0)
    
    # Individuelle Hazard Ratios
    individual_hrs = preds_t_flat / preds_c_flat
    
    mean_hr = np.mean(individual_hrs)
    median_hr = np.median(individual_hrs)
    
    print("\n==========================================================================")
    print("   COUNTERFACTUAL INFERENCE (KERAS GRU - EXAM LEVEL)")
    print("==========================================================================")
    print(f"  Ø Vorhergesagtes Dropout-Risiko (Ohne Support)  : {np.mean(preds_c_flat):.4f}")
    print(f"  Ø Vorhergesagtes Dropout-Risiko (Mit Support)   : {np.mean(preds_t_flat):.4f}")
    print(f"  Kausale Hazard Ratio (Mean HR)                  : {mean_hr:.4f}")
    print(f"  Kausale Hazard Ratio (Median HR)                : {median_hr:.4f}")
    print("==========================================================================")
    print("Interpretation:")
    if mean_hr < 1.0:
        print(f"  -> Support SENKT das Dropout-Risiko relativ um {(1 - mean_hr)*100:.1f}%.")
    else:
        print(f"  -> Support ERHÖHT das Dropout-Risiko relativ um {(mean_hr - 1)*100:.1f}%.")
    
    # JSON export
    metrics = {
        "Mean_Control_Risk": float(np.mean(preds_c_flat)),
        "Mean_Treated_Risk": float(np.mean(preds_t_flat)),
        "Mean_HR": float(mean_hr),
        "Median_HR": float(median_hr),
        "Support_Effect_Percent": float((1 - mean_hr) * 100) if mean_hr < 1.0 else float((mean_hr - 1) * 100)
    }
    
    out_file = data_dir / 'metrics' / 'counterfactual_gru_metrics.json'
    with open(out_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nCounterfactual Metriken gespeichert in {out_file}")

if __name__ == "__main__":
    run_counterfactual_inference()
