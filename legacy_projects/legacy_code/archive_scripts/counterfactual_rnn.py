import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from recurrent_exam_survival_v2 import build_recurrent_exam_dataset_v2, masked_binary_crossentropy, PADDING_VALUE

def main():
    data_dir = Path('../output_dl') if Path('../output_dl').exists() else Path('output_dl')
    model_path = data_dir / 'recurrent_exam_survival_v2.keras'
    if not model_path.exists():
        model_path = data_dir / 'models' / 'recurrent_exam_survival_v2.keras'
    
    if not model_path.exists():
        print(f"Modell nicht gefunden: {model_path}")
        return
        
    print("Lade Datensatz...")
    studis, X_seq, y_seq, studi_events = build_recurrent_exam_dataset_v2(data_dir)
    N, K_max, F = X_seq.shape
    
    # Re-create Train/Test Split to fit scaler exactly as in training
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
    
    # Lade Modell
    print("Lade GRU-Modell...")
    custom_objects = {'masked_binary_crossentropy': masked_binary_crossentropy}
    model = load_model(model_path, custom_objects=custom_objects)
    
    print("\nFühre Kontrafaktische Simulation auf Testdaten aus (N={})...".format(len(test_idx)))
    
    # X_test_base (ohne Support)
    X_ohne = X_test.copy()
    valid_mask_test = (X_test[:, :, 0] != PADDING_VALUE)
    
    # Setze Support auf 0
    X_ohne[valid_mask_test, 3] = 0.0 # fach
    X_ohne[valid_mask_test, 4] = 0.0 # uebf
    X_ohne[valid_mask_test, 5] = 0.0 # psych
    
    # X_test_mit (Dauerhafter Support ab der ersten Prüfung)
    X_mit = X_test.copy()
    X_mit[valid_mask_test, 3] = 1.0
    X_mit[valid_mask_test, 4] = 1.0
    X_mit[valid_mask_test, 5] = 1.0
    
    # Skalieren
    X_ohne[valid_mask_test] = scaler.transform(X_ohne[valid_mask_test])
    X_mit[valid_mask_test] = scaler.transform(X_mit[valid_mask_test])
    
    # Vorhersagen (Risk = Wahrscheinlichkeit Abbruch in jedem Schritt)
    y_pred_ohne = model.predict(X_ohne, verbose=0)
    y_pred_mit = model.predict(X_mit, verbose=0)
    
    # Extrahieren der Prognose für die LETZTE beobachtete Prüfung jedes Studenten
    risk_ohne_list = []
    risk_mit_list = []
    
    for i in range(len(test_idx)):
        # Finde den letzten gültigen Zeitschritt
        valid_steps = np.where(X_test[i, :, 0] != PADDING_VALUE)[0]
        if len(valid_steps) > 0:
            last_step = valid_steps[-1]
            risk_ohne_list.append(y_pred_ohne[i, last_step, 0])
            risk_mit_list.append(y_pred_mit[i, last_step, 0])
            
    risk_ohne = np.array(risk_ohne_list)
    risk_mit = np.array(risk_mit_list)
    
    # Relative Risk (analog zu HR in diskreter Zeit: P_mit / P_ohne)
    # Vermeide Division durch Null
    relative_risk = np.mean(risk_mit / (risk_ohne + 1e-7))
    mean_risk_ohne = np.mean(risk_ohne)
    mean_risk_mit = np.mean(risk_mit)
    
    print("==========================================================================")
    print("   KONTRAFAKTISCHE ANALYSE: RECURRENT GRU (Panel-Daten)")
    print("==========================================================================")
    print(f"  Durchschnittliches Abbruchrisiko (Ohne Support) : {mean_risk_ohne:.4f}")
    print(f"  Durchschnittliches Abbruchrisiko (Dauer-Support): {mean_risk_mit:.4f}")
    print(f"  Relative Risk (Pseudo-HR)                       : {relative_risk:.4f}")
    print("==========================================================================")
    if relative_risk > 1.0:
        print("  -> Das Modell schätzt, dass Dauer-Support das Risiko ERHÖHT (RR > 1)!")
        print("     Grund: Schlechte Extrapolation. Das Netz hat 'Dauer-Support' bei")
        print("     vielen Fehlversuchen in den Trainingsdaten fast nie gesehen und")
        print("     assoziiert Support immer noch fälschlicherweise als Krisensignal.")
    else:
        print("  -> Das Modell schätzt, dass Dauer-Support das Risiko SENKT.")

if __name__ == '__main__':
    main()
