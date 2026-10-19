import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

from deep_survival import load_raw_data, build_preprocessor, breslow_cox_partial_loss, compute_bootstrap_hazard_ratios

def main():
    data_dir = Path('../output_dl') if Path('../output_dl').exists() else Path('output_dl')
    model_path = data_dir / 'models' / 'deepsurv_landmark.keras'
    
    if not model_path.exists():
        print(f"Modell nicht gefunden: {model_path}")
        return
        
    print("Lade Daten und Preprocessor (wie im Training)...")
    data_file = data_dir / 'agg_abschluesse.csv'
    if not data_file.exists():
        data_file = Path('../output_dl/agg_abschluesse.csv')
    df_raw, feature_cols = load_raw_data(data_file)
    
    # 1. Group-consistent 3-Wege-Split (70% Train, 15% Val, 15% Test) on student IDs
    unique_students = df_raw['studierenden_id'].unique()
    train_students, temp_students = train_test_split(unique_students, test_size=0.30, random_state=42)
    val_students, test_students = train_test_split(temp_students, test_size=0.50, random_state=42)
    
    df_train = df_raw[df_raw['studierenden_id'].isin(train_students)]
    df_val = df_raw[df_raw['studierenden_id'].isin(val_students)]
    df_test = df_raw[df_raw['studierenden_id'].isin(test_students)]
    
    X_train_df = df_train[feature_cols]
    
    # Fit preprocessor strictly on Training Data
    preprocessor = build_preprocessor(X_train_df)
    preprocessor.fit(X_train_df)
    
    print("Lade DeepSurv Modell...")
    custom_objects = {'breslow_cox_partial_loss': breslow_cox_partial_loss}
    model = load_model(model_path, custom_objects=custom_objects)
    
    print("\n" + "=" * 65)
    print("COUNTERFACTUAL SIMULATION: EXTENDED DEEPSURV")
    print("=" * 65)
    
    # Führe die Funktion für die HR-Berechnung aus dem bestehenden Code aus
    ci_results = compute_bootstrap_hazard_ratios(model, df_test, preprocessor, n_boot=100)
    
    for supp_name, label in [('Fach_supp', 'Fachlicher Support (Modulbezogen)'),
                              ('Uebf_supp', 'Überfachlicher Support (Coaching)  '),
                              ('Psych_supp', 'Psychosozialer Support (Beratung)  ')]:
        mean_hr, lower_ci, upper_ci = ci_results[supp_name]
        if mean_hr < 1.0:
            risk_red = (1.0 - mean_hr) * 100.0
            effekt = f"Senkt Risiko um ca. {risk_red:.1f}%"
        else:
            risk_inc = (mean_hr - 1.0) * 100.0
            effekt = f"Erhöht Risiko um ca. {risk_inc:.1f}%"
            
        print(f"  • {label}: Pseudo-HR = {mean_hr:.4f} [95%-KI: {lower_ci:.4f} – {upper_ci:.4f}]")
        print(f"    -> {effekt}")
        
    print("=" * 65)

if __name__ == '__main__':
    main()
