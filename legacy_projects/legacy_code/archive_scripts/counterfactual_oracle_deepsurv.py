"""
Counterfactual Hazard Ratio Analysis für Oracle DeepSurv
=========================================================
Berechnet empirische kontrafaktische Hazard Ratios für die semester-lokalen
Support-Interventionen (fachlich, überfachlich, psychosozial) basierend auf
dem Oracle DeepSurv Modell mit latenten Variablen.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import tensorflow as tf

from extended_cox_delta import build_delta_panel
from extended_deep_survival_delta import breslow_cox_loss
from metrics_logger import save_metrics

def analyze_counterfactual_oracle_deepsurv(data_dir: Path):
    print("\n==========================================================================")
    print("   COUNTERFACTUAL HAZARD RATIO ANALYSIS (ORACLE DEEPSURV)")
    print("==========================================================================")
    
    panel_df = build_delta_panel(data_dir)
    
    num_cols_base = ['hzb_note', 'erwerbstaetigkeit_std', 't_stop', 't_start', 'fails_prev', 'delta_cp_prev', 'cp_rueckstand']
    cat_cols = ['stg_name', 'erstakademiker']
    treatment_cols = ['fach_supp_count', 'uebf_supp_count', 'psych_supp_count']
    oracle_cols = ['hidden_motivation_prev', 'hidden_soziale_integration_prev', 'hidden_erwartete_note_prev']
    
    feature_cols = num_cols_base + oracle_cols + cat_cols + treatment_cols
    
    unique_studis = np.array(panel_df['studierenden_id'].unique().tolist())
    train_ids, test_ids = train_test_split(unique_studis, test_size=0.20, random_state=42)
    
    train_panel = panel_df[panel_df['studierenden_id'].isin(train_ids)].copy()
    test_panel  = panel_df[panel_df['studierenden_id'].isin(test_ids)].copy()
    
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols_base + oracle_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))]), cat_cols),
        ('treatments', 'passthrough', treatment_cols)
    ])
    preprocessor.fit(train_panel[feature_cols])
    
    model_path = data_dir / "models" / "oracle_deepsurv.keras"
    if not model_path.exists():
        print(f"Modell {model_path} nicht gefunden! Bitte zuerst train_oracle_models.py ausführen.")
        return
        
    model = tf.keras.models.load_model(model_path, custom_objects={'breslow_cox_loss': breslow_cox_loss})
    print(f"Orakel-Modell geladen: {model_path.name}  |  Input-Dim: {model.input_shape}")

    metrics_all = {}

    for supp_col, prefix, label in [
        ('fach_supp_count',  'fach',  'Fachlicher Support (Oracle DeepSurv)'),
        ('uebf_supp_count',  'uebf',  'Überfachlicher Support (Oracle DeepSurv)'),
        ('psych_supp_count', 'psych', 'Psychosozialer Support (Oracle DeepSurv)'),
    ]:
        # 1. PARTIELL (≙ A vs C/D/E): Ziel-Support 0 vs. beobachtet, andere beobachtet
        control_part = test_panel.copy()
        treated_part = test_panel.copy()
        control_part[supp_col] = 0
        
        X_c_p = preprocessor.transform(control_part[feature_cols])
        X_t_p = preprocessor.transform(treated_part[feature_cols])
        h0_p = model.predict(X_c_p, verbose=0).flatten()
        h1_p = model.predict(X_t_p, verbose=0).flatten()
        hrs_p = np.exp(h1_p - h0_p)
        
        mean_hr_p   = float(np.mean(hrs_p))
        median_hr_p = float(np.median(hrs_p))
        q05_p       = float(np.quantile(hrs_p, 0.05))
        q95_p       = float(np.quantile(hrs_p, 0.95))
        
        # 2. ISOLIERT REALISTISCH (≙ B vs F/G/H): Alle 0 vs. nur Ziel beobachtet, andere 0
        control_iso = test_panel.copy()
        treated_iso = test_panel.copy()
        for c in treatment_cols:
            control_iso[c] = 0
            treated_iso[c] = 0
        treated_iso[supp_col] = test_panel[supp_col] # beobachtete Dosis
        
        X_c_i = preprocessor.transform(control_iso[feature_cols])
        X_t_i = preprocessor.transform(treated_iso[feature_cols])
        h0_i = model.predict(X_c_i, verbose=0).flatten()
        h1_i = model.predict(X_t_i, verbose=0).flatten()
        hrs_i = np.exp(h1_i - h0_i)
        
        mean_hr_i   = float(np.mean(hrs_i))
        median_hr_i = float(np.median(hrs_i))
        q05_i       = float(np.quantile(hrs_i, 0.05))
        q95_i       = float(np.quantile(hrs_i, 0.95))

        print(f"\n--- {label} ({supp_col}) ---")
        print(f"  PARTIELL:           Mean HR = {mean_hr_p:.4f}, Median HR = {median_hr_p:.4f} [{q05_p:.4f}, {q95_p:.4f}]")
        print(f"  ISOLIERT (realist): Mean HR = {mean_hr_i:.4f}, Median HR = {median_hr_i:.4f} [{q05_i:.4f}, {q95_i:.4f}]")

        metrics_all[f"{prefix}_partial"] = {"mean_hr": mean_hr_p, "median_hr": median_hr_p, "q05": q05_p, "q95": q95_p}
        metrics_all[f"{prefix}_isolated"] = {"mean_hr": mean_hr_i, "median_hr": median_hr_i, "q05": q05_i, "q95": q95_i}
        
        metrics_all[f"Mean_HR_{prefix}"]   = mean_hr_p
        metrics_all[f"Median_HR_{prefix}"] = median_hr_p
        metrics_all[f"Q05_HR_{prefix}"]    = q05_p
        metrics_all[f"Q95_HR_{prefix}"]    = q95_p

    print("\n" + "=" * 74)
    save_metrics("counterfactual_oracle_deepsurv_metrics", metrics_all, data_dir)
    print("Counterfactual Hazard Ratio Analysis für Oracle DeepSurv abgeschlossen.")

if __name__ == '__main__':
    possible_dirs = [Path("src/output_dl"), Path("output_dl"), Path("../src/output_dl"), Path("../output_dl")]
    target_dir = None
    for p in possible_dirs:
        if (p / "models" / "oracle_deepsurv.keras").exists():
            target_dir = p
            break
    if target_dir is None:
        target_dir = Path("output_dl")
    analyze_counterfactual_oracle_deepsurv(target_dir)
