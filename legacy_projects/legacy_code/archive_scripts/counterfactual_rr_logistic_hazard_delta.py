"""
Counterfactual Relative Risk Analysis für Extended Logistic Hazard Delta
==========================================================================
Berechnet das Relative Risiko (RR) für jeden der drei semester-lokalen Support-Typen
auf dem leistungsstarken Extended Logistic Hazard Delta Modell (ROC-AUC = 0.7992).
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from extended_cox_delta import build_delta_panel
from metrics_logger import save_metrics

def analyze_counterfactual_rr_logistic_hazard_delta(data_dir: Path):
    print("\n==========================================================================")
    print("   COUNTERFACTUAL RELATIVE RISK ANALYSIS (EXTENDED LOGISTIC HAZARD DELTA)")
    print("==========================================================================")
    
    panel_df = build_delta_panel(data_dir)
    
    num_cols = ['hzb_note', 'erwerbstaetigkeit_std', 't_stop', 't_start', 'fails_prev', 'delta_cp_prev', 'cp_rueckstand']
    cat_cols = ['stg_name', 'erstakademiker']
    treatment_cols = ['fach_supp_count', 'uebf_supp_count', 'psych_supp_count']
    
    feature_cols = num_cols + cat_cols + treatment_cols
    
    unique_studis = np.array(panel_df['studierenden_id'].unique().tolist())
    train_ids, test_ids = train_test_split(unique_studis, test_size=0.20, random_state=42)
    
    train_panel = panel_df[panel_df['studierenden_id'].isin(train_ids)].copy()
    test_panel  = panel_df[panel_df['studierenden_id'].isin(test_ids)].copy()
    
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))]), cat_cols),
        ('treatments', 'passthrough', treatment_cols)
    ])
    preprocessor.fit(train_panel[feature_cols])
    
    model_path = data_dir / "models" / "extended_logistic_hazard_delta.keras"
    if not model_path.exists():
        print(f"Modell {model_path} nicht gefunden!")
        return
        
    model = tf.keras.models.load_model(model_path)
    print(f"Modell geladen: {model_path.name}")

    metrics_all = {}

    for supp_col, prefix, label in [
        ('fach_supp_count',  'fach',  'Fachlicher Support'),
        ('uebf_supp_count',  'uebf',  'Überfachlicher Support'),
        ('psych_supp_count', 'psych', 'Psychosozialer Support'),
    ]:
        # 1. PARTIELL (≙ A vs C/D/E): Ziel-Support 0 vs. beobachtet, andere beobachtet
        control_part = test_panel.copy()
        treated_part = test_panel.copy()
        control_part[supp_col] = 0
        
        X_c_p = preprocessor.transform(control_part[feature_cols])
        X_t_p = preprocessor.transform(treated_part[feature_cols])
        p0_p = model.predict(X_c_p, verbose=0).flatten()
        p1_p = model.predict(X_t_p, verbose=0).flatten()
        rrs_p = p1_p / np.clip(p0_p, 1e-7, 1.0)
        
        mean_rr_p   = float(np.mean(rrs_p))
        median_rr_p = float(np.median(rrs_p))
        q05_p       = float(np.quantile(rrs_p, 0.05))
        q95_p       = float(np.quantile(rrs_p, 0.95))
        
        # 2. ISOLIERT REALISTISCH (≙ B vs F/G/H): Alle 0 vs. nur Ziel beobachtet, andere 0
        control_iso = test_panel.copy()
        treated_iso = test_panel.copy()
        for c in treatment_cols:
            control_iso[c] = 0
            treated_iso[c] = 0
        treated_iso[supp_col] = test_panel[supp_col] # beobachtete Dosis
        
        X_c_i = preprocessor.transform(control_iso[feature_cols])
        X_t_i = preprocessor.transform(treated_iso[feature_cols])
        p0_i = model.predict(X_c_i, verbose=0).flatten()
        p1_i = model.predict(X_t_i, verbose=0).flatten()
        rrs_i = p1_i / np.clip(p0_i, 1e-7, 1.0)
        
        mean_rr_i   = float(np.mean(rrs_i))
        median_rr_i = float(np.median(rrs_i))
        q05_i       = float(np.quantile(rrs_i, 0.05))
        q95_i       = float(np.quantile(rrs_i, 0.95))

        print(f"\n--- {label} ({supp_col}) ---")
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
    save_metrics("counterfactual_rr_logistic_hazard_delta", metrics_all, data_dir)
    print("Counterfactual Relative Risk Analysis für Logistic Hazard Delta abgeschlossen.")

if __name__ == '__main__':
    data_dir = Path('../output_dl')
    if not data_dir.exists():
        data_dir = Path('output_dl')
    analyze_counterfactual_rr_logistic_hazard_delta(data_dir)
