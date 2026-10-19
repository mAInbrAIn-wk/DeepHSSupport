"""
Counterfactual Relative Risk Analysis für Oracle Logistic Hazard
=================================================================
Berechnet das Relative Risiko (RR) für die drei Support-Typen auf dem
Oracle Logistic Hazard Modell, welches direkten Zugriff auf die latenten
Variablen (hidden_motivation_prev, hidden_soziale_integration_prev,
hidden_erwartete_note_prev) hat.

Damit wird quantifiziert, wie stark der kausale Identifikationslift
durch Entzerrung der latenten Confounder ist.
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

import feature_builder as fb
from metrics_logger import save_metrics

def analyze_counterfactual_oracle_logistic_hazard(data_dir: Path):
    print("\n==========================================================================")
    print("   COUNTERFACTUAL RELATIVE RISK ANALYSIS (ORACLE LOGISTIC HAZARD)")
    print("==========================================================================")
    
    data_dir = Path(data_dir)
    if not data_dir.exists() and (Path('src') / data_dir).exists():
        data_dir = Path('src') / data_dir
        
    panel_df, feature_cols, target_col, _ = fb.build_semester_panel_df(data_dir, mode='oracle', temporal='prev')
    treatment_cols = ['fach_supp_count', 'uebf_supp_count', 'psych_supp_count']
    
    unique_studis = np.array(panel_df['studierenden_id'].unique().tolist())
    train_ids, test_ids = train_test_split(unique_studis, test_size=0.20, random_state=42)
    
    train_panel = panel_df[panel_df['studierenden_id'].isin(train_ids)].copy()
    test_panel  = panel_df[panel_df['studierenden_id'].isin(test_ids)].copy()
    
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    scaler.fit(imputer.fit_transform(train_panel[feature_cols]))
    
    def transform_df(df_in):
        return scaler.transform(imputer.transform(df_in[feature_cols]))
    
    model_path = data_dir / "models" / "oracle_logistic_hazard.keras"
    if not model_path.exists():
        print(f"Modell {model_path} nicht gefunden! Bitte zuerst train_oracle_models.py ausführen.")
        return
        
    model = tf.keras.models.load_model(model_path)
    print(f"Orakel-Modell geladen: {model_path.name} | Features: {len(feature_cols)}")

    metrics_all = {}

    for supp_col, prefix, label in [
        ('fach_supp_count',  'fach',  'Fachlicher Support (Oracle)'),
        ('uebf_supp_count',  'uebf',  'Überfachlicher Support (Oracle)'),
        ('psych_supp_count', 'psych', 'Psychosozialer Support (Oracle)'),
    ]:
        # 1. PARTIELL (≙ A vs C/D/E): Ziel-Support 0 vs. beobachtet, andere beobachtet
        control_part = test_panel.copy()
        treated_part = test_panel.copy()
        control_part[supp_col] = 0
        
        X_c_p = transform_df(control_part)
        X_t_p = transform_df(treated_part)
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
        
        X_c_i = transform_df(control_iso)
        X_t_i = transform_df(treated_iso)
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
        
        metrics_all[f"Mean_RR_{prefix}"]   = mean_rr_p
        metrics_all[f"Median_RR_{prefix}"] = median_rr_p
        metrics_all[f"Q05_RR_{prefix}"]    = q05_p
        metrics_all[f"Q95_RR_{prefix}"]    = q95_p

    print("\n" + "=" * 74)
    save_metrics("counterfactual_oracle_logistic_hazard_metrics", metrics_all, data_dir)
    print("Counterfactual Relative Risk Analysis für Oracle Logistic Hazard abgeschlossen.")

main = analyze_counterfactual_oracle_logistic_hazard

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    args = parser.parse_args()
    if args.data_dir:
        target_dir = Path(args.data_dir)
    else:
        possible_dirs = [Path("src/output_dl"), Path("output_dl"), Path("../src/output_dl"), Path("../output_dl")]
        target_dir = None
        for p in possible_dirs:
            if (p / "models" / "oracle_logistic_hazard.keras").exists():
                target_dir = p
                break
        if target_dir is None:
            target_dir = Path("output_dl")
    main(target_dir)
