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

import feature_builder as fb
from extended_deep_survival import breslow_cox_loss
from metrics_logger import save_metrics

def analyze_counterfactual_hr(data_dir: Path, mode: str = 'standard'):
    print("\n==========================================================================")
    print(f"   COUNTERFACTUAL HAZARD RATIO ANALYSIS (EXTENDED DEEPSURV PANEL | mode={mode})")
    print("==========================================================================")
    
    model_path = None
    for candidate in [
        f"extended_deepsurv_prev_{mode}.keras",
        f"extended_deepsurv_cum_{mode}.keras",
        "extended_deepsurv_panel.keras",
        "extended_deepsurv_prev.keras",
        "extended_deepsurv_delta.keras",
        "extended_deepsurv_cum.keras",
        "extended_deepsurv.keras"
    ]:
        p = data_dir / "models" / candidate
        if p.exists():
            model_path = p
            break
    if model_path is None:
        print(f"Kein Extended DeepSurv Modell unter {data_dir / 'models'} gefunden!")
        return
        
    model = tf.keras.models.load_model(model_path, custom_objects={'breslow_cox_loss': breslow_cox_loss})
    print(f"Modell geladen: {model_path.name}  |  Input-Dim: {model.input_shape}")
    expected_dim = model.input_shape[-1]
    
    panel_df, feature_cols, target_col, _ = fb.build_semester_panel_df(data_dir, mode=mode, temporal='prev')
    
    cat_candidates = ['stg_name', 'erstakademiker', 'hzb_typ', 'migrationshintergrund']
    cat_cols = [c for c in cat_candidates if c in feature_cols]
    treatment_candidates = ['fach_supp_count', 'uebf_supp_count', 'psych_supp_count']
    treatment_cols = [c for c in treatment_candidates if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols and c not in treatment_cols]
    
    unique_studis = np.array(panel_df['studierenden_id'].unique().tolist())
    train_ids, test_ids = train_test_split(unique_studis, test_size=0.20, random_state=42)
    
    train_panel = panel_df[panel_df['studierenden_id'].isin(train_ids)].copy()
    test_panel  = panel_df[panel_df['studierenden_id'].isin(test_ids)].copy()
    
    transformers = []
    if num_cols:
        transformers.append(('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols))
    if cat_cols:
        transformers.append(('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))]), cat_cols))
    if treatment_cols:
        transformers.append(('treatments', 'passthrough', treatment_cols))
        
    preprocessor = ColumnTransformer(transformers)
    preprocessor.fit(train_panel[feature_cols])
    
    X_test_chk = preprocessor.transform(test_panel[feature_cols])
    if X_test_chk.shape[1] != expected_dim:
        alt_mode = 'gradeblind' if mode == 'standard' else 'standard'
        panel_df, feature_cols, target_col, _ = fb.build_semester_panel_df(data_dir, mode=alt_mode, temporal='prev')
        cat_cols = [c for c in cat_candidates if c in feature_cols]
        treatment_cols = [c for c in treatment_candidates if c in feature_cols]
        num_cols = [c for c in feature_cols if c not in cat_cols and c not in treatment_cols]
        train_panel = panel_df[panel_df['studierenden_id'].isin(train_ids)].copy()
        test_panel  = panel_df[panel_df['studierenden_id'].isin(test_ids)].copy()
        transformers = []
        if num_cols:
            transformers.append(('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols))
        if cat_cols:
            transformers.append(('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))]), cat_cols))
        if treatment_cols:
            transformers.append(('treatments', 'passthrough', treatment_cols))
        preprocessor = ColumnTransformer(transformers)
        preprocessor.fit(train_panel[feature_cols])

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
        
        # Abwärtskompatible Keys
        metrics_all[f"Mean_HR_{prefix}_supp"]   = mean_hr_p
        metrics_all[f"Median_HR_{prefix}_supp"] = median_hr_p

    print("\n" + "=" * 74)
    save_metrics("counterfactual_hr_analyzer", metrics_all, data_dir)
    print("Counterfactual HR Analyse abgeschlossen.")

if __name__ == '__main__':
    data_dir = Path('../output_dl')
    if not data_dir.exists():
        data_dir = Path('output_dl')
    analyze_counterfactual_hr(data_dir)
