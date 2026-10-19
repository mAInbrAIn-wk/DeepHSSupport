"""
Lineare Noteneffekt-Analyse (OLS / Ridge auf Prüfungsebene)
===========================================================
Analysiert den empirischen Effekt der verschiedenen Support-Formen
(fachlich, überfachlich, psychosozial; jeweils vorher vs. gleichzeitig)
auf die Prüfungsnoten unter strikter Kontrolle aller beobachtbaren Confounder.

Vergleicht die geschätzten Regressionskoeffizienten direkt mit der
Ground Truth (A vs C/D/E und F/G/H vs B).
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from metrics_logger import save_metrics

def analyze_grade_effects_linear(data_dir: Path):
    print("\n==========================================================================")
    print("   LINEARE NOTENEFFEKT-ANALYSE (OLS AUF PRÜFUNGSEBENE)")
    print("==========================================================================")
    
    # Pfad auflösen
    possible_dirs = [data_dir, Path("src/output_dl"), Path("output_dl"), Path("../output_dl")]
    resolved_dir = None
    for p in possible_dirs:
        if (p / "agg_pruefungen.csv").exists():
            resolved_dir = p
            break
    if resolved_dir is None:
        print("agg_pruefungen.csv nicht gefunden!")
        return
        
    df_pruef = pd.read_csv(resolved_dir / "agg_pruefungen.csv")
    df_studi = pd.read_csv(resolved_dir / "studierende.csv")
    
    # Demographie mergen
    cols_studi = ['studierenden_id', 'hzb_note', 'erwerbstaetigkeit_std', 'erstakademiker', 'studiengang_id']
    if 'migrationshintergrund' in df_studi.columns:
        cols_studi.append('migrationshintergrund')
    df_merged = df_pruef.merge(df_studi[cols_studi], on='studierenden_id', how='left')
    
    # Nur gültige Prüfungen mit Noten (1.0 bis 5.0)
    df_valid = df_merged.dropna(subset=['note']).copy()
    df_valid = df_valid[df_valid['note'] >= 1.0].copy()
    
    # Berechne gelaggte Features
    df_valid = df_valid.sort_values(['studierenden_id', 'pruefung_id']).reset_index(drop=True)
    df_valid['is_fail'] = (~df_valid['bestanden']).astype(int)
    df_valid['fails_cum_lag'] = df_valid.groupby('studierenden_id')['is_fail'].shift(1).fillna(0)
    
    print(f"Datensatz: {len(df_valid):,} Prüfungen von {df_valid['studierenden_id'].nunique():,} Studierenden.")
    
    # OLS Formel
    formel = (
        "note ~ support_glz_fachlich + support_vorher_fachlich + "
        "support_glz_ueberfachlich + support_vorher_ueberfachlich + "
        "support_glz_psychosozial + support_vorher_psychosozial + "
        "schwierigkeit + cp + versuch + hzb_note + erwerbstaetigkeit_std + erstakademiker + "
        "fails_cum_lag + C(studiengang_id)"
    )
    
    print("\nSchätze OLS Regressionsmodell ...")
    model = smf.ols(formula=formel, data=df_valid).fit(cov_type='cluster', cov_kwds={'groups': df_valid['studierenden_id']})
    
    print("\n--- OLS ERGEBNISSE (Cluster-robuste Standardfehler nach Studierenden) ---")
    support_keys = [
        'support_glz_fachlich', 'support_vorher_fachlich',
        'support_glz_ueberfachlich', 'support_vorher_ueberfachlich',
        'support_glz_psychosozial', 'support_vorher_psychosozial'
    ]
    
    metrics_all = {
        "R2": float(model.rsquared),
        "R2_adj": float(model.rsquared_adj),
        "N_obs": int(model.nobs),
        "coefficients": {}
    }
    
    for k in support_keys:
        if k in model.params:
            coef = float(model.params[k])
            se = float(model.bse[k])
            pval = float(model.pvalues[k])
            print(f"  {k:<30}: Coef = {coef:+.4f} (SE={se:.4f}, p={pval:.4e})")
            metrics_all["coefficients"][k] = {"coef": coef, "se": se, "pvalue": pval}
            
    for k in ['schwierigkeit', 'versuch', 'hzb_note', 'erwerbstaetigkeit_std', 'erstakademiker', 'fails_cum_lag']:
        if k in model.params:
            coef = float(model.params[k])
            se = float(model.bse[k])
            pval = float(model.pvalues[k])
            metrics_all["coefficients"][k] = {"coef": coef, "se": se, "pvalue": pval}

    # Kontrafaktische Vorhersagen auf Prüfungsebene:
    # 1. Partieller Noteneffekt (A vs C/D/E): Support auf 0 setzen vs. beobachtet lassen
    print("\n--- KONTRAFAKTISCHE NOTENDIFFERENZEN (DELTA NOTE = TREATED - CONTROL) ---")
    print("    (Negative Differenz = Notenverbesserung durch Support)")
    
    df_pred_treated = df_valid.copy()
    pred_obs = model.predict(df_pred_treated)
    
    for prefix, cols_pair, label in [
        ('fach',  ['support_glz_fachlich', 'support_vorher_fachlich'],       'Fachlicher Support'),
        ('uebf',  ['support_glz_ueberfachlich', 'support_vorher_ueberfachlich'], 'Überfachlicher Support'),
        ('psych', ['support_glz_psychosozial', 'support_vorher_psychosozial'],   'Psychosozialer Support')
    ]:
        # Partiell: Dieser Support auf 0, alle anderen beobachtet
        df_c_p = df_valid.copy()
        for c in cols_pair:
            df_c_p[c] = 0.0
        pred_c_p = model.predict(df_c_p)
        delta_p = float(np.mean(pred_obs - pred_c_p))
        
        # Isoliert realistisch: Alle anderen auf 0, nur dieser beobachtet vs. alle 0
        df_c_i = df_valid.copy()
        for c in support_keys:
            df_c_i[c] = 0.0
        pred_c_i = model.predict(df_c_i)
        
        df_t_i = df_valid.copy()
        for c in support_keys:
            if c not in cols_pair:
                df_t_i[c] = 0.0
        pred_t_i = model.predict(df_t_i)
        delta_i = float(np.mean(pred_t_i - pred_c_i))
        
        print(f"  {label:<28}: Partiell = {delta_p:+.4f} Notenpunkte | Isoliert = {delta_i:+.4f} Notenpunkte")
        metrics_all[f"{prefix}_partial"] = {"mean_delta_note": delta_p}
        metrics_all[f"{prefix}_isolated"] = {"mean_delta_note": delta_i}

    save_metrics("grade_effect_linear_metrics", metrics_all, resolved_dir)
    print(f"\nErgebnisse gespeichert in: {resolved_dir / 'metrics' / 'grade_effect_linear_metrics.json'}")

if __name__ == '__main__':
    analyze_grade_effects_linear(Path("output_dl"))
