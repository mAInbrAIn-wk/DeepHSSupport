"""
Oracle Mediationsanalyse (Imai / Pearl Framework)
========================================================================
Testet die strukturellen Mediations-Hypothesen unter Hinzunahme der
versteckten Variablen (hidden_motivation, hidden_soziale_integration).

Führt drei Tests pro Supportart aus:
1. Baseline: Ohne hidden vars (Standard)
2. Hidden as Confounder: Kontrolliert für die Selektions-Verzerrung.
3. Hidden as Mediator: Zwingt den Effekt durch den wahren Pfad.
4. Both: Confounder + Mediator
"""

import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf

# Import metrics_logger and feature_builder
sys.path.insert(0, str(Path(__file__).parent))
from metrics_logger import save_metrics
import feature_builder as fb

def run_oracle_mediation(data_dir: Path):
    print("\n" + "=" * 74)
    print("   ORACLE MEDIATIONSANALYSE (MIT VERSTECKTEN VARIABLEN)")
    print("=" * 74)

    # Oracle Mode liefert die hidden_..._prev Spalten!
    panel_df, feature_cols, target_col, _ = fb.build_semester_panel_df(
        data_dir, mode='standard', temporal='prev', oracle=True
    )

    valid_cols = ['event', 'fach_supp_count', 'uebf_supp_count', 'psych_supp_count',
                  'delta_cp_prev', 'fails_prev', 'gpa_prev', 'hzb_note', 'erwerbstaetigkeit_std', 
                  'erstakademiker', 'hidden_motivation_prev', 'hidden_soziale_integration_prev']
    
    missing = [c for c in valid_cols if c not in panel_df.columns]
    if missing:
        print(f"[FEHLER] Fehlende Spalten in panel_df: {missing}")
        return
        
    df_clean = panel_df.dropna(subset=valid_cols).copy()
    print(f"Analysiere Oracle Mediationspfade für {len(df_clean):,} Person-Semester Zeilen...")

    # Generischer Leistungs-Mediator
    df_clean['mediator_performance'] = df_clean['delta_cp_prev'] - df_clean['fails_prev'] * 5.0

    treatments = {
        "fachlich": ("fach_supp_count", "mediator_performance"),
        "ueberfachlich": ("uebf_supp_count", "hidden_motivation_prev"),
        "psychosozial": ("psych_supp_count", "hidden_soziale_integration_prev")
    }

    results = {}

    for t_name, (t_col, true_mediator) in treatments.items():
        print(f"\n--- Oracle Analyse für Treatment: {t_name.upper()} ({t_col}) ---")
        
        configs = {
            "1_Realistic": {
                "med_col": "mediator_performance",
                "confounders": "hzb_note + erwerbstaetigkeit_std + erstakademiker"
            },
            "2_Oracle_Confounder": {
                "med_col": "mediator_performance",
                "confounders": "hzb_note + erwerbstaetigkeit_std + erstakademiker + hidden_motivation_prev + hidden_soziale_integration_prev"
            },
            "3_Oracle_Mediator": {
                "med_col": true_mediator,
                "confounders": "hzb_note + erwerbstaetigkeit_std + erstakademiker"
            },
            "4_Oracle_Both": {
                "med_col": true_mediator,
                "confounders": "hzb_note + erwerbstaetigkeit_std + erstakademiker + hidden_motivation_prev + hidden_soziale_integration_prev"
            }
        }
        
        results[t_name] = {}
        
        for cfg_name, cfg in configs.items():
            med_col = cfg["med_col"]
            conf = cfg["confounders"]
            
            med_formula = f"{med_col} ~ {t_col} + {conf}"
            med_model = smf.ols(med_formula, data=df_clean).fit()
            gamma_t = med_model.params[t_col]
            
            out_formula = f"event ~ {t_col} + {med_col} + {conf}"
            out_model = smf.logit(out_formula, data=df_clean).fit(disp=False)
            beta_t = out_model.params[t_col]
            beta_m = out_model.params[med_col]
            
            acme = float(gamma_t * beta_m)
            ade = float(beta_t)
            total = acme + ade
            hr_total = np.exp(total)
            hr_direct = np.exp(ade)
            hr_indirect = np.exp(acme)
            
            results[t_name][cfg_name] = {
                "total_or": float(hr_total),
                "direct_or": float(hr_direct),
                "mediated_or": float(hr_indirect)
            }
            
            print(f"  [{cfg_name}] Total OR: {hr_total:.3f} | Direct OR (ADE): {hr_direct:.3f} | Mediated OR (ACME): {hr_indirect:.3f}")

    print("\n" + "=" * 74)
    print("   ZUSAMMENFASSUNG ORACLE MEDIATIONSANALYSE")
    print("=" * 74)
    print(f"{'Support':<14} | {'Modus':<20} | {'Total OR':<10} | {'Direct OR':<10} | {'Mediated OR'}")
    print("-" * 74)
    for t_name, cfgs in results.items():
        for cfg_name, r in cfgs.items():
            print(f"{t_name.capitalize():<14} | {cfg_name:<20} | {r['total_or']:<10.3f} | {r['direct_or']:<10.3f} | {r['mediated_or']:<10.3f}")
        print("-" * 74)
        
    save_metrics("oracle_mediation_analysis", results, data_dir)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='src/output_dl_seed99999')
    args = parser.parse_args()
    run_oracle_mediation(Path(args.data_dir))
