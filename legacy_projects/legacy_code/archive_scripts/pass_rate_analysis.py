"""
Bestehensquoten-Analyse (Logistische Regression auf Prüfungsebene)
==================================================================
Analysiert den empirischen Effekt der Support-Maßnahmen auf die
Bestehenswahrscheinlichkeit von Prüfungen unter Kontrolle aller Confounder.

Berechnet den kontrafaktischen Lift der Bestehensquote in Prozentpunkten (pp)
sowohl im partiellen als auch im isolierten Strang.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from metrics_logger import save_metrics

def analyze_pass_rates(data_dir: Path):
    print("\n==========================================================================")
    print("   BESTEHENSQUOTEN-ANALYSE (LOGISTISCHE REGRESSION AUF PRÜFUNGSEBENE)")
    print("==========================================================================")
    
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
    
    cols_studi = ['studierenden_id', 'hzb_note', 'erwerbstaetigkeit_std', 'erstakademiker', 'studiengang_id']
    if 'migrationshintergrund' in df_studi.columns:
        cols_studi.append('migrationshintergrund')
    df_merged = df_pruef.merge(df_studi[cols_studi], on='studierenden_id', how='left')
    
    df_valid = df_merged.dropna(subset=['bestanden']).copy()
    df_valid['bestanden_int'] = df_valid['bestanden'].astype(int)
    
    df_valid = df_valid.sort_values(['studierenden_id', 'pruefung_id']).reset_index(drop=True)
    df_valid['is_fail'] = (1 - df_valid['bestanden_int'])
    df_valid['fails_cum_lag'] = df_valid.groupby('studierenden_id')['is_fail'].shift(1).fillna(0)
    
    print(f"Datensatz: {len(df_valid):,} Prüfungen von {df_valid['studierenden_id'].nunique():,} Studierenden.")
    print(f"Basis-Bestehensquote: {df_valid['bestanden_int'].mean():.2%}")
    
    formel = (
        "bestanden_int ~ support_glz_fachlich + support_vorher_fachlich + "
        "support_glz_ueberfachlich + support_vorher_ueberfachlich + "
        "support_glz_psychosozial + support_vorher_psychosozial + "
        "schwierigkeit + cp + versuch + hzb_note + erwerbstaetigkeit_std + erstakademiker + "
        "fails_cum_lag + C(studiengang_id)"
    )
    
    print("\nSchätze Logistisches Regressionsmodell ...")
    model = smf.logit(formula=formel, data=df_valid).fit(disp=False)
    
    support_keys = [
        'support_glz_fachlich', 'support_vorher_fachlich',
        'support_glz_ueberfachlich', 'support_vorher_ueberfachlich',
        'support_glz_psychosozial', 'support_vorher_psychosozial'
    ]
    
    metrics_all = {
        "prsquared": float(model.prsquared),
        "N_obs": int(model.nobs),
        "coefficients": {}
    }
    
    print("\n--- LOGIT ERGEBNISSE (Odds Ratios) ---")
    for k in support_keys:
        if k in model.params:
            coef = float(model.params[k])
            odds_ratio = float(np.exp(coef))
            pval = float(model.pvalues[k])
            print(f"  {k:<30}: Coef = {coef:+.4f} | OR = {odds_ratio:.4f} (p={pval:.4e})")
            metrics_all["coefficients"][k] = {"coef": coef, "odds_ratio": odds_ratio, "pvalue": pval}

    # Kontrafaktische Bestehensquoten-Differenzen
    print("\n--- KONTRAFAKTISCHER BESTEHENSQUOTEN-LIFT (DELTA BESTEHENSRATE IN PP) ---")
    
    pred_obs = model.predict(df_valid)
    
    for prefix, cols_pair, label in [
        ('fach',  ['support_glz_fachlich', 'support_vorher_fachlich'],       'Fachlicher Support'),
        ('uebf',  ['support_glz_ueberfachlich', 'support_vorher_ueberfachlich'], 'Überfachlicher Support'),
        ('psych', ['support_glz_psychosozial', 'support_vorher_psychosozial'],   'Psychosozialer Support')
    ]:
        # Partiell: Dieser Support auf 0, andere beobachtet
        df_c_p = df_valid.copy()
        for c in cols_pair:
            df_c_p[c] = 0.0
        pred_c_p = model.predict(df_c_p)
        delta_pp_p = float((np.mean(pred_obs) - np.mean(pred_c_p)) * 100)
        
        # Isoliert realistisch: Alle auf 0 vs. nur dieser beobachtet
        df_c_i = df_valid.copy()
        for c in support_keys:
            df_c_i[c] = 0.0
        pred_c_i = model.predict(df_c_i)
        
        df_t_i = df_valid.copy()
        for c in support_keys:
            if c not in cols_pair:
                df_t_i[c] = 0.0
        pred_t_i = model.predict(df_t_i)
        delta_pp_i = float((np.mean(pred_t_i) - np.mean(pred_c_i)) * 100)
        
        print(f"  {label:<28}: Partiell = {delta_pp_p:+.2f}pp | Isoliert = {delta_pp_i:+.2f}pp")
        metrics_all[f"{prefix}_partial"] = {"mean_delta_pass_rate_pp": delta_pp_p}
        metrics_all[f"{prefix}_isolated"] = {"mean_delta_pass_rate_pp": delta_pp_i}

    save_metrics("pass_rate_counterfactual_metrics", metrics_all, resolved_dir)
    print(f"\nErgebnisse gespeichert in: {resolved_dir / 'metrics' / 'pass_rate_counterfactual_metrics.json'}")

if __name__ == '__main__':
    analyze_pass_rates(Path("output_dl"))
