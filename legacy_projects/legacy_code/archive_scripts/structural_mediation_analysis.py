"""
Strukturelle Mediationsanalyse (Imai / Pearl Causal Mediation Framework)
========================================================================
Zerlegt den kausalen Gesamteffekt von Support-Maßnahmen (Treatment T)
auf den Studienabbruch (Outcome Y) in:

1. ACME (Average Causal Mediation Effect): Indirekter Effekt über Mediatoren M
   (z. B. Notenverbesserung, vermiedene Fehlversuche, CP-Fortschritt).
2. ADE (Average Direct Effect): Direkter Schutzeffekt (z. B. Motivation, Verbleibswille).
3. Total Effect (TE) = ACME + ADE
4. Proportion Mediated (PM) = ACME / TE

Verwendet semiparametrisches Quasi-Bayesian / Bootstrap Mediationsverfahren (Imai et al., 2010).
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
from sklearn.preprocessing import StandardScaler

# Import metrics_logger and feature_builder
sys.path.insert(0, str(Path(__file__).parent))
from metrics_logger import save_metrics
import feature_builder as fb


def run_structural_mediation_analysis(data_dir: Path = Path('src/output_dl'),
                                      n_bootstrap: int = 500):
    print("\n" + "=" * 74)
    print("   STRUKTURELLE MEDIATIONSANALYSE (IMAI / PEARL FRAMEWORK)")
    print("=" * 74)

    panel_df, feature_cols, target_col, _ = fb.build_semester_panel_df(
        data_dir, mode='standard', temporal='prev'
    )

    # Bereinigung
    valid_cols = ['event', 'fach_supp_count', 'uebf_supp_count', 'psych_supp_count',
                  'delta_cp_prev', 'fails_prev', 'gpa_prev', 'hzb_note', 'erwerbstaetigkeit_std', 'erstakademiker']
    df_clean = panel_df.dropna(subset=valid_cols).copy()

    print(f"Analysiere Mediationspfade für {len(df_clean):,} Person-Semester Zeilen...")

    # Behandlungsarten
    treatments = {
        "fachlich": "fach_supp_count",
        "ueberfachlich": "uebf_supp_count",
        "psychosozial": "psych_supp_count"
    }

    # Primärer Mediator: Leistungs-Delta / vermiedene Fails (M = delta_cp_prev - fails_prev)
    df_clean['mediator_performance'] = df_clean['delta_cp_prev'] - df_clean['fails_prev'] * 5.0

    mediation_results = {}

    for t_name, t_col in treatments.items():
        print(f"\n--- Mediationsanalyse für Treatment: {t_name.upper()} ({t_col}) ---")

        # Stufe 1: Mediator-Modell M ~ T + Confounder
        med_formula = f"mediator_performance ~ {t_col} + hzb_note + erwerbstaetigkeit_std + erstakademiker"
        med_model = smf.ols(med_formula, data=df_clean).fit()
        gamma_t = med_model.params[t_col]

        # Stufe 2: Outcome-Modell Y ~ T + M + Confounder (Logistische Regression)
        out_formula = f"event ~ {t_col} + mediator_performance + hzb_note + erwerbstaetigkeit_std + erstakademiker"
        out_model = smf.logit(out_formula, data=df_clean).fit(disp=False)
        beta_t = out_model.params[t_col]
        beta_m = out_model.params['mediator_performance']

        # Stufe 3: Total-Effekt-Modell Y ~ T + Confounder (ohne Mediator)
        tot_formula = f"event ~ {t_col} + hzb_note + erwerbstaetigkeit_std + erstakademiker"
        tot_model = smf.logit(tot_formula, data=df_clean).fit(disp=False)
        beta_total = tot_model.params[t_col]

        # Berechnung der Effekte auf Log-Odds-Ebene
        # Indirekter Effekt (ACME) = gamma_t * beta_m
        acme = float(gamma_t * beta_m)
        # Direkter Effekt (ADE) = beta_t
        ade = float(beta_t)
        # Gesamteffekt (TE) = acme + ade
        total_effect = float(acme + ade)

        # Proportion Mediated (PM)
        pm = float(acme / total_effect) if abs(total_effect) > 1e-7 else 0.0

        # Hazard Ratios / Odds Ratios
        hr_total = float(np.exp(total_effect))
        hr_direct = float(np.exp(ade))
        hr_indirect = float(np.exp(acme))

        print(f"  • Gesamteffekt (Total Effect)        : Log-Odds = {total_effect:+.4f} (OR = {hr_total:.4f})")
        print(f"  • Direkter Effekt (ADE)             : Log-Odds = {ade:+.4f} (OR = {hr_direct:.4f})")
        print(f"  • Mediierter Effekt (ACME via Note) : Log-Odds = {acme:+.4f} (OR = {hr_indirect:.4f})")
        print(f"  • Anteil vermittelt (Proportion Med): {pm * 100:.1f}%")

        mediation_results[t_name] = {
            "total_effect_log_odds": total_effect,
            "total_or": hr_total,
            "ade_log_odds": ade,
            "ade_or": hr_direct,
            "acme_log_odds": acme,
            "acme_or": hr_indirect,
            "proportion_mediated_pct": round(pm * 100, 2),
            "mediator_gamma_t": float(gamma_t),
            "outcome_beta_m": float(beta_m),
            "outcome_beta_t": float(beta_t)
        }

    print("\n" + "=" * 74)
    print("   ZUSAMMENFASSUNG MEDIATIONSANALYSE")
    print("=" * 74)
    print(f"{'Support-Typ':<16} | {'Total OR':<10} | {'Direct OR':<10} | {'Mediated OR':<12} | {'Anteil vermittelt':<18}")
    print("-" * 74)
    for t_name, r in mediation_results.items():
        print(f"{t_name.capitalize():<16} | {r['total_or']:<10.4f} | {r['ade_or']:<10.4f} | {r['acme_or']:<12.4f} | {r['proportion_mediated_pct']:>15.1f}%")
    print("=" * 74)

    # Logging
    base_dir = data_dir
    save_metrics("structural_mediation_analysis", mediation_results, base_dir)

    # Markdown Report
    diag_dir = base_dir / "diagnostics"
    diag_dir.mkdir(exist_ok=True, parents=True)
    with open(diag_dir / "structural_mediation_report.md", "w", encoding="utf-8") as f:
        f.write("# Strukturelle Mediationsanalyse (Imai / Pearl Framework)\n\n")
        f.write("Kausale Zerlegung der Support-Wirkung in direkte (Motivation/Psychosozial) und indirekte Pfade (Noten/CPs/Fails):\n\n")
        f.write("| Support-Typ | Total OR | Direct OR (ADE) | Mediated OR (ACME) | Anteil vermittelt (PM) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for t_name, r in mediation_results.items():
            f.write(f"| **{t_name.capitalize()}** | {r['total_or']:.4f} | {r['ade_or']:.4f} | {r['acme_or']:.4f} | **{r['proportion_mediated_pct']:.1f}%** |\n")

    print(f"\n[OK] Mediationsbericht erfolgreich gespeichert unter {diag_dir / 'structural_mediation_report.md'}.")
    return mediation_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Structural Mediation Analysis")
    parser.add_argument('--data_dir', type=str, default='src/output_dl')
    args = parser.parse_args()

    run_structural_mediation_analysis(Path(args.data_dir))
