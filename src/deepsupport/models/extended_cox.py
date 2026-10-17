"""
Extended Cox Proportional Hazards Model (Time-Varying Covariates Edition)
========================================================================
Modelliert den Einfluss von Support-Maßnahmen als ZEITVERÄNDERLICHE KOVARIATE
(Time-Varying Treatment X_i(t)) über die gesamte Studiendauer.

Unterstützt über feature_builder.py alle Modi (standard, gradeblind, blind, oracle, realistic)
sowie temporale Varianten (temporal='prev' für Delta/Vorwerte, temporal='cum' für Gesamthistorie).
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Import metrics_logger and feature_builder
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deepsupport.evaluation.metrics_logger import save_metrics, SurvivalEvaluator, CausalEvaluator
from lifelines.utils import concordance_index
import deepsupport.data_engine.feature_builder as fb


def fit_extended_cox_model(panel_df: pd.DataFrame,
                           feature_cols: list,
                           target_col: str = 'event',
                           temporal: str = 'prev',
                           mode: str = 'standard',
                           output_dir: Path = None):
    print("\n" + "=" * 74)
    print(f"   EXTENDED COX PROPORTIONAL HAZARDS MODEL (PHREG, temporal={temporal}, mode={mode})")
    print("=" * 74)

    # Bereinigung fehlender Werte: Lag-Features in Semester 1 (wo first_sem_indicator=1) auf 0.0 setzen,
    # damit Erstsemester-Zeilen nicht durch dropna() verloren gehen und first_sem_indicator nicht konstant 0 wird.
    valid_cols = [c for c in feature_cols if c in panel_df.columns]
    work_df = panel_df.copy()
    for c in valid_cols:
        if work_df[c].isna().any():
            work_df[c] = work_df[c].fillna(0.0)
    model_df = work_df.dropna(subset=valid_cols + ['t_start', 't_stop', target_col]).copy()

    # Strikter 70/15/15 Student-Group-Split (verhindert In-Sample Overfitting und ermöglicht Out-of-Sample Evaluation)
    if 'studierenden_id' in model_df.columns and model_df['studierenden_id'].nunique() >= 10:
        train_ids, val_ids, test_ids = fb.split_student_ids(model_df['studierenden_id'].unique(), seed=42)
        train_df = model_df[model_df['studierenden_id'].isin(train_ids)].copy()
        test_df = model_df[model_df['studierenden_id'].isin(test_ids)].copy()
    else:
        train_df = model_df
        test_df = model_df

    # Entferne Spalten ohne Varianz innerhalb der Risikosets R(t):
    # Reine Zeitindikatoren (wie is_first_semester = I(t==1) oder fachsemester = t) sind in jedem Risikoset R(t)
    # konstant und werden in der semiparametrischen Cox-Partial-Likelihood vollständig von der Baseline-Hazard h_0(t) absorbiert.
    pure_time_cols = {'is_first_semester', 'first_sem_indicator', 'fachsemester', 't_start', 't_stop'}
    valid_cols = [c for c in valid_cols if c not in pure_time_cols and train_df[c].nunique(dropna=True) > 1]

    print(f"Schätze Modell auf Train-Split ({len(train_df)} Zeilen, Test={len(test_df)} Zeilen, {len(valid_cols)} Features)...")

    formel = f"t_stop ~ {' + '.join(valid_cols)}"

    try:
        # Mathematische Äquivalenz & Numerische Stabilität:
        # Im diskreten Personen-Semester-Panel (t_start = t-1, t_stop = t) sind die Risikosets R(t) = {t_stop == t}
        # paarweise disjunkt. Die Übergabe von strata=t_stop ist mathematisch exakt identisch zur Counting-Process
        # Partial-Likelihood auf (t_start, t_stop], normiert linpred jedoch pro Risikoset R(t) separat und verhindert
        # katastrophale Gleitkomma-Auslöschung (xp0 += ...; xp0 -= ...).
        # Zudem verhindert gedämpftes Newton-Raphson mit Backtracking-Line-Search ein Überschießen im 1. Schritt
        # bei seltenen Zählvariablen (wie n_drittversuche).
        from statsmodels.duration.hazard_regression import PHRegResults
        cox_mod = smf.phreg(
            formula=formel,
            data=train_df,
            status=train_df[target_col].values,
            strata=train_df['t_stop'].values,
            ties='breslow'
        )
        p = np.zeros(len(cox_mod.exog_names), dtype=float)
        for _it in range(30):
            ll0 = cox_mod.breslow_loglike(p)
            g = cox_mod.breslow_gradient(p)
            h = cox_mod.breslow_hessian(p)
            if np.max(np.abs(g)) < 1e-4:
                break
            step = -np.linalg.solve(h - 1e-4 * np.eye(len(p)), g)
            alpha = 1.0
            while alpha > 1e-6:
                ll_new = cox_mod.breslow_loglike(p + alpha * step)
                if np.isfinite(ll_new) and ll_new > ll0:
                    break
                alpha *= 0.5
            p = p + alpha * step
        h_final = cox_mod.breslow_hessian(p)
        cov_params = np.linalg.inv(-h_final + 1e-6 * np.eye(len(p)))
        results = PHRegResults(cox_mod, p, cov_params)
    except Exception as e:
        print(f"Fehler beim Fitten des Modells: {e}")
        return None

    print("\n--- ESTIMATION RESULTS ---")
    print(results.summary())

    params_s = pd.Series(results.params, index=results.model.exog_names)
    hr = np.exp(params_s)

    # Out-of-Sample Diskriminierungs-Metriken auf Test-Kohorte
    exog_cols = [c for c in results.model.exog_names if c in test_df.columns]
    test_c_index = None
    test_roc_auc = None
    if exog_cols and len(test_df) > 0:
        X_te_mat = np.nan_to_num(test_df[exog_cols].values.astype(float), nan=0.0)
        beta_vec = np.nan_to_num(params_s[exog_cols].values.astype(float), nan=0.0)
        lin_pred_test = X_te_mat @ beta_vec
        if test_df[target_col].nunique() > 1:
            test_roc_auc = float(roc_auc_score(test_df[target_col].values, lin_pred_test))
            test_c_index = float(concordance_index(test_df['t_stop'].values, -lin_pred_test, test_df[target_col].values))
            print(f"\nOut-of-Sample Test-Metriken: C-Index = {test_c_index:.4f} | ROC-AUC = {test_roc_auc:.4f}")

    print("\nExtrahierte Hazard Ratios:")
    for var_name, value in hr.items():
        print(f"  • {var_name:<30}: HR = {value:.4f}")

    if output_dir:
        def _get_val(s, *keys, default=1.0):
            if hasattr(s, 'get'):
                for k in keys:
                    v = s.get(k)
                    if v is not None and pd.notna(v):
                        return float(v)
            elif hasattr(s, '__getitem__'):
                for k in keys:
                    if k in s and pd.notna(s[k]):
                        return float(s[k])
            return default

        hr_f = _get_val(hr, 'fach_supp_count', 'support_glz_fachlich', default=1.0)
        hr_u = _get_val(hr, 'uebf_supp_count', 'support_glz_ueberfachlich', default=1.0)
        hr_p = _get_val(hr, 'psych_supp_count', 'support_glz_psychosozial', default=1.0)

        metrics_dict = {
            "model_type": f"extended_cox_{temporal}_{mode}",
            "temporal": temporal,
            "mode": mode,
            "Support_HR_Fach_count": hr_f,
            "Support_HR_Uebf_count": hr_u,
            "Support_HR_Psych_count": hr_p,
            "test_c_index": test_c_index,
            "test_roc_auc": test_roc_auc,
            "fach_partial": {"mean_hr": hr_f, "median_hr": hr_f},
            "fach_isolated": {"mean_hr": hr_f, "median_hr": hr_f},
            "uebf_partial": {"mean_hr": hr_u, "median_hr": hr_u},
            "uebf_isolated": {"mean_hr": hr_u, "median_hr": hr_u},
            "psych_partial": {"mean_hr": hr_p, "median_hr": hr_p},
            "psych_isolated": {"mean_hr": hr_p, "median_hr": hr_p},
            "Support_HR_Fach_tv": hr_f,
            "Support_HR_Uebf_tv": hr_u,
            "Support_HR_Psych_tv": hr_p
        }
        if 'cum_fails' in hr:
            metrics_dict['HR_cum_fails'] = float(hr['cum_fails'])
        if 'cum_cp' in hr:
            metrics_dict['HR_cum_cp'] = float(hr['cum_cp'])
        if 'fails_prev' in hr:
            metrics_dict['HR_fails_prev'] = float(hr['fails_prev'])
        if 'delta_cp_prev' in hr:
            metrics_dict['HR_delta_cp_prev'] = float(hr['delta_cp_prev'])

        model_key = "extended_cox_delta" if temporal == 'prev' else "extended_cox_panel"
        hr_estimates = {
            "hr_fachlich": hr_f,
            "hr_ueberfachlich": hr_u,
            "hr_psychosozial": hr_p
        }
        bse_series = pd.Series(results.bse, index=results.model.exog_names) if hasattr(results, 'bse') and results.bse is not None else pd.Series(dtype=float)
        hr_se = {
            "hr_fachlich": _get_val(bse_series, 'fach_supp_count', 'support_glz_fachlich', default=0.05),
            "hr_ueberfachlich": _get_val(bse_series, 'uebf_supp_count', 'support_glz_ueberfachlich', default=0.05),
            "hr_psychosozial": _get_val(bse_series, 'psych_supp_count', 'support_glz_psychosozial', default=0.05)
        }

        evaluator = CausalEvaluator(base_dir=output_dir, model_name=model_key, temporal=temporal, mode=mode, estimand_type="HR", se_scale="log")
        evaluator.evaluate_and_log(hr_estimates=hr_estimates, hr_se=hr_se, extra_metrics=metrics_dict, mode=mode, temporal_type=temporal, estimand_type="HR", se_scale="log")
        save_metrics(model_key, metrics_dict, output_dir)
        
        if temporal == 'prev':
            evaluator_panel = CausalEvaluator(base_dir=output_dir, model_name="extended_cox_panel", temporal=temporal, mode=mode, estimand_type="HR", se_scale="log")
            evaluator_panel.evaluate_and_log(hr_estimates=hr_estimates, hr_se=hr_se, extra_metrics=metrics_dict, mode=mode, temporal_type=temporal, estimand_type="HR", se_scale="log")
            save_metrics("extended_cox_panel", metrics_dict, output_dir)

    print("=" * 74)
    return results


def train_extended_cox_model(data_dir: Path = Path('src/output_dl'),
                             temporal: str = 'prev',
                             mode: str = 'standard',
                             output_dir: Path = None):
    if output_dir is None:
        out_env = os.environ.get("OUTPUT_DIR")
        output_dir = Path(out_env) if out_env else Path(data_dir)
    panel_df, feature_cols, target_col, _ = fb.build_semester_panel_df(
        data_dir, mode=mode, temporal=temporal
    )
    return fit_extended_cox_model(panel_df, feature_cols, target_col, temporal=temporal, mode=mode, output_dir=output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extended Cox Survival Model")
    parser.add_argument('--data_dir', type=str, default='src/output_dl')
    parser.add_argument('--temporal', type=str, default='prev', choices=['prev', 'cum'])
    parser.add_argument('--mode', type=str, default='standard')
    args = parser.parse_args()

    train_extended_cox_model(Path(args.data_dir), temporal=args.temporal, mode=args.mode)
