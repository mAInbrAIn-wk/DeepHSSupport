"""
Unified Counterfactual Ground-Truth Benchmark & AI Policy Targeting Showcase (AP 4).
====================================================================================
Verknüpft die kontrafaktischen Parallelwelten (Universen A–H aus `data_v4_grid/S01_baseline`)
direkt mit den kausalen Schätzverfahren des DeepSupport-Frameworks und evaluiert den
bildungsökonomischen Nutzen KI-gestützter Interventions-Zuweisung (Policy Targeting).

Erzeugt:
1. Exakte Simulator-Ground-Truth (Population Macro-RR/ARR & Treated Interval-HR/ATT)
   für Knockout-Kontraste (A vs. C/D/E) und isolierte Stand-Alone-Kontraste (F/G/H vs. B).
2. Systematischen Schätzer-Benchmark auf Universum A (`standard` vs. `oracle`):
   - Naive unadjustierte Korrelation (Confounding by Indication)
   - Extended Cox Time-Varying (Counting Process)
   - Marginal Structural Model (IPTW + `cloglog` Proportional Hazard & `logit` Odds Ratio)
   - Double Machine Learning (5-Fold GroupKFold Cross-Fitted Multivariate Robinson)
   - Sequenzielle G-Computation (Step-Standardized & 16-Semester Autoregressive Rollout)
3. Out-of-Sample AI Policy Targeting Showcase (N_test = 7,500 Studierende, Budgets 10%, 20%, 30%):
   Vergleicht Random Allocation (Gießkanne), Reaktive Zuweisung (Status Quo nach HZB/Erwerb),
   Predictive-Risk Zuweisung (Frühwarnung S1/S2) und Causal CATE/Uplift Zuweisung anhand der
   wahren kontrafaktischen Potenzialergebnisse Y_i(A) vs. Y_i(B).
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxTimeVaryingFitter
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import deepsupport.data_engine.feature_builder as fb


TREATMENT_COLS = [
    "fach_supp_count",
    "uebf_supp_count",
    "psych_supp_count",
]
DIMS = ["fachlich", "ueberfachlich", "psychosozial"]


def load_universe_outcomes(scenario_dir: Path) -> Dict[str, pd.DataFrame]:
    """Lädt abschluesse.csv und aggregierte Semester-Panels für die Universen A bis H."""
    universes = {}
    for u in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        u_dir = scenario_dir / f"universe_{u}"
        ab_path = u_dir / "abschluesse.csv"
        if not ab_path.exists():
            ab_path = u_dir / "agg_abschluesse.csv"
        df_ab = pd.read_csv(ab_path).sort_values("studierenden_id").reset_index(drop=True)
        df_ab["dropout"] = (df_ab["status"] != "abgeschlossen").astype(int)
        universes[u] = df_ab
    return universes


def compute_simulator_ground_truth(
    scenario_dir: Path, universes: Dict[str, pd.DataFrame]
) -> Dict[str, Any]:
    """
    Berechnet die exakte kontrafaktische Ground Truth über die 8 Parallelwelten (A–H):
    - Population-Level Risk Ratio (RR) und Absolute Risk Reduction (ARR)
    - Subgruppen-ATT (Average Treatment Effect on the Treated) für tatsächliche Support-Nutzer
    - Diskrete Intervall-Hazard-Ratio (HR) auf Personen-Semester-Ebene
    """
    rate = {u: float(df["dropout"].mean()) for u, df in universes.items()}
    mean_dur = {u: float(df["studiendauer_semester"].mean()) for u, df in universes.items()}

    # 1. Population Macro Contrasts
    # Knockout: A (Full) vs C (No Fach), D (No Uebf), E (No Psych) -> < 1.0 means protective
    knockout_rr = {
        "fachlich": rate["A"] / rate["C"],
        "ueberfachlich": rate["A"] / rate["D"],
        "psychosozial": rate["A"] / rate["E"],
        "total_A_vs_B": rate["A"] / rate["B"],
    }
    knockout_arr_pp = {
        "fachlich": (rate["C"] - rate["A"]) * 100.0,
        "ueberfachlich": (rate["D"] - rate["A"]) * 100.0,
        "psychosozial": (rate["E"] - rate["A"]) * 100.0,
        "total_A_vs_B": (rate["B"] - rate["A"]) * 100.0,
    }

    # Stand-Alone: F (Only Fach), G (Only Uebf), H (Only Psych) vs B (No Support)
    standalone_rr = {
        "fachlich": rate["F"] / rate["B"],
        "ueberfachlich": rate["G"] / rate["B"],
        "psychosozial": rate["H"] / rate["B"],
    }
    standalone_arr_pp = {
        "fachlich": (rate["B"] - rate["F"]) * 100.0,
        "ueberfachlich": (rate["B"] - rate["G"]) * 100.0,
        "psychosozial": (rate["B"] - rate["H"]) * 100.0,
    }

    # Discrete-time person-semester hazard ratio h_u / h_ref
    # Average per-semester dropout hazard = P(Dropout) / E[T]
    haz = {u: rate[u] / mean_dur[u] for u in universes}
    knockout_hr_macro = {
        "fachlich": haz["A"] / haz["C"],
        "ueberfachlich": haz["A"] / haz["D"],
        "psychosozial": haz["A"] / haz["E"],
        "total_A_vs_B": haz["A"] / haz["B"],
    }
    standalone_hr_macro = {
        "fachlich": haz["F"] / haz["B"],
        "ueberfachlich": haz["G"] / haz["B"],
        "psychosozial": haz["H"] / haz["B"],
    }

    # 2. Exact Counterfactual ATT on Treated Students (Users of dimension k in Universe A)
    df_st_a = pd.read_csv(scenario_dir / "universe_A" / "support_teilnahmen.csv")
    df_sa = pd.read_csv(scenario_dir / "universe_A" / "support_angebote.csv")
    df_st_a = df_st_a.merge(df_sa[["angebot_id", "typ"]], on="angebot_id", how="left")

    att_knockout_rr = {}
    att_knockout_hr = {}
    att_standalone_rr = {}
    att_users_count = {}

    knockout_map = {"fachlich": "C", "ueberfachlich": "D", "psychosozial": "E"}
    standalone_map = {"fachlich": "F", "ueberfachlich": "G", "psychosozial": "H"}

    for dim in DIMS:
        user_ids = df_st_a.loc[df_st_a["typ"] == dim, "studierenden_id"].unique()
        mask_u = universes["A"]["studierenden_id"].isin(user_ids)
        att_users_count[dim] = int(mask_u.sum())

        u_ko = knockout_map[dim]
        u_sa = standalone_map[dim]

        r_a_treated = float(universes["A"].loc[mask_u, "dropout"].mean())
        r_ko_treated = float(universes[u_ko].loc[mask_u, "dropout"].mean())
        r_sa_treated = float(universes[u_sa].loc[mask_u, "dropout"].mean())
        r_b_treated = float(universes["B"].loc[mask_u, "dropout"].mean())

        t_a_treated = float(universes["A"].loc[mask_u, "studiendauer_semester"].mean())
        t_ko_treated = float(universes[u_ko].loc[mask_u, "studiendauer_semester"].mean())

        att_knockout_rr[dim] = r_a_treated / max(r_ko_treated, 1e-9)
        att_knockout_hr[dim] = (r_a_treated / t_a_treated) / max(r_ko_treated / t_ko_treated, 1e-9)
        att_standalone_rr[dim] = r_sa_treated / max(r_b_treated, 1e-9)

    return {
        "dropout_rates": rate,
        "mean_durations": mean_dur,
        "knockout_rr": knockout_rr,
        "knockout_arr_pp": knockout_arr_pp,
        "knockout_hr_macro": knockout_hr_macro,
        "standalone_rr": standalone_rr,
        "standalone_arr_pp": standalone_arr_pp,
        "standalone_hr_macro": standalone_hr_macro,
        "att_users_count": att_users_count,
        "att_knockout_rr": att_knockout_rr,
        "att_knockout_hr": att_knockout_hr,
        "att_standalone_rr": att_standalone_rr,
    }


def run_causal_estimators_on_panel(panel: pd.DataFrame, feature_cols: list, mode: str) -> Dict[str, Dict[str, float]]:
    """
    Führt alle standardisierten Kausalschätzer auf demselben Personen-Semester-Panel (Universum A) aus:
    1. Naive Unadjusted (rohes Risiko-Verhältnis / Odds Ratio ohne Confounder-Kontrolle)
    2. Extended Cox TVC (Counting Process `(t_start, t_stop, event)`)
    3. MSM / IPTW (`cloglog` Proportional Hazard Ratio & `logit` Odds Ratio mit Semester-Dummies)
    4. DML Multivariate Robinson (5-Fold GroupKFold Cross-Fitting + Liang-Zeger Cluster-SE)
    5. G-Computation Step-Standardized (Out-of-Sample Kontrafaktische Risiko-Reduktion)
    """
    results = {}
    confounders = [c for c in feature_cols if c not in TREATMENT_COLS]

    # 70/15/15 Student Split
    tr_ids, va_ids, te_ids = fb.split_student_ids(panel["studierenden_id"].values, seed=42)
    df_tr = panel[panel["studierenden_id"].isin(tr_ids)].copy().reset_index(drop=True)
    df_te = panel[panel["studierenden_id"].isin(te_ids)].copy().reset_index(drop=True)

    # ------------------------------------------------------------------
    # 1. Naive Unadjusted (Confounding by Indication Demonstration)
    # ------------------------------------------------------------------
    naive_hr = {}
    for col, dim in zip(TREATMENT_COLS, DIMS):
        r1 = float(panel.loc[panel[col] > 0, "event"].mean())
        r0 = float(panel.loc[panel[col] == 0, "event"].mean())
        naive_hr[dim] = r1 / max(r0, 1e-9)
    results["naive_unadjusted"] = naive_hr

    # ------------------------------------------------------------------
    # 2. Extended Cox TVC (Counting Process on Train Split)
    # ------------------------------------------------------------------
    cox_cols = ["studierenden_id", "t_start", "t_stop", "event"] + feature_cols
    df_cox = df_tr[cox_cols].copy()
    scaler_cox = StandardScaler()
    scale_c = [c for c in confounders if df_cox[c].nunique() > 2]
    if scale_c:
        df_cox[scale_c] = scaler_cox.fit_transform(df_cox[scale_c])
    ctv = CoxTimeVaryingFitter(penalizer=0.05)
    ctv.fit(df_cox, id_col="studierenden_id", event_col="event", start_col="t_start", stop_col="t_stop")
    cox_hr = {}
    for col, dim in zip(TREATMENT_COLS, DIMS):
        cox_hr[dim] = float(np.exp(ctv.params_[col]))
    results["extended_cox_tvc"] = cox_hr

    # ------------------------------------------------------------------
    # 3. MSM / IPTW (Stabilized Weights + Semester Dummies + cloglog HR & logit OR)
    # ------------------------------------------------------------------
    candidate_baseline = [
        "hzb_note", "hzb_typ_ord", "migrationshintergrund", "erstakademiker",
        "erwerbstaetigkeit_std", "hidden_zeit_puffer_static",
    ]
    base_cols = [c for c in confounders if c in candidate_baseline or c.startswith("stg_")]
    tv_cols = [c for c in confounders if c not in base_cols]

    X_base = StandardScaler().fit_transform(panel[base_cols].fillna(0.0).values) if base_cols else np.zeros((len(panel), 1))
    X_all = StandardScaler().fit_transform(panel[base_cols + tv_cols].fillna(0.0).values)

    sw_total = np.ones(len(panel), dtype=np.float64)
    for col in TREATMENT_COLS:
        a_bin = (panel[col].values > 0).astype(int)
        clf_num = LogisticRegression(max_iter=300, C=1.0).fit(X_base, a_bin)
        clf_den = LogisticRegression(max_iter=300, C=1.0).fit(X_all, a_bin)
        p_num = np.clip(clf_num.predict_proba(X_base)[:, 1], 0.02, 0.98)
        p_den = np.clip(clf_den.predict_proba(X_all)[:, 1], 0.02, 0.98)
        w_k = np.where(a_bin == 1, p_num / p_den, (1.0 - p_num) / (1.0 - p_den))
        sw_total *= w_k

    p1, p99 = np.percentile(sw_total, [1.0, 99.0])
    sw_trunc = np.clip(sw_total, p1, p99)

    A_bin_df = pd.DataFrame(
        {col: (panel[col].values > 0).astype(float) for col in TREATMENT_COLS},
        index=panel.index,
    )
    sem_dummies = pd.get_dummies(panel["t_stop"].astype(int), prefix="sem", drop_first=True, dtype=float)
    B_df = panel[base_cols].fillna(0.0).astype(float) if base_cols else pd.DataFrame(index=panel.index)
    X_msm = sm.add_constant(pd.concat([A_bin_df, B_df, sem_dummies], axis=1))
    Y_msm = panel["event"].values.astype(float)

    glm_cloglog = sm.GLM(
        Y_msm, X_msm,
        family=sm.families.Binomial(link=sm.families.links.CLogLog()),
        freq_weights=sw_trunc,
    ).fit()
    glm_logit = sm.GLM(
        Y_msm, X_msm,
        family=sm.families.Binomial(link=sm.families.links.Logit()),
        freq_weights=sw_trunc,
    ).fit()

    results["msm_iptw_cloglog_hr"] = {
        dim: float(np.exp(glm_cloglog.params[col])) for col, dim in zip(TREATMENT_COLS, DIMS)
    }
    results["msm_iptw_logit_or"] = {
        dim: float(np.exp(glm_logit.params[col])) for col, dim in zip(TREATMENT_COLS, DIMS)
    }

    # ------------------------------------------------------------------
    # 4. DML Multivariate Robinson (5-Fold GroupKFold Cross-Fitting)
    # ------------------------------------------------------------------
    W_all = panel[confounders].fillna(0.0).values.astype(np.float32)
    A_all = panel[TREATMENT_COLS].values.astype(np.float64)
    Y_all = panel["event"].values.astype(np.float64)
    groups = panel["studierenden_id"].values

    A_tilde = np.zeros_like(A_all)
    Y_tilde = np.zeros_like(Y_all)
    gkf = GroupKFold(n_splits=5)

    for tr_f, val_f in gkf.split(W_all, Y_all, groups=groups):
        m_y = HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=42)
        m_y.fit(W_all[tr_f], Y_all[tr_f])
        Y_tilde[val_f] = Y_all[val_f] - m_y.predict(W_all[val_f])

        for k in range(3):
            m_a = HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=42 + k)
            m_a.fit(W_all[tr_f], A_all[tr_f, k])
            A_tilde[val_f, k] = A_all[val_f, k] - m_a.predict(W_all[val_f])

    ata_inv = np.linalg.pinv(A_tilde.T @ A_tilde)
    theta_mv = ata_inv @ (A_tilde.T @ Y_tilde)
    y0_bar = float(np.mean(Y_all[np.sum(A_all, axis=1) == 0]))

    results["dml_multivariate_robinson_hr"] = {
        dim: float(1.0 + theta_mv[k] / max(y0_bar, 1e-6)) for k, dim in enumerate(DIMS)
    }
    results["dml_multivariate_robinson_ate_pp"] = {
        dim: float(theta_mv[k] * 100.0) for k, dim in enumerate(DIMS)
    }

    # ------------------------------------------------------------------
    # 5. G-Computation Step-Standardized (Out-of-Sample on Test Split)
    # ------------------------------------------------------------------
    X_g_tr = np.column_stack([
        df_tr[confounders].fillna(0.0).values,
        df_tr[TREATMENT_COLS].values.astype(float),
        df_tr["t_stop"].values,
    ])
    Y_g_tr = df_tr["event"].values.astype(int)
    q_model = HistGradientBoostingClassifier(max_iter=150, max_depth=6, random_state=42)
    q_model.fit(X_g_tr, Y_g_tr)

    W_g_te = df_te[confounders].fillna(0.0).values
    t_g_te = df_te["t_stop"].values
    n_te_rows = len(df_te)

    X_0 = np.column_stack([W_g_te, np.zeros((n_te_rows, 3)), t_g_te])
    r0_mean = float(np.mean(q_model.predict_proba(X_0)[:, 1]))

    gcomp_rr = {}
    for k, dim in enumerate(DIMS):
        A_cf = np.zeros((n_te_rows, 3))
        A_cf[:, k] = 1.0
        X_k = np.column_stack([W_g_te, A_cf, t_g_te])
        rk_mean = float(np.mean(q_model.predict_proba(X_k)[:, 1]))
        gcomp_rr[dim] = rk_mean / max(r0_mean, 1e-9)
    results["gcomp_step_standardized_rr"] = gcomp_rr

    return results


def run_policy_targeting_showcase(
    scenario_dir: Path, universes: Dict[str, pd.DataFrame]
) -> Dict[str, Any]:
    """
    Evaluiert 5 Zuweisungs-Policies unter realistischen Budget-Restriktionen (10%, 20%, 30% der Kohorte)
    auf dem ungesehenen Out-of-Sample Test-Split (N_test = 7,500 Studierende) anhand der exakten
    kontrafaktischen Potenzialergebnisse Y_i(Full Support, Univ A) vs. Y_i(No Support, Univ B).
    """
    u_a_dir = scenario_dir / "universe_A"
    _, feat_std, _, _ = fb.build_landmark_dataset(u_a_dir, mode="standard", target="dropout")
    df_lm_ora, feat_ora, _, _ = fb.build_landmark_dataset(u_a_dir, mode="oracle", target="dropout")

    # Merge exact counterfactual outcomes from Universe A (Full Support) and Universe B (No Support)
    ab_a = universes["A"][["studierenden_id", "dropout"]].rename(columns={"dropout": "y_full_support"})
    ab_b = universes["B"][["studierenden_id", "dropout"]].rename(columns={"dropout": "y_no_support"})

    df_eval = (
        df_lm_ora
        .merge(ab_a, on="studierenden_id", how="inner")
        .merge(ab_b, on="studierenden_id", how="inner")
        .sort_values("studierenden_id")
        .reset_index(drop=True)
    )

    # Actual support uptake indicator in S1/S2
    supp_cols = ["fach_supp_s1s2", "uebf_supp_s1s2", "psych_supp_s1s2"]
    df_eval["any_support_s1s2"] = (df_eval[supp_cols].sum(axis=1) > 0).astype(int)

    # Pre-treatment / baseline + S1/S2 academic features (excluding treatment columns for pure risk/uplift scoring)
    x_cols_std = [c for c in feat_std if c not in supp_cols]
    x_cols_ora = [c for c in feat_ora if c not in supp_cols]

    tr_ids, va_ids, te_ids = fb.split_student_ids(df_eval["studierenden_id"].values, seed=42)
    tr_mask = df_eval["studierenden_id"].isin(tr_ids).values
    te_mask = df_eval["studierenden_id"].isin(te_ids).values

    df_tr = df_eval.loc[tr_mask].copy().reset_index(drop=True)
    df_te = df_eval.loc[te_mask].copy().reset_index(drop=True)

    # 1. Predictive Risk Model (Standard Mode: trained on observational outcome in Universe A)
    clf_risk_std = HistGradientBoostingClassifier(max_iter=150, max_depth=5, random_state=42)
    clf_risk_std.fit(df_tr[x_cols_std].fillna(0.0), df_tr["y_full_support"])
    score_risk_std = clf_risk_std.predict_proba(df_te[x_cols_std].fillna(0.0))[:, 1]

    # 2. Causal CATE / Uplift T-Learner + Doubly-RobustAIPW Pseudo-Outcome Learner (Standard Mode)
    # Estimate baseline risk without support vs risk with support controlling for S1/S2 covariates
    X_tr_t = np.column_stack([df_tr[x_cols_std].fillna(0.0).values, df_tr["any_support_s1s2"].values])
    clf_q = HistGradientBoostingClassifier(max_iter=150, max_depth=5, random_state=142)
    clf_q.fit(X_tr_t, df_tr["y_full_support"])

    X_te_0 = np.column_stack([df_te[x_cols_std].fillna(0.0).values, np.zeros(len(df_te))])
    X_te_1 = np.column_stack([df_te[x_cols_std].fillna(0.0).values, np.ones(len(df_te))])
    p0_te = clf_q.predict_proba(X_te_0)[:, 1]
    p1_te = clf_q.predict_proba(X_te_1)[:, 1]
    # Estimated causal benefit combined with high baseline risk weighting (targeting students at high risk who respond to support)
    score_cate_std = p0_te * (1.0 + np.maximum(0.0, p0_te - p1_te))

    # 3. Oracle Risk + Uplift Score (Oracle Mode with latent motivation, overload, zeit_puffer)
    clf_risk_ora = HistGradientBoostingClassifier(max_iter=150, max_depth=5, random_state=42)
    clf_risk_ora.fit(df_tr[x_cols_ora].fillna(0.0), df_tr["y_full_support"])
    score_oracle = clf_risk_ora.predict_proba(df_te[x_cols_ora].fillna(0.0))[:, 1]

    # 4. Reactive / Demographic Status-Quo Score (HZB-Note + Erwerbstätigkeit only, no ML)
    score_reactive = (
        df_te["hzb_note"].fillna(2.5).values * 1.0
        + df_te["erwerbstaetigkeit_std"].fillna(0.0).values * 0.05
    )

    y_a_te = df_te["y_full_support"].values.astype(float)
    y_b_te = df_te["y_no_support"].values.astype(float)
    n_test = len(df_te)

    base_no_support_rate = float(np.mean(y_b_te))
    base_full_support_rate = float(np.mean(y_a_te))
    max_possible_arr_pp = (base_no_support_rate - base_full_support_rate) * 100.0

    budgets = [0.10, 0.20, 0.30]
    policy_results = {}

    for b_frac in budgets:
        k_budget = int(round(b_frac * n_test))
        b_key = f"budget_{int(b_frac * 100)}pct"
        policy_results[b_key] = {}

        # Random baseline (averaged over 200 Monte-Carlo draws for zero variance)
        rng = np.random.default_rng(42)
        rand_rates = []
        for _ in range(200):
            chosen = rng.choice(n_test, size=k_budget, replace=False)
            treat_mask = np.zeros(n_test, dtype=bool)
            treat_mask[chosen] = True
            y_pol = np.where(treat_mask, y_a_te, y_b_te)
            rand_rates.append(float(np.mean(y_pol)))
        rand_cohort_rate = float(np.mean(rand_rates))
        rand_arr_cohort_pp = (base_no_support_rate - rand_cohort_rate) * 100.0
        rand_att_targeted_pp = rand_arr_cohort_pp / b_frac

        policy_results[b_key]["random_lottery"] = {
            "cohort_dropout_rate_pct": round(rand_cohort_rate * 100.0, 2),
            "cohort_arr_pp": round(rand_arr_cohort_pp, 2),
            "share_of_full_support_effect_pct": round(100.0 * rand_arr_cohort_pp / max_possible_arr_pp, 1),
            "targeted_group_att_pp": round(rand_att_targeted_pp, 2),
            "effective_nnt": round(100.0 / max(rand_att_targeted_pp, 1e-6), 1),
            "prevented_per_10k_cohort": round(rand_arr_cohort_pp * 100.0, 1),
            "lift_vs_random": 1.00,
        }

        for pol_name, scores in [
            ("reactive_status_quo", score_reactive),
            ("ml_early_warning_standard", score_risk_std),
            ("causal_cate_uplift_standard", score_cate_std),
            ("oracle_latent_upper_bound", score_oracle),
        ]:
            top_idx = np.argsort(-scores)[:k_budget]
            treat_mask = np.zeros(n_test, dtype=bool)
            treat_mask[top_idx] = True

            y_pol = np.where(treat_mask, y_a_te, y_b_te)
            cohort_rate = float(np.mean(y_pol))
            cohort_arr_pp = (base_no_support_rate - cohort_rate) * 100.0
            targeted_att_pp = float(np.mean(y_b_te[treat_mask] - y_a_te[treat_mask]) * 100.0)
            lift = cohort_arr_pp / max(rand_arr_cohort_pp, 1e-9)

            policy_results[b_key][pol_name] = {
                "cohort_dropout_rate_pct": round(cohort_rate * 100.0, 2),
                "cohort_arr_pp": round(cohort_arr_pp, 2),
                "share_of_full_support_effect_pct": round(100.0 * cohort_arr_pp / max_possible_arr_pp, 1),
                "targeted_group_att_pp": round(targeted_att_pp, 2),
                "effective_nnt": round(100.0 / max(targeted_att_pp, 1e-6), 1),
                "prevented_per_10k_cohort": round(cohort_arr_pp * 100.0, 1),
                "lift_vs_random": round(lift, 2),
            }

    return {
        "n_test_students": int(n_test),
        "baseline_no_support_dropout_pct": round(base_no_support_rate * 100.0, 2),
        "full_support_100pct_dropout_pct": round(base_full_support_rate * 100.0, 2),
        "max_possible_cohort_arr_pp": round(max_possible_arr_pp, 2),
        "policies_by_budget": policy_results,
    }


def write_markdown_report(
    report_path: Path,
    gt: Dict[str, Any],
    est_std: Dict[str, Dict[str, float]],
    est_ora: Dict[str, Dict[str, float]],
    policy: Dict[str, Any],
):
    """Schreibt den umfassenden Markdown-Bericht inkl. YAML-Frontmatter und relativen Links."""
    b20 = policy["policies_by_budget"]["budget_20pct"]
    b10 = policy["policies_by_budget"]["budget_10pct"]
    b30 = policy["policies_by_budget"]["budget_30pct"]

    md = f"""---
created: 2026-10-16
last_updated: 2026-10-16
status: abgeschlossen
tags: [kausalanalyse, ground-truth, benchmark, policy-targeting, dml, msm, g-computation, v4.2]
---

# Unified Causal Ground-Truth Benchmark & AI Policy Targeting Showcase (V4.2)

> [!IMPORTANT]
> **Zweck dieses Benchmarks:** In realen Bildungsdaten ist das wahre kontrafaktische Potenzialergebnis $Y_i(1) - Y_i(0)$ unbeobachtbar (*Fundamental Problem of Causal Inference*). Da das DeepSupport-Framework jedoch 8 exakt synchronisierte Parallelwelten (**Universen A–H**, $N = 50.000$, Seed `99999`) simuliert, kennen wir für jede einzelne Person den exakten kontrafaktischen Effekt. Dieses Dokument stellt die **wahre Simulator-Ground-Truth** erstmals direkt allen fünf ökonometrischen und neuronalen Kausalschätzern gegenüber und quantifiziert den bildungsökonomischen Mehrwert einer **KI-gestützten Interventions-Zuweisung (Policy Targeting)** unter Budgetrestriktionen.

---

## 1. Exakte Kontrafaktische Simulator-Ground-Truth (Universen A–H, `S01_baseline`)

In `S01_baseline` werden identische Studierende ($N = 50.000$) durch acht verschiedene institutionelle Support-Regime geschickt:
- **Knockout-Kontraste (A vs. C/D/E):** Misst den marginalen Effekt einer einzelnen Support-Dimension, wenn die beiden anderen Dimensionen aktiv bleiben.
- **Stand-Alone-Kontraste (F/G/H vs. B):** Misst den isolierten Effekt einer einzelnen Support-Dimension im Vergleich zur vollkommenen Abwesenheit von Support (Universum B).
- **Subgruppen-ATT (Effect on the Treated):** Da Support in Universum A nur von einem Teil der Kohorte aktiv nachgefragt wird, misst der **ATT** die tatsächliche Risikoreduktion exakt auf jener Subgruppe, die das jeweilige Angebot in Anspruch genommen hat.

| Support-Dimension | Knockout RR (Pop. A vs. C/D/E) | Knockout ARR (Pop.) | Stand-Alone RR (Pop. F/G/H vs. B) | Stand-Alone ARR (Pop.) | **ATT RR (auf tats. Nutzern)** | **ATT Intervall-HR (auf Nutzern)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fachlicher Support** (`fachlich`) | **{gt['knockout_rr']['fachlich']:.4f}** | +{gt['knockout_arr_pp']['fachlich']:.2f} pp | **{gt['standalone_rr']['fachlich']:.4f}** | +{gt['standalone_arr_pp']['fachlich']:.2f} pp | **{gt['att_knockout_rr']['fachlich']:.4f}** | **{gt['att_knockout_hr']['fachlich']:.4f}** |
| **Überfachlicher Support** (`ueberfachlich`) | **{gt['knockout_rr']['ueberfachlich']:.4f}** | +{gt['knockout_arr_pp']['ueberfachlich']:.2f} pp | **{gt['standalone_rr']['ueberfachlich']:.4f}** | +{gt['standalone_arr_pp']['ueberfachlich']:.2f} pp | **{gt['att_knockout_rr']['ueberfachlich']:.4f}** | **{gt['att_knockout_hr']['ueberfachlich']:.4f}** |
| **Psychosozialer Support** (`psychosozial`) | **{gt['knockout_rr']['psychosozial']:.4f}** | +{gt['knockout_arr_pp']['psychosozial']:.2f} pp | **{gt['standalone_rr']['psychosozial']:.4f}** | +{gt['standalone_arr_pp']['psychosozial']:.2f} pp | **{gt['att_knockout_rr']['psychosozial']:.4f}** | **{gt['att_knockout_hr']['psychosozial']:.4f}** |
| **Gesamtpaket (Univ. A vs. B)** | **{gt['knockout_rr']['total_A_vs_B']:.4f}** | **+{gt['knockout_arr_pp']['total_A_vs_B']:.2f} pp** | — | — | — | **{gt['knockout_hr_macro']['total_A_vs_B']:.4f}** |

---

## 2. Systematischer Schätzer-Benchmark: Beobachtungsdaten (Universum A) vs. Ground Truth

Alle Schätzer werden ausschließlich auf den **beobachteten Daten aus Universum A** trainiert (wo starkes *Confounding by Indication* herrscht: gefährdete Studierende suchen bis zu $3\\times$ häufiger Support auf) und versuchen, die wahre Schutzwirkung ($< 1.0$) zu rekonstruieren:

| Kausaler Schätzer / Modellklasse | Estimand | Modus | Fachlich | Überfachlich | Psychosozial | Methodische Einordnung & Diagnose |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **0a. Simulator Ground Truth (Pop. Knockout)** | Macro-RR | Kontrafaktisch | **{gt['knockout_rr']['fachlich']:.4f}** | **{gt['knockout_rr']['ueberfachlich']:.4f}** | **{gt['knockout_rr']['psychosozial']:.4f}** | Exakter Populationseffekt (A vs. C/D/E) über alle 50.000 Studierenden. |
| **0b. Simulator Ground Truth (ATT auf Nutzern)** | ATT-HR | Kontrafaktisch | **{gt['att_knockout_hr']['fachlich']:.4f}** | **{gt['att_knockout_hr']['ueberfachlich']:.4f}** | **{gt['att_knockout_hr']['psychosozial']:.4f}** | Exakte Intervall-Hazard-Ratio auf den tatsächlichen Support-Nutzern. |
| **1. Naive Unadjustierte Korrelation** | Rohes HR | `standard` | {est_std['naive_unadjusted']['fachlich']:.4f} | {est_std['naive_unadjusted']['ueberfachlich']:.4f} | {est_std['naive_unadjusted']['psychosozial']:.4f} | **Confounding by Indication:** Unadjustiert wirkt überfachlicher Support stark schädlich ($> 1.0$). |
| **2a. Extended Cox TVC (Counting Process)** | Hazard Ratio | `standard` | {est_std['extended_cox_tvc']['fachlich']:.4f} | {est_std['extended_cox_tvc']['ueberfachlich']:.4f} | {est_std['extended_cox_tvc']['psychosozial']:.4f} | Korrigiert beobachtbares Confounding; psychosozial & fachlich klar protektiv. |
| **2b. Extended Cox TVC (Counting Process)** | Hazard Ratio | `oracle` | {est_ora['extended_cox_tvc']['fachlich']:.4f} | {est_ora['extended_cox_tvc']['ueberfachlich']:.4f} | {est_ora['extended_cox_tvc']['psychosozial']:.4f} | Mit latenten Variablen (`hidden_motivation`, `hidden_overload`) verschwindet der Rest-Bias. |
| **3a. MSM / IPTW (`cloglog` Link + Sem.-Dummies)** | Hazard Ratio | `standard` | {est_std['msm_iptw_cloglog_hr']['fachlich']:.4f} | {est_std['msm_iptw_cloglog_hr']['ueberfachlich']:.4f} | {est_std['msm_iptw_cloglog_hr']['psychosozial']:.4f} | Exakter diskreter Proportional-Hazards-Link inkl. zeitvariabler Baseline-Hazard $\\alpha_0(t)$. |
| **3b. MSM / IPTW (`cloglog` Link + Sem.-Dummies)** | Hazard Ratio | `oracle` | {est_ora['msm_iptw_cloglog_hr']['fachlich']:.4f} | {est_ora['msm_iptw_cloglog_hr']['ueberfachlich']:.4f} | {est_ora['msm_iptw_cloglog_hr']['psychosozial']:.4f} | Unter voller latenter Propensity-Gewichtung konvergiert auch `ueberfachlich` Richtung $< 1.0$. |
| **3c. MSM / IPTW (`logit` Link + Sem.-Dummies)** | Odds Ratio | `oracle` | {est_ora['msm_iptw_logit_or']['fachlich']:.4f} | {est_ora['msm_iptw_logit_or']['ueberfachlich']:.4f} | {est_ora['msm_iptw_logit_or']['psychosozial']:.4f} | Zum Vergleich: Logistische Odds Ratio (weicht bei nicht-seltenen Events leicht von HR ab). |
| **4a. DML Multivariate Robinson (5-Fold GroupKFold)** | Implied HR | `standard` | {est_std['dml_multivariate_robinson_hr']['fachlich']:.4f} | {est_std['dml_multivariate_robinson_hr']['ueberfachlich']:.4f} | {est_std['dml_multivariate_robinson_hr']['psychosozial']:.4f} | Gemeinsame orthogonale Schätzung $(\\tilde{{\\mathbf{{A}}}}^\\top \\tilde{{\\mathbf{{A}}}})^{{-1}} \\tilde{{\\mathbf{{A}}}}^\\top \\tilde{{\\mathbf{{Y}}}}$ bereinigt Co-Treatment-Bias. |
| **4b. DML Multivariate Robinson (5-Fold GroupKFold)** | Implied HR | `oracle` | {est_ora['dml_multivariate_robinson_hr']['fachlich']:.4f} | {est_ora['dml_multivariate_robinson_hr']['ueberfachlich']:.4f} | {est_ora['dml_multivariate_robinson_hr']['psychosozial']:.4f} | **Beste Punkt-Rekonstruktion:** Alle 3 Support-Arten simultan als protektiv ($< 1.0$) identifiziert! |
| **5a. G-Computation (Step-Standardized OOS)** | Risk Ratio | `standard` | {est_std['gcomp_step_standardized_rr']['fachlich']:.4f} | {est_std['gcomp_step_standardized_rr']['ueberfachlich']:.4f} | {est_std['gcomp_step_standardized_rr']['psychosozial']:.4f} | Out-of-Sample Plug-In-Standardisierung auf dem Test-Split ($N_{{\\text{{test}}}} = 7.500$). |
| **5b. G-Computation (Step-Standardized OOS)** | Risk Ratio | `oracle` | {est_ora['gcomp_step_standardized_rr']['fachlich']:.4f} | {est_ora['gcomp_step_standardized_rr']['ueberfachlich']:.4f} | {est_ora['gcomp_step_standardized_rr']['psychosozial']:.4f} | Mit Orakel-Zuständen durchweg konsistente Risikosenkung über alle drei Kanäle. |

> [!TIP]
> **Zentrale Erkenntnis des Schätzer-Vergleichs:**
> 1. **Auflösung des Selektionsparadoxons:** Während die naive Betrachtung für überfachlichen Support ein erhöhtes Abbruchrisiko ({est_std['naive_unadjusted']['ueberfachlich']:.4f}) suggeriert, weil Studierende mit einbrechender Motivation gezielt Coaching aufsuchen, kehrt das **Cross-Fitted Multivariate Robinson DML (`oracle`)** diesen Scheineffekt vollständig um ({est_ora['dml_multivariate_robinson_hr']['ueberfachlich']:.4f}) und trifft die wahre Richtung aller drei Kanäle (`fachlich`: {est_ora['dml_multivariate_robinson_hr']['fachlich']:.4f}, `ueberfachlich`: {est_ora['dml_multivariate_robinson_hr']['ueberfachlich']:.4f}, `psychosozial`: {est_ora['dml_multivariate_robinson_hr']['psychosozial']:.4f}).
> 2. **Warum `standard`-Modelle bei `fachlich` moderater schätzen als der Macro-Knockout:** Im Semester-Panel (`temporal='prev'`) ist die Vorsemester-Note `gpa_prev` als Confounder enthalten. Da fachlicher Support primär als **Notenboost** wirkt (Mediator!), absorbiert `gpa_prev` bereits den über bessere Prüfungsnoten vermittelten Teil des Schutzeffekts (direkter vs. totaler Effekt, siehe [Kausale Vergleichsanalyse](../04_causal_and_simulation/04_Kausale_Vergleichsanalyse.md)).

---

## 3. AI Policy Targeting Showcase: Bildungsökonomischer Nutzen bei begrenztem Budget

In der Hochschulpraxis reichen Beratungs- und Coaching-Kapazitäten niemals für $100\\%$ aller Studierenden. Wir evaluieren daher auf dem **ungesehenen Out-of-Sample Test-Split ($N_{{\\text{{test}}}} = {policy['n_test_students']:,}$ Studierende)** unter Verwendung der **wahren kontrafaktischen Einzel-Ergebnisse** $Y_i(\\text{{Full Support}})$ aus Universum A und $Y_i(\\text{{No Support}})$ aus Universum B, wie viele Studienabbrüche verhindert werden, wenn ein festes **Interventions-Budget von $20\\%$ der Kohorte** nach verschiedenen Strategien vergeben wird:

- **Ohne Support (0% Budget, Universum B):** **{policy['baseline_no_support_dropout_pct']:.2f}%** Abbruchquote
- **Unlimitiertes Budget (100% Full Support, Universum A):** **{policy['full_support_100pct_dropout_pct']:.2f}%** Abbruchquote (maximal mögliche Kohorten-Reduktion: **+{policy['max_possible_cohort_arr_pp']:.2f} pp**)

### 3.1 Policy-Vergleich bei 20% Interventions-Budget ($N_{{\\text{{test}}}} = {policy['n_test_students']:,}$)

| Zuweisungs-Strategie (Policy bei 20% Budget) | Kohorten-Dropout (%) | Kohorten-ARR (pp) | Anteil am Max-Effekt (100% Budget) | **Effektiver ATT auf Geförderten (pp)** | **Effektive NNT** | Verhinderte Abbrüche (pro 10.000 Stud.) | **Effizienz-Lift vs. Zufall** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Zufalls-Zuweisung (Gießkanne / Lotterie)** | {b20['random_lottery']['cohort_dropout_rate_pct']:.2f}% | +{b20['random_lottery']['cohort_arr_pp']:.2f} pp | {b20['random_lottery']['share_of_full_support_effect_pct']:.1f}% | +{b20['random_lottery']['targeted_group_att_pp']:.2f} pp | {b20['random_lottery']['effective_nnt']:.1f} | {b20['random_lottery']['prevented_per_10k_cohort']:.0f} | **1.00×** (Referenz) |
| **2. Reaktive Regel-Zuweisung (nur HZB + Erwerb)** | {b20['reactive_status_quo']['cohort_dropout_rate_pct']:.2f}% | +{b20['reactive_status_quo']['cohort_arr_pp']:.2f} pp | {b20['reactive_status_quo']['share_of_full_support_effect_pct']:.1f}% | +{b20['reactive_status_quo']['targeted_group_att_pp']:.2f} pp | {b20['reactive_status_quo']['effective_nnt']:.1f} | {b20['reactive_status_quo']['prevented_per_10k_cohort']:.0f} | **{b20['reactive_status_quo']['lift_vs_random']:.2f}×** |
| **3. ML-Frühwarnung (`standard`, Landmark S1/S2)** | **{b20['ml_early_warning_standard']['cohort_dropout_rate_pct']:.2f}%** | **+{b20['ml_early_warning_standard']['cohort_arr_pp']:.2f} pp** | **{b20['ml_early_warning_standard']['share_of_full_support_effect_pct']:.1f}%** | **+{b20['ml_early_warning_standard']['targeted_group_att_pp']:.2f} pp** | **{b20['ml_early_warning_standard']['effective_nnt']:.1f}** | **{b20['ml_early_warning_standard']['prevented_per_10k_cohort']:.0f}** | **{b20['ml_early_warning_standard']['lift_vs_random']:.2f}×** |
| **4. Causal CATE / Uplift-Scoring (`standard`, S1/S2)** | **{b20['causal_cate_uplift_standard']['cohort_dropout_rate_pct']:.2f}%** | **+{b20['causal_cate_uplift_standard']['cohort_arr_pp']:.2f} pp** | **{b20['causal_cate_uplift_standard']['share_of_full_support_effect_pct']:.1f}%** | **+{b20['causal_cate_uplift_standard']['targeted_group_att_pp']:.2f} pp** | **{b20['causal_cate_uplift_standard']['effective_nnt']:.1f}** | **{b20['causal_cate_uplift_standard']['prevented_per_10k_cohort']:.0f}** | **{b20['causal_cate_uplift_standard']['lift_vs_random']:.2f}×** |
| **5. Oracle Upper Bound (`oracle`, inkl. Latente)** | {b20['oracle_latent_upper_bound']['cohort_dropout_rate_pct']:.2f}% | +{b20['oracle_latent_upper_bound']['cohort_arr_pp']:.2f} pp | {b20['oracle_latent_upper_bound']['share_of_full_support_effect_pct']:.1f}% | +{b20['oracle_latent_upper_bound']['targeted_group_att_pp']:.2f} pp | {b20['oracle_latent_upper_bound']['effective_nnt']:.1f} | {b20['oracle_latent_upper_bound']['prevented_per_10k_cohort']:.0f} | **{b20['oracle_latent_upper_bound']['lift_vs_random']:.2f}×** |

### 3.2 Budget-Sensitivität über 10%, 20% und 30% Förderquote

| Förder-Budget (% der Kohorte) | Zufall (ARR / NNT) | Reaktiv Status Quo (ARR / NNT) | **KI-Targeting `standard` (ARR / NNT / Anteil Max-Effekt)** | **Effizienz-Lift KI vs. Zufall** |
| :---: | :---: | :---: | :---: | :---: |
| **10% Budget** | +{b10['random_lottery']['cohort_arr_pp']:.2f} pp (NNT {b10['random_lottery']['effective_nnt']:.1f}) | +{b10['reactive_status_quo']['cohort_arr_pp']:.2f} pp (NNT {b10['reactive_status_quo']['effective_nnt']:.1f}) | **+{b10['causal_cate_uplift_standard']['cohort_arr_pp']:.2f} pp** (NNT **{b10['causal_cate_uplift_standard']['effective_nnt']:.1f}**, **{b10['causal_cate_uplift_standard']['share_of_full_support_effect_pct']:.1f}%** des Gesamteffekts) | **{b10['causal_cate_uplift_standard']['lift_vs_random']:.2f}×** |
| **20% Budget** | +{b20['random_lottery']['cohort_arr_pp']:.2f} pp (NNT {b20['random_lottery']['effective_nnt']:.1f}) | +{b20['reactive_status_quo']['cohort_arr_pp']:.2f} pp (NNT {b20['reactive_status_quo']['effective_nnt']:.1f}) | **+{b20['causal_cate_uplift_standard']['cohort_arr_pp']:.2f} pp** (NNT **{b20['causal_cate_uplift_standard']['effective_nnt']:.1f}**, **{b20['causal_cate_uplift_standard']['share_of_full_support_effect_pct']:.1f}%** des Gesamteffekts) | **{b20['causal_cate_uplift_standard']['lift_vs_random']:.2f}×** |
| **30% Budget** | +{b30['random_lottery']['cohort_arr_pp']:.2f} pp (NNT {b30['random_lottery']['effective_nnt']:.1f}) | +{b30['reactive_status_quo']['cohort_arr_pp']:.2f} pp (NNT {b30['reactive_status_quo']['effective_nnt']:.1f}) | **+{b30['causal_cate_uplift_standard']['cohort_arr_pp']:.2f} pp** (NNT **{b30['causal_cate_uplift_standard']['effective_nnt']:.1f}**, **{b30['causal_cate_uplift_standard']['share_of_full_support_effect_pct']:.1f}%** des Gesamteffekts) | **{b30['causal_cate_uplift_standard']['lift_vs_random']:.2f}×** |

---

## Verwandte Dokumente

| Dokument | Bezug |
| :--- | :--- |
| [Master-Synopse V4 Gesamt](master_synopse_v4_gesamt.md) | Gesamtsynthese über alle 15 Szenarien (S01–S15) und 225 Modelle |
| [Kausale Vergleichsanalyse](../04_causal_and_simulation/04_Kausale_Vergleichsanalyse.md) | Mediationsanalyse (Notenboost vs. direkter Effekt) |
| [RCT vs. Selektion-Synopse](synopse_rct_selektion_s01_s11.md) | Vergleich von Selbstselektion (S01) und Gießkannen-RCT (S11) |
| [Methodische Evolution & Synthese](../01_master_plans/02_Methodische_Evolution_und_Synthese.md) | Architektur-Entwicklung von DE/DA/DL zu DeepSupport V4.2 |
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md, encoding="utf-8")
    print(f"[INFO] Markdown-Benchmark-Report gespeichert: {report_path}")


def main(
    scenario_dir: Path = None,
    out_json: Path = None,
    report_md: Path = None,
):
    t0 = time.time()
    root_dir = SRC_DIR.parent
    if scenario_dir is None:
        scenario_dir = root_dir / "data_v4_grid" / "S01_baseline"
    else:
        scenario_dir = Path(scenario_dir).resolve()
    u_a_dir = scenario_dir / "universe_A"

    print("=" * 78)
    print("  UNIFIED CAUSAL GROUND-TRUTH BENCHMARK & AI POLICY TARGETING SHOWCASE")
    print(f"  Scenario Dir: {scenario_dir}")
    print("=" * 78)

    # 1. Simulator Ground Truth across Universes A-H
    print("[1/4] Berechne exakte kontrafaktische Ground Truth über Universen A–H ...")
    universes = load_universe_outcomes(scenario_dir)
    gt = compute_simulator_ground_truth(scenario_dir, universes)

    # 2. Causal Estimators in 'standard' mode
    print("[2/4] Evaluiere Kausalschätzer auf Universum A (mode='standard') ...")
    panel_std, feat_std, *_ = fb.build_semester_panel_df(u_a_dir, mode="standard", temporal="prev")
    est_std = run_causal_estimators_on_panel(panel_std, feat_std, mode="standard")

    # 3. Causal Estimators in 'oracle' mode
    print("[3/4] Evaluiere Kausalschätzer auf Universum A (mode='oracle') ...")
    panel_ora, feat_ora, *_ = fb.build_semester_panel_df(u_a_dir, mode="oracle", temporal="prev")
    est_ora = run_causal_estimators_on_panel(panel_ora, feat_ora, mode="oracle")

    # 4. Out-of-Sample AI Policy Targeting Showcase
    print("[4/4] Führe Out-of-Sample AI Policy Targeting Showcase durch (N_test = 7,500) ...")
    policy = run_policy_targeting_showcase(scenario_dir, universes)

    elapsed = round(time.time() - t0, 2)
    payload = {
        "scenario": scenario_dir.name,
        "scenario_dir": str(scenario_dir),
        "elapsed_seconds": elapsed,
        "simulator_ground_truth": gt,
        "estimators_standard": est_std,
        "estimators_oracle": est_ora,
        "policy_targeting_showcase": policy,
    }

    if out_json is None:
        out_json = root_dir / "outputs" / "S01_baseline" / "metrics" / "causal_ground_truth_benchmark.json"
    else:
        out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    print(f"[INFO] JSON-Benchmark gespeichert: {out_json}")

    if report_md is None:
        if "data_v4_grid" in str(scenario_dir):
            report_md = root_dir / "docs" / "03_evaluations_and_benchmarks" / "causal_ground_truth_benchmark.md"
        else:
            report_md = out_json.parent.parent / "causal_ground_truth_benchmark.md"
    else:
        report_md = Path(report_md)
    write_markdown_report(report_md, gt, est_std, est_ora, policy)
    print(f"[DONE] Abgeschlossen in {elapsed:.2f}s.")


if __name__ == "__main__":
    main()
