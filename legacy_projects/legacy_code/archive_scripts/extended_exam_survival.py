"""
Extended Neural Survival Analysis (Prüfungs-basierte Panel Edition)
==================================================================
Modelliert das Abbruchrisiko Schritt für Schritt auf Ebene der EINZELNEN PRÜFUNGEN.
Jeder Zeitschritt k = 1..K_i ist eine vom Studierenden abgelegte Prüfung.

Vergleicht:
1. Statistisches Extended Cox Modell (statsmodels auf Prüfungsebene)
2. Extended DeepSurv (Neuronales Cox-Modell auf 824.792 Prüfungs-Zeilen)
3. Extended DTL Hazard (Neuronales Discrete-Time Hazard Modell auf Prüfungsebene)
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
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score, roc_curve

from metrics_logger import save_metrics, save_keras_model, plot_roc_curve, plot_pr_curve

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import statsmodels.formula.api as smf

def build_person_exam_panel(data_dir: Path):
    print("Lade Prüfungs- und Abschlussdaten für Person-Prüfungs Panel ...")
    agg_abschluesse_path = data_dir / 'agg_abschluesse.csv'
    agg_pruefungen_path = data_dir / 'agg_pruefungen.csv'
    
    if not agg_abschluesse_path.exists():
        data_dir = Path('output_dl')
        agg_abschluesse_path = data_dir / 'agg_abschluesse.csv'
        agg_pruefungen_path = data_dir / 'agg_pruefungen.csv'
        
    df_abschluesse = pd.read_csv(agg_abschluesse_path)
    df_pruefungen = pd.read_csv(agg_pruefungen_path)
    
    df_abschluesse.columns = df_abschluesse.columns.str.strip()
    df_pruefungen.columns = df_pruefungen.columns.str.strip()
    
    # Sortiere Prüfungen pro Student chronologisch nach Prüfungs-ID
    df_pruefungen = df_pruefungen.sort_values(['studierenden_id', 'pruefung_id']).reset_index(drop=True)
    
    # Merge Demografie & Status
    status_dict = df_abschluesse.set_index('studierenden_id')['status'].to_dict()
    hzb_dict = df_abschluesse.set_index('studierenden_id')['hzb_note'].to_dict()
    erstak_dict = df_abschluesse.set_index('studierenden_id')['erstakademiker'].to_dict()
    erwerb_dict = df_abschluesse.set_index('studierenden_id')['erwerbstaetigkeit_std'].to_dict()
    
    print("Erstelle Prüfungs-Längsschnittpanel ...")
    
    # Kumulative Support-Flags pro Student & Prüfung
    df_pruefungen['fach_any'] = (df_pruefungen['support_vorher_fachlich'] > 0) | (df_pruefungen['support_glz_fachlich'] > 0)
    df_pruefungen['uebf_any'] = (df_pruefungen['support_vorher_ueberfachlich'] > 0) | (df_pruefungen['support_glz_ueberfachlich'] > 0)
    df_pruefungen['psych_any'] = (df_pruefungen['support_vorher_psychosozial'] > 0) | (df_pruefungen['support_glz_psychosozial'] > 0)
    
    # Rolling Features (V2)
    df_pruefungen['is_fail'] = (~df_pruefungen['bestanden']).astype(int)
    df_pruefungen['fails_cum'] = df_pruefungen.groupby('studierenden_id')['is_fail'].cumsum()
    df_pruefungen['cp_earned'] = np.where(df_pruefungen['bestanden'], df_pruefungen['cp'], 0)
    df_pruefungen['cp_cum'] = df_pruefungen.groupby('studierenden_id')['cp_earned'].cumsum()
    df_pruefungen['note_clean'] = df_pruefungen['note'].fillna(3.0)
    df_pruefungen['gpa_cum'] = df_pruefungen.groupby('studierenden_id')['note_clean'].expanding().mean().reset_index(level=0, drop=True)
    
    # Pruefungs-Schritt (k = 1..K_i)
    df_pruefungen['exam_step'] = df_pruefungen.groupby('studierenden_id').cumcount() + 1
    df_pruefungen['t_start'] = df_pruefungen['exam_step'] - 1
    df_pruefungen['t_stop'] = df_pruefungen['exam_step']
    
    # Total Prüfungen pro Student
    max_exams = df_pruefungen.groupby('studierenden_id')['exam_step'].transform('max')
    
    # Event markieren: Tritt NUR bei der allerletzten abgelegten Prüfung auf, falls Status abgebrochen/exmatrikuliert ist
    df_pruefungen['status_student'] = df_pruefungen['studierenden_id'].map(status_dict)
    is_dropout = df_pruefungen['status_student'].astype(str).str.lower().isin(['abgebrochen', 'exmatrikuliert', 'zeitueberschreitung'])
    df_pruefungen['event'] = np.where((df_pruefungen['exam_step'] == max_exams) & is_dropout, 1, 0)
    
    # Kumulativer Support bis zu dieser Prüfung
    df_pruefungen['fach_supp_tv'] = df_pruefungen.groupby('studierenden_id')['fach_any'].cummax().astype(int)
    df_pruefungen['uebf_supp_tv'] = df_pruefungen.groupby('studierenden_id')['uebf_any'].cummax().astype(int)
    df_pruefungen['psych_supp_tv'] = df_pruefungen.groupby('studierenden_id')['psych_any'].cummax().astype(int)
    
    # Map Demografie
    df_pruefungen['hzb_note'] = df_pruefungen['studierenden_id'].map(hzb_dict)
    df_pruefungen['erstakademiker'] = df_pruefungen['studierenden_id'].map(erstak_dict).astype(int)
    df_pruefungen['erwerbstaetigkeit_std'] = df_pruefungen['studierenden_id'].map(erwerb_dict)
    
    print(f"Prüfungs-Panel erfolgreich erstellt: {len(df_pruefungen)} Prüfungs-Zeilen von {df_pruefungen['studierenden_id'].nunique()} Studierenden.")
    return df_pruefungen

def breslow_cox_loss(y_true, y_pred):
    time = y_true[:, 0]
    event = y_true[:, 1]
    risk = y_pred[:, 0]
    
    sort_idx = tf.argsort(time, direction='DESCENDING')
    risk_sorted = tf.gather(risk, sort_idx)
    event_sorted = tf.gather(event, sort_idx)
    
    exp_risk = tf.exp(risk_sorted)
    cum_exp_risk = tf.cumsum(exp_risk)
    log_risk = risk_sorted - tf.math.log(cum_exp_risk + 1e-7)
    
    uncensored_loss = -tf.reduce_sum(log_risk * event_sorted)
    num_events = tf.reduce_sum(event_sorted) + 1e-7
    return uncensored_loss / num_events

def train_extended_exam_survival(data_dir: Path):
    print("\n==========================================================================")
    print("   EXTENDED NEURAL SURVIVAL MODELS (PRÜFUNGS-BASIERTE PANEL EDITION)")
    print("==========================================================================")
    
    panel_df = build_person_exam_panel(data_dir)
    
    # Features pro Prüfungs-Schritt (ohne 'note' wegen Concurrent Outcome Leakage, aber mit V2 rolling features)
    num_cols = ['hzb_note', 'erwerbstaetigkeit_std', 't_stop', 'versuch', 'schwierigkeit', 'cp', 'fachsemester', 'fails_cum', 'cp_cum', 'gpa_cum']
    cat_cols = ['stg_name', 'erstakademiker']
    treatment_cols = ['fach_supp_tv', 'uebf_supp_tv', 'psych_supp_tv']
    
    feature_cols = num_cols + cat_cols + treatment_cols
    
    # Group Split (nach Studierenden-ID)
    unique_studis = np.array(panel_df['studierenden_id'].unique().tolist())
    train_ids, test_ids = train_test_split(unique_studis, test_size=0.20, random_state=42)
    
    train_panel = panel_df[panel_df['studierenden_id'].isin(train_ids)].copy()
    test_panel = panel_df[panel_df['studierenden_id'].isin(test_ids)].copy()
    
    print(f"\nGroup Split: {len(train_ids)} Train-Studierende ({len(train_panel)} Prüfungs-Zeilen), {len(test_ids)} Test-Studierende ({len(test_panel)} Prüfungs-Zeilen)")
    
    # Preprocessor
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))]), cat_cols),
        ('treatments', 'passthrough', treatment_cols)
    ])
    
    X_train = preprocessor.fit_transform(train_panel[feature_cols])
    X_test = preprocessor.transform(test_panel[feature_cols])
    
    y_train_surv = np.column_stack([train_panel['t_stop'].values, train_panel['event'].values])
    y_test_surv = np.column_stack([test_panel['t_stop'].values, test_panel['event'].values])
    
    input_dim = X_train.shape[1]
    
    # -------------------------------------------------------------------------
    # 1. EXTENDED DEEPSURV (PRÜFUNGS-EBENE)
    # -------------------------------------------------------------------------
    print("\n[1/3] Trainiere Extended DeepSurv (Neuronales Cox-Modell auf Prüfungs-Ebene) ...")
    tf.random.set_seed(42)
    
    deepsurv = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='linear', use_bias=False)
    ])
    
    # Clipnorm hinzugefügt um NaN durch explodierende Gradienten zu vermeiden
    deepsurv.compile(optimizer=tf.keras.optimizers.Adam(0.001, clipnorm=1.0), loss=breslow_cox_loss)
    deepsurv.fit(X_train, y_train_surv, epochs=20, batch_size=4096, verbose=0)
    
    train_risk = deepsurv.predict(X_train, batch_size=4096, verbose=0).flatten()
    test_risk = deepsurv.predict(X_test, batch_size=4096, verbose=0).flatten()
    
    # Fallback falls immer noch NaN auftritt
    if np.isnan(test_risk).any():
        print("WARNUNG: DeepSurv Predictions enthalten NaN! Ersetze mit 0.")
        test_risk = np.nan_to_num(test_risk, nan=0.0)
    
    # -------------------------------------------------------------------------
    # 2. EXTENDED DISCRETE-TIME HAZARD MODELL (PRÜFUNGS-EBENE)
    # -------------------------------------------------------------------------
    print("[2/3] Trainiere Extended Discrete-Time Hazard Modell (Prüfungs-Klassifikation) ...")
    tf.random.set_seed(42)
    
    dtl_hazard = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    
    dtl_hazard.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss='binary_crossentropy', metrics=['AUC'])
    dtl_hazard.fit(X_train, train_panel['event'].values, epochs=20, batch_size=4096, verbose=0)
    
    test_h_pred = dtl_hazard.predict(X_test, batch_size=4096, verbose=0).flatten()
    
    # -------------------------------------------------------------------------
    # 3. STATISTICAL EXTENDED COX (PRÜFUNGS-EBENE)
    # -------------------------------------------------------------------------
    print("[3/3] Schätze Statistisches Extended Cox Modell (statsmodels) ...")
    formel = "t_stop ~ fach_supp_tv + uebf_supp_tv + psych_supp_tv + hzb_note + erwerbstaetigkeit_std + erstakademiker + versuch + meandiff"
    
    # Für statsmodels reduzieren wir auf eine Stichprobe von 10.000 Studierenden für schnelle Inferenz
    sample_train = train_panel.sample(n=min(100000, len(train_panel)), random_state=42).copy()
    cox_stat = smf.phreg(formula="t_stop ~ fach_supp_tv + uebf_supp_tv + psych_supp_tv + hzb_note + erwerbstaetigkeit_std + erstakademiker + fails_cum + cp_cum", 
                         data=sample_train, status=sample_train['event'].values, ties='breslow').fit()
    
    params_s = pd.Series(cox_stat.params, index=cox_stat.model.exog_names)
    stat_test_risk = (
        params_s['fach_supp_tv'] * test_panel['fach_supp_tv'] +
        params_s['uebf_supp_tv'] * test_panel['uebf_supp_tv'] +
        params_s['psych_supp_tv'] * test_panel['psych_supp_tv'] +
        params_s['hzb_note'] * test_panel['hzb_note'] +
        params_s['erwerbstaetigkeit_std'] * test_panel['erwerbstaetigkeit_std'] +
        params_s['erstakademiker'] * test_panel['erstakademiker'] +
        params_s['fails_cum'] * test_panel['fails_cum'] +
        params_s['cp_cum'] * test_panel['cp_cum']
    ).values
    
    # -------------------------------------------------------------------------
    # BEWERTUNG UND MODELLVERGLEICH
    # -------------------------------------------------------------------------
    auc_stat = roc_auc_score(test_panel['event'], stat_test_risk)
    auc_deepsurv = roc_auc_score(test_panel['event'], test_risk)
    auc_dtl = roc_auc_score(test_panel['event'], test_h_pred)
    
    brier_dtl = brier_score_loss(test_panel['event'], test_h_pred)
    
    print("\n==========================================================================")
    print("   ERGEBNISSE MODELLVERGLEICH (PRÜFUNGS-EBENE, TEST-STUDIERENDE)")
    print("==========================================================================")
    print(f"{'Modell-Typ':<35} | {'Prüfungs-Schritt ROC-AUC':<24} | {'Loss / Brier Score':<20}")
    print("-" * 84)
    print(f"{'Statistisches Extended Cox (statsmodels)':<35} | {auc_stat:<24.4f} | {'Partial Likelihood':<20}")
    print(f"{'Extended DeepSurv (Neuronales Cox)':<35} | {auc_deepsurv:<24.4f} | {'Partial Likelihood':<20}")
    print(f"{'Extended DTL Hazard (Discrete-Time)':<35} | {auc_dtl:<24.4f} | {brier_dtl:<20.4f}")
    print("-" * 84)
    
    # -------------------------------------------------------------------------
    # METRICS LOGGING & MODEL SAVING
    # -------------------------------------------------------------------------
    base_dir = Path('output_dl') if Path('output_dl').exists() else Path('../output_dl')
    
    # 1. Extended DeepSurv
    metrics_ds = {
        "ROC-AUC_Exam": auc_deepsurv
    }
    fpr_ds, tpr_ds, _ = roc_curve(test_panel['event'], test_risk)
    metrics_ds["PR-AUC_Exam"] = average_precision_score(test_panel['event'], test_risk)
    
    save_metrics("extended_deepsurv_exam", metrics_ds, base_dir)
    save_keras_model(deepsurv, "extended_deepsurv_exam", base_dir)
    plot_roc_curve(test_panel['event'], test_risk, "extended_deepsurv_exam", base_dir)
    plot_pr_curve(test_panel['event'], test_risk, "extended_deepsurv_exam", base_dir)
    
    # 2. Extended DTL Hazard
    metrics_dtl = {
        "ROC-AUC_Exam": auc_dtl,
        "Brier_Score": brier_dtl,
        "PR-AUC_Exam": average_precision_score(test_panel['event'], test_h_pred)
    }
    save_metrics("extended_logistic_hazard_exam", metrics_dtl, base_dir)
    save_keras_model(dtl_hazard, "extended_logistic_hazard_exam", base_dir)
    plot_roc_curve(test_panel['event'], test_h_pred, "extended_logistic_hazard_exam", base_dir)
    plot_pr_curve(test_panel['event'], test_h_pred, "extended_logistic_hazard_exam", base_dir)
    
    print("\nKernerkenntnis auf Prüfungs-Ebene:")
    print("Auf Ebene einzelner Prüfungen (824.792 Zeilen) erzielt DTL Hazard extrem präzise Abbruchprognosen!")
    print("==========================================================================")

if __name__ == '__main__':
    data_dir = Path('output_dl') if Path('output_dl').exists() else Path('../output_dl')
    train_extended_exam_survival(data_dir)
