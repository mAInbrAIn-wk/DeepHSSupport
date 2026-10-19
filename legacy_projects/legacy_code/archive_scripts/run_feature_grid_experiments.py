"""
Feature Grid Master Evaluation & Benchmark Pipeline
===================================================
Trainiert und evaluiert die zentralen Modell-Klassen über das vollständige
Feature-Grid:
  1. standard    (Vollständige Baseline)
  2. gradeblind  (Ohne Notenmerkmale, nur CP & Fehlversuche)
  3. blind       (Präventiv: Ohne jegliche Verlaufsleistungen)
  4. oracle      (Ground-Truth Orakel mit latenten Variablen)
  5. realistic   (Datenschutzkonform: Ohne Migration, Erstakad., Erwerb, Psych. Supp)

Erfasst PR-AUC, ROC-AUC, Brier-Score sowie die Kausaleffekte (RR/HR für Fachlich, Überfachlich, Psychosozial).
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Masking, GRU, LayerNormalization, TimeDistributed,
    MultiHeadAttention, Add
)
import statsmodels.formula.api as smf

from feature_builder import (
    build_semester_sequence_tensor,
    build_exam_sequence_tensor,
    build_semester_panel_df,
    build_landmark_dataset,
    PADDING_VALUE
)
from recurrent_survival_model import masked_binary_crossentropy
from transformer_survival_model import PositionalEncoding
from metrics_logger import save_metrics

MODES = ['standard', 'gradeblind', 'blind', 'oracle', 'realistic']


# =========================================================================
# 1. SEMESTER GRU GRID
# =========================================================================
def evaluate_semester_gru_grid(data_dir: Path) -> Dict[str, Any]:
    print("\n" + "="*80)
    print("   [GRID EVALUATION] SEMESTER GRU DELTA (KLASSE 6)")
    print("="*80)
    
    results = {}
    
    for mode in MODES:
        print(f"\n>>> Trainiere Semester GRU im Modus: [{mode.upper()}] ...")
        studis, X_seq, y_seq, studi_events, f_names, f_indices = build_semester_sequence_tensor(
            data_dir, max_semesters=16, mode=mode
        )
        N, T, F = X_seq.shape
        
        train_idx, temp_idx, _, y_temp_event = train_test_split(
            np.arange(N), studi_events, test_size=0.30, random_state=42, stratify=studi_events
        )
        val_idx, test_idx, _, _ = train_test_split(
            temp_idx, y_temp_event, test_size=0.50, random_state=42, stratify=y_temp_event
        )
        
        X_train, X_val, X_test = X_seq[train_idx].copy(), X_seq[val_idx].copy(), X_seq[test_idx].copy()
        y_train, y_val, y_test = y_seq[train_idx], y_seq[val_idx], y_seq[test_idx]
        
        scaler = StandardScaler()
        valid_mask_train = (X_train[:, :, 0] != PADDING_VALUE)
        scaler.fit(X_train[valid_mask_train])
        
        for X_split in [X_train, X_val, X_test]:
            valid_mask = (X_split[:, :, 0] != PADDING_VALUE)
            X_split[valid_mask] = scaler.transform(X_split[valid_mask])
            
        tf.random.set_seed(42)
        model = Sequential([
            Input(shape=(T, F)),
            Masking(mask_value=PADDING_VALUE),
            GRU(32, return_sequences=True),
            LayerNormalization(),
            Dropout(0.2),
            TimeDistributed(Dense(16, activation='relu')),
            TimeDistributed(Dense(1, activation='sigmoid'))
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss=masked_binary_crossentropy)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=25, batch_size=256, verbose=0)
        
        preds = model.predict(X_test, verbose=0)
        mask = (y_test.flatten() != PADDING_VALUE)
        y_true_flat = y_test.flatten()[mask]
        y_pred_flat = preds.flatten()[mask]
        
        auc = float(roc_auc_score(y_true_flat, y_pred_flat))
        prauc = float(average_precision_score(y_true_flat, y_pred_flat))
        brier = float(brier_score_loss(y_true_flat, y_pred_flat))
        
        cf_results = {}
        valid_mask_test = (X_test[:, :, 0] != PADDING_VALUE)
        
        supp_configs = [
            ('fach', 'fach_supp', 'Fachlicher Support'),
            ('uebf', 'uebf_supp', 'Überfachlicher Support'),
            ('psych', 'psych_supp', 'Psychosozialer Support')
        ]
        
        for prefix, key, s_name in supp_configs:
            feat_idx = f_indices.get(key)
            if feat_idx is None:
                cf_results[f"{prefix}_partial"] = {"mean_rr": None, "median_rr": None}
                cf_results[f"{prefix}_isolated"] = {"mean_rr": None, "median_rr": None}
                continue
                
            # Partiell
            X_c_part = X_test.copy()
            X_t_part = X_test.copy()
            X_c_part[valid_mask_test, feat_idx] = 0.0
            
            p0_p = model.predict(X_c_part, verbose=0).flatten()[valid_mask_test.flatten()]
            p1_p = model.predict(X_t_part, verbose=0).flatten()[valid_mask_test.flatten()]
            rrs_p = p1_p / np.clip(p0_p, 1e-7, 1.0)
            
            # Isoliert
            X_c_iso = X_test.copy()
            X_t_iso = X_test.copy()
            for _, other_key, _ in supp_configs:
                other_idx = f_indices.get(other_key)
                if other_idx is not None:
                    X_c_iso[valid_mask_test, other_idx] = 0.0
                    X_t_iso[valid_mask_test, other_idx] = 0.0
            X_t_iso[valid_mask_test, feat_idx] = X_test[valid_mask_test, feat_idx]
            
            p0_i = model.predict(X_c_iso, verbose=0).flatten()[valid_mask_test.flatten()]
            p1_i = model.predict(X_t_iso, verbose=0).flatten()[valid_mask_test.flatten()]
            rrs_i = p1_i / np.clip(p0_i, 1e-7, 1.0)
            
            cf_results[f"{prefix}_partial"] = {"mean_rr": float(np.mean(rrs_p)), "median_rr": float(np.median(rrs_p))}
            cf_results[f"{prefix}_isolated"] = {"mean_rr": float(np.mean(rrs_i)), "median_rr": float(np.median(rrs_i))}
            
        print(f"   --> PR-AUC: {prauc:.4f} | ROC-AUC: {auc:.4f} | Brier: {brier:.4f}")
        if cf_results['fach_partial']['mean_rr'] is not None:
            print(f"       CF-RR Fach: {cf_results['fach_partial']['mean_rr']:.4f} (part) / {cf_results['fach_isolated']['mean_rr']:.4f} (iso)")
        if cf_results['uebf_partial']['mean_rr'] is not None:
            print(f"       CF-RR Uebf: {cf_results['uebf_partial']['mean_rr']:.4f} (part) / {cf_results['uebf_isolated']['mean_rr']:.4f} (iso)")
        if cf_results['psych_partial']['mean_rr'] is not None:
            print(f"       CF-RR Psych: {cf_results['psych_partial']['mean_rr']:.4f} (part) / {cf_results['psych_isolated']['mean_rr']:.4f} (iso)")
            
        results[mode] = {
            "n_features": F,
            "roc_auc": auc,
            "pr_auc": prauc,
            "brier_score": brier,
            "counterfactual": cf_results
        }
        
        save_metrics(f"grid_semester_gru_{mode}", results[mode], data_dir)
        
    return results


# =========================================================================
# 2. SEMESTER CAUSAL TRANSFORMER GRID
# =========================================================================
def evaluate_semester_transformer_grid(data_dir: Path) -> Dict[str, Any]:
    print("\n" + "="*80)
    print("   [GRID EVALUATION] SEMESTER CAUSAL TRANSFORMER (KLASSE 6)")
    print("="*80)
    
    results = {}
    
    for mode in MODES:
        print(f"\n>>> Trainiere Semester Transformer im Modus: [{mode.upper()}] ...")
        studis, X_seq, y_seq, studi_events, f_names, f_indices = build_semester_sequence_tensor(
            data_dir, max_semesters=16, mode=mode
        )
        N, T, F = X_seq.shape
        
        train_idx, temp_idx, _, y_temp_event = train_test_split(
            np.arange(N), studi_events, test_size=0.30, random_state=42, stratify=studi_events
        )
        val_idx, test_idx, _, _ = train_test_split(
            temp_idx, y_temp_event, test_size=0.50, random_state=42, stratify=y_temp_event
        )
        
        X_train, X_val, X_test = X_seq[train_idx].copy(), X_seq[val_idx].copy(), X_seq[test_idx].copy()
        y_train, y_val, y_test = y_seq[train_idx], y_seq[val_idx], y_seq[test_idx]
        
        scaler = StandardScaler()
        valid_mask_train = (X_train[:, :, 0] != PADDING_VALUE)
        scaler.fit(X_train[valid_mask_train])
        
        for X_split in [X_train, X_val, X_test]:
            valid_mask = (X_split[:, :, 0] != PADDING_VALUE)
            X_split[valid_mask] = scaler.transform(X_split[valid_mask])
            
        tf.random.set_seed(42)
        d_model = 32
        inputs = Input(shape=(T, F))
        masked_inputs = Masking(mask_value=PADDING_VALUE)(inputs)
        x_proj = Dense(d_model)(masked_inputs)
        x_pos = PositionalEncoding(sequence_length=T, d_model=d_model)(x_proj)
        
        attn_out = MultiHeadAttention(num_heads=2, key_dim=d_model, dropout=0.1)(
            query=x_pos, value=x_pos, use_causal_mask=True
        )
        x_norm1 = LayerNormalization()(Add()([x_pos, attn_out]))
        ffn = Dense(64, activation='relu')(x_norm1)
        ffn = Dense(d_model)(ffn)
        x_norm2 = LayerNormalization()(Add()([x_norm1, ffn]))
        outputs = TimeDistributed(Dense(1, activation='sigmoid'))(x_norm2)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(0.003), loss=masked_binary_crossentropy)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=256, verbose=0)
        
        preds = model.predict(X_test, verbose=0)
        mask = (y_test.flatten() != PADDING_VALUE)
        y_true_flat = y_test.flatten()[mask]
        y_pred_flat = preds.flatten()[mask]
        
        auc = float(roc_auc_score(y_true_flat, y_pred_flat))
        prauc = float(average_precision_score(y_true_flat, y_pred_flat))
        brier = float(brier_score_loss(y_true_flat, y_pred_flat))
        
        cf_results = {}
        valid_mask_test = (X_test[:, :, 0] != PADDING_VALUE)
        supp_configs = [
            ('fach', 'fach_supp', 'Fachlicher Support'),
            ('uebf', 'uebf_supp', 'Überfachlicher Support'),
            ('psych', 'psych_supp', 'Psychosozialer Support')
        ]
        
        for prefix, key, s_name in supp_configs:
            feat_idx = f_indices.get(key)
            if feat_idx is None:
                cf_results[f"{prefix}_partial"] = {"mean_rr": None, "median_rr": None}
                cf_results[f"{prefix}_isolated"] = {"mean_rr": None, "median_rr": None}
                continue
                
            X_c_part = X_test.copy()
            X_t_part = X_test.copy()
            X_c_part[valid_mask_test, feat_idx] = 0.0
            
            p0_p = model.predict(X_c_part, verbose=0).flatten()[valid_mask_test.flatten()]
            p1_p = model.predict(X_t_part, verbose=0).flatten()[valid_mask_test.flatten()]
            rrs_p = p1_p / np.clip(p0_p, 1e-7, 1.0)
            
            X_c_iso = X_test.copy()
            X_t_iso = X_test.copy()
            for _, other_key, _ in supp_configs:
                other_idx = f_indices.get(other_key)
                if other_idx is not None:
                    X_c_iso[valid_mask_test, other_idx] = 0.0
                    X_t_iso[valid_mask_test, other_idx] = 0.0
            X_t_iso[valid_mask_test, feat_idx] = X_test[valid_mask_test, feat_idx]
            
            p0_i = model.predict(X_c_iso, verbose=0).flatten()[valid_mask_test.flatten()]
            p1_i = model.predict(X_t_iso, verbose=0).flatten()[valid_mask_test.flatten()]
            rrs_i = p1_i / np.clip(p0_i, 1e-7, 1.0)
            
            cf_results[f"{prefix}_partial"] = {"mean_rr": float(np.mean(rrs_p)), "median_rr": float(np.median(rrs_p))}
            cf_results[f"{prefix}_isolated"] = {"mean_rr": float(np.mean(rrs_i)), "median_rr": float(np.median(rrs_i))}
            
        print(f"   --> PR-AUC: {prauc:.4f} | ROC-AUC: {auc:.4f} | Brier: {brier:.4f}")
        results[mode] = {
            "n_features": F,
            "roc_auc": auc,
            "pr_auc": prauc,
            "brier_score": brier,
            "counterfactual": cf_results
        }
        save_metrics(f"grid_semester_transformer_{mode}", results[mode], data_dir)
        
    return results


# =========================================================================
# 3. SEMESTER PANEL COX & LOGISTIC HAZARD GRID
# =========================================================================
def evaluate_semester_panel_grid(data_dir: Path) -> Dict[str, Any]:
    print("\n" + "="*80)
    print("   [GRID EVALUATION] SEMESTER PANEL MODELS (EXTENDED COX & LOGISTIC HAZARD)")
    print("="*80)
    
    cox_results = {}
    hazard_results = {}
    
    for mode in MODES:
        print(f"\n>>> Trainiere Semester Panel im Modus: [{mode.upper()}] ...")
        panel_df, f_cols, target_col, f_indices = build_semester_panel_df(data_dir, mode=mode)
        
        # 1. Extended Cox (Statsmodels PHReg)
        formula = f"t_stop ~ {' + '.join(f_cols)}"
        try:
            mod_cox = smf.phreg(
                formula=formula,
                data=panel_df,
                status=panel_df['event'].values,
                entry=panel_df['t_start'].values,
                ties='breslow'
            )
            res_cox = mod_cox.fit()
            params_s = pd.Series(res_cox.params, index=res_cox.model.exog_names)
            
            hr_fach = float(np.exp(params_s.get('fach_supp_count', 0.0)))
            hr_uebf = float(np.exp(params_s.get('uebf_supp_count', 0.0)))
            hr_psych = float(np.exp(params_s.get('psych_supp_count', 0.0))) if 'psych_supp_count' in f_cols else None
            
            cox_results[mode] = {
                "n_features": len(f_cols),
                "hr_fach": hr_fach,
                "hr_uebf": hr_uebf,
                "hr_psych": hr_psych
            }
            print(f"   [Cox PHReg] HR Fach: {hr_fach:.4f} | HR Uebf: {hr_uebf:.4f} | HR Psych: {hr_psych if hr_psych is not None else 'N/A'}")
        except Exception as e:
            print(f"   [Cox PHReg Error in {mode}]: {e}")
            cox_results[mode] = {"error": str(e)}
            
        # 2. Extended Logistic Hazard (Neural)
        studis = panel_df['studierenden_id'].unique()
        train_studis, test_studis = train_test_split(studis, test_size=0.30, random_state=42)
        
        train_df = panel_df[panel_df['studierenden_id'].isin(train_studis)].copy()
        test_df = panel_df[panel_df['studierenden_id'].isin(test_studis)].copy()
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[f_cols].values)
        X_test = scaler.transform(test_df[f_cols].values)
        y_train = train_df['event'].values
        y_test = test_df['event'].values
        
        haz_model = Sequential([
            Input(shape=(len(f_cols),)),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        haz_model.compile(optimizer='adam', loss='binary_crossentropy')
        haz_model.fit(X_train, y_train, epochs=10, batch_size=512, verbose=0)
        
        p_pred = haz_model.predict(X_test, verbose=0).flatten()
        auc = float(roc_auc_score(y_test, p_pred))
        prauc = float(average_precision_score(y_test, p_pred))
        brier = float(brier_score_loss(y_test, p_pred))
        
        cf_haz = {}
        for prefix, col_name in [('fach', 'fach_supp_count'), ('uebf', 'uebf_supp_count'), ('psych', 'psych_supp_count')]:
            if col_name not in f_cols:
                cf_haz[prefix] = None
                continue
            idx = f_cols.index(col_name)
            
            X_c = X_test.copy()
            X_t = X_test.copy()
            X_c[:, idx] = 0.0
            
            p0 = haz_model.predict(X_c, verbose=0).flatten()
            p1 = haz_model.predict(X_t, verbose=0).flatten()
            rrs = p1 / np.clip(p0, 1e-7, 1.0)
            cf_haz[prefix] = float(np.mean(rrs))
            
        print(f"   [Neural Hazard] PR-AUC: {prauc:.4f} | ROC-AUC: {auc:.4f} | RR Fach: {cf_haz.get('fach', 'N/A')}")
        hazard_results[mode] = {
            "n_features": len(f_cols),
            "roc_auc": auc,
            "pr_auc": prauc,
            "brier_score": brier,
            "counterfactual": cf_haz
        }
        
    return {"cox": cox_results, "hazard": hazard_results}


# =========================================================================
# 4. EXAM GRU V2 GRID
# =========================================================================
def evaluate_exam_gru_grid(data_dir: Path) -> Dict[str, Any]:
    print("\n" + "="*80)
    print("   [GRID EVALUATION] EXAM GRU V2 (KLASSE 7)")
    print("="*80)
    
    results = {}
    
    for mode in MODES:
        print(f"\n>>> Trainiere Exam GRU im Modus: [{mode.upper()}] ...")
        studis, X_seq, y_seq, studi_events, f_names, f_indices = build_exam_sequence_tensor(
            data_dir, max_exams=50, mode=mode
        )
        N, K_max, F = X_seq.shape
        
        train_idx, temp_idx, _, y_temp_event = train_test_split(
            np.arange(N), studi_events, test_size=0.30, random_state=42, stratify=studi_events
        )
        val_idx, test_idx, _, _ = train_test_split(
            temp_idx, y_temp_event, test_size=0.50, random_state=42, stratify=y_temp_event
        )
        
        X_train, X_val, X_test = X_seq[train_idx].copy(), X_seq[val_idx].copy(), X_seq[test_idx].copy()
        y_train, y_val, y_test = y_seq[train_idx], y_seq[val_idx], y_seq[test_idx]
        
        scaler = StandardScaler()
        valid_mask_train = (X_train[:, :, 0] != PADDING_VALUE)
        scaler.fit(X_train[valid_mask_train])
        
        for X_split in [X_train, X_val, X_test]:
            valid_mask = (X_split[:, :, 0] != PADDING_VALUE)
            X_split[valid_mask] = scaler.transform(X_split[valid_mask])
            
        tf.random.set_seed(42)
        model = Sequential([
            Input(shape=(K_max, F)),
            Masking(mask_value=PADDING_VALUE),
            GRU(32, return_sequences=True),
            LayerNormalization(),
            Dropout(0.2),
            TimeDistributed(Dense(16, activation='relu')),
            TimeDistributed(Dense(1, activation='sigmoid'))
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss=masked_binary_crossentropy)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=256, verbose=0)
        
        preds = model.predict(X_test, verbose=0)
        mask = (y_test.flatten() != PADDING_VALUE)
        y_true_flat = y_test.flatten()[mask]
        y_pred_flat = preds.flatten()[mask]
        
        auc = float(roc_auc_score(y_true_flat, y_pred_flat))
        prauc = float(average_precision_score(y_true_flat, y_pred_flat))
        brier = float(brier_score_loss(y_true_flat, y_pred_flat))
        
        print(f"   --> PR-AUC: {prauc:.4f} | ROC-AUC: {auc:.4f} | Brier: {brier:.4f}")
        results[mode] = {
            "n_features": F,
            "roc_auc": auc,
            "pr_auc": prauc,
            "brier_score": brier
        }
        save_metrics(f"grid_exam_gru_{mode}", results[mode], data_dir)
        
    return results


def generate_master_markdown_report(master_summary: Dict[str, Any], output_path: Path):
    md_lines = [
        "# Master Feature-Grid Benchmark Report",
        "",
        "Dieses Dokument enthält die systematische Gegenüberstellung aller Modell-Klassen und Feature-Modi.",
        "",
        "| Modell-Klasse | Feature-Modus | Features | PR-AUC | ROC-AUC | Brier-Score | RR Fachlich (part/iso) | RR Überfachlich (part/iso) | RR Psychosozial (part/iso) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    
    for model_name, model_data in master_summary.items():
        if model_name == "panel_models":
            for sub_name, sub_data in model_data.items():
                for mode, metrics in sub_data.items():
                    n_f = metrics.get("n_features", "-")
                    pr = f"{metrics.get('pr_auc', 0.0):.4f}" if metrics.get('pr_auc') is not None else "-"
                    roc = f"{metrics.get('roc_auc', 0.0):.4f}" if metrics.get('roc_auc') is not None else "-"
                    br = f"{metrics.get('brier_score', 0.0):.4f}" if metrics.get('brier_score') is not None else "-"
                    
                    cf = metrics.get("counterfactual", {})
                    fach = f"{cf.get('fach_partial', {}).get('mean_rr', '-')}/{cf.get('fach_isolated', {}).get('mean_rr', '-')}"
                    uebf = f"{cf.get('uebf_partial', {}).get('mean_rr', '-')}/{cf.get('uebf_isolated', {}).get('mean_rr', '-')}"
                    psych = f"{cf.get('psych_partial', {}).get('mean_rr', '-')}/{cf.get('psych_isolated', {}).get('mean_rr', '-')}"
                    
                    md_lines.append(f"| Panel: {sub_name.upper()} | {mode} | {n_f} | {pr} | {roc} | {br} | {fach} | {uebf} | {psych} |")
        else:
            for mode, metrics in model_data.items():
                n_f = metrics.get("n_features", "-")
                pr = f"{metrics.get('pr_auc', 0.0):.4f}" if metrics.get('pr_auc') is not None else "-"
                roc = f"{metrics.get('roc_auc', 0.0):.4f}" if metrics.get('roc_auc') is not None else "-"
                br = f"{metrics.get('brier_score', 0.0):.4f}" if metrics.get('brier_score') is not None else "-"
                
                cf = metrics.get("counterfactual", {})
                fach = f"{cf.get('fach_partial', {}).get('mean_rr', '-')}/{cf.get('fach_isolated', {}).get('mean_rr', '-')}" if cf else "-"
                uebf = f"{cf.get('uebf_partial', {}).get('mean_rr', '-')}/{cf.get('uebf_isolated', {}).get('mean_rr', '-')}" if cf else "-"
                psych = f"{cf.get('psych_partial', {}).get('mean_rr', '-')}/{cf.get('psych_isolated', {}).get('mean_rr', '-')}" if cf else "-"
                
                md_lines.append(f"| {model_name.replace('_', ' ').title()} | {mode} | {n_f} | {pr} | {roc} | {br} | {fach} | {uebf} | {psych} |")
                
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines) + "\n")
    print(f" Markdown-Report erfolgreich gespeichert unter: {output_path}")


def main(data_dir: Optional[Union[str, Path]] = None):
    if data_dir is not None:
        data_dir = Path(data_dir)
    elif os.environ.get('DATA_DIR'):
        data_dir = Path(os.environ['DATA_DIR'])
    else:
        data_dir = Path('output_dl')
        for candidate in [Path('output_dl'), Path('src/output_dl'), Path('../output_dl')]:
            if (candidate / 'agg_abschluesse.csv').exists():
                data_dir = candidate
                break
            
    print(f"Starte Master Grid Evaluation Pipeline (Data Dir: {data_dir}) ...")
    
    sem_gru_res = evaluate_semester_gru_grid(data_dir)
    sem_trans_res = evaluate_semester_transformer_grid(data_dir)
    panel_res = evaluate_semester_panel_grid(data_dir)
    exam_gru_res = evaluate_exam_gru_grid(data_dir)
    
    master_summary = {
        "semester_gru": sem_gru_res,
        "semester_transformer": sem_trans_res,
        "panel_models": panel_res,
        "exam_gru": exam_gru_res
    }
    
    metrics_dir = data_dir / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_path_json = metrics_dir / 'feature_grid_master_benchmark.json'
    summary_path_md = metrics_dir / 'feature_grid_master_benchmark.md'
    
    with open(summary_path_json, 'w', encoding='utf-8') as f:
        json.dump(master_summary, f, indent=4)
        
    generate_master_markdown_report(master_summary, summary_path_md)
    print(f"\n Master Feature-Grid Benchmark JSON erfolgreich gespeichert unter: {summary_path_json}")

if __name__ == '__main__':
    main()