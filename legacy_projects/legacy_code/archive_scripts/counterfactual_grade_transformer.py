"""
Kontrafaktische Noteninferenz für Deep Exam Transformer Regressor
==================================================================
Evaluiert den Kausaleffekt der Supportmaßnahmen auf die vorhergesagte
Abschlussnote / Prüfungsnote unter Verwendung des hochkapazitären
Deep Exam Transformer Regressors (R² ≈ 0.90).

Berechnet die erwartete Notendifferenz (Delta Note = Treated - Control)
im partiellen und isolierten Strang.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import feature_builder as fb
from transformer_survival_model import PositionalEncoding
from metrics_logger import save_metrics

def analyze_counterfactual_grade_transformer(data_dir: Path):
    print("\n==========================================================================")
    print("   COUNTERFACTUAL GRADE ANALYSIS (EXAM TRANSFORMER REGRESSOR)")
    print("==========================================================================")
    
    data_dir = Path(data_dir)
    if not data_dir.exists() and (Path('src') / data_dir).exists():
        data_dir = Path('src') / data_dir
        
    possible_models = [
        data_dir / "models" / "timeseries_exam_transformer.keras",
        data_dir / "models" / "timeseries_exam_transformer_prev_gradeblind.keras",
        data_dir / "models" / "timeseries_exam_transformer_cum_standard.keras",
        data_dir / "models" / "timeseries_exam_transformer_cum_gradeblind.keras",
        data_dir / "models" / "deep_exam_transformer_regressor.keras"
    ]
    model_path = None
    for mp in possible_models:
        if mp.exists():
            model_path = mp
            break
            
    if model_path is None:
        print(f"Kein passendes Exam-Transformer-Modell in {data_dir}/models gefunden!")
        return
        
    # Load model first to detect expected feature dimension
    custom_objects = {'PositionalEncoding': PositionalEncoding}
    try:
        from deep_transformer_regression import AttentionPooling
        custom_objects['AttentionPooling'] = AttentionPooling
    except Exception:
        pass
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"Modell geladen: {model_path.name} | Input Shape: {model.input_shape}")
    expected_dim = model.input_shape[-1]
    eval_mode = 'standard' if expected_dim == 24 else 'gradeblind'

    print(f"Lade Prüfungs-Zeitreihen-Datensatz (mode={eval_mode}, expected_dim={expected_dim}) ...")
    studis, X_exam_3d, y_seq, studi_events, feature_names, PADDING_VALUE = fb.build_exam_sequence_tensor(
        data_dir, max_exams=40, mode=eval_mode, temporal='prev', target_type='grade'
    )
    
    df_abschluesse, _ = fb._load_raw_data(data_dir)
    note_dict = df_abschluesse.set_index('studierenden_id')['abschlussnote'].to_dict()
    y_student = np.array([note_dict.get(s, np.nan) for s in studis])
    valid_grad_mask = ~np.isnan(y_student)
    
    X_exam_3d = X_exam_3d[valid_grad_mask]
    y_gpa_ex = y_student[valid_grad_mask]
    
    X_tr_e, X_temp_e, y_tr_e, y_temp_e = train_test_split(X_exam_3d, y_gpa_ex, test_size=0.30, random_state=42)
    X_va_e, X_te_e, y_va_e, y_te_e = train_test_split(X_temp_e, y_temp_e, test_size=0.50, random_state=42)
    
    scaler = StandardScaler()
    valid_mask_tr = (X_tr_e[:, :, 0] != PADDING_VALUE)
    scaler.fit(X_tr_e[valid_mask_tr])
    valid_mask_te = (X_te_e[:, :, 0] != PADDING_VALUE)
    
    # Support Feature Indizes im 3D Exam Tensor (aus timeseries_exam.py)
    # [0:fachsem, 1:versuch, 2:cp, 3:schwierigkeit, 4:supp_vor_fach, 5:supp_vor_uebf, 6:supp_vor_psych, 7:supp_glz_fach, 8:supp_glz_uebf, 9:supp_glz_psych, 10:fails_lag, 11:cp_lag, ...]
    ALL_SUPP_IDXS = [4, 5, 6, 7, 8, 9]
    
    metrics_all = {}
    
    for (idx_vor, idx_glz), prefix, label in [
        ((4, 7), 'fach',  'Fachlicher Support'),
        ((5, 8), 'uebf',  'Überfachlicher Support'),
        ((6, 9), 'psych', 'Psychosozialer Support'),
    ]:
        # 1. PARTIELL: Ziel-Support auf 0 setzen vs. beobachtet lassen
        X_c_p = X_te_e.copy()
        X_t_p = X_te_e.copy()
        X_c_p[valid_mask_te, idx_vor] = 0.0
        X_c_p[valid_mask_te, idx_glz] = 0.0
        
        X_c_p[valid_mask_te] = scaler.transform(X_c_p[valid_mask_te])
        X_t_p[valid_mask_te] = scaler.transform(X_t_p[valid_mask_te])
        
        pred_c_p = model.predict(X_c_p, verbose=0).flatten()
        pred_t_p = model.predict(X_t_p, verbose=0).flatten()
        
        diffs_p = pred_t_p - pred_c_p # Negativ = Note verbessert sich durch Support
        mean_diff_p   = float(np.mean(diffs_p))
        median_diff_p = float(np.median(diffs_p))
        q05_p         = float(np.quantile(diffs_p, 0.05))
        q95_p         = float(np.quantile(diffs_p, 0.95))
        
        # 2. ISOLIERT REALISTISCH: Alle Supports 0 vs. nur Ziel beobachtet
        X_c_i = X_te_e.copy()
        X_t_i = X_te_e.copy()
        for idx in ALL_SUPP_IDXS:
            X_c_i[valid_mask_te, idx] = 0.0
            X_t_i[valid_mask_te, idx] = 0.0
        X_t_i[valid_mask_te, idx_vor] = X_te_e[valid_mask_te, idx_vor]
        X_t_i[valid_mask_te, idx_glz] = X_te_e[valid_mask_te, idx_glz]
        
        X_c_i[valid_mask_te] = scaler.transform(X_c_i[valid_mask_te])
        X_t_i[valid_mask_te] = scaler.transform(X_t_i[valid_mask_te])
        
        pred_c_i = model.predict(X_c_i, verbose=0).flatten()
        pred_t_i = model.predict(X_t_i, verbose=0).flatten()
        
        diffs_i = pred_t_i - pred_c_i
        mean_diff_i   = float(np.mean(diffs_i))
        median_diff_i = float(np.median(diffs_i))
        q05_i         = float(np.quantile(diffs_i, 0.05))
        q95_i         = float(np.quantile(diffs_i, 0.95))
        
        print(f"\n--- {label} ({prefix}) ---")
        print(f"  PARTIELL:           Mean Delta = {mean_diff_p:+.4f} Notenpunkte [{q05_p:+.4f}, {q95_p:+.4f}]")
        print(f"  ISOLIERT (realist): Mean Delta = {mean_diff_i:+.4f} Notenpunkte [{q05_i:+.4f}, {q95_i:+.4f}]")
        
        metrics_all[f"{prefix}_partial"] = {"mean_delta_note": mean_diff_p, "median_delta_note": median_diff_p, "q05": q05_p, "q95": q95_p}
        metrics_all[f"{prefix}_isolated"] = {"mean_delta_note": mean_diff_i, "median_delta_note": median_diff_i, "q05": q05_i, "q95": q95_i}
        
    save_metrics("counterfactual_grade_transformer_metrics", metrics_all, data_dir)
    print(f"\nErgebnisse gespeichert in: {data_dir / 'metrics' / 'counterfactual_grade_transformer_metrics.json'}")

main = analyze_counterfactual_grade_transformer

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='output_dl')
    args = parser.parse_args()
    main(Path(args.data_dir))
