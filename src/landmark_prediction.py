import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier as XGBClassifier, HistGradientBoostingRegressor as XGBRegressor
from sklearn.metrics import r2_score, accuracy_score, classification_report, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

import deepsupport.data_engine.feature_builder as fb
from deepsupport.models.autoregressive_transformer import SinCosPositionalEncoding
from deepsupport.models.autoregressive_gru import PADDING_VALUE

from deepsupport.evaluation.metrics_logger import MulticlassEvaluator, RegressionEvaluator, save_metrics

def build_landmark_dataset(df_ab, df_pr, max_semester=2, max_seq_len=30):
    # Nur Studierende, die das Landmark-Semester überlebt haben (Studiendauer > 2)
    valid_studis = df_ab[df_ab['studiendauer_semester'] > max_semester]['studierenden_id'].unique()
    df_ab_valid = df_ab[df_ab['studierenden_id'].isin(valid_studis)].set_index('studierenden_id')
    df_pr_valid = df_pr[df_pr['studierenden_id'].isin(valid_studis)].sort_values(['studierenden_id', 'pruefung_id'])
    
    # Filter auf Prüfungen bis zum Landmark-Semester
    df_pr_lm = df_pr_valid[df_pr_valid['fachsemester'] <= max_semester].copy()
    
    # Status Mapping
    status_map = {'abgeschlossen': 0, 'abgebrochen': 1, 'exmatrikuliert': 2, 'zeitueberschreitung': 3}
    
    # Features für Zweig A (exakt gleiche Reihenfolge wie in fb.prepare_next_exam_dataset!)
    hist_feats = [
        'versuch', 'schwierigkeit', 'cp', 'bestanden_int', 'note_clean',
        'support_vorher_fachlich', 'support_vorher_ueberfachlich',
        'support_glz_fachlich', 'support_glz_ueberfachlich',
        'support_vorher_psychosozial', 'support_glz_psychosozial',
        'fachsemester',
    ]
                  
    df_pr_lm['bestanden_int'] = df_pr_lm['bestanden'].astype(int)
    df_pr_lm['note_clean'] = np.where(
        ~df_pr_lm['bestanden'],
        df_pr_lm['note'].fillna(5.0),
        df_pr_lm['note'].fillna(3.0),
    )
    
    X_list = []
    y_status = []
    y_grade = []
    valid_ids = []
    
    for studi_id, group in df_pr_lm.groupby('studierenden_id', sort=False):
        records = group[hist_feats].values.astype(np.float32)
        pad_seq = np.full((max_seq_len, len(hist_feats)), PADDING_VALUE, dtype=np.float32)
        hist_len = min(len(records), max_seq_len)
        pad_seq[:hist_len] = records[-hist_len:]
            
        X_list.append(pad_seq)
        
        studi_ab = df_ab_valid.loc[studi_id]
        y_status.append(status_map[studi_ab['status']])
        y_grade.append(studi_ab['abschlussnote'])
        valid_ids.append(studi_id)
        
    X = np.array(X_list, dtype=np.float32)
    return X, np.array(y_status), np.array(y_grade), np.array(valid_ids)

def main(data_dir=None, output_dir=None):
    if data_dir is not None:
        data_dir = Path(data_dir)
    elif os.environ.get('DATA_DIR'):
        data_dir = Path(os.environ['DATA_DIR'])
    else:
        data_dir = Path('data_v4_grid/S01_baseline/universe_A')
        
    if output_dir is not None:
        output_dir = Path(output_dir)
    elif os.environ.get('OUTPUT_DIR'):
        output_dir = Path(os.environ['OUTPUT_DIR'])
    else:
        output_dir = data_dir
        
    model_path = output_dir / 'models' / 'autoregressive_deep_transformer.keras'
    if not model_path.exists():
        model_path = data_dir / 'models' / 'autoregressive_deep_transformer.keras'
    
    print(f"1. Lade Deep Transformer Autoregressor aus {model_path}...")
    with tf.keras.utils.custom_object_scope({'SinCosPositionalEncoding': SinCosPositionalEncoding}):
        full_model = load_model(model_path)
    
    # 2. Modell "köpfen": Output NACH dem GlobalAveragePooling als Feature-Vektor
    pooling_layer = None
    for layer in full_model.layers:
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling1D):
            pooling_layer = layer
            break
            
    if pooling_layer is None:
        raise ValueError("Konnte GlobalAveragePooling1D Layer nicht finden!")
        
    # Input 0 ist exam_history
    feature_extractor = Model(inputs=full_model.inputs[0], outputs=pooling_layer.output)
    
    print("2. Lade und formatiere Landmark-Daten (Ende Semester 2)...")
    df_ab, df_pr = fb._load_raw_data(data_dir)
    X_lm, y_status, y_grade, valid_ids = build_landmark_dataset(df_ab, df_pr, max_semester=2, max_seq_len=30)
    print(f" -> Gefunden: {len(X_lm)} Studierende, die Sem 2 überlebt haben.")
    
    # Konsistenter 70/15/15 Split auf Studierenden-IDs (ohne Scaler-Leakage!)
    tr_students, va_students, te_students = fb.split_student_ids(valid_ids, seed=42)
    tr_idx = np.where(np.isin(valid_ids, tr_students))[0]
    va_idx = np.where(np.isin(valid_ids, va_students))[0]
    te_idx = np.where(np.isin(valid_ids, te_students))[0]
    
    scaled_lm, _ = fb.scale_3d_sequences(
        {"train": X_lm[tr_idx], "val": X_lm[va_idx], "test": X_lm[te_idx]},
        fit_key="train",
        padding_value=PADDING_VALUE,
    )
    X_lm_scaled = np.empty_like(X_lm)
    X_lm_scaled[tr_idx] = scaled_lm["train"]
    X_lm_scaled[va_idx] = scaled_lm["val"]
    X_lm_scaled[te_idx] = scaled_lm["test"]
    
    print("3. Feature Extraction (Deep Transformer Embedding)...")
    embeddings = feature_extractor.predict(X_lm_scaled, batch_size=256, verbose=0)
    print(f" -> Embedding-Shape: {embeddings.shape}")
    
    print("4. Trainiere Status-Klassifikator (4 Klassen)...")
    try:
        xgb_class = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        xgb_class.fit(embeddings[tr_idx], y_status[tr_idx])
    except TypeError:
        xgb_class = XGBClassifier(max_iter=100, max_depth=4, learning_rate=0.1, random_state=42)
        xgb_class.fit(embeddings[tr_idx], y_status[tr_idx])
    
    preds_status = xgb_class.predict(embeddings[te_idx])
    probs_status = xgb_class.predict_proba(embeddings[te_idx])
    acc = accuracy_score(y_status[te_idx], preds_status)
    class_names = ['Absolviert', 'Abbruch (Freiw)', 'Exma (Zwang)', 'Zeitueberschr.']
    print(f"\n=> Genauigkeit Status-Vorhersage: {acc:.4f}")
    print(classification_report(y_status[te_idx], preds_status, target_names=class_names))
    
    mc_eval = MulticlassEvaluator(base_dir=output_dir, model_name="landmark_representation_status", class_names=class_names)
    mc_metrics = mc_eval.evaluate_and_log(y_true=y_status[te_idx], y_prob_matrix=probs_status, mode="standard", temporal_type="flat")
    
    print("\n5. Trainiere Regressor (Nur fuer Absolventen / Hurdle Stage 2)...")
    tr_grad = tr_idx[(y_status[tr_idx] == 0) & (~np.isnan(y_grade[tr_idx]))]
    te_grad = te_idx[(y_status[te_idx] == 0) & (~np.isnan(y_grade[te_idx]))]
    
    try:
        xgb_reg = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        xgb_reg.fit(embeddings[tr_grad], y_grade[tr_grad])
    except TypeError:
        xgb_reg = XGBRegressor(max_iter=100, max_depth=4, learning_rate=0.1, random_state=42)
        xgb_reg.fit(embeddings[tr_grad], y_grade[tr_grad])
    
    preds_grade = xgb_reg.predict(embeddings[te_grad])
    r2 = r2_score(y_grade[te_grad], preds_grade)
    print(f"=> R2 Score fuer Abschlussnote (conditional auf Abschluss): {r2:.4f}")

    reg_eval = RegressionEvaluator(base_dir=output_dir, model_name="landmark_representation_grade")
    reg_metrics = reg_eval.evaluate_and_log(y_true=y_grade[te_grad], y_pred=preds_grade, mode="standard", temporal_type="flat")

    metrics_out = {
        'Landmark_Status_Accuracy': float(acc),
        'Landmark_Status_Macro_F1': mc_metrics.get('macro_f1'),
        'Landmark_Status_ROC_AUC_OVR': mc_metrics.get('roc_auc_ovr_macro'),
        'Landmark_Grad_R2': float(r2),
        'Landmark_Grad_RMSE': reg_metrics.get('rmse'),
        'Landmark_Grad_MAE': reg_metrics.get('mae'),
    }
    save_metrics("landmark_representation_learning", metrics_out, base_dir=output_dir, mode="standard", temporal_type="flat")
    (output_dir / 'metrics').mkdir(exist_ok=True, parents=True)
    import json
    with open(output_dir / 'metrics' / 'landmark_representation_learning_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_out, f, indent=4)

if __name__ == '__main__':
    main()

