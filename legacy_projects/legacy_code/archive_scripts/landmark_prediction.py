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

sys.path.insert(0, str(Path('src').absolute()))
import feature_builder as fb
from autoregressive_deep_transformer import SinCosPositionalEncoding
from autoregressive_next_exam import PADDING_VALUE

def build_landmark_dataset(df_ab, df_pr, max_semester=2, max_seq_len=30):
    # Nur Studenten, die das Landmark-Semester berlebt haben (Studiendauer > 2)
    valid_studis = df_ab[df_ab['studiendauer_semester'] > max_semester]['studierenden_id'].unique()
    df_ab_valid = df_ab[df_ab['studierenden_id'].isin(valid_studis)].set_index('studierenden_id')
    df_pr_valid = df_pr[df_pr['studierenden_id'].isin(valid_studis)]
    
    # Filter auf Prfungen bis zum Landmark-Semester
    df_pr_lm = df_pr_valid[df_pr_valid['fachsemester'] <= max_semester].copy()
    
    # Status Mapping
    status_map = {'abgeschlossen': 0, 'abgebrochen': 1, 'exmatrikuliert': 2, 'zeitueberschreitung': 3}
    
    # Features fr Zweig A (mssen EXAKT matchen mit Training!)
    hist_feats = ['versuch', 'schwierigkeit', 'cp', 'bestanden_int', 'note_clean',
                  'support_vorher_fachlich', 'support_vorher_ueberfachlich', 'support_vorher_psychosozial',
                  'support_glz_fachlich', 'support_glz_ueberfachlich', 'support_glz_psychosozial', 'fachsemester']
                  
    # Clean up notes / bestanden
    df_pr_lm['bestanden_int'] = df_pr_lm['bestanden'].astype(int)
    df_pr_lm['note_clean'] = df_pr_lm['note'].fillna(5.0)
    
    X_list = []
    y_status = []
    y_grade = []
    valid_ids = []
    
    for studi_id, group in df_pr_lm.groupby('studierenden_id'):
        group = group.sort_values(['fachsemester', 'modul_id']) # chronologisch (grob)
        records = group[hist_feats].values
        
        if len(records) > max_seq_len:
            records = records[-max_seq_len:]
            
        pad_len = max_seq_len - len(records)
        if pad_len > 0:
            pad_mat = np.full((pad_len, len(hist_feats)), PADDING_VALUE, dtype=np.float32)
            seq = np.vstack([pad_mat, records])
        else:
            seq = records
            
        X_list.append(seq)
        
        studi_ab = df_ab_valid.loc[studi_id]
        y_status.append(status_map[studi_ab['status']])
        y_grade.append(studi_ab['abschlussnote'])
        valid_ids.append(studi_id)
        
    X = np.array(X_list, dtype=np.float32)
    return X, np.array(y_status), np.array(y_grade), valid_ids

def main(data_dir=None):
    if data_dir is not None:
        data_dir = Path(data_dir)
    elif os.environ.get('DATA_DIR'):
        data_dir = Path(os.environ['DATA_DIR'])
    else:
        data_dir = Path('src/output_dl') if Path('src/output_dl').exists() else Path('output_dl')
        
    model_path = data_dir / 'models' / 'autoregressive_deep_transformer.keras'
    
    print("1. Lade Deep Transformer Autoregressor...")
    with tf.keras.utils.custom_object_scope({'SinCosPositionalEncoding': SinCosPositionalEncoding}):
        full_model = load_model(model_path)
    
    # 2. Modell "koepfen": Output NACH dem GlobalAveragePooling als Feature-Vektor
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
    print(f" -> Gefunden: {len(X_lm)} Studierende, die Sem 2 berlebt haben.")
    
    scaler = StandardScaler()
    vm = X_lm[:, :, 0] != PADDING_VALUE
    scaler.fit(X_lm[vm])
    X_lm_scaled = X_lm.copy()
    X_lm_scaled[vm] = scaler.transform(X_lm[vm])
    
    print("3. Feature Extraction (Deep Transformer Embedding)...")
    embeddings = feature_extractor.predict(X_lm_scaled, batch_size=256)
    print(f" -> Embedding-Shape: {embeddings.shape}")
    
    idx = np.arange(len(embeddings))
    tr_idx, te_idx = train_test_split(idx, test_size=0.3, random_state=42, stratify=y_status)
    
    print("4. Trainiere XGBoost Status-Klassifikator (4 Klassen)...")
    xgb_class = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)
    xgb_class.fit(embeddings[tr_idx], y_status[tr_idx])
    
    preds_status = xgb_class.predict(embeddings[te_idx])
    acc = accuracy_score(y_status[te_idx], preds_status)
    print(f"\n=> Genauigkeit Status-Vorhersage: {acc:.4f}")
    print(classification_report(y_status[te_idx], preds_status, 
                                target_names=['Absolviert', 'Abbruch (Freiw)', 'Exma (Zwang)', 'Zeitueberschr.']))
    
    print("\n5. Trainiere XGBoost Regressor (Nur fr Absolventen)...")
    tr_grad = tr_idx[y_status[tr_idx] == 0]
    te_grad = te_idx[y_status[te_idx] == 0]
    
    xgb_reg = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
    xgb_reg.fit(embeddings[tr_grad], y_grade[tr_grad])
    
    preds_grade = xgb_reg.predict(embeddings[te_grad])
    r2 = r2_score(y_grade[te_grad], preds_grade)
    print(f"=> R2 Score fr Abschlussnote (conditional auf Abschluss): {r2:.4f}")

if __name__ == '__main__':
    main()
