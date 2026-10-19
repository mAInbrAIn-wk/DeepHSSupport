import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import numpy as np
from pathlib import Path
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, average_precision_score
from tensorflow.keras.models import load_model
import tensorflow as tf

sys.path.insert(0, str(Path('src').absolute()))
from autoregressive_next_exam import prepare_next_exam_dataset
from sklearn.preprocessing import StandardScaler

def main():
    print("="*60)
    print(" TRANSFER LEARNING PoC: Autoregressive Dual-Head ")
    print("="*60)
    
    v35_data_dir = Path('src/output_dl')
    print("Lade V3.5 Datensatz...")
    
    X_hist, X_ctx, y_grade, y_pass, student_ids = prepare_next_exam_dataset(v35_data_dir)
    
    # Group-consistent Split on student IDs
    unique_students = np.unique(student_ids)
    tr_students, temp_students = train_test_split(unique_students, test_size=0.30, random_state=42)
    va_students, te_students = train_test_split(temp_students, test_size=0.50, random_state=42)
    
    tr_idx = np.where(np.isin(student_ids, tr_students))[0]
    va_idx = np.where(np.isin(student_ids, va_students))[0]
    te_idx = np.where(np.isin(student_ids, te_students))[0]
    
    scaler_seq = StandardScaler()
    scaler_ctx = StandardScaler()
    
    vm_tr = X_hist[tr_idx, :, 0] != -99.0
    scaler_seq.fit(X_hist[tr_idx][vm_tr])
    scaler_ctx.fit(X_ctx[tr_idx])
    
    X_hist_scaled = X_hist.copy()
    X_ctx_scaled = X_ctx.copy()
    for s_idx in [tr_idx, va_idx, te_idx]:
        vm = X_hist[s_idx, :, 0] != -99.0
        X_hist_scaled[s_idx][vm] = scaler_seq.transform(X_hist[s_idx][vm])
        X_ctx_scaled[s_idx] = scaler_ctx.transform(X_ctx[s_idx])
    
    X_te_hist = X_hist_scaled[te_idx]
    X_te_ctx = X_ctx_scaled[te_idx]
    y_te_grade = y_grade[te_idx]
    y_te_pass = y_pass[te_idx]
    
    print("\nLade originales V3.5 Modell als Baseline...")
    base_model = load_model(v35_data_dir / 'models' / 'autoregressive_next_exam_dual_head.keras')
    preds_base = base_model.predict({'exam_history': X_te_hist, 'next_exam_context': X_te_ctx}, verbose=0)
    
    r2_base = r2_score(y_te_grade, preds_base[0].flatten())
    auc_base = roc_auc_score(y_te_pass, preds_base[1].flatten())
    print(f"BASELINE (V3.5 Model): R2 = {r2_base:.4f}, ROC-AUC = {auc_base:.4f}")
    
    v36_model_path = Path('src/output_dl_seed99999/models/autoregressive_next_exam_dual_head.keras')
    print("\nLade pre-trainiertes V3.6 Modell fr Transfer Learning...")
    transfer_model = load_model(v36_model_path)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    transfer_model.compile(optimizer=optimizer,
                           loss={'out_grade': 'mse', 'out_pass': 'binary_crossentropy'},
                           loss_weights={'out_grade': 1.0, 'out_pass': 0.8},
                           metrics={'out_grade': ['mae'], 'out_pass': ['accuracy']})
    
    print("Starte Finetuning (3 Epochen)...")
    transfer_model.fit(
        {'exam_history': X_hist_scaled[tr_idx], 'next_exam_context': X_ctx_scaled[tr_idx]},
        {'out_grade': y_grade[tr_idx], 'out_pass': y_pass[tr_idx]},
        validation_data=(
            {'exam_history': X_hist_scaled[va_idx], 'next_exam_context': X_ctx_scaled[va_idx]},
            {'out_grade': y_grade[va_idx], 'out_pass': y_pass[va_idx]}
        ),
        epochs=3, batch_size=256, verbose=1
    )
    
    preds_trans = transfer_model.predict({'exam_history': X_te_hist, 'next_exam_context': X_te_ctx}, verbose=0)
    r2_trans = r2_score(y_te_grade, preds_trans[0].flatten())
    auc_trans = roc_auc_score(y_te_pass, preds_trans[1].flatten())
    
    print("\n" + "="*60)
    print(" VERGLEICH AUF V3.5 TEST SET ")
    print("="*60)
    print(f"V3.5 Base Model       -> R2: {r2_base:.4f}, AUC: {auc_base:.4f}")
    print(f"V3.6 Transfer Finetune-> R2: {r2_trans:.4f}, AUC: {auc_trans:.4f}")
    print(f"Lift                  -> R2: {r2_trans - r2_base:+.4f}, AUC: {auc_trans - auc_base:+.4f}")
    
    out_dir = Path('src/output_dl_transfer')
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / 'transfer_learning_results.json', 'w') as f:
        json.dump({
            'base_r2': float(r2_base),
            'base_auc': float(auc_base),
            'transfer_r2': float(r2_trans),
            'transfer_auc': float(auc_trans)
        }, f, indent=4)
        
    print(f"Results saved to {out_dir}")

if __name__ == '__main__':
    main()
