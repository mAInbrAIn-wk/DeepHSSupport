import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

from deepsupport.models.autoregressive_gru import prepare_next_exam_dataset, PADDING_VALUE
from sklearn.preprocessing import StandardScaler

import json

def main(data_dir=None, output_dir=None):
    if data_dir is None:
        data_dir = Path(os.environ.get('DATA_DIR', 'data_v4_grid/S01_baseline/universe_A'))
    else:
        data_dir = Path(data_dir)
        
    if output_dir is not None:
        output_dir = Path(output_dir)
    elif os.environ.get('OUTPUT_DIR'):
        output_dir = Path(os.environ['OUTPUT_DIR'])
    else:
        output_dir = data_dir
        
    print(f"Loading data from {data_dir}...")
    import deepsupport.data_engine.feature_builder as fb
    X_hist, X_ctx, y_grades, y_pass, student_ids = fb.prepare_next_exam_dataset(data_dir)
    
    # Group-consistent Split on student IDs via feature_builder
    tr_students, va_students, te_students = fb.split_student_ids(student_ids, seed=42)
    
    train_idx = np.where(np.isin(student_ids, tr_students))[0]
    val_idx = np.where(np.isin(student_ids, va_students))[0]
    test_idx = np.where(np.isin(student_ids, te_students))[0]
    
    scaled_hist, _ = fb.scale_3d_sequences(
        {"train": X_hist[train_idx], "val": X_hist[val_idx], "test": X_hist[test_idx]},
        fit_key="train",
        padding_value=PADDING_VALUE,
    )
    X_test_hist = scaled_hist["test"]
    
    scaler_ctx = StandardScaler()
    scaler_ctx.fit(X_ctx[train_idx])
    X_test_ctx = scaler_ctx.transform(X_ctx[test_idx]).astype(np.float32)
    
    y_pass_test = y_pass[test_idx]
    y_fail_test = 1 - y_pass_test
    
    model_path = output_dir / 'models' / 'autoregressive_next_exam_dual_head.keras'
    if not model_path.exists():
        model_path = data_dir / 'models' / 'autoregressive_next_exam_dual_head.keras'
        
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
        
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    print("Predicting...")
    preds = model.predict({'exam_history': X_test_hist, 'next_exam_context': X_test_ctx}, verbose=0)
    pred_pass = preds[1].flatten()
    pred_fail = 1.0 - pred_pass
    
    pr_auc_fail = float(average_precision_score(y_fail_test, pred_fail))
    print(f"Next_Exam_Fail_PR_AUC: {pr_auc_fail:.4f}")
    print(f"Prevalence of Fail: {float(np.mean(y_fail_test)):.4f}")
    
    metrics_out = {
        'Next_Exam_Fail_PR_AUC': pr_auc_fail,
        'Prevalence_of_Fail': float(np.mean(y_fail_test))
    }
    (output_dir / 'metrics').mkdir(exist_ok=True, parents=True)
    with open(output_dir / 'metrics' / 'autoregressive_fail_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_out, f, indent=4)

if __name__ == '__main__':
    main()

