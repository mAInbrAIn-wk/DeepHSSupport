import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

sys.path.insert(0, 'src')
import feature_builder as fb
from train_mlp_regression import run_regression_training
from timeseries_semester_transformer import train_timeseries_semester_transformer
from timeseries_exam_transformer import train_timeseries_exam_transformer

data_dir = Path('src/output_dl_v4')

print('=== 1. BASELINE PREVALENZEN IN V4 ===')
df_ab, df_pr = fb._load_raw_data(data_dir)
total_studis = len(df_ab)
dropout_studis = (df_ab['status'] != 'abgeschlossen').sum()
grad_studis = (df_ab['status'] == 'abgeschlossen').sum()
print(f'Studierenden-Ebene: Dropout = {dropout_studis}/{total_studis} ({dropout_studis/total_studis*100:.2f}%), Abschluss = {grad_studis}/{total_studis} ({grad_studis/total_studis*100:.2f}%)')

# Panel Ebene
X_panel, y_event, t_start, t_stop, feat_names, _ = fb.build_semester_panel_df(data_dir)
print(f'Semester-Panel-Ebene: Events (Dropouts) = {y_event.sum():.0f}/{len(y_event)} ({y_event.mean()*100:.2f}%)')

# Exam Ebene
X_exam, y_exam_drop, y_exam_time, y_exam_event, _, _ = fb.build_exam_sequence_tensor(data_dir)
# Flatten valid steps
vm = X_exam[:, :, 0] != fb.PADDING_VALUE
y_exam_flat = y_exam_drop[vm]
print(f'Pruefungs-Sequenz-Ebene: Event Rate = {y_exam_flat.mean()*100:.2f}%')

print('\n=== 2. GRADEBLIND REGRESSIONEN AUF V4 ===')
print('-> A. Landmark Abschlussnoten-Regression (gradeblind)...')
res_landmark_gb = run_regression_training(data_dir, use_landmark=True, mode='gradeblind')

print('-> B. Semester Timeseries Transformer Abschlussnoten-Regression (gradeblind)...')
res_sem_trans_gb = train_timeseries_semester_transformer(data_dir, max_semesters=16, temporal='prev', mode='gradeblind')

print('-> C. Exam Timeseries Transformer Noten-Regression (gradeblind)...')
res_exam_trans_gb = train_timeseries_exam_transformer(data_dir, max_exams=40, temporal='prev', mode='gradeblind')
