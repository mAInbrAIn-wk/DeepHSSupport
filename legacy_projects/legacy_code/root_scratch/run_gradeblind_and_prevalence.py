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
from train_mlp_baseline import run_baseline_training

data_dir = Path('src/output_dl_v4')

print('=' * 80)
print('1. BASELINE PREVALENZEN IN V4 (Klassen-Verteilungen)')
print('=' * 80)
df_ab, df_pr = fb._load_raw_data(data_dir)
total_studis = len(df_ab)
dropout_studis = (df_ab['status'] != 'abgeschlossen').sum()
grad_studis = (df_ab['status'] == 'abgeschlossen').sum()
print(f'Studierenden-Ebene (N=50.000):')
print(f'  - Dropout (Minoritaet)   : {dropout_studis:,} ({dropout_studis/total_studis*100:.2f}%)')
print(f'  - Abschluss (Mehrheit)   : {grad_studis:,} ({grad_studis/total_studis*100:.2f}%)')
print(f'  -> Random Baseline PR-AUC Dropout: {dropout_studis/total_studis:.4f}')
print(f'  -> Random Baseline PR-AUC Abschluss: {grad_studis/total_studis:.4f}')

# Panel Ebene
df_panel, p_feat, p_target, _ = fb.build_semester_panel_df(data_dir)
panel_events = df_panel['event'].sum()
panel_total = len(df_panel)
print(f'\nSemester-Panel-Ebene (N={panel_total:,} Person-Semester Zeilen):')
print(f'  - Event / Dropout im Semester : {panel_events:,} ({panel_events/panel_total*100:.2f}%)')
print(f'  - Kein Event im Semester      : {panel_total - panel_events:,} ({(panel_total - panel_events)/panel_total*100:.2f}%)')
print(f'  -> Random Baseline PR-AUC Panel Event: {panel_events/panel_total:.4f}')

# Exam Sequenz Ebene
_, X_exam, y_exam_drop, _, _, _ = fb.build_exam_sequence_tensor(data_dir)
vm = X_exam[:, :, 0] != fb.PADDING_VALUE
y_exam_flat = y_exam_drop[vm]
print(f'\nPruefungs-Ebene (N={len(y_exam_flat):,} Pruefungen):')
print(f'  - Event / Dropout nach Pruefung: {(y_exam_flat == 1).sum():,} ({y_exam_flat.mean()*100:.2f}%)')
print(f'  -> Random Baseline PR-AUC Exam Event: {y_exam_flat.mean():.4f}')

print('\n' + '=' * 80)
print('2. GRADEBLIND REGRESSIONEN AUF V4 (Standard vs Gradeblind)')
print('=' * 80)

print('\n--- A. Landmark Abschlussnoten-Regression (Semester 2) ---')
print('Modus: standard')
res_lm_std = run_regression_training(data_dir, use_landmark=True, mode='standard')
print('\nModus: gradeblind (ohne Noten-Historie, nur CP, Versuche, HZB)')
res_lm_gb = run_regression_training(data_dir, use_landmark=True, mode='gradeblind')

print('\n--- B. Semester Timeseries Transformer ---')
print('Modus: gradeblind')
res_sem_trans_gb = train_timeseries_semester_transformer(data_dir, max_semesters=16, temporal='prev', mode='gradeblind')

print('\n--- C. Exam Timeseries Transformer ---')
print('Modus: gradeblind')
res_exam_trans_gb = train_timeseries_exam_transformer(data_dir, max_exams=40, temporal='prev', mode='gradeblind')

print('\n--- D. Landmark Dropout-Klassifikation (standard vs gradeblind) ---')
print('Modus: gradeblind')
res_lm_class_gb = run_baseline_training(data_dir, use_landmark=True, mode='gradeblind')
