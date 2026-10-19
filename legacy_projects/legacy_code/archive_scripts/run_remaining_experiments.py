"""
Runner Script: Führt verbleibende Experimente ab Schritt 7 aus
================================================================
Da Schritte 1-6 bereits fertig berechnet wurden, startet dieses Skript
direkt ab Schritt 7 (Recurrent Survival, Transformers, Dynamic DeepHit, DeepSurv).
"""

import os
import sys
from pathlib import Path

# Add src to sys.path
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

def run_remaining():
    print("=" * 80)
    print("   STARTE REKORRENTE & TRANSFORMER SURVIVAL MODELLE (AB SCHRITT 6)")
    print("=" * 80)
    
    data_dir = Path('output_dl') if Path('output_dl').exists() else Path('../output_dl')
    
    # 6-8: Bereits in task-1977 erfolgreich beendet & gespeichert!
    # print("\n>>> [6/12] Trainiere Exam-Transformer Regressor ...")
    # from timeseries_exam_transformer import main as run_ts_exam_tf
    # run_ts_exam_tf()
    # 
    # print("\n>>> [7/12] Trainiere Recurrent Survival GRU (Standard & Blind) ...")
    # from recurrent_survival_model import train_recurrent_survival_model
    # train_recurrent_survival_model(data_dir, blind=False)
    # train_recurrent_survival_model(data_dir, blind=True)
    # 
    # print("\n>>> [8/12] Trainiere Recurrent Exam Survival GRU ...")
    # from recurrent_exam_survival import train_recurrent_exam_survival
    # train_recurrent_exam_survival(data_dir)
    
    # 9. Causal Transformer Survival (Standard & Blind)
    print("\n>>> [9/12] Trainiere Causal Transformer Survival (Standard & Blind) ...")
    from transformer_survival_model import train_causal_transformer_survival
    train_causal_transformer_survival(data_dir, blind=False)
    train_causal_transformer_survival(data_dir, blind=True)
    
    # 10. Exam Causal Transformer Survival
    print("\n>>> [10/12] Trainiere Causal Exam-Transformer Survival ...")
    from transformer_exam_survival import main as run_tf_exam_surv
    run_tf_exam_surv()
    
    # 11. Dynamic DeepHit Competing Risks
    print("\n>>> [11/12] Trainiere Dynamic DeepHit Competing Risks ...")
    from dynamic_deephit_model import train_dynamic_deephit_model
    train_dynamic_deephit_model(data_dir)
    
    # 12. DeepSurv Landmark Survival
    print("\n>>> [12/12] Trainiere DeepSurv & Logistic Hazard Landmark ...")
    from deep_survival import main as run_deep_surv
    run_deep_surv()
    
    print("\n" + "=" * 80)
    print("   ALLE VERBLEIBENDEN MODELL-TRAININGS ERFOLGREICH BEENDET!")
    print("=" * 80)

if __name__ == '__main__':
    run_remaining()
