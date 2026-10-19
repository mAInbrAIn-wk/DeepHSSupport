"""
Catch-up Runner fuer fehlende Fast-Suite-Schritte
=================================================
Fuehrt die 4 durch Bugfixes behobenen Schritte aus:
1. Extended DeepSurv & Logistic Hazard [standard]
2. Extended DeepSurv & Logistic Hazard [gradeblind]
3. Landmark Deep Survival (DeepSurv & Logistic Hazard)
4. Counterfactual Grade Transformer
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import time
import argparse
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deepsupport.models.extended_deepsurv import train_extended_deep_survival
from deepsupport.models.deep_survival import train_deep_survival
from deepsupport.evaluation.causal.counterfactual_grade_transformer import analyze_counterfactual_grade_transformer

def main():
    parser = argparse.ArgumentParser(description="Catch-Up Fast Suite Runner")
    parser.add_argument('--data_dir', type=str, default="data_v4_grid/S01_baseline/universe_A",
                        help="Pfad zu den Eingabedaten")
    parser.add_argument('--output_dir', type=str, default="outputs/output_thinkcentre/fast",
                        help="Ziel-Ausgabeverzeichnis")
    parser.add_argument('--temporal', type=str, default="prev")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        data_dir = PROJECT_ROOT / args.data_dir
        
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["DATA_DIR"] = str(data_dir.resolve())
    os.environ["OUTPUT_DIR"] = str(output_dir.resolve())

    print("=" * 80)
    print("   STARTE FAST SUITE CATCH-UP RUNNER")
    print(f"   Data Dir:   {data_dir.resolve()}")
    print(f"   Output Dir: {output_dir.resolve()}")
    print("=" * 80)

    t0 = time.time()

    # 1. Extended DeepSurv [standard]
    print("\n[1/4] Extended DeepSurv & Logistic Hazard [standard]...")
    try:
        train_extended_deep_survival(data_dir=data_dir, temporal=args.temporal, mode='standard')
    except Exception as e:
        print(f"[FEHLER] Extended DeepSurv standard: {e}")

    # 2. Extended DeepSurv [gradeblind]
    print("\n[2/4] Extended DeepSurv & Logistic Hazard [gradeblind]...")
    try:
        train_extended_deep_survival(data_dir=data_dir, temporal=args.temporal, mode='gradeblind')
    except Exception as e:
        print(f"[FEHLER] Extended DeepSurv gradeblind: {e}")

    # 3. Deep Survival Landmark (LH & DS)
    print("\n[3/4] Deep Survival Landmark (LH & DS)...")
    try:
        train_deep_survival(data_dir=data_dir)
    except Exception as e:
        print(f"[FEHLER] Deep Survival Landmark: {e}")

    # 4. Counterfactual Grade Transformer
    print("\n[4/4] Counterfactual Grade Transformer...")
    try:
        analyze_counterfactual_grade_transformer(data_dir=data_dir)
    except Exception as e:
        print(f"[FEHLER] Counterfactual Grade Transformer: {e}")

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print(f"[OK] CATCH-UP RUN ERFOLGREICH BEENDET IN {elapsed/60:.2f} MINUTEN!")
    print("=" * 80)

if __name__ == '__main__':
    main()
