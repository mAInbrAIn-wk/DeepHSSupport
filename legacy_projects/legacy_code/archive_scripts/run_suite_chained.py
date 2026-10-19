import sys
import time
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.chdir(SRC_DIR)

from run_overnight_v41 import run_master_overnight_pipeline_v41

def main():
    total_start = time.time()
    print("\n" + "#" * 80)
    print("   STARTE PHASE 1: V3.6 DATEN (BEREINIGTE EVALUATION)")
    print(f"   Startzeit: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 80 + "\n")

    v36_dir = Path("output_dl")
    try:
        run_master_overnight_pipeline_v41(
            data_dir=v36_dir,
            temporal="prev",
            population_seed=42
        )
        print("\n[OK] Phase 1 (V3.6) erfolgreich abgeschlossen!")
    except Exception as e:
        print(f"\n[WARNUNG] Fehler in Phase 1 (V3.6): {e}")

    print("\n" + "#" * 80)
    print("   STARTE PHASE 2: V4.1 BASELINE (S01 / UNIVERSE A)")
    print(f"   Startzeit: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 80 + "\n")

    v41_dir = Path("output_v4_grid_v41/S01_baseline/universe_A")
    try:
        run_master_overnight_pipeline_v41(
            data_dir=v41_dir,
            temporal="prev",
            population_seed=42
        )
        print("\n[OK] Phase 2 (V4.1) erfolgreich abgeschlossen!")
    except Exception as e:
        print(f"\n[WARNUNG] Fehler in Phase 2 (V4.1): {e}")

    total_elapsed = time.time() - total_start
    print("\n" + "#" * 80)
    print(f"   GESAMTPROZESS BEENDET NACH {total_elapsed/60:.2f} MINUTEN ({total_elapsed/3600:.2f} STUNDEN)")
    print("#" * 80 + "\n")

if __name__ == '__main__':
    main()
