"""
Master Suite Orchestration (V4.1)
=================================
Zentraler Einstiegspunkt zur Steuerung der Modell-Landschaft:
  --suite fast   : Schnelle Core-Suite (25+ Modelle, Inferenz, Kontrafaktik; ~20 Min.)
  --suite heavy  : Schwere Deep-Suite (Deep Transformer, Autoregressoren; ~2.5 Std.)
  --suite all    : Gesamtlauf beider Suites in logischer Abfolge
"""

import os
import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.chdir(SRC_DIR)

from run_fast_suite import run_fast_suite
from run_heavy_suite import run_heavy_suite

def main():
    parser = argparse.ArgumentParser(description="Master Suite Runner V4.1")
    parser.add_argument('--suite', type=str, default='fast', choices=['fast', 'heavy', 'all'],
                        help="Auswahl der Suite: 'fast' (empfohlen fuer Sensitivitaet), 'heavy' (fuer Baseline), oder 'all'")
    parser.add_argument('--data_dir', type=str, default="output_v4_grid_v41/S01_baseline/universe_A")
    parser.add_argument('--temporal', type=str, default='prev', choices=['prev', 'cum'],
                        help="Temporaler Modus: 'prev' (Deltas/Vorsemester) oder 'cum' (Historie)")
    parser.add_argument('--modes', type=str, default='standard,gradeblind',
                        help="Kommagetrennte Liste der Modi (z.B. 'standard,gradeblind,oracle')")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    mode_list = [m.strip() for m in args.modes.split(',') if m.strip()]

    print("\n" + "#" * 80)
    print("   DEEPSUPPORT MASTER SUITE ORCHESTRATION")
    print(f"   Suite: {args.suite.upper()} | Temporal: {args.temporal} | Modi: {mode_list}")
    print(f"   Ziel-Verzeichnis: {data_dir.resolve()}")
    print("#" * 80 + "\n")

    total_start = time.time()

    if args.suite in ['fast', 'all']:
        print("\n>>> STARTE FAST CORE SUITE ...")
        run_fast_suite(data_dir=data_dir, temporal=args.temporal, modes=mode_list, population_seed=args.seed)

    if args.suite in ['heavy', 'all']:
        print("\n>>> STARTE HEAVY DEEP SUITE ...")
        run_heavy_suite(data_dir=data_dir, temporal=args.temporal, modes=mode_list, population_seed=args.seed)

    total_elapsed = time.time() - total_start
    print("\n" + "#" * 80)
    print(f"   MASTER SUITE ABGESCHLOSSEN IN {total_elapsed/60:.2f} MINUTEN ({total_elapsed/3600:.2f} STUNDEN)")
    print("#" * 80 + "\n")

if __name__ == '__main__':
    main()
