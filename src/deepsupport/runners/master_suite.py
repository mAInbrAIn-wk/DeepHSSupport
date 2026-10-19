"""
Master Suite Orchestration (V4.2)
=================================
Zentraler Einstiegspunkt zur Steuerung der Modell-Landschaft:
  --suite fast   : Schnelle Core-Suite (25+ Modelle, Inferenz, Kontrafaktik; ~20 Min.)
  --suite heavy  : Schwere Deep-Suite (Deep Transformer, Autoregressoren; ~1.8 Std.)
  --suite all    : Gesamtlauf beider Suites in logischer Abfolge
"""

import os
import sys
import time
import argparse
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deepsupport.runners.fast_suite import run_fast_suite
from deepsupport.runners.heavy_suite import run_heavy_suite

def main():
    parser = argparse.ArgumentParser(description="Master Suite Runner V4.2")
    parser.add_argument('--suite', type=str, default='all', choices=['fast', 'heavy', 'all'],
                        help="Auswahl der Suite: 'fast', 'heavy', oder 'all'")
    parser.add_argument('--data_dir', type=str, default="data_v4_grid/S01_baseline/universe_A",
                        help="Pfad zu den Roh-/Aggregatdaten des Ziel-Universums")
    parser.add_argument('--output_dir', type=str, default="outputs/output_thinkcentre",
                        help="Basis-Verzeichnis fuer Metriken und Reports")
    parser.add_argument('--temporal', type=str, default='prev', choices=['prev', 'cum'],
                        help="Temporaler Modus: 'prev' (Deltas/Vorsemester) oder 'cum' (Historie)")
    parser.add_argument('--modes', type=str, default='standard,gradeblind',
                        help="Kommagetrennte Liste der Modi (z.B. 'standard,gradeblind,oracle')")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        # Fallback auf Projekt-Root
        data_dir = PROJECT_ROOT / args.data_dir
        
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_list = [m.strip() for m in args.modes.split(',') if m.strip()]

    print("\n" + "#" * 80)
    print("   DEEPSUPPORT MASTER SUITE ORCHESTRATION")
    print(f"   Suite:       {args.suite.upper()}")
    print(f"   Temporal:    {args.temporal}")
    print(f"   Modi:        {mode_list}")
    print(f"   Daten-Pfad:  {data_dir.resolve()}")
    print(f"   Output-Pfad: {output_dir.resolve()}")
    print("#" * 80 + "\n")

    total_start = time.time()

    if args.suite in ['fast', 'all']:
        print("\n>>> STARTE FAST CORE SUITE ...")
        fast_out = output_dir / "fast"
        fast_out.mkdir(parents=True, exist_ok=True)
        run_fast_suite(data_dir=data_dir, output_dir=fast_out, temporal=args.temporal, modes=mode_list, population_seed=args.seed)

    if args.suite in ['heavy', 'all']:
        print("\n>>> STARTE HEAVY DEEP SUITE ...")
        heavy_out = output_dir / "heavy"
        heavy_out.mkdir(parents=True, exist_ok=True)
        run_heavy_suite(data_dir=data_dir, output_dir=heavy_out, temporal=args.temporal, modes=mode_list, population_seed=args.seed)

    total_elapsed = time.time() - total_start
    print("\n" + "#" * 80)
    print(f"   MASTER SUITE ABGESCHLOSSEN IN {total_elapsed/60:.2f} MINUTEN ({total_elapsed/3600:.2f} STUNDEN)")
    print(f"   Ergebnisse archiviert in: {output_dir.resolve()}")
    print("#" * 80 + "\n")

if __name__ == '__main__':
    main()
