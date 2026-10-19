"""
Master Retraining & Counterfactual Analysis Pipeline (V3.6)
============================================================
Leitet alle Retraining-Aufrufe an den Master-Orchestrator `run_overnight.py` weiter.
"""

import sys
import argparse
from pathlib import Path

# Projekt-Pfade einbinden
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_overnight import run_master_overnight_pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Master Retraining Pipeline")
    parser.add_argument('--data_dir', type=str, default='src/output_dl')
    parser.add_argument('--temporal', type=str, default='prev', choices=['prev', 'cum'])
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    run_master_overnight_pipeline(
        data_dir=Path(args.data_dir),
        skip_sim=True,
        temporal=args.temporal,
        population_seed=args.seed
    )