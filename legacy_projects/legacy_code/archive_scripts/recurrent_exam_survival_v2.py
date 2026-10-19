"""
DEPRECATED / CONSOLIDATED WRAPPER: recurrent_exam_survival_v2.py
================================================================
Dieses Skript wurde in `src/recurrent_exam_survival.py` konsolidiert.
Leitet Aufrufe transparent an das konsolidierte Skript weiter.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from recurrent_exam_survival import train_recurrent_exam_survival_model

if __name__ == '__main__':
    data_dir = Path('src/output_dl')
    if not data_dir.exists():
        data_dir = Path('output_dl')
    train_recurrent_exam_survival_model(data_dir, temporal='prev', mode='standard')
