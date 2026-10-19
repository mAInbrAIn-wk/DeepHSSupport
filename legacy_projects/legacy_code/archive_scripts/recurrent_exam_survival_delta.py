"""
DEPRECATED / CONSOLIDATED WRAPPER: recurrent_exam_survival_delta.py
===================================================================
Dieses Skript wurde in `src/recurrent_exam_survival.py` konsolidiert.
Leitet Aufrufe transparent an das konsolidierte Skript weiter.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import feature_builder as fb
from recurrent_exam_survival import train_recurrent_exam_survival_model

def build_recurrent_exam_dataset_delta(data_dir=None, max_exams=40, mode='standard'):
    if data_dir is None:
        data_dir = Path('src/output_dl') if Path('src/output_dl').exists() else Path('output_dl')
    return fb.build_exam_sequence_tensor(data_dir, max_exams=max_exams, mode=mode, temporal='prev')

def train_recurrent_exam_survival_delta(data_dir=None, mode='standard'):
    if data_dir is None:
        data_dir = Path('src/output_dl') if Path('src/output_dl').exists() else Path('output_dl')
    return train_recurrent_exam_survival_model(data_dir, temporal='prev', mode=mode)

if __name__ == '__main__':
    data_dir = Path('src/output_dl')
    if not data_dir.exists():
        data_dir = Path('output_dl')
    train_recurrent_exam_survival_model(data_dir, temporal='prev', mode='standard')
