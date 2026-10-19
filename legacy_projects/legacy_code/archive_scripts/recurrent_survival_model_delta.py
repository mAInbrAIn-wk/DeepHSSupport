"""
DEPRECATED / CONSOLIDATED WRAPPER: recurrent_survival_model_delta.py
====================================================================
Dieses Skript wurde in `src/recurrent_survival_model.py` konsolidiert.
Leitet Aufrufe transparent an das konsolidierte Skript weiter.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import feature_builder as fb
from recurrent_survival_model import train_recurrent_survival_model, masked_binary_crossentropy

def build_recurrent_survival_dataset_delta(data_dir=None, max_semesters=16, mode='standard'):
    if data_dir is None:
        data_dir = Path('src/output_dl') if Path('src/output_dl').exists() else Path('output_dl')
    return fb.build_semester_sequence_tensor(data_dir, max_semesters=max_semesters, mode=mode, temporal='prev')

def train_recurrent_survival_model_delta(data_dir=None, mode='standard'):
    if data_dir is None:
        data_dir = Path('src/output_dl') if Path('src/output_dl').exists() else Path('output_dl')
    return train_recurrent_survival_model(data_dir, temporal='prev', mode=mode)

if __name__ == '__main__':
    data_dir = Path('src/output_dl')
    if not data_dir.exists():
        data_dir = Path('output_dl')
    train_recurrent_survival_model(data_dir, temporal='prev', mode='standard')
