"""
DEPRECATED / CONSOLIDATED WRAPPER: dynamic_deephit_delta_model.py
==================================================================
Dieses Skript wurde in `src/dynamic_deephit_model.py` konsolidiert.
Leitet Aufrufe transparent an das konsolidierte Skript weiter.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import feature_builder as fb
from dynamic_deephit_model import train_dynamic_deephit_model

def build_competing_risks_dataset_delta(data_dir=None, max_semesters=16, mode='standard'):
    if data_dir is None:
        data_dir = Path('src/output_dl') if Path('src/output_dl').exists() else Path('output_dl')
    return fb.build_semester_sequence_tensor(data_dir, max_semesters=max_semesters, mode=mode, temporal='prev', target_type='competing_risks')

def train_dynamic_deephit_delta_model(data_dir=None, mode='standard'):
    if data_dir is None:
        data_dir = Path('src/output_dl') if Path('src/output_dl').exists() else Path('output_dl')
    return train_dynamic_deephit_model(data_dir, temporal='prev', mode=mode)

if __name__ == '__main__':
    data_dir = Path('src/output_dl')
    if not data_dir.exists():
        data_dir = Path('output_dl')
    train_dynamic_deephit_model(data_dir, temporal='prev', mode='standard')
