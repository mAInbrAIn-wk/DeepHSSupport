"""
DEPRECATED / CONSOLIDATED WRAPPER: extended_cox_delta.py
=========================================================
Dieses Skript wurde in `src/extended_cox_survival.py` konsolidiert.
Leitet Aufrufe transparent an das konsolidierte Skript und feature_builder weiter.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import feature_builder as fb
from extended_cox_survival import train_extended_cox_model

def build_delta_panel(data_dir=None, mode='standard'):
    if data_dir is None:
        data_dir = Path('src/output_dl') if Path('src/output_dl').exists() else Path('output_dl')
    panel_df, _, _, _ = fb.build_semester_panel_df(data_dir, mode=mode, temporal='prev')
    return panel_df

def fit_extended_cox_delta(data_dir=None, mode='standard'):
    if data_dir is None:
        data_dir = Path('src/output_dl') if Path('src/output_dl').exists() else Path('output_dl')
    return train_extended_cox_model(data_dir, temporal='prev', mode=mode)

if __name__ == '__main__':
    data_dir = Path('src/output_dl')
    if not data_dir.exists():
        data_dir = Path('output_dl')
    train_extended_cox_model(data_dir, temporal='prev', mode='standard')
