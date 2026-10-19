"""
DEPRECATED / CONSOLIDATED WRAPPER: extended_deep_survival_delta.py
==================================================================
Dieses Skript wurde in `src/extended_deep_survival.py` konsolidiert.
Leitet Aufrufe transparent an das konsolidierte Skript weiter.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extended_deep_survival import train_extended_deep_survival, breslow_cox_loss

def train_extended_deep_survival_delta(data_dir=None, mode='standard'):
    if data_dir is None:
        data_dir = Path('src/output_dl') if Path('src/output_dl').exists() else Path('output_dl')
    return train_extended_deep_survival(data_dir, temporal='prev', mode=mode)

if __name__ == '__main__':
    data_dir = Path('src/output_dl')
    if not data_dir.exists():
        data_dir = Path('output_dl')
    train_extended_deep_survival(data_dir, temporal='prev', mode='standard')
