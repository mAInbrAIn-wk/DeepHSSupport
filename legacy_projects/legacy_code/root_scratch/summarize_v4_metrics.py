import json
from pathlib import Path
import pandas as pd

metrics_dir = Path('src/output_dl_v4/metrics')
all_metrics = {}

for f in sorted(metrics_dir.glob('*.json')):
    try:
        with open(f, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
            all_metrics[f.stem] = data
    except Exception as e:
        print(f'Fehler bei {f}: {e}')

print(f'Gefundene Metrik-Dateien: {len(all_metrics)}')
for name, data in all_metrics.items():
    print(f'=== {name} ===')
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (int, float)):
                print(f'  {k}: {v:.4f}')
            elif isinstance(v, dict):
                print(f'  {k}: {v}')
    print()
