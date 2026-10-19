import json
from pathlib import Path

for name in ['structural_mediation_analysis', 'mlp_baseline', 'mlp_regression']:
    p = Path(f'src/output_dl_v4/metrics/{name}_metrics.json')
    if p.exists():
        print(f'=== {name} ===')
        print(p.read_text(encoding='utf-8'))
