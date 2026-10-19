import json
from pathlib import Path

# Check if autoregressive deep transformer was run in output_dl_v4
models_v4 = list(Path('src/output_dl_v4/models').glob('*.keras'))
print('Modelle in output_dl_v4/models:')
for m in models_v4:
    print(' ', m.name)

# Check gradeblind in output_dl_v4/metrics
metrics_v4 = list(Path('src/output_dl_v4/metrics').glob('*gradeblind*.json'))
print('\nGradeblind Metriken in output_dl_v4/metrics:')
for m in metrics_v4:
    print(' ', m.name)
