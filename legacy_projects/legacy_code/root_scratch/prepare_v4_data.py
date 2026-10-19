import os
import shutil
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path('src').absolute()))
from aggregate import aggregiere_daten

src_universes = Path('src/output_v4_universes')
target_dir = Path('src/output_dl_v4')

target_dir.mkdir(parents=True, exist_ok=True)
metrics_dir = target_dir / 'metrics'
metrics_dir.mkdir(parents=True, exist_ok=True)

# 1. Universe A als Root-Datensatz kopieren
print('1. Kopiere Universe A als Root-Datensatz nach', target_dir)
for f in (src_universes / 'universe_A').glob('*.csv'):
    shutil.copy(f, target_dir / f.name)

# 2. Sub-Universen B bis H kopieren
for u in ['B', 'C', 'D', 'E', 'F', 'G', 'H']:
    u_src = src_universes / f'universe_{u}'
    u_dst = target_dir / f'universe_{u}'
    print(f'2. Kopiere Universe {u}...')
    if u_dst.exists():
        shutil.rmtree(u_dst)
    shutil.copytree(u_src, u_dst)

# 3. Ground Truth JSON kopieren
if (src_universes / 'metrics' / 'true_macro_effects_v4.json').exists():
    shutil.copy(src_universes / 'metrics' / 'true_macro_effects_v4.json', metrics_dir / 'true_macro_effects_v4.json')
    shutil.copy(src_universes / 'metrics' / 'true_macro_effects_v4.json', metrics_dir / 'true_macro_effects_v3.json')

# 4. Aggregiere Root (Universe A)
print('4. Aggregiere Root (Universe A) mit DuckDB...')
aggregiere_daten(target_dir, backend='duckdb')

# 5. Aggregiere Sub-Universen B bis H
for u in ['B', 'C', 'D', 'E', 'F', 'G', 'H']:
    print(f'5. Aggregiere Universe {u} mit DuckDB...')
    aggregiere_daten(target_dir / f'universe_{u}', backend='duckdb')

print('\n[OK] Vorbereitung des V4-Datensatzes output_dl_v4 erfolgreich abgeschlossen!')
