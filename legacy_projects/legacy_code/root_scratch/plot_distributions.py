import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
N = 200000

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- 1. ALTER ---
ax = axes[0, 0]
v3_alter = np.clip(rng.normal(20.5, 2.8, N), 17, 45).astype(int)
rng2 = np.random.default_rng(42)
v4_alter = (17 + rng2.beta(0.125 * 20.0, (1.0 - 0.125) * 20.0, N) * (45 - 17)).astype(int)
ax.hist(v3_alter, bins=range(17, 46), alpha=0.5, density=True, label='V3.6 clip(N(20.5, 2.8), 17, 45)', color='blue')
ax.hist(v4_alter, bins=range(17, 46), alpha=0.5, density=True, label='V4 Beta(a=2.5, b=17.5) -> [17, 45]', color='orange')
ax.set_title('Alter bei Immatrikulation')
ax.set_xlabel('Alter')
ax.legend(fontsize=8)

# --- 2. HZB-NOTE ---
ax = axes[0, 1]
rng3 = np.random.default_rng(42)
v3_hzb = np.clip(rng3.normal(2.4, 0.55, N), 1.0, 4.0).round(1)
rng4 = np.random.default_rng(42)
v4_hzb = (1.0 + rng4.beta(0.466 * 20.0, (1.0 - 0.466) * 20.0, N) * 3.0).round(1)
ax.hist(v3_hzb, bins=30, alpha=0.5, density=True, label='V3.6 clip(N(2.4, 0.55), 1.0, 4.0)', color='blue')
ax.hist(v4_hzb, bins=30, alpha=0.5, density=True, label='V4 Beta -> [1.0, 4.0]', color='orange')
ax.set_title('HZB-Note')
ax.set_xlabel('Note')
ax.legend(fontsize=8)

# --- 3. MOTIVATION (Initialwert) ---
ax = axes[1, 0]
# V3: motivation_startwert=0.7, gewicht_motivation_hzb=0.08, gewicht_motivation_erwerb=0.2, rauschen=0.1
# mit hzb_note ~ 2.4, erwerb ~ 0.3
rng5 = np.random.default_rng(42)
v3_mot = np.clip(0.7 + (2.5 - v3_hzb) * 0.08 - 0.3 * 0.2 + rng5.normal(0, 0.1, N), 0.05, 1.0)
rng6 = np.random.default_rng(42)
mean_mot_v4 = np.clip(0.7 + (2.5 - v4_hzb) * 0.08 - 0.3 * 0.2, 0.01, 0.99)
v4_mot = rng6.beta(mean_mot_v4 * 20.0, (1.0 - mean_mot_v4) * 20.0)
ax.hist(v3_mot, bins=50, alpha=0.5, density=True, label='V3.6 clip(Normal + Noise, 0.05, 1.0)', color='blue')
ax.hist(v4_mot, bins=50, alpha=0.5, density=True, label='V4 Beta(mean*20, (1-mean)*20)', color='orange')
ax.set_title('Motivation (Startwert)')
ax.set_xlabel('Motivation')
ax.legend(fontsize=8)

# --- 4. SOZIALE INTEGRATION (Random-Walk-Schritt) ---
ax = axes[1, 1]
# V3: clip(current + N(0, 0.05), 0.05, 1.0) - simulieren ab current=0.6
rng7 = np.random.default_rng(42)
current = 0.6
v3_walk = np.clip(current + rng7.normal(0, 0.05, N), 0.05, 1.0)
rng8 = np.random.default_rng(42)
mean_walk = np.clip(current, 0.01, 0.99)
v4_walk = rng8.beta(mean_walk * 40.0, (1.0 - mean_walk) * 40.0, N)
ax.hist(v3_walk, bins=50, alpha=0.5, density=True, label='V3.6 clip(0.6 + N(0, 0.05), 0.05, 1.0)', color='blue')
ax.hist(v4_walk, bins=50, alpha=0.5, density=True, label='V4 Beta(0.6*40, 0.4*40)', color='orange')
ax.set_title('Soziale Integration (Random Walk ab 0.6)')
ax.set_xlabel('Soziale Integration')
ax.legend(fontsize=8)

plt.suptitle('V3.6 (Clipped Normal) vs. V4 (Beta) Verteilungsvergleich', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('src/output_v4_universes/plots_v3_vs_v4_distributions.png', dpi=150, bbox_inches='tight')
print('Plot gespeichert!')

# Statistiken ausgeben
print(f'\nAlter:       V3 mean={v3_alter.mean():.2f} std={v3_alter.std():.2f}  |  V4 mean={v4_alter.mean():.2f} std={v4_alter.std():.2f}')
print(f'HZB-Note:    V3 mean={v3_hzb.mean():.2f} std={v3_hzb.std():.2f}  |  V4 mean={v4_hzb.mean():.2f} std={v4_hzb.std():.2f}')
print(f'Motivation:  V3 mean={v3_mot.mean():.3f} std={v3_mot.std():.3f}  |  V4 mean={v4_mot.mean():.3f} std={v4_mot.std():.3f}')
print(f'  V3 Mot < 0.2: {(v3_mot < 0.2).mean()*100:.2f}%  |  V4 Mot < 0.2: {(v4_mot < 0.2).mean()*100:.2f}%')
print(f'  V3 Mot < 0.1: {(v3_mot < 0.1).mean()*100:.2f}%  |  V4 Mot < 0.1: {(v4_mot < 0.1).mean()*100:.2f}%')
print(f'  V3 Mot = 0.05 (am Clip): {(v3_mot <= 0.06).mean()*100:.2f}%  |  V4 Mot <= 0.06: {(v4_mot <= 0.06).mean()*100:.2f}%')
