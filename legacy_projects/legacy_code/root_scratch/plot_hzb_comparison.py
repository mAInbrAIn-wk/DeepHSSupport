import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

v3_stud = pd.read_csv('src/output_dl_seed99999/studierende.csv')
v4_stud = pd.read_csv('src/output_v4_test/studierende.csv')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# HZB-Note aus den ECHTEN Daten
ax = axes[0]
ax.hist(v3_stud['hzb_note'], bins=30, alpha=0.5, density=True, label='V3.6 (clipped Normal)', color='blue')
ax.hist(v4_stud['hzb_note'], bins=30, alpha=0.5, density=True, label='V4 (Beta)', color='orange')
ax.axvline(3.5, color='red', linestyle='--', label='Grenze >= 3.5')
ax.set_title('HZB-Note (Echte Sim-Daten, N=50k)')
ax.set_xlabel('Note')
ax.legend()

# Detail an den Raendern
ax = axes[1]
bins_edge = np.arange(1.0, 4.1, 0.1)
v3_counts, _ = np.histogram(v3_stud['hzb_note'], bins=bins_edge)
v4_counts, _ = np.histogram(v4_stud['hzb_note'], bins=bins_edge)
bin_centers = (bins_edge[:-1] + bins_edge[1:]) / 2
ax.bar(bin_centers - 0.02, v3_counts, width=0.04, alpha=0.7, label='V3.6', color='blue')
ax.bar(bin_centers + 0.02, v4_counts, width=0.04, alpha=0.7, label='V4', color='orange')
ax.set_title('HZB-Note Detail (Rand-Verhalten)')
ax.set_xlabel('Note')
ax.set_ylabel('Anzahl')
ax.legend()

plt.tight_layout()
plt.savefig('src/output_v4_universes/plots_hzb_comparison.png', dpi=150, bbox_inches='tight')
print('Plot gespeichert!')

# Quantile
print('\nHZB-Note Quantile:')
for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
    v3q = v3_stud["hzb_note"].quantile(q)
    v4q = v4_stud["hzb_note"].quantile(q)
    print(f'  {q*100:5.1f}%: V3={v3q:.2f}  V4={v4q:.2f}')

# Wie viele Studierende mit SEHR schlechten HZB Noten?
print(f'\nV3 HZB >= 3.5: {(v3_stud["hzb_note"] >= 3.5).sum()} ({(v3_stud["hzb_note"] >= 3.5).mean()*100:.1f}%)')
print(f'V4 HZB >= 3.5: {(v4_stud["hzb_note"] >= 3.5).sum()} ({(v4_stud["hzb_note"] >= 3.5).mean()*100:.1f}%)')
print(f'V3 HZB >= 3.0: {(v3_stud["hzb_note"] >= 3.0).sum()} ({(v3_stud["hzb_note"] >= 3.0).mean()*100:.1f}%)')
print(f'V4 HZB >= 3.0: {(v4_stud["hzb_note"] >= 3.0).sum()} ({(v4_stud["hzb_note"] >= 3.0).mean()*100:.1f}%)')
print(f'V3 HZB <= 1.5: {(v3_stud["hzb_note"] <= 1.5).sum()} ({(v3_stud["hzb_note"] <= 1.5).mean()*100:.1f}%)')
print(f'V4 HZB <= 1.5: {(v4_stud["hzb_note"] <= 1.5).sum()} ({(v4_stud["hzb_note"] <= 1.5).mean()*100:.1f}%)')
