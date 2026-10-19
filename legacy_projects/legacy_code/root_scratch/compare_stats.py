import pandas as pd
import numpy as np
from pathlib import Path

def get_stats(data_dir):
    df_ab = pd.read_csv(Path(data_dir) / 'abschluesse.csv')
    
    n_total = len(df_ab)
    status_counts = df_ab['status'].value_counts(normalize=True) * 100
    
    absolventen = df_ab[df_ab['status'] == 'abgeschlossen']
    avg_grade = absolventen['abschlussnote'].mean()
    avg_duration = absolventen['studiendauer_semester'].mean()
    
    df_pr = pd.read_csv(Path(data_dir) / 'pruefungen.csv')
    avg_fails = df_pr[df_pr['bestanden'] == False].groupby('studierenden_id').size().mean()
    
    df_supp = pd.read_csv(Path(data_dir) / 'support_teilnahmen.csv')
    supp_per_studi = len(df_supp) / n_total
    
    return {
        'N': n_total,
        'Absolviert (%)': status_counts.get('abgeschlossen', 0),
        'Freiw. Abbruch (%)': status_counts.get('abgebrochen', 0),
        'Exmatrikuliert (%)': status_counts.get('exmatrikuliert', 0),
        'Zeitueberschr. (%)': status_counts.get('zeitueberschreitung', 0),
        'Notendurchschnitt': avg_grade,
        'Studiendauer (Sem)': avg_duration,
        'Support-Nutzungen p.P.': supp_per_studi
    }

stats_v3 = get_stats('src/output_dl_seed99999')
stats_v4 = get_stats('src/output_v4_test')

print(f"{'Metrik':<25} | {'V3.6 (Welt A)':<20} | {'V4 (Welt A)':<20} | {'Delta'}")
print("-" * 80)
for k in stats_v3.keys():
    if k == 'N': continue
    v3_val = stats_v3[k]
    v4_val = stats_v4[k]
    diff = v4_val - v3_val
    print(f"{k:<25} | {v3_val:<20.2f} | {v4_val:<20.2f} | {diff:+.2f}")
