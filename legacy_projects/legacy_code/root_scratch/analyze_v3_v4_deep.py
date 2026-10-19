import pandas as pd
import numpy as np
from pathlib import Path

# Echte simulierte Daten laden
v3_stud = pd.read_csv('src/output_dl_seed99999/studierende.csv')
v4_stud = pd.read_csv('src/output_v4_test/studierende.csv')
v3_ab = pd.read_csv('src/output_dl_seed99999/abschluesse.csv')
v4_ab = pd.read_csv('src/output_v4_test/abschluesse.csv')
v3_pr = pd.read_csv('src/output_dl_seed99999/pruefungen.csv')
v4_pr = pd.read_csv('src/output_v4_test/pruefungen.csv')
v3_supp = pd.read_csv('src/output_dl_seed99999/support_teilnahmen.csv')
v4_supp = pd.read_csv('src/output_v4_test/support_teilnahmen.csv')

print('=== STUDIERENDE (Initialwerte) ===')
for col in ['alter', 'hzb_note', 'motivation', 'soziale_integration', 'erwerbstaetigkeit_std']:
    if col in v3_stud.columns and col in v4_stud.columns:
        v3m = v3_stud[col].mean()
        v3s = v3_stud[col].std()
        v4m = v4_stud[col].mean()
        v4s = v4_stud[col].std()
        print(f'  {col:30s}  V3: mean={v3m:.3f} std={v3s:.3f}  |  V4: mean={v4m:.3f} std={v4s:.3f}')

print('\n=== ABSCHLUSS-STATUS ===')
print('V3.6:')
print(v3_ab['status'].value_counts(normalize=True).sort_index() * 100)
print('\nV4:')
print(v4_ab['status'].value_counts(normalize=True).sort_index() * 100)

print('\n=== PRUEFUNGEN ===')
v3_pass_rate = v3_pr['bestanden'].mean() * 100
v4_pass_rate = v4_pr['bestanden'].mean() * 100
v3_note_pass = v3_pr[v3_pr['bestanden'] == True]['note'].mean()
v4_note_pass = v4_pr[v4_pr['bestanden'] == True]['note'].mean()
print(f'V3: N={len(v3_pr):,}  Bestanden={v3_pass_rate:.1f}%  Note(bestanden)={v3_note_pass:.2f}')
print(f'V4: N={len(v4_pr):,}  Bestanden={v4_pass_rate:.1f}%  Note(bestanden)={v4_note_pass:.2f}')

print('\n=== DURCHFALLER-ANALYSE ===')
v3_fails = v3_pr[v3_pr['bestanden'] == False].groupby('studierenden_id').size()
v4_fails = v4_pr[v4_pr['bestanden'] == False].groupby('studierenden_id').size()
n3 = len(v3_stud)
n4 = len(v4_stud)
print(f'V3: Studi mit >= 1 Fail: {len(v3_fails)}/{n3} ({len(v3_fails)/n3*100:.1f}%)')
print(f'V4: Studi mit >= 1 Fail: {len(v4_fails)}/{n4} ({len(v4_fails)/n4*100:.1f}%)')
print(f'V3: Mean Fails pro Studi (wenn fail): {v3_fails.mean():.2f}')
print(f'V4: Mean Fails pro Studi (wenn fail): {v4_fails.mean():.2f}')

# Drittversuch-Exma
v3_max_versuch = v3_pr.groupby(['studierenden_id', 'modul_id'])['versuch'].max()
v4_max_versuch = v4_pr.groupby(['studierenden_id', 'modul_id'])['versuch'].max()
print(f'\nV3: Modul-Versuche >= 3 (Exma-Gefahr): {(v3_max_versuch >= 3).sum():,}')
print(f'V4: Modul-Versuche >= 3 (Exma-Gefahr): {(v4_max_versuch >= 3).sum():,}')

print('\n=== SUPPORT ANALYSE ===')
v3_sa = pd.read_csv('src/output_dl_seed99999/support_angebote.csv')
v4_sa = pd.read_csv('src/output_v4_test/support_angebote.csv')

v3_supp_typed = v3_supp.merge(v3_sa[['angebot_id', 'typ']], on='angebot_id')
v4_supp_typed = v4_supp.merge(v4_sa[['angebot_id', 'typ']], on='angebot_id')

print('V3.6 Support-Nutzung nach Typ:')
print(v3_supp_typed['typ'].value_counts())
print(f'  Pro Kopf: {len(v3_supp)/n3:.2f}')

print('\nV4 Support-Nutzung nach Typ:')
print(v4_supp_typed['typ'].value_counts())
print(f'  Pro Kopf: {len(v4_supp)/n4:.2f}')

# HZB-Note Verteilung genauer
print('\n=== HZB-NOTE Verteilung ===')
print(f'V3: mean={v3_stud["hzb_note"].mean():.3f} std={v3_stud["hzb_note"].std():.3f} min={v3_stud["hzb_note"].min()} max={v3_stud["hzb_note"].max()}')
print(f'V4: mean={v4_stud["hzb_note"].mean():.3f} std={v4_stud["hzb_note"].std():.3f} min={v4_stud["hzb_note"].min()} max={v4_stud["hzb_note"].max()}')

# Wie viele "schlechte" HZB-Noten (>= 3.5)?
print(f'V3: HZB >= 3.5: {(v3_stud["hzb_note"] >= 3.5).mean()*100:.1f}%')
print(f'V4: HZB >= 3.5: {(v4_stud["hzb_note"] >= 3.5).mean()*100:.1f}%')
print(f'V3: HZB <= 1.5: {(v3_stud["hzb_note"] <= 1.5).mean()*100:.1f}%')
print(f'V4: HZB <= 1.5: {(v4_stud["hzb_note"] <= 1.5).mean()*100:.1f}%')

# Motivation Verteilung
print('\n=== MOTIVATION Verteilung ===')
print(f'V3: mean={v3_stud["motivation"].mean():.3f} std={v3_stud["motivation"].std():.3f}')
print(f'V4: mean={v4_stud["motivation"].mean():.3f} std={v4_stud["motivation"].std():.3f}')
print(f'V3: Mot < 0.3: {(v3_stud["motivation"] < 0.3).mean()*100:.2f}%  Mot > 0.8: {(v3_stud["motivation"] > 0.8).mean()*100:.2f}%')
print(f'V4: Mot < 0.3: {(v4_stud["motivation"] < 0.3).mean()*100:.2f}%  Mot > 0.8: {(v4_stud["motivation"] > 0.8).mean()*100:.2f}%')

# Soziale Integration
print('\n=== SOZIALE INTEGRATION ===')
print(f'V3: mean={v3_stud["soziale_integration"].mean():.3f} std={v3_stud["soziale_integration"].std():.3f}')
print(f'V4: mean={v4_stud["soziale_integration"].mean():.3f} std={v4_stud["soziale_integration"].std():.3f}')

# Fachlicher Support-Effekt: Noten mit/ohne fachlichem Support
print('\n=== FACHLICHER SUPPORT EFFEKT AUF NOTEN ===')
v3_fach_ids = set(v3_supp_typed[v3_supp_typed['typ'] == 'fachlich']['studierenden_id'].unique())
v4_fach_ids = set(v4_supp_typed[v4_supp_typed['typ'] == 'fachlich']['studierenden_id'].unique())

v3_with = v3_ab[v3_ab['studierenden_id'].isin(v3_fach_ids)]
v3_without = v3_ab[~v3_ab['studierenden_id'].isin(v3_fach_ids)]
v4_with = v4_ab[v4_ab['studierenden_id'].isin(v4_fach_ids)]
v4_without = v4_ab[~v4_ab['studierenden_id'].isin(v4_fach_ids)]

print(f'V3: Fachlichen Support genutzt: {len(v3_fach_ids)} ({len(v3_fach_ids)/n3*100:.1f}%)')
print(f'V4: Fachlichen Support genutzt: {len(v4_fach_ids)} ({len(v4_fach_ids)/n4*100:.1f}%)')

v3_drop_w = (v3_with['status'] != 'abgeschlossen').mean() * 100
v3_drop_wo = (v3_without['status'] != 'abgeschlossen').mean() * 100
v4_drop_w = (v4_with['status'] != 'abgeschlossen').mean() * 100
v4_drop_wo = (v4_without['status'] != 'abgeschlossen').mean() * 100
print(f'V3: Dropout MIT fach. Support: {v3_drop_w:.1f}%  OHNE: {v3_drop_wo:.1f}%')
print(f'V4: Dropout MIT fach. Support: {v4_drop_w:.1f}%  OHNE: {v4_drop_wo:.1f}%')
