import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, 'src')

output_dir = "C:/Users/wilfr/.gemini/antigravity/brain/16832ed6-a522-415e-9395-ef24e16fef79"
base_dir = "C:/GitHub_public/Abschlussprojekt/src/output_v4_grid"

def load_abschluesse(scenario, uni):
    path = os.path.join(base_dir, scenario, f"universe_{uni}", "abschluesse.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def is_dropout(status):
    return status in ['abgebrochen', 'exmatrikuliert', 'zeitueberschreitung']

def compare_universes(df1, df2, name1, name2):
    if df1 is None or df2 is None:
        return None
    merged = pd.merge(df1[['studierenden_id', 'status']], df2[['studierenden_id', 'status']], on='studierenden_id', suffixes=('_1', '_2'))
    
    grad_to_drop = 0
    drop_to_grad = 0
    
    for _, row in merged.iterrows():
        s1 = row['status_1']
        s2 = row['status_2']
        if s1 == 'abgeschlossen' and is_dropout(s2):
            grad_to_drop += 1
        elif is_dropout(s1) and s2 == 'abgeschlossen':
            drop_to_grad += 1
            
    net = drop_to_grad - grad_to_drop
    
    return {
        'Vergleich': f"{name1} vs {name2}",
        'Abschluss zu Dropout': grad_to_drop,
        'Dropout zu Abschluss': drop_to_grad,
        'Netto-Effekt': net
    }

results_t1 = []
universes = ['B', 'C', 'D', 'E', 'F', 'G', 'H']
df_a_s01 = load_abschluesse('S01_baseline', 'A')

for u in universes:
    df_u = load_abschluesse('S01_baseline', u)
    res = compare_universes(df_a_s01, df_u, 'S01-A', f'S01-{u}')
    if res:
        results_t1.append(res)

df_a_s07 = load_abschluesse('S07_noise_half', 'A')
df_b_s07 = load_abschluesse('S07_noise_half', 'B')
res = compare_universes(df_a_s07, df_b_s07, 'S07-A', 'S07-B')
if res: results_t1.append(res)
res = compare_universes(df_a_s01, df_a_s07, 'S01-A', 'S07-A')
if res: results_t1.append(res)

df_a_s08 = load_abschluesse('S08_noise_double', 'A')
df_b_s08 = load_abschluesse('S08_noise_double', 'B')
res = compare_universes(df_a_s08, df_b_s08, 'S08-A', 'S08-B')
if res: results_t1.append(res)
res = compare_universes(df_a_s01, df_a_s08, 'S01-A', 'S08-A')
if res: results_t1.append(res)

df_res_t1 = pd.DataFrame(results_t1)
with open(os.path.join(output_dir, 'migrationsanalyse_v4.md'), 'w') as f:
    f.write("# Migrationsanalyse: Statuswechsel zwischen Welten (V4)\n\n")
    f.write(df_res_t1.to_markdown(index=False))
    f.write("\n\n_Hinweis: Positiver Netto-Effekt bedeutet mehr Studierende mit Abschluss in der zweiten Gruppe als in der ersten._\n")

scenarios = {'S01_baseline': 'Baseline (20h)', 'S09_timecost_0h': '0h', 'S10_timecost_60h': '60h'}
all_stats = []
plot_data = []

for sc_id, sc_name in scenarios.items():
    df_abs = load_abschluesse(sc_id, 'A')
    p_path = os.path.join(base_dir, sc_id, "universe_A", "pruefungen.csv")
    if df_abs is not None and os.path.exists(p_path):
        df_pruef = pd.read_csv(p_path)
        
        counts = df_pruef.groupby('studierenden_id').size().reset_index(name='versuche')
        last_grades = df_pruef.sort_values(['semester_id'], ascending=True).groupby('studierenden_id').last().reset_index()[['studierenden_id', 'note']]
        
        df_merged = df_abs.merge(counts, on='studierenden_id', how='left').merge(last_grades, on='studierenden_id', how='left')
        df_merged['szenario'] = sc_name
        df_merged['status_group'] = df_merged['status'].apply(lambda x: 'dropout' if is_dropout(x) else 'abgeschlossen')
        
        for st in ['abgeschlossen', 'dropout']:
            sub = df_merged[df_merged['status_group'] == st]
            if len(sub) == 0: continue
            
            avg_dur = sub['studiendauer_semester'].mean()
            avg_vers = sub['versuche'].mean()
            avg_grade = sub['abschlussnote'].mean() if st == 'abgeschlossen' else sub['note'].mean()
            
            all_stats.append({
                'Szenario': sc_name,
                'Status': st,
                'Ø Studiendauer': avg_dur,
                'Ø Versuche': avg_vers,
                'Ø Note (letzte/Abschluss)': avg_grade
            })
            
        plot_data.append(df_merged)

df_plot = pd.concat(plot_data)

df_res_t2 = pd.DataFrame(all_stats)
with open(os.path.join(output_dir, 'zeitkosten_studienebene_v4.md'), 'w') as f:
    f.write("# Zeitkosten-Verteilung auf Studierendenebene (Uni A)\n\n")
    f.write(df_res_t2.to_markdown(index=False))

fig, axes = plt.subplots(3, 1, figsize=(10, 15))

sns.boxplot(data=df_plot, x='studiendauer_semester', y='szenario', hue='status_group', ax=axes[0])
axes[0].set_title('Studiendauer-Verteilung nach Status')

sns.boxplot(data=df_plot, x='versuche', y='szenario', hue='status_group', ax=axes[1])
axes[1].set_title('Prüfungsversuche pro Studierendem')

sns.histplot(data=df_plot[df_plot['status_group'] == 'abgeschlossen'], x='abschlussnote', hue='szenario', element='step', ax=axes[2])
axes[2].set_title('Abschlussnoten der Absolventen')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plots_zeitkosten_studienebene.png'))
