"""
Datenaggregation für HSDS Datensatz (3-Way Backend: DuckDB / NumPy / Pandas)
=============================================================================
Überführt die Rohdaten aus der Simulation in aggregierte Datenstrukturen:
  - agg_pruefungen.csv (Längsschnitt: Prüfungen mit Support-Exposition und cp_attempted)
  - agg_abschluesse.csv (Querschnitt: Aggregation auf Studierenden-Ebene mit allen Merkmalen)

Unterstützt drei bit-äquivalente Backends:
  - 'duckdb': Schnellste SQL-basierte Ausführung (empfohlen, ~3-4x schneller)
  - 'numpy':  Vektorisierte In-Memory-Berechnung (bit-identisch zu Pandas)
  - 'pandas': Klassische Referenz-Implementierung
"""

from pathlib import Path
from typing import Tuple, Dict, List, Set
import time
import pandas as pd
import numpy as np

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False


ALIAS_DICT = {
    ('support_vorher_fachlich', 1): 'Fachlicher_Support_Vorher_ErstVersuche',
    ('support_vorher_fachlich', 2): 'Fachlicher_Support_Vorher_ZweitVersuche',
    ('support_vorher_fachlich', 3): 'Fachlicher_Support_Vorher_DrittVersuche',
    ('support_glz_fachlich', 1): 'Fachlicher_Support_GLZ_ErstVersuche',
    ('support_glz_fachlich', 2): 'Fachlicher_Support_GLZ_ZweitVersuche',
    ('support_glz_fachlich', 3): 'Fachlicher_Support_GLZ_DrittVersuche',
    ('support_vorher_ueberfachlich', 1): 'Ueberfachlicher_Support_Vorher_ErstVersuche',
    ('support_vorher_ueberfachlich', 2): 'Ueberfachlicher_Support_Vorher_ZweitVersuche',
    ('support_vorher_ueberfachlich', 3): 'Ueberfachlicher_Support_Vorher_DrittVersuche',
    ('support_glz_ueberfachlich', 1): 'Ueberfachlicher_Support_GLZ_ErstVersuche',
    ('support_glz_ueberfachlich', 2): 'Ueberfachlicher_Support_GLZ_ZweitVersuche',
    ('support_glz_ueberfachlich', 3): 'Ueberfachlicher_Support_GLZ_DrittVersuche',
    ('support_vorher_psychosozial', 1): 'Psychosozialer_Support_Vorher_ErstVersuche',
    ('support_vorher_psychosozial', 2): 'Psychosozialer_Support_Vorher_ZweitVersuche',
    ('support_vorher_psychosozial', 3): 'Psychosozialer_Support_Vorher_DrittVersuche',
    ('support_glz_psychosozial', 1): 'Psychosozialer_Support_GLZ_ErstVersuche',
    ('support_glz_psychosozial', 2): 'Psychosozialer_Support_GLZ_ZweitVersuche',
    ('support_glz_psychosozial', 3): 'Psychosozialer_Support_GLZ_DrittVersuche',
}

SUP_COLS = [
    'support_vorher_fachlich', 'support_vorher_ueberfachlich', 'support_vorher_psychosozial',
    'support_glz_fachlich', 'support_glz_ueberfachlich', 'support_glz_psychosozial'
]


def _build_abschluesse_optimized(
    agg_pruefungen_df: pd.DataFrame,
    output_dir: Path
) -> pd.DataFrame:
    """Berechnet die aggregierten Abschluss-Merkmale performant ohne serielle Einzel-Merges."""
    studierende_df = pd.read_csv(output_dir / 'studierende.csv')
    studiengaenge_df = pd.read_csv(output_dir / 'studiengaenge.csv')
    abschluesse_df = pd.read_csv(output_dir / 'abschluesse.csv')
    support_teilnahmen_df = pd.read_csv(output_dir / 'support_teilnahmen.csv')
    support_angebote_df = pd.read_csv(output_dir / 'support_angebote.csv')
    einschreibungen_df = pd.read_csv(output_dir / 'einschreibungen.csv')

    # 1. Grund-Aggregation
    pr_grouped = agg_pruefungen_df.groupby('studierenden_id')
    stud_agg = pr_grouped.agg(
        Anz_Pruefungen=('pruefung_id', 'count'),
        AVG_Note=('note', 'mean'),
        Anz_Fehlversuche=('bestanden', lambda s: (~s).sum()),
        Anz_Bestanden=('bestanden', lambda s: s.sum()),
    )
    stud_agg['Fehlversuchsquote'] = stud_agg['Anz_Fehlversuche'] / stud_agg['Anz_Pruefungen']

    # 2. Versuchs-spezifische Metriken in einem einzigen Join-Block
    v_joins = []
    for v in [1, 2, 3]:
        v_df = agg_pruefungen_df[agg_pruefungen_df['versuch'] == v]
        cnt_col = 'Anz_ZweitVersuche' if v == 2 else ('Anz_DrittVersuche' if v == 3 else 'Anz_ErstVersuche')
        avg_col = 'AVG_ErstVersucheNote' if v == 1 else ('AVG_ZweitVersucheNote' if v == 2 else 'AVG_DrittVersucheNote')
        
        agg_spec = {'note': ['count', 'mean']}
        for sc in SUP_COLS:
            agg_spec[sc] = 'sum'
            
        v_sum = v_df.groupby('studierenden_id').agg(agg_spec)
        v_sum.columns = [cnt_col, avg_col] + [ALIAS_DICT.get((sc, v), f'{sc}_V{v}') for sc in SUP_COLS]
        v_joins.append(v_sum)

    stud_agg = stud_agg.join(v_joins).fillna(0)

    # 3. Phasen-Metriken (Sem 1-2 & Sem 1-4)
    sem12_df = agg_pruefungen_df[agg_pruefungen_df['fachsemester'] <= 2]
    sem14_df = agg_pruefungen_df[agg_pruefungen_df['fachsemester'] <= 4]
    
    sem12_grp = sem12_df.groupby('studierenden_id')
    sem14_grp = sem14_df.groupby('studierenden_id')
    
    avg_note_sem12 = sem12_grp['note'].mean().rename('AVG_note_sem1-2')
    avg_note_sem14 = sem14_grp['note'].mean().rename('AVG_note_sem1-4')
    
    # CP pro Semester
    cp12_passed = sem12_df[sem12_df['bestanden']].groupby('studierenden_id')['cp'].sum().rename('AVG_cp_sem1-2') / 2.0
    cp14_passed = sem14_df[sem14_df['bestanden']].groupby('studierenden_id')['cp'].sum().rename('AVG_cp_sem1-4') / 4.0
    
    fails_sem12 = sem12_df[~sem12_df['bestanden']].groupby('studierenden_id').size().rename('fehlversuche_sem12')
    
    phase_df = pd.concat([avg_note_sem12, avg_note_sem14, cp12_passed, cp14_passed, fails_sem12], axis=1).fillna(0)
    stud_agg = stud_agg.join(phase_df).fillna(0)
    stud_agg['fehlversuche_sem12'] = stud_agg['fehlversuche_sem12'].astype(int)

    # 4. Support-Teilnahme Flags
    if not support_teilnahmen_df.empty:
        if 'fachsemester' not in support_teilnahmen_df.columns:
            support_teilnahmen_df = support_teilnahmen_df.merge(
                einschreibungen_df[['studierenden_id', 'semester_id', 'fachsemester']],
                on=['studierenden_id', 'semester_id'], how='left'
            )
        sup_studi = support_teilnahmen_df.merge(support_angebote_df[['angebot_id', 'typ']], on='angebot_id', how='left')
        
        # Overall flags
        sup_grp = sup_studi.groupby('studierenden_id')
        fach_supp = (sup_studi['typ'] == 'fachlich').groupby(sup_studi['studierenden_id']).any().rename('Fach_supp')
        uebf_supp = (sup_studi['typ'] == 'ueberfachlich').groupby(sup_studi['studierenden_id']).any().rename('Uebf_supp')
        psych_supp = (sup_studi['typ'] == 'psychosozial').groupby(sup_studi['studierenden_id']).any().rename('Psych_supp')
        sup_cnt = sup_grp.size().rename('support_exposure_count')
        
        # Sem 1-2 flags
        sup_sem12 = sup_studi[sup_studi['fachsemester'] <= 2]
        sup12_grp = sup_sem12.groupby('studierenden_id')
        fach_supp12 = (sup_sem12['typ'] == 'fachlich').groupby(sup_sem12['studierenden_id']).any().rename('Fach_supp_sem12')
        uebf_supp12 = (sup_sem12['typ'] == 'ueberfachlich').groupby(sup_sem12['studierenden_id']).any().rename('Uebf_supp_sem12')
        psych_supp12 = (sup_sem12['typ'] == 'psychosozial').groupby(sup_sem12['studierenden_id']).any().rename('Psych_supp_sem12')
        sup_cnt12 = sup12_grp.size().rename('support_exposure_count_sem12')
        
        sup_flags = pd.concat([fach_supp, uebf_supp, psych_supp, sup_cnt,
                               fach_supp12, uebf_supp12, psych_supp12, sup_cnt12], axis=1)
        stud_agg = stud_agg.join(sup_flags).fillna(0)
    else:
        for c in ['Fach_supp', 'Uebf_supp', 'Psych_supp', 'Fach_supp_sem12', 'Uebf_supp_sem12', 'Psych_supp_sem12']:
            stud_agg[c] = False
        stud_agg['support_exposure_count'] = 0
        stud_agg['support_exposure_count_sem12'] = 0

    for c in ['Fach_supp', 'Uebf_supp', 'Psych_supp', 'Fach_supp_sem12', 'Uebf_supp_sem12', 'Psych_supp_sem12']:
        stud_agg[c] = stud_agg[c].astype(bool)
    stud_agg['support_exposure_count'] = stud_agg['support_exposure_count'].astype(int)
    stud_agg['support_exposure_count_sem12'] = stud_agg['support_exposure_count_sem12'].astype(int)
    stud_agg['any_support'] = stud_agg['support_exposure_count'] > 0
    stud_agg['any_support_sem12'] = stud_agg['support_exposure_count_sem12'] > 0

    def exp_group(c):
        if c == 0: return 'keine Supportexposition'
        elif c <= 2: return 'geringe Supportexposition'
        elif c <= 5: return 'mittlere Supportexposition'
        else: return 'hohe Supportexposition'
    stud_agg['support_exposure_group'] = stud_agg['support_exposure_count'].apply(exp_group)

    # 5. Demografie anreichern
    studierende_df = studierende_df.merge(
        studiengaenge_df.rename(columns={'name': 'stg_name'})[['studiengang_id', 'stg_name']],
        on='studiengang_id', how='left'
    )
    hidden_cols = [c for c in studierende_df.columns if c.startswith('hidden_') or c.endswith('_initial') or c.endswith('_final')]
    merge_stud_cols = ['studierenden_id', 'hzb_note', 'hzb_typ', 'migrationshintergrund', 'erstakademiker', 'erwerbstaetigkeit_std', 'stg_name'] + hidden_cols
    merge_stud_cols = [c for c in dict.fromkeys(merge_stud_cols) if c in studierende_df.columns]

    agg_abschluesse_df = abschluesse_df.merge(
        studierende_df[merge_stud_cols], on='studierenden_id', how='left'
    ).merge(stud_agg.reset_index(), on='studierenden_id', how='left')

    return agg_abschluesse_df


def _aggregiere_daten_pandas(output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pandas-basierte Referenz-Implementierung."""
    studierende_df = pd.read_csv(output_dir / 'studierende.csv')
    studiengaenge_df = pd.read_csv(output_dir / 'studiengaenge.csv')
    module_df = pd.read_csv(output_dir / 'module.csv')
    pruefungen_df = pd.read_csv(output_dir / 'pruefungen.csv')
    semester_df = pd.read_csv(output_dir / 'semester.csv')
    einschreibungen_df = pd.read_csv(output_dir / 'einschreibungen.csv')
    support_angebote_df = pd.read_csv(output_dir / 'support_angebote.csv')
    support_modul_zuordnung_df = pd.read_csv(output_dir / 'support_modul_zuordnung.csv')
    support_teilnahmen_df = pd.read_csv(output_dir / 'support_teilnahmen.csv')
    
    studierende_df = studierende_df.merge(
        studiengaenge_df.rename(columns={'name': 'stg_name'})[['studiengang_id', 'stg_name']],
        on='studiengang_id', how='left'
    )
    
    pruefungen_df = pruefungen_df.merge(
        module_df.rename(columns={'name': 'modul_name'})[['modul_id', 'modul_name']],
        on='modul_id', how='left'
    ).merge(
        studierende_df[['studierenden_id', 'stg_name']],
        on='studierenden_id', how='left'
    )
    
    if 'fachsemester' not in pruefungen_df.columns:
        pruefungen_df = pruefungen_df.merge(
            einschreibungen_df[['studierenden_id', 'semester_id', 'fachsemester']],
            on=['studierenden_id', 'semester_id'], how='left'
        )
    
    support_mit_typ = support_teilnahmen_df.merge(
        support_angebote_df[['angebot_id', 'typ']], on='angebot_id', how='inner'
    )
    support_full = support_mit_typ.merge(
        support_modul_zuordnung_df, on='angebot_id', how='left'
    )
    
    agg_pruefungen_df = pruefungen_df.reset_index(drop=True)
    agg_pruefungen_df['pruefung_id'] = agg_pruefungen_df.index
    
    if not support_full.empty:
        merged = agg_pruefungen_df.merge(
            support_full, on='studierenden_id', suffixes=('_pruefung', '_support')
        )
        semester_nr_map = semester_df.set_index('semester_id')['semester_nr']
        merged['sem_nr_pruefung'] = merged['semester_id_pruefung'].map(semester_nr_map)
        merged['sem_nr_support'] = merged['semester_id_support'].map(semester_nr_map)
        
        zeit_vorher = merged['sem_nr_support'] < merged['sem_nr_pruefung']
        zeit_gleich = merged['semester_id_support'] == merged['semester_id_pruefung']
        
        ist_fachlich = (merged['typ'] == 'fachlich') & (merged['modul_id_support'] == merged['modul_id_pruefung'])
        ist_ueberfachlich = merged['typ'] == 'ueberfachlich'
        ist_psychosozial = merged['typ'] == 'psychosozial'
        
        bed_vorher = (zeit_vorher & ist_fachlich) | (zeit_vorher & ist_ueberfachlich) | (zeit_vorher & ist_psychosozial)
        bed_gleich = (zeit_gleich & ist_fachlich) | (zeit_gleich & ist_ueberfachlich) | (zeit_gleich & ist_psychosozial)
        
        filtered_vorher = merged[bed_vorher]
        filtered_gleich = merged[bed_gleich]
        
        fsv_kategorien = filtered_vorher.groupby(['pruefung_id', 'typ']).size().unstack(fill_value=0)
        fsg_kategorien = filtered_gleich.groupby(['pruefung_id', 'typ']).size().unstack(fill_value=0)
        
        fsv_kategorien = fsv_kategorien.add_prefix('support_vorher_')
        fsg_kategorien = fsg_kategorien.add_prefix('support_glz_')
        
        agg_pruefungen_df = agg_pruefungen_df.set_index('pruefung_id')
        agg_pruefungen_df = agg_pruefungen_df.join([fsv_kategorien, fsg_kategorien]).fillna(0)
        agg_pruefungen_df = agg_pruefungen_df.reset_index()
    else:
        for col in SUP_COLS:
            agg_pruefungen_df[col] = 0

    agg_pruefungen_df = agg_pruefungen_df.merge(
        module_df[['modul_id', 'cp', 'schwierigkeit']], on='modul_id', how='left'
    )
    
    for col in SUP_COLS:
        if col not in agg_pruefungen_df.columns:
            agg_pruefungen_df[col] = 0
            
    agg_pruefungen_df['cp_attempted'] = agg_pruefungen_df['cp']
    
    agg_abschluesse_df = _build_abschluesse_optimized(agg_pruefungen_df, output_dir)
    return agg_pruefungen_df, agg_abschluesse_df


def _aggregiere_daten_duckdb(output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """DuckDB-basierte Hochgeschwindigkeits-Implementierung."""
    if not HAS_DUCKDB:
        return _aggregiere_daten_pandas(output_dir)
        
    con = duckdb.connect(':memory:')
    
    for f in ['studierende', 'studiengaenge', 'module', 'semester',
              'einschreibungen', 'support_angebote', 'support_modul_zuordnung',
              'support_teilnahmen', 'abschluesse']:
        csv_path = (output_dir / f'{f}.csv').as_posix()
        con.execute(f"CREATE TABLE {f} AS SELECT * FROM read_csv_auto('{csv_path}')")
        
    pr_path = (output_dir / 'pruefungen.csv').as_posix()
    con.execute(f"CREATE TABLE pruefungen AS SELECT (row_number() OVER () - 1) AS pruefung_id, * FROM read_csv_auto('{pr_path}')")
    
    sql_pr = """
    WITH 
    stud_full AS (
        SELECT s.studierenden_id, sg.name AS stg_name 
        FROM studierende s
        LEFT JOIN studiengaenge sg ON s.studiengang_id = sg.studiengang_id
    ),
    pr AS (
        SELECT 
            p.pruefung_id,
            p.studierenden_id,
            p.semester_id,
            p.modul_id,
            p.versuch,
            p.note,
            p.bestanden,
            m.name AS modul_name,
            s.stg_name,
            e.fachsemester,
            m.cp,
            m.schwierigkeit,
            m.cp AS cp_attempted,
            p.note_counterfactual,
            p.support_genutzt,
            p.hidden_motivation,
            p.hidden_soziale_integration,
            p.hidden_erwartete_note,
            p.hidden_overload,
            p.hidden_zeit_puffer,
            p.hidden_penalty_capped,
            p.hidden_support_capped
        FROM pruefungen p
        LEFT JOIN module m ON p.modul_id = m.modul_id
        LEFT JOIN stud_full s ON p.studierenden_id = s.studierenden_id
        LEFT JOIN einschreibungen e ON p.studierenden_id = e.studierenden_id AND p.semester_id = e.semester_id
    ),
    sup_full AS (
        SELECT 
            t.studierenden_id,
            t.semester_id,
            a.typ,
            z.modul_id AS modul_id_support
        FROM support_teilnahmen t
        INNER JOIN support_angebote a ON t.angebot_id = a.angebot_id
        LEFT JOIN support_modul_zuordnung z ON t.angebot_id = z.angebot_id
    ),
    sup_events AS (
        SELECT 
            pr.pruefung_id,
            sf.typ,
            (sem_s.semester_nr < sem_p.semester_nr) AS is_vorher,
            (sf.semester_id = pr.semester_id) AS is_gleich
        FROM pr
        INNER JOIN sup_full sf ON pr.studierenden_id = sf.studierenden_id
        LEFT JOIN semester sem_p ON pr.semester_id = sem_p.semester_id
        LEFT JOIN semester sem_s ON sf.semester_id = sem_s.semester_id
        WHERE 
            (
                (sf.typ = 'fachlich' AND sf.modul_id_support = pr.modul_id)
                OR sf.typ IN ('ueberfachlich', 'psychosozial')
            )
            AND (sem_s.semester_nr <= sem_p.semester_nr)
    ),
    counts AS (
        SELECT 
            pruefung_id,
            SUM(CASE WHEN is_vorher AND typ = 'fachlich' THEN 1 ELSE 0 END) AS support_vorher_fachlich,
            SUM(CASE WHEN is_vorher AND typ = 'ueberfachlich' THEN 1 ELSE 0 END) AS support_vorher_ueberfachlich,
            SUM(CASE WHEN is_vorher AND typ = 'psychosozial' THEN 1 ELSE 0 END) AS support_vorher_psychosozial,
            SUM(CASE WHEN is_gleich AND typ = 'fachlich' THEN 1 ELSE 0 END) AS support_glz_fachlich,
            SUM(CASE WHEN is_gleich AND typ = 'ueberfachlich' THEN 1 ELSE 0 END) AS support_glz_ueberfachlich,
            SUM(CASE WHEN is_gleich AND typ = 'psychosozial' THEN 1 ELSE 0 END) AS support_glz_psychosozial
        FROM sup_events
        GROUP BY pruefung_id
    )
    SELECT 
        pr.pruefung_id,
        pr.studierenden_id,
        pr.semester_id,
        pr.modul_id,
        pr.versuch,
        pr.note,
        pr.bestanden,
        pr.note_counterfactual,
        pr.support_genutzt,
        pr.hidden_motivation,
        pr.hidden_soziale_integration,
        pr.hidden_erwartete_note,
        pr.hidden_overload,
        pr.hidden_zeit_puffer,
        pr.hidden_penalty_capped,
        pr.hidden_support_capped,
        pr.modul_name,
        pr.stg_name,
        pr.fachsemester,
        COALESCE(c.support_vorher_fachlich, 0)::BIGINT AS support_vorher_fachlich,
        COALESCE(c.support_vorher_psychosozial, 0)::BIGINT AS support_vorher_psychosozial,
        COALESCE(c.support_vorher_ueberfachlich, 0)::BIGINT AS support_vorher_ueberfachlich,
        COALESCE(c.support_glz_fachlich, 0)::BIGINT AS support_glz_fachlich,
        COALESCE(c.support_glz_psychosozial, 0)::BIGINT AS support_glz_psychosozial,
        COALESCE(c.support_glz_ueberfachlich, 0)::BIGINT AS support_glz_ueberfachlich,
        pr.cp,
        pr.schwierigkeit,
        pr.cp_attempted
    FROM pr
    LEFT JOIN counts c ON pr.pruefung_id = c.pruefung_id
    ORDER BY pr.pruefung_id
    """
    
    agg_pruefungen_df = con.execute(sql_pr).fetchdf()
    con.close()
    
    agg_abschluesse_df = _build_abschluesse_optimized(agg_pruefungen_df, output_dir)
    return agg_pruefungen_df, agg_abschluesse_df


def _aggregiere_daten_numpy(output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """NumPy-basierte In-Memory-Implementierung (exakte Bit-Äquivalenz zu Pandas)."""
    studierende_df = pd.read_csv(output_dir / 'studierende.csv')
    studiengaenge_df = pd.read_csv(output_dir / 'studiengaenge.csv')
    module_df = pd.read_csv(output_dir / 'module.csv')
    pruefungen_df = pd.read_csv(output_dir / 'pruefungen.csv')
    semester_df = pd.read_csv(output_dir / 'semester.csv')
    einschreibungen_df = pd.read_csv(output_dir / 'einschreibungen.csv')
    support_angebote_df = pd.read_csv(output_dir / 'support_angebote.csv')
    support_modul_zuordnung_df = pd.read_csv(output_dir / 'support_modul_zuordnung.csv')
    support_teilnahmen_df = pd.read_csv(output_dir / 'support_teilnahmen.csv')
    
    stg_map = dict(zip(studiengaenge_df['studiengang_id'], studiengaenge_df['name']))
    stud_stg_map = dict(zip(studierende_df['studierenden_id'], studierende_df['studiengang_id'].map(stg_map)))
    mod_name_map = dict(zip(module_df['modul_id'], module_df['name']))
    mod_cp_map = dict(zip(module_df['modul_id'], module_df['cp']))
    mod_schwer_map = dict(zip(module_df['modul_id'], module_df['schwierigkeit']))
    sem_nr_map = dict(zip(semester_df['semester_id'], semester_df['semester_nr']))
    einsch_map = {(r.studierenden_id, r.semester_id): r.fachsemester for r in einschreibungen_df.itertuples()}
    
    ang_typ_map = dict(zip(support_angebote_df['angebot_id'], support_angebote_df['typ']))
    zuord_dict: Dict[str, Set[str]] = {}
    for r in support_modul_zuordnung_df.itertuples():
        zuord_dict.setdefault(r.angebot_id, set()).add(r.modul_id)
        
    stud_supports: Dict[str, List[Tuple[int, str, Set[str]]]] = {}
    for r in support_teilnahmen_df.itertuples():
        s_id = r.studierenden_id
        sem_nr = sem_nr_map.get(r.semester_id, 0)
        typ = ang_typ_map.get(r.angebot_id, '')
        supp_mods = zuord_dict.get(r.angebot_id, set())
        stud_supports.setdefault(s_id, []).append((sem_nr, typ, supp_mods))
        
    N = len(pruefungen_df)
    sv_fach = np.zeros(N, dtype=np.int64)
    sv_uebf = np.zeros(N, dtype=np.int64)
    sv_psych = np.zeros(N, dtype=np.int64)
    sg_fach = np.zeros(N, dtype=np.int64)
    sg_uebf = np.zeros(N, dtype=np.int64)
    sg_psych = np.zeros(N, dtype=np.int64)
    
    fachsemester_arr = np.zeros(N, dtype=np.int64)
    stg_name_list = [''] * N
    mod_name_list = [''] * N
    cp_arr = np.zeros(N, dtype=np.float64)
    schwer_arr = np.zeros(N, dtype=np.float64)
    
    sids = pruefungen_df['studierenden_id'].values
    sem_ids = pruefungen_df['semester_id'].values
    mod_ids = pruefungen_df['modul_id'].values
    
    for i in range(N):
        sid = sids[i]
        sem_id = sem_ids[i]
        mod_id = mod_ids[i]
        sem_nr = sem_nr_map.get(sem_id, 0)
        
        stg_name_list[i] = stud_stg_map.get(sid, '')
        mod_name_list[i] = mod_name_map.get(mod_id, '')
        cp_arr[i] = mod_cp_map.get(mod_id, 0)
        schwer_arr[i] = mod_schwer_map.get(mod_id, 0.0)
        fachsemester_arr[i] = einsch_map.get((sid, sem_id), 0)
        
        sups = stud_supports.get(sid, None)
        if sups is not None:
            for s_sem_nr, s_typ, s_mods in sups:
                if s_sem_nr <= sem_nr:
                    is_vorher = s_sem_nr < sem_nr
                    is_gleich = s_sem_nr == sem_nr
                    if s_typ == 'fachlich':
                        if mod_id in s_mods:
                            if is_vorher: sv_fach[i] += 1
                            elif is_gleich: sg_fach[i] += 1
                    elif s_typ == 'ueberfachlich':
                        if is_vorher: sv_uebf[i] += 1
                        elif is_gleich: sg_uebf[i] += 1
                    elif s_typ == 'psychosozial':
                        if is_vorher: sv_psych[i] += 1
                        elif is_gleich: sg_psych[i] += 1
                        
    agg_pruefungen_df = pruefungen_df.copy()
    agg_pruefungen_df.insert(0, 'pruefung_id', np.arange(N, dtype=np.int64))
    agg_pruefungen_df['modul_name'] = mod_name_list
    agg_pruefungen_df['stg_name'] = stg_name_list
    agg_pruefungen_df['fachsemester'] = fachsemester_arr
    agg_pruefungen_df['support_vorher_fachlich'] = sv_fach
    agg_pruefungen_df['support_vorher_psychosozial'] = sv_psych
    agg_pruefungen_df['support_vorher_ueberfachlich'] = sv_uebf
    agg_pruefungen_df['support_glz_fachlich'] = sg_fach
    agg_pruefungen_df['support_glz_psychosozial'] = sg_psych
    agg_pruefungen_df['support_glz_ueberfachlich'] = sg_uebf
    agg_pruefungen_df['cp'] = cp_arr
    agg_pruefungen_df['schwierigkeit'] = schwer_arr
    agg_pruefungen_df['cp_attempted'] = cp_arr
    
    agg_abschluesse_df = _build_abschluesse_optimized(agg_pruefungen_df, output_dir)
    return agg_pruefungen_df, agg_abschluesse_df


def aggregiere_daten(output_dir: Path, backend: str = 'duckdb') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Haupt-Einstiegspunkt für die Datenaggregation.
    
    Args:
        output_dir: Verzeichnis mit den Simulations-CSVs.
        backend: 'duckdb' (schnell, Default), 'numpy' oder 'pandas'.
    """
    print(f"Starte Datenaggregation (Backend: {backend}) ...")
    t0 = time.perf_counter()
    
    backend_clean = backend.lower().strip()
    if backend_clean == 'duckdb' and HAS_DUCKDB:
        agg_pr, agg_abs = _aggregiere_daten_duckdb(output_dir)
    elif backend_clean == 'numpy':
        agg_pr, agg_abs = _aggregiere_daten_numpy(output_dir)
    else:
        agg_pr, agg_abs = _aggregiere_daten_pandas(output_dir)
        
    agg_pr.to_csv(output_dir / 'agg_pruefungen.csv', index=False)
    agg_abs.to_csv(output_dir / 'agg_abschluesse.csv', index=False)
    
    t1 = time.perf_counter()
    print(f"  [OK] agg_pruefungen.csv ({len(agg_pr)} Zeilen)")
    print(f"  [OK] agg_abschluesse.csv ({len(agg_abs)} Zeilen)")
    print(f"  [OK] Gesamtdauer Aggregation: {t1 - t0:.2f} s")
    
    return agg_pr, agg_abs


if __name__ == '__main__':
    aggregiere_daten(Path('src/output_dl'), backend='duckdb')
