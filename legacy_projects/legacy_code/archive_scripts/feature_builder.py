"""
Feature Builder & Harmonization Module (Feature Factory)
=========================================================
Stellt standardisierte, hochperformante und konsistente Datenstrukturen (3D-Tensoren für Sequenzmodelle,
2D-DataFrames für Panel- & Landmark-Modelle) für alle Modell-Klassen bereit.

Unterstützt beliebig kombinierbare Flags bzw. Modi:
- standard:    Vollständiges, bereinigtes Standard-Feature-Set.
- gradeblind:  Entfernt alle Noten-Leistungsdaten (gpa, note, delta_gpa), behält aber CP, Fehlversuche und HZB.
- blind:       Entfernt jeglichen akademischen Fortschritt (Noten, CP, Fails, Rückstand). Reine Eingangsprognose.
- oracle:      Fügt latente DGP-Variablen (Motivation, Soziale Integration, Erwartete Note) hinzu.
- realistic:   Entfernt sensible/nicht erfassbare Merkmale (Migration, Erstakademiker, Erwerb, Psych. Support, Schwierigkeit).

Temporale Steuerung:
- temporal='prev' (Default): Nutzt lokale Vorsemester-Merkmale (fails_prev, delta_cp_prev, gpa_prev).
- temporal='cum': Nutzt aufgelaufene Historie (cum_fails, cum_cp, gpa_cum).
(Support-Variablen bleiben als Treatment-Variablen vom temporal-Switch unberührt.)
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import pandas as pd

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

PADDING_VALUE = -99.0

HZB_ORDINAL_MAP = {
    'Allg. Hochschulreife': 3.0,
    'Fachgebundene HR': 2.0,
    'Fachhochschulreife': 1.0,
    'Berufl. Qualifikation': 0.0
}

STUDIENGAENGE_LIST = ['Informatik', 'BWL', 'Maschinenbau', 'Psychologie', 'Soziale Arbeit']


def _resolve_modes(mode: str = 'standard',
                   gradeblind: bool = False,
                   blind: bool = False,
                   oracle: bool = False,
                   realistic: bool = False) -> Tuple[bool, bool, bool, bool]:
    """Löst mode-String und boolesche Flags konsistent auf."""
    m = mode.lower().strip()
    if 'gradeblind' in m:
        gradeblind = True
    if 'blind' in m and 'gradeblind' not in m:
        blind = True
    if 'oracle' in m:
        oracle = True
    if 'realistic' in m or 'dsgvo' in m:
        realistic = True

    if blind:
        gradeblind = True

    return gradeblind, blind, oracle, realistic


def _load_raw_data(data_dir: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Lädt agg_abschluesse.csv und agg_pruefungen.csv mit Fallback-Prüfung."""
    data_dir = Path(data_dir)
    agg_abschluesse_path = data_dir / 'agg_abschluesse.csv'
    agg_pruefungen_path = data_dir / 'agg_pruefungen.csv'

    if not agg_abschluesse_path.exists() or not agg_pruefungen_path.exists():
        # Falls Rohdaten vorhanden sind, automatisch on-the-fly aggregieren
        if (data_dir / 'abschluesse.csv').exists() and (data_dir / 'pruefungen.csv').exists():
            print(f"[FeatureBuilder] Aggregierte Daten fehlen in {data_dir}. Starte automatische Aggregation ...")
            try:
                from deepsupport.data_engine.aggregate import aggregiere_daten
            except ImportError:
                from aggregate import aggregiere_daten
            aggregiere_daten(data_dir)
        else:
            candidates = [Path('output_dl'), Path('../output_dl'), Path('output_dl_v3')]
            for c in candidates:
                if (c / 'agg_abschluesse.csv').exists():
                    agg_abschluesse_path = c / 'agg_abschluesse.csv'
                    agg_pruefungen_path = c / 'agg_pruefungen.csv'
                    break

    df_abschluesse = pd.read_csv(agg_abschluesse_path)
    df_pruefungen = pd.read_csv(agg_pruefungen_path)

    df_abschluesse.columns = df_abschluesse.columns.str.strip()
    df_pruefungen.columns = df_pruefungen.columns.str.strip()

    studi_path = agg_abschluesse_path.parent / 'studierende.csv'
    if studi_path.exists():
        df_studi = pd.read_csv(studi_path)
        df_studi.columns = df_studi.columns.str.strip()
        latent_initial_cols = [
            'studierenden_id', 'migrationshintergrund', 'erstakademiker', 'erwerbstaetigkeit_std',
            'hzb_typ', 'hzb_note', 'motivation_initial', 'soziale_integration_initial',
            'hidden_erwartete_note_initial', 'hidden_motivation_initial', 'hidden_soziale_integration_initial',
            'hidden_zeit_puffer'
        ]
        merge_cols = [c for c in latent_initial_cols if c in df_studi.columns and (c == 'studierenden_id' or c not in df_abschluesse.columns)]
        if len(merge_cols) > 1:
            df_abschluesse = pd.merge(df_abschluesse, df_studi[merge_cols], on='studierenden_id', how='left')

    return df_abschluesse, df_pruefungen


# =========================================================================
# 1. SEMESTER SEQUENCE TENSOR (Klasse 6: GRU, Transformer, DeepHit, Regr.)
# =========================================================================
def build_semester_sequence_tensor(
    data_dir: Union[str, Path],
    max_semesters: int = 16,
    mode: str = 'standard',
    temporal: str = 'prev',
    target_type: str = 'dropout',
    gradeblind: bool = False,
    blind: bool = False,
    oracle: bool = False,
    realistic: bool = False,
    backend: str = 'duckdb'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Optional[int]]]:
    """
    Erstellt den 3D-Sequenztensor für Semester-Modelle: (N, max_semesters, n_features).
    """
    gradeblind, blind, oracle, realistic = _resolve_modes(mode, gradeblind, blind, oracle, realistic)
    df_abschluesse, df_pruefungen = _load_raw_data(data_dir)

    df_pruefungen['cp_earned'] = np.where(df_pruefungen['bestanden'], df_pruefungen['cp'], 0)
    df_pruefungen['is_fail'] = np.where(~df_pruefungen['bestanden'], 1, 0)
    cp_att_col = 'cp_attempted' if 'cp_attempted' in df_pruefungen.columns else 'cp'

    agg_dict = {
        'sem_cp': ('cp_earned', 'sum'),
        'sem_cp_attempted': (cp_att_col, 'sum'),
        'sem_fails': ('is_fail', 'sum'),
        'sem_gpa': ('note', 'mean'),
        'fach_supp_count': ('support_glz_fachlich', 'sum'),
        'uebf_supp_count': ('support_glz_ueberfachlich', 'sum'),
        'psych_supp_count': ('support_glz_psychosozial', 'sum')
    }
    if 'hidden_motivation' in df_pruefungen.columns:
        agg_dict['hidden_motivation'] = ('hidden_motivation', 'mean')
        agg_dict['hidden_soziale_integration'] = ('hidden_soziale_integration', 'mean')
        agg_dict['hidden_erwartete_note'] = ('hidden_erwartete_note', 'mean')
        agg_dict['hidden_overload'] = ('hidden_overload', 'mean')

    sem_agg = df_pruefungen.groupby(['studierenden_id', 'fachsemester']).agg(**agg_dict).reset_index()
    sem_agg = sem_agg.sort_values(['studierenden_id', 'fachsemester']).reset_index(drop=True)

    sem_agg['cum_cp'] = sem_agg.groupby('studierenden_id')['sem_cp'].cumsum()
    sem_agg['cum_fails'] = sem_agg.groupby('studierenden_id')['sem_fails'].cumsum()
    sem_agg['gpa_clean'] = sem_agg['sem_gpa'].fillna(3.0)
    sem_agg['gpa_cum'] = sem_agg.groupby('studierenden_id')['gpa_clean'].expanding().mean().reset_index(level=0, drop=True)

    sem_agg['cum_cp_vorher'] = sem_agg.groupby('studierenden_id')['cum_cp'].shift(1).fillna(0.0)
    sem_agg['cum_fails_vorher'] = sem_agg.groupby('studierenden_id')['cum_fails'].shift(1).fillna(0.0)
    sem_agg['gpa_cum_vorher'] = sem_agg.groupby('studierenden_id')['gpa_cum'].shift(1).fillna(3.0)

    sem_agg['fails_prev'] = sem_agg.groupby('studierenden_id')['sem_fails'].shift(1).fillna(0.0)
    sem_agg['delta_cp_prev'] = sem_agg.groupby('studierenden_id')['sem_cp'].shift(1).fillna(0.0)
    sem_agg['gpa_prev'] = sem_agg.groupby('studierenden_id')['gpa_clean'].shift(1).fillna(3.0)
    sem_agg['cp_rueckstand_vorher'] = np.maximum(0.0, (sem_agg['fachsemester'] - 1) * 30.0 - sem_agg['cum_cp_vorher'])

    if 'hidden_motivation' in sem_agg.columns:
        sem_agg['hidden_motivation_prev'] = sem_agg.groupby('studierenden_id')['hidden_motivation'].shift(1).fillna(0.5)
        sem_agg['hidden_soziale_integration_prev'] = sem_agg.groupby('studierenden_id')['hidden_soziale_integration'].shift(1).fillna(0.5)
        sem_agg['hidden_erwartete_note_prev'] = sem_agg.groupby('studierenden_id')['hidden_erwartete_note'].shift(1).fillna(3.0)
        sem_agg['hidden_overload_prev'] = sem_agg.groupby('studierenden_id')['hidden_overload'].shift(1).fillna(0.0)

    feature_names: List[str] = [
        'hzb_note',
        'hzb_typ_ord',
        'stg_Informatik', 'stg_BWL', 'stg_Maschinenbau', 'stg_Psychologie', 'stg_Soziale_Arbeit'
    ]

    if not realistic:
        feature_names.extend(['migrationshintergrund', 'erstakademiker', 'erwerbstaetigkeit_std'])

    if not blind:
        if temporal == 'cum':
            feature_names.extend(['cum_fails_vorher', 'cum_cp_vorher', 'cp_rueckstand_vorher', 'sem_cp_attempted'])
            if not gradeblind:
                feature_names.append('gpa_cum_vorher')
        else:
            feature_names.extend(['fails_prev', 'delta_cp_prev', 'cp_rueckstand_vorher', 'sem_cp_attempted'])
            if not gradeblind:
                feature_names.append('gpa_prev')

    feature_names.extend(['fach_supp_count', 'uebf_supp_count'])
    if not realistic:
        feature_names.append('psych_supp_count')

    if oracle:
        feature_names.extend([
            'hidden_motivation_prev',
            'hidden_soziale_integration_prev',
            'hidden_erwartete_note_prev',
            'hidden_overload_prev',
            'hidden_zeit_puffer'
        ])

    feature_indices: Dict[str, Optional[int]] = {
        'fach_supp': feature_names.index('fach_supp_count') if 'fach_supp_count' in feature_names else None,
        'uebf_supp': feature_names.index('uebf_supp_count') if 'uebf_supp_count' in feature_names else None,
        'psych_supp': feature_names.index('psych_supp_count') if 'psych_supp_count' in feature_names else None,
    }

    n_features = len(feature_names)
    studis = df_abschluesse['studierenden_id'].unique()
    num_studis = len(studis)

    target_dim = 2 if target_type == 'competing_risks' else 1
    X_seq = np.full((num_studis, max_semesters, n_features), PADDING_VALUE, dtype=np.float32)
    y_seq = np.full((num_studis, max_semesters, target_dim), PADDING_VALUE, dtype=np.float32)

    stud_id_map = {sid: idx for idx, sid in enumerate(studis)}
    sem_agg['i_idx'] = sem_agg['studierenden_id'].map(stud_id_map)
    sem_agg['t_idx'] = sem_agg['fachsemester'] - 1

    valid_mask = (sem_agg['t_idx'] >= 0) & (sem_agg['t_idx'] < max_semesters)
    v_i = sem_agg.loc[valid_mask, 'i_idx'].values
    v_t = sem_agg.loc[valid_mask, 't_idx'].values

    # Demografische Werte vorbereiten
    df_abschluesse['hzb_typ_ord'] = df_abschluesse['hzb_typ'].map(HZB_ORDINAL_MAP).fillna(3.0)
    for s_name in STUDIENGAENGE_LIST:
        df_abschluesse[f"stg_{s_name.replace(' ', '_')}"] = (df_abschluesse['stg_name'] == s_name).astype(float)
    df_abschluesse['mig_f'] = df_abschluesse['migrationshintergrund'].fillna(False).astype(float)
    df_abschluesse['erst_f'] = df_abschluesse['erstakademiker'].fillna(False).astype(float)
    df_abschluesse['erw_f'] = pd.to_numeric(df_abschluesse['erwerbstaetigkeit_std'], errors='coerce').fillna(0.0)

    # 1. Statische Werte über alle gültigen Zeitschritte broadcasten
    X_seq[v_i, v_t, feature_names.index('hzb_note')] = df_abschluesse.loc[v_i, 'hzb_note'].values
    if oracle and 'hidden_zeit_puffer' in feature_names:
        X_seq[v_i, v_t, feature_names.index('hidden_zeit_puffer')] = df_abschluesse.loc[v_i, 'hidden_zeit_puffer'].fillna(60.0).values
    X_seq[v_i, v_t, feature_names.index('hzb_typ_ord')] = df_abschluesse.loc[v_i, 'hzb_typ_ord'].values
    for s_name in STUDIENGAENGE_LIST:
        f_name = f"stg_{s_name.replace(' ', '_')}"
        X_seq[v_i, v_t, feature_names.index(f_name)] = df_abschluesse.loc[v_i, f_name].values

    if not realistic:
        X_seq[v_i, v_t, feature_names.index('migrationshintergrund')] = df_abschluesse.loc[v_i, 'mig_f'].values
        X_seq[v_i, v_t, feature_names.index('erstakademiker')] = df_abschluesse.loc[v_i, 'erst_f'].values
        X_seq[v_i, v_t, feature_names.index('erwerbstaetigkeit_std')] = df_abschluesse.loc[v_i, 'erw_f'].values

    # 2. Dynamische Semester-Features
    if not blind:
        if temporal == 'cum':
            X_seq[v_i, v_t, feature_names.index('cum_fails_vorher')] = sem_agg.loc[valid_mask, 'cum_fails_vorher'].values
            X_seq[v_i, v_t, feature_names.index('cum_cp_vorher')] = sem_agg.loc[valid_mask, 'cum_cp_vorher'].values
            X_seq[v_i, v_t, feature_names.index('cp_rueckstand_vorher')] = sem_agg.loc[valid_mask, 'cp_rueckstand_vorher'].values
            if not gradeblind:
                X_seq[v_i, v_t, feature_names.index('gpa_cum_vorher')] = sem_agg.loc[valid_mask, 'gpa_cum_vorher'].values
        else:
            X_seq[v_i, v_t, feature_names.index('fails_prev')] = sem_agg.loc[valid_mask, 'fails_prev'].values
            X_seq[v_i, v_t, feature_names.index('delta_cp_prev')] = sem_agg.loc[valid_mask, 'delta_cp_prev'].values
            X_seq[v_i, v_t, feature_names.index('cp_rueckstand_vorher')] = sem_agg.loc[valid_mask, 'cp_rueckstand_vorher'].values
            if not gradeblind:
                X_seq[v_i, v_t, feature_names.index('gpa_prev')] = sem_agg.loc[valid_mask, 'gpa_prev'].values
        X_seq[v_i, v_t, feature_names.index('sem_cp_attempted')] = sem_agg.loc[valid_mask, 'sem_cp_attempted'].values

    X_seq[v_i, v_t, feature_names.index('fach_supp_count')] = sem_agg.loc[valid_mask, 'fach_supp_count'].values
    X_seq[v_i, v_t, feature_names.index('uebf_supp_count')] = sem_agg.loc[valid_mask, 'uebf_supp_count'].values
    if not realistic:
        X_seq[v_i, v_t, feature_names.index('psych_supp_count')] = sem_agg.loc[valid_mask, 'psych_supp_count'].values

    if oracle and 'hidden_motivation_prev' in sem_agg.columns:
        X_seq[v_i, v_t, feature_names.index('hidden_motivation_prev')] = sem_agg.loc[valid_mask, 'hidden_motivation_prev'].values
        X_seq[v_i, v_t, feature_names.index('hidden_soziale_integration_prev')] = sem_agg.loc[valid_mask, 'hidden_soziale_integration_prev'].values
        X_seq[v_i, v_t, feature_names.index('hidden_erwartete_note_prev')] = sem_agg.loc[valid_mask, 'hidden_erwartete_note_prev'].values
        X_seq[v_i, v_t, feature_names.index('hidden_overload_prev')] = sem_agg.loc[valid_mask, 'hidden_overload_prev'].values

    # 3. Targets zuweisen (Exakte DGP-Werte: 'abgeschlossen' vs. ['abgebrochen', 'exmatrikuliert', 'zeitueberschreitung'])
    df_abschluesse['is_dropout'] = df_abschluesse['status'].str.strip().isin(['abgebrochen', 'exmatrikuliert', 'zeitueberschreitung']).astype(int)
    df_abschluesse['is_grad'] = (df_abschluesse['status'].str.strip() == 'abgeschlossen').astype(int)
    
    studi_events = np.where(df_abschluesse['is_dropout'] == 1, 1, np.where(df_abschluesse['is_grad'] == 1, 2, 0))

    if target_type == 'gpa':
        y_seq[v_i, v_t, 0] = sem_agg.loc[valid_mask, 'gpa_clean'].values
    else:
        # Standard: 0.0 an allen Zeitschritten mit Daten
        y_seq[v_i, v_t, 0] = 0.0
        if target_type == 'competing_risks':
            y_seq[v_i, v_t, 1] = 0.0
            
        # Event am letzten Semester
        for i, row in enumerate(df_abschluesse.itertuples(index=False)):
            max_sem = min(int(row.studiendauer_semester), max_semesters)
            t_last = max_sem - 1
            if 0 <= t_last < max_semesters:
                if target_type == 'competing_risks':
                    y_seq[i, t_last, 0] = 1.0 if row.is_dropout == 1 else 0.0
                    y_seq[i, t_last, 1] = 1.0 if row.is_grad == 1 else 0.0
                else:
                    y_seq[i, t_last, 0] = 1.0 if row.is_dropout == 1 else 0.0

    return studis, X_seq, y_seq, studi_events, feature_names, feature_indices


# =========================================================================
# 2. EXAM SEQUENCE TENSOR (Klasse 7: Exam-GRU, Exam-Transformer, Regr.)
# =========================================================================
def build_exam_sequence_tensor(
    data_dir: Union[str, Path],
    max_exams: int = 50,
    mode: str = 'standard',
    temporal: str = 'prev',
    target_type: str = 'dropout',
    gradeblind: bool = False,
    blind: bool = False,
    oracle: bool = False,
    realistic: bool = False,
    backend: str = 'duckdb'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Optional[int]]]:
    """
    Erstellt den 3D-Sequenztensor für Prüfungs-Modelle: (N, max_exams, n_features).
    """
    gradeblind, blind, oracle, realistic = _resolve_modes(mode, gradeblind, blind, oracle, realistic)
    df_abschluesse, df_pruefungen = _load_raw_data(data_dir)

    df_pruefungen = df_pruefungen.sort_values(['studierenden_id', 'pruefung_id']).reset_index(drop=True)
    df_pruefungen['is_fail'] = (~df_pruefungen['bestanden']).astype(int)
    df_pruefungen['cp_earned'] = np.where(df_pruefungen['bestanden'], df_pruefungen['cp'], 0)
    df_pruefungen['fails_cum'] = df_pruefungen.groupby('studierenden_id')['is_fail'].cumsum()
    df_pruefungen['cp_cum'] = df_pruefungen.groupby('studierenden_id')['cp_earned'].cumsum()
    df_pruefungen['note_clean'] = df_pruefungen['note'].fillna(3.0)
    df_pruefungen['gpa_cum'] = df_pruefungen.groupby('studierenden_id')['note_clean'].expanding().mean().reset_index(level=0, drop=True)

    df_pruefungen['k_idx'] = df_pruefungen.groupby('studierenden_id').cumcount()
    df_pruefungen['fails_prev_exam'] = df_pruefungen.groupby('studierenden_id')['is_fail'].shift(1).fillna(0)
    df_pruefungen['cp_earned_prev_exam'] = df_pruefungen.groupby('studierenden_id')['cp_earned'].shift(1).fillna(0.0)
    df_pruefungen['note_prev_exam'] = df_pruefungen.groupby('studierenden_id')['note_clean'].shift(1).fillna(3.0)
    
    # Note: fails_cum, cp_cum, gpa_cum are ONLY used in temporal='cum' mode. 
    # They are inclusive of the current exam, which is technically leakage, 
    # but since temporal='cum' represents 'history up to and including now', this is a design choice.
    df_pruefungen['cp_cum_prev'] = df_pruefungen.groupby('studierenden_id')['cp_earned'].cumsum().groupby(df_pruefungen['studierenden_id']).shift(1).fillna(0.0)
    df_pruefungen['cp_rueckstand'] = np.maximum(0.0, (df_pruefungen['fachsemester'] - 1) * 30.0 - df_pruefungen['cp_cum_prev'])

    feature_names: List[str] = [
        'hzb_note',
        'hzb_typ_ord',
        'stg_Informatik', 'stg_BWL', 'stg_Maschinenbau', 'stg_Psychologie', 'stg_Soziale_Arbeit'
    ]

    if not realistic:
        feature_names.extend(['migrationshintergrund', 'erstakademiker', 'erwerbstaetigkeit_std'])

    feature_names.extend(['fachsemester', 'versuch', 'cp_value'])
    if not realistic:
        feature_names.append('schwierigkeit')

    if not blind:
        if temporal == 'cum':
            feature_names.extend(['fails_cum', 'cp_cum', 'cp_rueckstand'])
            if not gradeblind:
                feature_names.append('gpa_cum')
        else:
            feature_names.extend(['fails_prev_exam', 'cp_earned_prev_exam', 'cp_rueckstand'])
            if not gradeblind:
                feature_names.append('note_prev_exam')

    feature_names.extend([
        'support_vorher_fachlich', 'support_glz_fachlich',
        'support_vorher_ueberfachlich', 'support_glz_ueberfachlich'
    ])
    if not realistic:
        feature_names.extend(['support_vorher_psychosozial', 'support_glz_psychosozial'])

    if oracle:
        feature_names.extend(['hidden_motivation', 'hidden_soziale_integration', 'hidden_erwartete_note',
                              'hidden_overload', 'hidden_zeit_puffer'])

    feature_indices: Dict[str, Optional[int]] = {
        'fach_glz': feature_names.index('support_glz_fachlich') if 'support_glz_fachlich' in feature_names else None,
        'fach_vorher': feature_names.index('support_vorher_fachlich') if 'support_vorher_fachlich' in feature_names else None,
        'uebf_glz': feature_names.index('support_glz_ueberfachlich') if 'support_glz_ueberfachlich' in feature_names else None,
        'uebf_vorher': feature_names.index('support_vorher_ueberfachlich') if 'support_vorher_ueberfachlich' in feature_names else None,
        'psych_glz': feature_names.index('support_glz_psychosozial') if 'support_glz_psychosozial' in feature_names else None,
        'psych_vorher': feature_names.index('support_vorher_psychosozial') if 'support_vorher_psychosozial' in feature_names else None,
    }

    n_features = len(feature_names)
    studis = df_abschluesse['studierenden_id'].unique()
    num_studis = len(studis)

    X_seq = np.full((num_studis, max_exams, n_features), PADDING_VALUE, dtype=np.float32)
    y_seq = np.full((num_studis, max_exams, 1), PADDING_VALUE, dtype=np.float32)

    stud_id_map = {sid: idx for idx, sid in enumerate(studis)}
    df_pruefungen['i_idx'] = df_pruefungen['studierenden_id'].map(stud_id_map)

    mask = df_pruefungen['k_idx'] < max_exams
    v_i = df_pruefungen.loc[mask, 'i_idx'].values
    v_k = df_pruefungen.loc[mask, 'k_idx'].values

    # Demografische Werte vorbereiten
    df_abschluesse['hzb_typ_ord'] = df_abschluesse['hzb_typ'].map(HZB_ORDINAL_MAP).fillna(3.0)
    for s_name in STUDIENGAENGE_LIST:
        df_abschluesse[f"stg_{s_name.replace(' ', '_')}"] = (df_abschluesse['stg_name'] == s_name).astype(float)
    df_abschluesse['mig_f'] = df_abschluesse['migrationshintergrund'].fillna(False).astype(float)
    df_abschluesse['erst_f'] = df_abschluesse['erstakademiker'].fillna(False).astype(float)
    df_abschluesse['erw_f'] = pd.to_numeric(df_abschluesse['erwerbstaetigkeit_std'], errors='coerce').fillna(0.0)

    # 1. Statische Merkmale
    X_seq[v_i, v_k, feature_names.index('hzb_note')] = df_abschluesse.loc[v_i, 'hzb_note'].values
    X_seq[v_i, v_k, feature_names.index('hzb_typ_ord')] = df_abschluesse.loc[v_i, 'hzb_typ_ord'].values
    for s_name in STUDIENGAENGE_LIST:
        f_name = f"stg_{s_name.replace(' ', '_')}"
        X_seq[v_i, v_k, feature_names.index(f_name)] = df_abschluesse.loc[v_i, f_name].values

    if not realistic:
        X_seq[v_i, v_k, feature_names.index('migrationshintergrund')] = df_abschluesse.loc[v_i, 'mig_f'].values
        X_seq[v_i, v_k, feature_names.index('erstakademiker')] = df_abschluesse.loc[v_i, 'erst_f'].values
        X_seq[v_i, v_k, feature_names.index('erwerbstaetigkeit_std')] = df_abschluesse.loc[v_i, 'erw_f'].values

    # 2. Prüfungs-Features
    X_seq[v_i, v_k, feature_names.index('fachsemester')] = df_pruefungen.loc[mask, 'fachsemester'].values
    X_seq[v_i, v_k, feature_names.index('versuch')] = df_pruefungen.loc[mask, 'versuch'].values
    X_seq[v_i, v_k, feature_names.index('cp_value')] = df_pruefungen.loc[mask, 'cp'].values

    if not realistic:
        X_seq[v_i, v_k, feature_names.index('schwierigkeit')] = df_pruefungen.loc[mask, 'schwierigkeit'].values

    if not blind:
        if temporal == 'cum':
            X_seq[v_i, v_k, feature_names.index('fails_cum')] = df_pruefungen.loc[mask, 'fails_cum'].values
            X_seq[v_i, v_k, feature_names.index('cp_cum')] = df_pruefungen.loc[mask, 'cp_cum'].values
            X_seq[v_i, v_k, feature_names.index('cp_rueckstand')] = df_pruefungen.loc[mask, 'cp_rueckstand'].values
            if not gradeblind:
                X_seq[v_i, v_k, feature_names.index('gpa_cum')] = df_pruefungen.loc[mask, 'gpa_cum'].values
        else:
            X_seq[v_i, v_k, feature_names.index('fails_prev_exam')] = df_pruefungen.loc[mask, 'fails_prev_exam'].values
            X_seq[v_i, v_k, feature_names.index('cp_earned_prev_exam')] = df_pruefungen.loc[mask, 'cp_earned_prev_exam'].values
            X_seq[v_i, v_k, feature_names.index('cp_rueckstand')] = df_pruefungen.loc[mask, 'cp_rueckstand'].values
            if not gradeblind:
                X_seq[v_i, v_k, feature_names.index('note_prev_exam')] = df_pruefungen.loc[mask, 'note_prev_exam'].values

    X_seq[v_i, v_k, feature_names.index('support_vorher_fachlich')] = df_pruefungen.loc[mask, 'support_vorher_fachlich'].values
    X_seq[v_i, v_k, feature_names.index('support_glz_fachlich')] = df_pruefungen.loc[mask, 'support_glz_fachlich'].values
    X_seq[v_i, v_k, feature_names.index('support_vorher_ueberfachlich')] = df_pruefungen.loc[mask, 'support_vorher_ueberfachlich'].values
    X_seq[v_i, v_k, feature_names.index('support_glz_ueberfachlich')] = df_pruefungen.loc[mask, 'support_glz_ueberfachlich'].values

    if not realistic:
        X_seq[v_i, v_k, feature_names.index('support_vorher_psychosozial')] = df_pruefungen.loc[mask, 'support_vorher_psychosozial'].values
        X_seq[v_i, v_k, feature_names.index('support_glz_psychosozial')] = df_pruefungen.loc[mask, 'support_glz_psychosozial'].values

    if oracle and 'hidden_motivation' in df_pruefungen.columns:
        X_seq[v_i, v_k, feature_names.index('hidden_motivation')] = df_pruefungen.loc[mask, 'hidden_motivation'].fillna(0.5).values
        X_seq[v_i, v_k, feature_names.index('hidden_soziale_integration')] = df_pruefungen.loc[mask, 'hidden_soziale_integration'].fillna(0.5).values
        X_seq[v_i, v_k, feature_names.index('hidden_erwartete_note')] = df_pruefungen.loc[mask, 'hidden_erwartete_note'].fillna(3.0).values
        X_seq[v_i, v_k, feature_names.index('hidden_overload')] = df_pruefungen.loc[mask, 'hidden_overload'].fillna(0.0).values
        X_seq[v_i, v_k, feature_names.index('hidden_zeit_puffer')] = df_pruefungen.loc[mask, 'hidden_zeit_puffer'].fillna(60.0).values

    # 3. Targets zuweisen
    df_abschluesse['is_dropout'] = df_abschluesse['status'].str.strip().str.lower().isin(['abgebrochen', 'exmatrikuliert', 'zeitueberschreitung']).astype(int)
    studi_events = df_abschluesse['is_dropout'].values

    if target_type == 'gpa':
        y_seq[v_i, v_k, 0] = df_pruefungen.loc[mask, 'note_clean'].values
    else:
        y_seq[v_i, v_k, 0] = 0.0
        df_pruefungen['exam_count_total'] = df_pruefungen.groupby('studierenden_id')['pruefung_id'].transform('count')
        last_exam_mask = mask & (df_pruefungen['k_idx'] == (df_pruefungen['exam_count_total'] - 1))
        last_i = df_pruefungen.loc[last_exam_mask, 'i_idx'].values
        last_k = df_pruefungen.loc[last_exam_mask, 'k_idx'].values
        y_seq[last_i, last_k, 0] = df_abschluesse.loc[last_i, 'is_dropout'].values

    return studis, X_seq, y_seq, studi_events, feature_names, feature_indices


# =========================================================================
# 3. SEMESTER PANEL DATAFRAME (Klasse 5: Extended Cox, DeepSurv, DML)
# =========================================================================
def build_semester_panel_df(
    data_dir: Union[str, Path],
    mode: str = 'standard',
    temporal: str = 'prev',
    gradeblind: bool = False,
    blind: bool = False,
    oracle: bool = False,
    realistic: bool = False,
    backend: str = 'duckdb'
) -> Tuple[pd.DataFrame, List[str], str, Dict[str, Optional[str]]]:
    """
    Erstellt ein Person-Semester Längsschnitt-Panel im Counting Process Format
    (t_start, t_stop, event, X_features...).
    """
    gradeblind, blind, oracle, realistic = _resolve_modes(mode, gradeblind, blind, oracle, realistic)
    df_abschluesse, df_pruefungen = _load_raw_data(data_dir)

    df_pruefungen['cp_earned'] = np.where(df_pruefungen['bestanden'], df_pruefungen['cp'], 0)
    df_pruefungen['is_fail'] = np.where(~df_pruefungen['bestanden'], 1, 0)

    pr_sem = df_pruefungen.groupby(['studierenden_id', 'fachsemester']).agg({
        'support_glz_fachlich': 'sum',
        'support_glz_ueberfachlich': 'sum',
        'support_glz_psychosozial': 'sum',
        'cp_earned': 'sum',
        'is_fail': 'sum',
        'note': 'mean',
        'hidden_motivation': 'mean',
        'hidden_soziale_integration': 'mean',
        'hidden_erwartete_note': 'mean',
        'hidden_overload': 'mean'
    }).reset_index().sort_values(['studierenden_id', 'fachsemester']).reset_index(drop=True)

    pr_sem['t_start'] = pr_sem['fachsemester'] - 1
    pr_sem['t_stop'] = pr_sem['fachsemester']

    pr_sem['cum_cp'] = pr_sem.groupby('studierenden_id')['cp_earned'].cumsum()
    pr_sem['cum_fails'] = pr_sem.groupby('studierenden_id')['is_fail'].cumsum()
    pr_sem['note_clean'] = pr_sem['note'].fillna(3.0)

    pr_sem['cum_cp_vorher'] = pr_sem.groupby('studierenden_id')['cum_cp'].shift(1).fillna(0.0)
    pr_sem['cum_fails_vorher'] = pr_sem.groupby('studierenden_id')['cum_fails'].shift(1).fillna(0.0)
    pr_sem['fails_prev'] = pr_sem.groupby('studierenden_id')['is_fail'].shift(1).fillna(0)
    pr_sem['delta_cp_prev'] = pr_sem.groupby('studierenden_id')['cp_earned'].shift(1).fillna(0.0)
    pr_sem['gpa_prev'] = pr_sem.groupby('studierenden_id')['note_clean'].shift(1).fillna(3.0)
    pr_sem['cp_rueckstand'] = np.maximum(0.0, (pr_sem['fachsemester'] - 1) * 30.0 - pr_sem['cum_cp_vorher'])

    pr_sem['fach_supp_count'] = pr_sem['support_glz_fachlich']
    pr_sem['uebf_supp_count'] = pr_sem['support_glz_ueberfachlich']
    pr_sem['psych_supp_count'] = pr_sem['support_glz_psychosozial']
    pr_sem['fach_supp_active'] = pr_sem['support_glz_fachlich']
    pr_sem['uebf_supp_active'] = pr_sem['support_glz_ueberfachlich']
    pr_sem['psych_supp_active'] = pr_sem['support_glz_psychosozial']

    if 'hidden_motivation' in pr_sem.columns:
        pr_sem['hidden_motivation_prev'] = pr_sem.groupby('studierenden_id')['hidden_motivation'].shift(1).fillna(0.5)
        pr_sem['hidden_soziale_integration_prev'] = pr_sem.groupby('studierenden_id')['hidden_soziale_integration'].shift(1).fillna(0.5)
        pr_sem['hidden_erwartete_note_prev'] = pr_sem.groupby('studierenden_id')['hidden_erwartete_note'].shift(1).fillna(3.0)
        pr_sem['hidden_overload_prev'] = pr_sem.groupby('studierenden_id')['hidden_overload'].shift(1).fillna(0.0)

    df_abschluesse['is_dropout'] = df_abschluesse['status'].str.strip().str.lower().isin(['abgebrochen', 'exmatrikuliert', 'zeitueberschreitung']).astype(int)
    df_abschluesse['hzb_typ_ord'] = df_abschluesse['hzb_typ'].map(HZB_ORDINAL_MAP).fillna(3.0)
    df_abschluesse['migrationshintergrund'] = df_abschluesse['migrationshintergrund'].fillna(False).astype(float)
    df_abschluesse['erstakademiker'] = df_abschluesse['erstakademiker'].fillna(False).astype(float)
    df_abschluesse['erwerbstaetigkeit_std'] = pd.to_numeric(df_abschluesse['erwerbstaetigkeit_std'], errors='coerce').fillna(0.0)
    for s_name in STUDIENGAENGE_LIST:
        df_abschluesse[f"stg_{s_name.replace(' ', '_')}"] = (df_abschluesse['stg_name'] == s_name).astype(float)

    stud_cols = ['studierenden_id', 'is_dropout', 'studiendauer_semester', 'hzb_note', 'hzb_typ_ord', 'stg_name', 'migrationshintergrund', 'erstakademiker', 'erwerbstaetigkeit_std', 'hidden_zeit_puffer'] + [f"stg_{s.replace(' ', '_')}" for s in STUDIENGAENGE_LIST]
    stud_cols = [c for c in stud_cols if c in df_abschluesse.columns]

    panel_df = pr_sem.merge(df_abschluesse[stud_cols], on='studierenden_id', how='left')
    panel_df['event'] = np.where((panel_df['fachsemester'] == panel_df['studiendauer_semester']) & (panel_df['is_dropout'] == 1), 1, 0)
    panel_df['delta_gpa'] = panel_df['gpa_prev'] - panel_df['hzb_note']

    feature_cols: List[str] = ['hzb_note', 'hzb_typ_ord'] + [f"stg_{s.replace(' ', '_')}" for s in STUDIENGAENGE_LIST[1:]]

    if not realistic:
        feature_cols.extend(['migrationshintergrund', 'erstakademiker', 'erwerbstaetigkeit_std'])

    if not blind:
        if temporal == 'cum':
            feature_cols.extend(['cum_fails', 'cum_cp', 'cp_rueckstand'])
        else:
            feature_cols.extend(['fails_prev', 'delta_cp_prev', 'cp_rueckstand'])
        if not gradeblind:
            feature_cols.append('gpa_prev')

    feature_cols.extend(['fach_supp_count', 'uebf_supp_count'])
    if not realistic:
        feature_cols.append('psych_supp_count')

    if oracle and 'hidden_motivation_prev' in panel_df.columns:
        feature_cols.extend([
            'hidden_motivation_prev',
            'hidden_soziale_integration_prev',
            'hidden_erwartete_note_prev',
            'hidden_overload_prev',
            'hidden_zeit_puffer_static'
        ])
        panel_df['hidden_zeit_puffer_static'] = panel_df['hidden_zeit_puffer'].fillna(60.0)

    feature_indices = {
        'fach_supp': 'fach_supp_count',
        'uebf_supp': 'uebf_supp_count',
        'psych_supp': 'psych_supp_count' if not realistic else None
    }

    return panel_df, feature_cols, 'event', feature_indices


# =========================================================================
# 3b. EXAM PANEL DATAFRAME (Klasse 5b: Extended Exam Survival)
# =========================================================================
def build_exam_panel_df(
    data_dir: Union[str, Path],
    mode: str = 'standard',
    temporal: str = 'prev',
    gradeblind: bool = False,
    blind: bool = False,
    oracle: bool = False,
    realistic: bool = False,
    backend: str = 'duckdb'
) -> Tuple[pd.DataFrame, List[str], str, Dict[str, Optional[str]]]:
    """
    Erstellt ein Person-Prüfung Counting Process Längsschnitt-Panel für Cox-Modelle auf Prüfungsebene.
    """
    gradeblind, blind, oracle, realistic = _resolve_modes(mode, gradeblind, blind, oracle, realistic)
    df_abschluesse, df_pruefungen = _load_raw_data(data_dir)

    df_pruefungen = df_pruefungen.sort_values(['studierenden_id', 'pruefung_id']).reset_index(drop=True)
    df_pruefungen['is_fail'] = (~df_pruefungen['bestanden']).astype(int)
    df_pruefungen['cp_earned'] = np.where(df_pruefungen['bestanden'], df_pruefungen['cp'], 0)
    df_pruefungen['fails_cum'] = df_pruefungen.groupby('studierenden_id')['is_fail'].cumsum()
    df_pruefungen['cp_cum'] = df_pruefungen.groupby('studierenden_id')['cp_earned'].cumsum()
    df_pruefungen['note_clean'] = df_pruefungen['note'].fillna(3.0)
    df_pruefungen['gpa_cum'] = df_pruefungen.groupby('studierenden_id')['note_clean'].expanding().mean().reset_index(level=0, drop=True)

    df_pruefungen['t_start'] = df_pruefungen.groupby('studierenden_id').cumcount()
    df_pruefungen['t_stop'] = df_pruefungen['t_start'] + 1
    df_pruefungen['exam_count_total'] = df_pruefungen.groupby('studierenden_id')['pruefung_id'].transform('count')
    df_pruefungen['is_last_exam'] = df_pruefungen['t_stop'] == df_pruefungen['exam_count_total']

    df_pruefungen['fails_prev'] = df_pruefungen.groupby('studierenden_id')['is_fail'].shift(1).fillna(0)
    df_pruefungen['delta_cp_prev'] = df_pruefungen.groupby('studierenden_id')['cp_earned'].shift(1).fillna(0.0)
    df_pruefungen['gpa_prev'] = df_pruefungen.groupby('studierenden_id')['note_clean'].shift(1).fillna(3.0)
    # Note: fails_cum, cp_cum, gpa_cum are inclusive of the current exam which is technically 
    # leakage, but since temporal='cum' represents history up to and including now, this is a design choice.
    df_pruefungen['cp_cum_prev'] = df_pruefungen.groupby('studierenden_id')['cp_earned'].cumsum().groupby(df_pruefungen['studierenden_id']).shift(1).fillna(0.0)
    df_pruefungen['cp_rueckstand'] = np.maximum(0.0, (df_pruefungen['fachsemester'] - 1) * 30.0 - df_pruefungen['cp_cum_prev'])
    df_pruefungen['cp_value'] = df_pruefungen['cp']

    df_abschluesse['is_dropout'] = df_abschluesse['status'].str.strip().str.lower().isin(
        ['abgebrochen', 'exmatrikuliert', 'zeitueberschreitung']
    ).astype(int)
    df_abschluesse['hzb_typ_ord'] = df_abschluesse['hzb_typ'].map(HZB_ORDINAL_MAP).fillna(3.0)
    for s_name in STUDIENGAENGE_LIST:
        df_abschluesse[f"stg_{s_name.replace(' ', '_')}"] = (df_abschluesse['stg_name'] == s_name).astype(float)

    stud_cols = ['studierenden_id', 'is_dropout', 'hzb_note', 'hzb_typ_ord', 'migrationshintergrund', 'erstakademiker', 'erwerbstaetigkeit_std'] + [f"stg_{s.replace(' ', '_')}" for s in STUDIENGAENGE_LIST]
    stud_cols = [c for c in stud_cols if c in df_abschluesse.columns]

    panel_df = df_pruefungen.merge(df_abschluesse[stud_cols], on='studierenden_id', how='left')
    panel_df['event'] = np.where(panel_df['is_last_exam'] & (panel_df['is_dropout'] == 1), 1, 0)

    feature_cols: List[str] = ['hzb_note', 'hzb_typ_ord'] + [f"stg_{s.replace(' ', '_')}" for s in STUDIENGAENGE_LIST[1:]]
    if not realistic:
        feature_cols.extend(['migrationshintergrund', 'erstakademiker', 'erwerbstaetigkeit_std'])
    feature_cols.extend(['fachsemester', 'versuch', 'cp_value'])
    if not realistic:
        feature_cols.append('schwierigkeit')

    if not blind:
        if temporal == 'cum':
            feature_cols.extend(['fails_cum', 'cp_cum', 'cp_rueckstand'])
            if not gradeblind:
                feature_cols.append('gpa_cum')
        else:
            feature_cols.extend(['fails_prev', 'delta_cp_prev', 'cp_rueckstand'])
            if not gradeblind:
                feature_cols.append('gpa_prev')

    feature_cols.extend(['support_vorher_fachlich', 'support_glz_fachlich',
                         'support_vorher_ueberfachlich', 'support_glz_ueberfachlich'])
    if not realistic:
        feature_cols.extend(['support_vorher_psychosozial', 'support_glz_psychosozial'])

    if oracle and 'hidden_motivation' in panel_df.columns:
        # hidden_zeit_puffer kommt aus df_pruefungen (identisch pro Student), nicht aus dem Merge
        feature_cols.extend(['hidden_motivation', 'hidden_soziale_integration', 'hidden_erwartete_note', 'hidden_overload', 'hidden_zeit_puffer'])
        panel_df['hidden_motivation'] = panel_df['hidden_motivation'].fillna(0.5)
        panel_df['hidden_soziale_integration'] = panel_df['hidden_soziale_integration'].fillna(0.5)
        panel_df['hidden_erwartete_note'] = panel_df['hidden_erwartete_note'].fillna(3.0)
        panel_df['hidden_overload'] = panel_df['hidden_overload'].fillna(0.0)
        panel_df['hidden_zeit_puffer'] = panel_df['hidden_zeit_puffer'].fillna(60.0)
    feature_indices = {
        'fach_glz': 'support_glz_fachlich',
        'fach_vorher': 'support_vorher_fachlich',
        'uebf_glz': 'support_glz_ueberfachlich',
        'uebf_vorher': 'support_vorher_ueberfachlich',
        'psych_glz': 'support_glz_psychosozial' if not realistic else None,
        'psych_vorher': 'support_vorher_psychosozial' if not realistic else None
    }

    return panel_df, feature_cols, 'event', feature_indices


# =========================================================================
# 4. LANDMARK DATASET (Klassen 1, 2a, 4: S1-S2 Landmark bis T0=2)
# =========================================================================
def build_landmark_dataset(
    data_dir: Union[str, Path],
    t0: int = 2,
    mode: str = 'standard',
    target: str = 'dropout',
    target_type: str = 'binary',
    graduates_only: bool = False,
    gradeblind: bool = False,
    blind: bool = False,
    oracle: bool = False,
    realistic: bool = False,
    backend: str = 'duckdb'
) -> Tuple[pd.DataFrame, List[str], str, Dict[str, Optional[str]]]:
    """
    Erstellt ein statisches Landmark-Dataset mit Aggregaten bis Semester T0 (Default: 2).
    """
    gradeblind, blind, oracle, realistic = _resolve_modes(mode, gradeblind, blind, oracle, realistic)
    df_abschluesse, df_pruefungen = _load_raw_data(data_dir)

    df_valid = df_abschluesse[df_abschluesse['studiendauer_semester'] >= t0].copy()
    if graduates_only:
        df_valid = df_valid[df_valid['status'].str.strip().str.lower().isin(
            ['abgeschlossen', 'absolviert', 'abschluss', 'bestanden', 'erfolgreich']
        )].copy()

    df_pruefungen['cp_earned'] = np.where(df_pruefungen['bestanden'], df_pruefungen['cp'], 0)
    df_pruefungen['is_fail'] = np.where(~df_pruefungen['bestanden'], 1, 0)
    pr_t0 = df_pruefungen[df_pruefungen['fachsemester'] <= t0]

    lm_agg = pr_t0.groupby('studierenden_id').agg({
        'cp_earned': 'sum',
        'is_fail': 'sum',
        'note': 'mean',
        'support_glz_fachlich': 'sum',
        'support_glz_ueberfachlich': 'sum',
        'support_glz_psychosozial': 'sum',
        'hidden_motivation': 'mean',
        'hidden_soziale_integration': 'mean',
        'hidden_erwartete_note': 'mean',
        'hidden_overload': 'mean'
    }).reset_index().rename(columns={
        'cp_earned': 'cp_s1s2',
        'is_fail': 'fails_s1s2',
        'note': 'gpa_s1s2',
        'support_glz_fachlich': 'fach_supp_s1s2',
        'support_glz_ueberfachlich': 'uebf_supp_s1s2',
        'support_glz_psychosozial': 'psych_supp_s1s2',
        'hidden_motivation': 'hidden_motivation_s1s2',
        'hidden_soziale_integration': 'hidden_soziale_integration_s1s2',
        'hidden_erwartete_note': 'hidden_erwartete_note_s1s2',
        'hidden_overload': 'hidden_overload_s1s2'
    })

    df_lm = pd.merge(df_valid, lm_agg, on='studierenden_id', how='left')
    df_lm['cp_s1s2'] = df_lm['cp_s1s2'].fillna(0.0)
    df_lm['fails_s1s2'] = df_lm['fails_s1s2'].fillna(0.0)
    df_lm['gpa_s1s2'] = df_lm['gpa_s1s2'].fillna(3.0)
    df_lm['fach_supp_s1s2'] = df_lm['fach_supp_s1s2'].fillna(0.0)
    df_lm['uebf_supp_s1s2'] = df_lm['uebf_supp_s1s2'].fillna(0.0)
    df_lm['psych_supp_s1s2'] = df_lm['psych_supp_s1s2'].fillna(0.0)

    if 'motivation_initial' in df_lm.columns:
        df_lm['hidden_motivation_s1s2'] = df_lm['hidden_motivation_s1s2'].fillna(df_lm['motivation_initial'])
        df_lm['hidden_soziale_integration_s1s2'] = df_lm['hidden_soziale_integration_s1s2'].fillna(df_lm['soziale_integration_initial'])
        df_lm['hidden_erwartete_note_s1s2'] = df_lm['hidden_erwartete_note_s1s2'].fillna(df_lm['hidden_erwartete_note_initial'])
    df_lm['hidden_motivation_s1s2'] = df_lm['hidden_motivation_s1s2'].fillna(0.5)
    df_lm['hidden_soziale_integration_s1s2'] = df_lm['hidden_soziale_integration_s1s2'].fillna(0.5)
    df_lm['hidden_erwartete_note_s1s2'] = df_lm['hidden_erwartete_note_s1s2'].fillna(3.0)
    df_lm['hidden_overload_s1s2'] = df_lm['hidden_overload_s1s2'].fillna(0.0)
    if 'hidden_zeit_puffer' in df_lm.columns:
        df_lm['hidden_zeit_puffer'] = df_lm['hidden_zeit_puffer'].fillna(60.0)

    df_lm['hzb_typ_ord'] = df_lm['hzb_typ'].map(HZB_ORDINAL_MAP).fillna(3.0)
    for s_name in STUDIENGAENGE_LIST:
        df_lm[f"stg_{s_name.replace(' ', '_')}"] = (df_lm['stg_name'] == s_name).astype(float)

    if target == 'abschlussnote':
        target_col = 'abschlussnote'
        df_lm[target_col] = pd.to_numeric(df_lm['abschlussnote'], errors='coerce').fillna(3.0)
    elif target == 'status':
        target_col = 'status'
    else:
        target_col = 'is_dropout'
        df_lm['is_dropout'] = df_lm['status'].str.strip().str.lower().isin(
            ['abgebrochen', 'exmatrikuliert', 'zeitueberschreitung']
        ).astype(int)

    feature_cols: List[str] = ['hzb_note', 'hzb_typ_ord'] + [f"stg_{s.replace(' ', '_')}" for s in STUDIENGAENGE_LIST]

    if not realistic:
        feature_cols.extend(['migrationshintergrund', 'erstakademiker', 'erwerbstaetigkeit_std'])

    if not blind:
        feature_cols.extend(['cp_s1s2', 'fails_s1s2'])
        if not gradeblind:
            feature_cols.append('gpa_s1s2')

    feature_cols.extend(['fach_supp_s1s2', 'uebf_supp_s1s2'])
    if not realistic:
        feature_cols.append('psych_supp_s1s2')

    if oracle:
        feature_cols.extend([
            'hidden_motivation_s1s2',
            'hidden_soziale_integration_s1s2',
            'hidden_erwartete_note_s1s2',
            'hidden_overload_s1s2',
            'hidden_zeit_puffer'
        ])

    feature_indices = {
        'fach_supp': 'fach_supp_s1s2',
        'uebf_supp': 'uebf_supp_s1s2',
        'psych_supp': 'psych_supp_s1s2' if not realistic else None
    }

    return df_lm, feature_cols, target_col, feature_indices