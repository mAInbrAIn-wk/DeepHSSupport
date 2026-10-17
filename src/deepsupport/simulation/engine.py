import math
import zlib
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from pathlib import Path
from models import Student, ModulState, PruefungsErgebnis
from deepsupport.data_engine.config import CONFIG, MODULE_CURRICULA, STUDIENGAENGE, SUPPORT_ANGEBOTE, SUPPORT_KEYWORDS, HZB_TYPEN, HZB_GEWICHTE

# V4.1: Deterministisches, positionsunabhängiges Prüfungsrauschen (restauriert aus V3)
#
# DESIGNENTSCHEIDUNG: Das Prüfungsrauschen ist deterministisch per (Student, Modul, Versuch).
# In allen 8 Universen erhält derselbe Student bei derselben Prüfung dasselbe Zufallsrauschen.
# Die Rauschrichtung (positiv/negativ) bleibt identisch; nur die Amplitude ändert sich
# (via gewicht_rauschen in S07/S08).
#
# Diese Synchronisation ist NICHT alternativlos: In der Realität gibt es einen
# "Schmetterlingseffekt" im unbeobachteten Teil der Welt — z.B. könnte ein Student
# in einer Welt mit Support besser schlafen und deshalb eine andere Tagesform haben.
# Wir entscheiden uns bewusst für die synchronisierte Variante, da sie die kausale
# Isolation der Support-Effekte maximiert und Statusdifferenzen zwischen Universen
# ausschließlich durch den simulierten Mechanismus entstehen.
def get_exam_noise(base_seed: int, modul_id: str, versuch: int, cfg: Dict = None) -> float:
    if cfg is None: cfg = CONFIG
    exam_seed = (base_seed ^ zlib.crc32(f"{modul_id}_{versuch}".encode('utf-8'))) & 0xFFFFFFFF
    return float(np.random.default_rng(exam_seed).normal(0, cfg["gewicht_rauschen"]))

def _erzeuge_semester_liste(start_jahr: int, end_jahr: int, puffer_semester: int = 16) -> List[Dict]:
    semesters = []
    nr = 1
    for jahr in range(start_jahr, end_jahr + 1 + math.ceil(puffer_semester / 2)):
        semesters.append({"semester_id": f"WS{jahr}", "semester_nr": nr, "typ": "WS", "jahr": jahr})
        nr += 1
        semesters.append({"semester_id": f"SS{jahr + 1}", "semester_nr": nr, "typ": "SS", "jahr": jahr + 1})
        nr += 1
    return semesters

def generiere_stammdaten() -> Dict[str, pd.DataFrame]:
    semester_df = pd.DataFrame(_erzeuge_semester_liste(CONFIG["start_jahr"], CONFIG["end_jahr"]))
    studiengaenge_df = pd.DataFrame(STUDIENGAENGE)
    
    module_rows, modul_sg_rows = [], []
    modul_id_counter = 1

    for sg_id, curriculum in MODULE_CURRICULA.items():
        for eintrag in curriculum:
            modul_id = f"MOD{modul_id_counter:04d}"
            module_rows.append({
                "modul_id": modul_id,
                "name": eintrag["name"],
                "cp": eintrag["cp"],
                "schwierigkeit": eintrag["schwierigkeit"],
                "turnus": eintrag.get("turnus", "beides"),
                "workload_h": eintrag.get("workload_h", eintrag["cp"] * 30),
            })
            modul_sg_rows.append({
                "modul_id": modul_id,
                "studiengang_id": sg_id,
                "empfohlenes_fachsemester": eintrag["fachsem"],
                "pflicht": eintrag["pflicht"],
            })
            modul_id_counter += 1

    module_df = pd.DataFrame(module_rows)
    modul_studiengang_df = pd.DataFrame(modul_sg_rows)
    support_angebote_df = pd.DataFrame(SUPPORT_ANGEBOTE)

    zuordnung_rows = []
    seen = set()
    for _, modul in module_df.iterrows():
        modul_name_lower = modul["name"].lower()
        for keyword, angebot_id, wirkung in SUPPORT_KEYWORDS:
            if keyword in modul_name_lower:
                key = (angebot_id, modul["modul_id"])
                if key not in seen:
                    zuordnung_rows.append({
                        "angebot_id": angebot_id,
                        "modul_id": modul["modul_id"],
                        "wirkungsstaerke": wirkung,
                    })
                    seen.add(key)

    support_modul_zuordnung_df = pd.DataFrame(zuordnung_rows)

    return {
        "semester_df": semester_df,
        "studiengaenge_df": studiengaenge_df,
        "module_df": module_df,
        "modul_studiengang_df": modul_studiengang_df,
        "support_angebote_df": support_angebote_df,
        "support_modul_zuordnung_df": support_modul_zuordnung_df,
    }

def generiere_studierende(stammdaten: Dict[str, pd.DataFrame], rng: np.random.Generator, cfg: Dict = None) -> List[Student]:
    if cfg is None:
        cfg = CONFIG
    n = cfg["n_studierende"]
    studiengaenge = stammdaten["studiengaenge_df"]
    semester_df = stammdaten["semester_df"]
    kohorten_semester = semester_df[
        (semester_df["typ"] == "WS")
        & (semester_df["jahr"] >= cfg["start_jahr"])
        & (semester_df["jahr"] <= cfg["end_jahr"])
    ]["semester_id"].tolist()

    sg_gewichte = np.array(cfg.get("studiengang_gewichte", CONFIG["studiengang_gewichte"]), dtype=float)
    sg_gewichte = sg_gewichte / sg_gewichte.sum()

    # Fachbereichsspezifische Geschlechtergewichte aus STUDIENGAENGE (V5 Destatis-Kalibrierung)
    sg_gender_map = {}
    for sg_entry in STUDIENGAENGE:
        sid_key = sg_entry["studiengang_id"]
        if cfg.get("dgp_version", "5.0") == "4.1":
            sg_gender_map[sid_key] = [0.68, 0.30, 0.02] if sid_key in ("SG01", "SG03") else [0.48, 0.50, 0.02]
        else:
            w = np.array(sg_entry.get("geschlecht_gewichte", [0.48, 0.50, 0.02]), dtype=float)
            sg_gender_map[sid_key] = (w / w.sum()).tolist()

    alter_min = cfg.get("alter_min", CONFIG["alter_min"])
    alter_max = cfg.get("alter_max", CONFIG["alter_max"])
    alter_norm_mean = cfg.get("alter_norm_mean", CONFIG["alter_norm_mean"])
    kappa_alter = cfg.get("kappa_alter", CONFIG["kappa_alter"])
    alter_beruf_min = cfg.get("alter_beruflich_min", CONFIG["alter_beruflich_min"])
    alter_beruf_max = cfg.get("alter_beruflich_max", CONFIG["alter_beruflich_max"])

    hzb_min = cfg.get("hzb_min", CONFIG["hzb_min"])
    hzb_max = cfg.get("hzb_max", CONFIG["hzb_max"])
    hzb_span = hzb_max - hzb_min
    hzb_norm_mean = cfg.get("hzb_norm_mean", CONFIG["hzb_norm_mean"])
    kappa_hzb = cfg.get("kappa_hzb", CONFIG["kappa_hzb"])
    hzb_off_allg = cfg.get("hzb_offset_allgemein", CONFIG["hzb_offset_allgemein"])
    hzb_off_fach = cfg.get("hzb_offset_fachgebunden_beruflich", CONFIG["hzb_offset_fachgebunden_beruflich"])

    mig_quote = cfg.get("migrations_quote", CONFIG["migrations_quote"])
    erst_quote = cfg.get("erstakademiker_quote", CONFIG["erstakademiker_quote"])

    erwerb_modell = cfg.get("erwerb_modell", CONFIG["erwerb_modell"])
    erwerb_quote = cfg.get("erwerb_erwerbstaetig_quote", CONFIG["erwerb_erwerbstaetig_quote"])
    erwerb_min_h = cfg.get("erwerb_min_h", CONFIG["erwerb_min_h"])
    erwerb_max_h = cfg.get("erwerb_max_h", CONFIG["erwerb_max_h"])
    erwerb_norm_mean = cfg.get("erwerb_aktiv_norm_mean", CONFIG["erwerb_aktiv_norm_mean"])
    kappa_erwerb = cfg.get("kappa_erwerb", CONFIG["kappa_erwerb"])

    kappa_mot = cfg.get("kappa_motivation", CONFIG["kappa_motivation"])
    mot_boost_beruf = cfg.get("motivation_boost_beruflich", CONFIG["motivation_boost_beruflich"])
    kappa_soz = cfg.get("kappa_integration", CONFIG["kappa_integration"])

    zp_max = cfg.get("zeit_puffer_max_h", CONFIG["zeit_puffer_max_h"])
    zp_mean = cfg.get("zeit_puffer_norm_mean", CONFIG["zeit_puffer_norm_mean"])
    kappa_zp = cfg.get("kappa_zeit_puffer", CONFIG["kappa_zeit_puffer"])

    studierende = []
    for i in range(n):
        sid = f"STUD{i + 1:06d}"
        sg_id = rng.choice(studiengaenge["studiengang_id"].values, p=sg_gewichte)
        koh = rng.choice(kohorten_semester)

        # Geschlecht (V5: fachbereichsspezifisch nach Destatis), Alter, HZB
        geschlecht = rng.choice(["m", "w", "d"], p=sg_gender_map[sg_id])
        alter = int(alter_min + rng.beta(alter_norm_mean * kappa_alter, (1.0 - alter_norm_mean) * kappa_alter) * (alter_max - alter_min))
        hzb_note = round(float(hzb_min + rng.beta(hzb_norm_mean * kappa_hzb, (1.0 - hzb_norm_mean) * kappa_hzb) * hzb_span), 1)
        hzb_typ = rng.choice(HZB_TYPEN, p=HZB_GEWICHTE)

        # HZB-Typ Offsets & Spezialbehandlung
        hzb_offset = 0.0
        if hzb_typ == 'Allg. Hochschulreife':
            hzb_offset = hzb_off_allg
        elif hzb_typ in ('Fachgebundene HR', 'Berufl. Qualifikation'):
            hzb_offset = hzb_off_fach

        if hzb_typ == 'Berufl. Qualifikation':
            alter = max(alter, int(rng.uniform(alter_beruf_min, alter_beruf_max)))

        erwartete_note = float(np.clip(hzb_note + hzb_offset, hzb_min, hzb_max))

        migration = bool(rng.random() < mig_quote)
        erstakademiker = bool(rng.random() < erst_quote)

        # V5: Zero-Inflated Erwerbstätigkeitsmodell nach 22. DSW-Sozialerhebung (37% 0h, 63% aktiv)
        if erwerb_modell == "discrete_v41":
            erwerb = int(np.clip(rng.choice([0, 5, 10, 15, 20, 25, 30], p=[0.25, 0.15, 0.20, 0.15, 0.13, 0.07, 0.05]), 0, 40))
        else:
            ist_erwerbstaetig = rng.random() < erwerb_quote
            erwerb_beta_draw = rng.beta(erwerb_norm_mean * kappa_erwerb, (1.0 - erwerb_norm_mean) * kappa_erwerb)
            if ist_erwerbstaetig:
                erwerb = int(np.clip(round(erwerb_min_h + erwerb_beta_draw * (erwerb_max_h - erwerb_min_h)), erwerb_min_h, 40))
            else:
                erwerb = 0

        mean_mot = np.clip(
            cfg["motivation_startwert"]
            + (2.5 - hzb_note) * cfg["gewicht_motivation_hzb"]
            - erwerb * cfg["gewicht_motivation_erwerb"],
            0.01,
            0.99,
        )
        motivation = float(rng.beta(mean_mot * kappa_mot, (1.0 - mean_mot) * kappa_mot))
        if hzb_typ == 'Berufl. Qualifikation':
            motivation = min(1.0, motivation + mot_boost_beruf)

        mean_soz = np.clip(
            cfg["integration_startwert"]
            - (cfg["gewicht_integration_erstakademiker"] if erstakademiker else 0.0)
            - (cfg["gewicht_integration_migration"] if migration else 0.0)
            - erwerb * cfg["gewicht_integration_erwerb"],
            0.01,
            0.99,
        )
        soz_int = float(rng.beta(mean_soz * kappa_soz, (1.0 - mean_soz) * kappa_soz))

        # Individueller Zeitpuffer als Beta-Verteilung
        zeit_puffer = round(float(rng.beta(zp_mean * kappa_zp, (1.0 - zp_mean) * kappa_zp) * zp_max), 1)

        studi = Student(
            studierenden_id=sid, studiengang_id=sg_id, kohorten_semester_id=koh, geschlecht=geschlecht, alter_immatrikulation=alter,
            hzb_note=hzb_note, hzb_typ=hzb_typ, migrationshintergrund=migration, erstakademiker=erstakademiker, erwerbstaetigkeit_std=erwerb,
            motivation=round(motivation, 3), soziale_integration=round(soz_int, 3), motivation_initial=round(motivation, 3), soziale_integration_initial=round(soz_int, 3),
            erwartete_note=erwartete_note, erwartete_note_initial=erwartete_note,
            hidden_zeit_puffer=zeit_puffer
        )

        sg_module = stammdaten["modul_studiengang_df"]
        my_modules = sg_module[sg_module["studiengang_id"] == sg_id]["modul_id"].tolist()
        for m_id in my_modules:
            studi.modul_states[m_id] = ModulState(modul_id=m_id)

        studierende.append(studi)
    return studierende

# ---- Simulation Logic ----

def simuliere_pruefung(schwierigkeit: float, erwartete_note: float, motivation: float, soz_int: float, fachlicher_boost: float, versuch: int, overload_penalty: float, exam_noise: float, cfg: Dict = None) -> Tuple[float, bool, float]:
    if cfg is None: cfg = CONFIG
    # Berechne latente Leistung ohne Support
    leistung_base = (
        cfg["leistung_startwert"] +
        (2.5 - erwartete_note) * cfg["gewicht_hzb"] +
        (motivation - 0.5) * cfg["gewicht_motivation"] +
        (soz_int - 0.5) * cfg["gewicht_integration"] -
        schwierigkeit * cfg["gewicht_schwierigkeit"] +
        (versuch - 1) * cfg["gewicht_lerneffekt"] -
        overload_penalty +
        exam_noise
    )

    leistung_mit_support = leistung_base + fachlicher_boost

    def leistung_zu_note(l: float) -> float:
        note_raw = float(np.clip(5.0 - l * 4.0, 1.0, 5.0))
        gueltige_noten = [1.0, 1.3, 1.7, 2.0, 2.3, 2.7, 3.0, 3.3, 3.7, 4.0, 5.0]
        return 5.0 if note_raw >= 4.0 else min(gueltige_noten, key=lambda x: abs(x - note_raw))

    note = leistung_zu_note(leistung_mit_support)
    note_counterfactual = leistung_zu_note(leistung_base)
    return note, (note <= 4.0), note_counterfactual

def berechne_dropout(motivation: float, soz_int: float, cp_rueckstand: float, durchgefallen_aktuell: int, fachsemester: int, overload_penalty: float, cfg: Dict = None) -> float:
    """Dropout-Wahrscheinlichkeit. Erwerbstätigkeit wirkt hier NICHT direkt,
    sondern indirekt über das Zeitkontomodell (overload_penalty)."""
    if cfg is None:
        cfg = CONFIG
    p_base = cfg.get("dropout_base", CONFIG["dropout_base"])
    mot_thresh = cfg.get("dropout_schwelle_motivation", CONFIG["dropout_schwelle_motivation"])
    mot_weight = cfg.get("dropout_gewicht_motivation", CONFIG["dropout_gewicht_motivation"])
    soz_thresh = cfg.get("dropout_schwelle_integration", CONFIG["dropout_schwelle_integration"])
    soz_weight = cfg.get("dropout_gewicht_integration", CONFIG["dropout_gewicht_integration"])
    cp_scale = cfg.get("dropout_cp_skala", CONFIG["dropout_cp_skala"])
    cp_weight = cfg.get("dropout_gewicht_cp_rueckstand", CONFIG["dropout_gewicht_cp_rueckstand"])
    fail_weight = cfg.get("dropout_gewicht_fehlversuch", CONFIG["dropout_gewicht_fehlversuch"])
    ovl_cap = cfg.get("dropout_overload_cap", CONFIG["dropout_overload_cap"])
    ovl_weight = cfg.get("dropout_gewicht_overload", CONFIG["dropout_gewicht_overload"])

    p = (
        p_base
        + max(0.0, (mot_thresh - motivation)) * mot_weight
        + max(0.0, (soz_thresh - soz_int)) * soz_weight
        + min(cp_rueckstand / cp_scale, 1.0) * cp_weight
        + durchgefallen_aktuell * fail_weight
        + min(overload_penalty, ovl_cap) * ovl_weight
    )
    if fachsemester == 1:
        p *= cfg.get("dropout_faktor_sem1", CONFIG["dropout_faktor_sem1"])
    if fachsemester >= 5:
        p *= cfg.get("dropout_faktor_ab_sem5", CONFIG["dropout_faktor_ab_sem5"])
    global_scale = cfg.get("dropout_global_skalierer", CONFIG["dropout_global_skalierer"])
    max_p = cfg.get("dropout_max_p", CONFIG["dropout_max_p"])
    return float(np.clip(p * global_scale, 0.0, max_p))

def simuliere_verlaeufe(studierende: List[Student], stammdaten: Dict[str, pd.DataFrame], rng: np.random.Generator, cfg: Dict = None, population_seed: int = 12345):
    if cfg is None: cfg = CONFIG
    module_df = stammdaten["module_df"]
    modul_sg_df = stammdaten["modul_studiengang_df"]
    support_df = stammdaten["support_angebote_df"]
    support_zuord_df = stammdaten["support_modul_zuordnung_df"]
    semester_df = stammdaten["semester_df"].sort_values("semester_nr").reset_index(drop=True)
    studiengaenge_df = stammdaten["studiengaenge_df"]

    semester_order = semester_df["semester_id"].tolist()
    semester_lookup = {sid: i for i, sid in enumerate(semester_order)}

    # --- PRECOMPUTE PHASE (V4/V5 PERFORMANCE OPTIMIZATION) ---
    modul_data = module_df.set_index("modul_id").to_dict("index")
    sg_infos = studiengaenge_df.set_index("studiengang_id").to_dict("index")

    # Precompute Studiengang-Module Lists and Sets
    sg_module_cache = {}
    sg_pflicht_cache = {}
    sg_ba_cache = {}
    for sg_id in sg_infos.keys():
        df = modul_sg_df[modul_sg_df["studiengang_id"] == sg_id]
        sg_module_cache[sg_id] = df.to_dict("records")
        sg_pflicht_cache[sg_id] = [r["modul_id"] for _, r in df.iterrows() if r["pflicht"]]
        sg_ba_cache[sg_id] = [r["modul_id"] for _, r in df.iterrows() if "bachelorarbeit" in modul_data[r["modul_id"]]["name"].lower()]

    # Precompute Support Offers
    support_list = support_df.to_dict("records")

    # Precompute Support Assignments (angebot_id -> list of modul_id)
    ang_to_mod = {}
    mod_to_ang_boost = {}
    for _, row in support_zuord_df.iterrows():
        a_id = row["angebot_id"]
        m_id = row["modul_id"]
        boost = row["wirkungsstaerke"]

        if a_id not in ang_to_mod: ang_to_mod[a_id] = []
        ang_to_mod[a_id].append(m_id)

        if m_id not in mod_to_ang_boost: mod_to_ang_boost[m_id] = {}
        mod_to_ang_boost[m_id][a_id] = boost

    # Semester types precomputed
    sem_types = semester_df.set_index("semester_id")["typ"].to_dict()

    # Support-Zuordnungs-Dict für Carry-over
    support_zuord_dict = {(r["modul_id"], r["angebot_id"]): float(r["wirkungsstaerke"]) for _, r in support_zuord_df.iterrows()}
    support_by_id = {r["angebot_id"]: r for r in support_df.to_dict("records")}

    # Vollständige Support-Liste (ALLE Angebote, auch blockierte) für Pad-Draws
    full_support_df = pd.DataFrame(SUPPORT_ANGEBOTE)
    support_list_all = full_support_df.to_dict("records")
    # Aktive Angebote (nach Universum-Filterung)
    active_support_ids = set(support_df["angebot_id"].tolist())

    # Konfigurierte Parameter (V5 Parametrisierung)
    anomalie_quote = cfg.get("anomalie_quote", CONFIG["anomalie_quote"])
    anomalie_typen = cfg.get("anomalie_typen", CONFIG["anomalie_typen"])
    anomalie_gewichte = cfg.get("anomalie_gewichte", CONFIG["anomalie_gewichte"])
    max_sem = cfg.get("max_simulations_semester", CONFIG["max_simulations_semester"])
    zeitkonto_budget_h = cfg.get("zeitkonto_budget_h", CONFIG["zeitkonto_budget_h"])
    sem_wochen = cfg.get("semester_wochen_faktor", CONFIG["semester_wochen_faktor"])
    zeitkonto_min_h = cfg.get("zeitkonto_min_h", CONFIG["zeitkonto_min_h"])
    ba_rest_cp = cfg.get("ba_zulassung_rest_cp", CONFIG["ba_zulassung_rest_cp"])
    drop_sig_const = cfg.get("modulabwurf_sigmoid_konstante_h", CONFIG["modulabwurf_sigmoid_konstante_h"])

    kosten_faktor = cfg.get("support_kosten_faktor", CONFIG["support_kosten_faktor"])
    rct_mode = cfg.get("rct_support_uptake", CONFIG["rct_support_uptake"])
    rct_rates = cfg.get("rct_rates", CONFIG["rct_rates"])
    disable_apathy = cfg.get("disable_apathy_dampening", CONFIG["disable_apathy_dampening"])
    apathy_thresh = cfg.get("apathy_threshold", CONFIG["apathy_threshold"])

    p_base_fach = cfg.get("support_p_base_fachlich", CONFIG["support_p_base_fachlich"])
    p_slope_fach = cfg.get("support_p_slope_fachlich", CONFIG["support_p_slope_fachlich"])
    p_rep_fach = cfg.get("support_p_repeat_bonus_fachlich", CONFIG["support_p_repeat_bonus_fachlich"])
    p_base_uebf = cfg.get("support_p_base_ueberfachlich", CONFIG["support_p_base_ueberfachlich"])
    p_slope_uebf = cfg.get("support_p_slope_ueberfachlich", CONFIG["support_p_slope_ueberfachlich"])
    p_base_psych = cfg.get("support_p_base_psychosozial", CONFIG["support_p_base_psychosozial"])
    p_slope_psych = cfg.get("support_p_slope_psychosozial", CONFIG["support_p_slope_psychosozial"])
    p_erst_bonus = cfg.get("support_p_erstakademiker_bonus", CONFIG["support_p_erstakademiker_bonus"])
    p_supp_max = cfg.get("support_p_max", CONFIG["support_p_max"])
    p_zeit_ausnahme = cfg.get("support_zeit_ausnahme_p", CONFIG["support_zeit_ausnahme_p"])

    boost_uebf_mot = cfg.get("support_boost_ueberfachlich_mot", CONFIG["support_boost_ueberfachlich_mot"])
    boost_uebf_soz = cfg.get("support_boost_ueberfachlich_soz", CONFIG["support_boost_ueberfachlich_soz"])
    boost_psych_mot = cfg.get("support_boost_psychosozial_mot", CONFIG["support_boost_psychosozial_mot"])
    boost_psych_soz = cfg.get("support_boost_psychosozial_soz", CONFIG["support_boost_psychosozial_soz"])
    carryover_faktor = cfg.get("support_carryover_faktor", CONFIG["support_carryover_faktor"])

    super_diff_thresh = cfg.get("super_klausur_diff_schwelle", CONFIG["super_klausur_diff_schwelle"])
    super_boost_base = cfg.get("super_klausur_boost_base", CONFIG["super_klausur_boost_base"])
    super_boost_slope = cfg.get("super_klausur_boost_slope", CONFIG["super_klausur_boost_slope"])
    glaett_alt = cfg.get("erwartete_note_glaettung_alt", CONFIG["erwartete_note_glaettung_alt"])
    glaett_neu = cfg.get("erwartete_note_glaettung_neu", CONFIG["erwartete_note_glaettung_neu"])
    mot_pen_fail = cfg.get("motivation_penalty_fehlversuch", CONFIG["motivation_penalty_fehlversuch"])
    mot_bon_pass = cfg.get("motivation_bonus_alle_bestanden", CONFIG["motivation_bonus_alle_bestanden"])
    soz_drift_std = cfg.get("soziale_integration_drift_std", CONFIG["soziale_integration_drift_std"])

    # --- Tracker ---
    tracker_modules_dropped = 0
    tracker_overload_hits = 0

    for idx, studi in enumerate(studierende):
        studi.stat_modules_dropped = 0  # Dynamisches Attribut fuer spaetere Auswertung
        if (idx + 1) % cfg.get("log_every_n_studis", 5000) == 0: print(f"Simuliert: {idx+1}/{len(studierende)}")

        # Per-Student-Seeds (synchronisiert über alle 8 Universen)
        base_seed = (zlib.crc32(studi.studierenden_id.encode('utf-8')) ^ population_seed) & 0xFFFFFFFF
        rng_support = np.random.default_rng((base_seed + 1) & 0xFFFFFFFF)
        rng_social  = np.random.default_rng((base_seed + 2) & 0xFFFFFFFF)
        rng_dropout = np.random.default_rng((base_seed + 3) & 0xFFFFFFFF)
        rng_anomalie = np.random.default_rng((base_seed + 4) & 0xFFFFFFFF)
        rng_workload = np.random.default_rng((base_seed + 5) & 0xFFFFFFFF)

        bisherige_fach_supports = set()

        sg_info = sg_infos[studi.studiengang_id]
        sg_module_list = sg_module_cache[studi.studiengang_id]

        koh_idx = semester_lookup[studi.kohorten_semester_id]

        fachsem = 1
        chron_sem_idx = koh_idx

        anomalie_mask = rng_anomalie.random() < anomalie_quote
        if anomalie_mask:
            studi.anomalie_typ = rng_anomalie.choice(anomalie_typen, p=anomalie_gewichte)

        plateau_pausen = 0

        while chron_sem_idx < len(semester_order) and fachsem <= max_sem and not studi.abgebrochen and not studi.abschluss_erreicht and not studi.exmatrikuliert:
            akt_sem_id = semester_order[chron_sem_idx]
            akt_sem_typ = semester_df.iloc[chron_sem_idx]["typ"]

            # Anomalie Plateau (pausiert Fachsemester)
            if studi.anomalie_typ == "plateau" and fachsem in (3, 4) and plateau_pausen < 2:
                plateau_pausen += 1
                studi.motivation = max(0.05, studi.motivation - 0.08)
                chron_sem_idx += 1
                continue

            studi.einschreibungen.append({"semester_id": akt_sem_id, "fachsemester": fachsem, "status": "aktiv"})

            # --- Zeitkonto berechnen ---
            verfuegbare_zeit = max(zeitkonto_min_h, zeitkonto_budget_h - (studi.erwerbstaetigkeit_std * sem_wochen))

            # --- Module auswählen (Beachtung von Turnus und Voraussetzungen) ---
            geplante_module = []
            for row in sg_module_list:
                m_id = row["modul_id"]
                m_state = studi.modul_states[m_id]
                m_info = modul_data[m_id]

                # Check Turnus
                if m_info["turnus"] not in ("beides", akt_sem_typ):
                    continue

                # Offen und empfohlen <= fachsem ODER Wiederholung
                if m_state.status == "offen" and (row["empfohlenes_fachsemester"] <= fachsem or m_state.versuche > 0):
                    if "bachelorarbeit" in m_info["name"].lower():
                        cp_bestanden = studi.cp_bestanden({m: modul_data[m]["cp"] for m in modul_data})
                        if cp_bestanden < sg_info["cp_gesamt"] - ba_rest_cp:
                            continue
                    geplante_module.append(m_id)

            # Optional: Limit modules by Anomalie
            if studi.anomalie_typ == "super_schnell":
                voraus = [r["modul_id"] for r in sg_module_list if r["empfohlenes_fachsemester"] == fachsem + 1 and studi.modul_states[r["modul_id"]].status == "offen"]
                for v_m in voraus:
                    if modul_data[v_m]["turnus"] in ("beides", akt_sem_typ):
                        geplante_module.append(v_m)
                        break
            elif studi.anomalie_typ == "sehr_lang" and rng_anomalie.random() < 0.4:
                geplante_module = geplante_module[:max(1, len(geplante_module)-1)]

            # --- Reaktive Support-Nutzung simulieren (mit Pad-Draws für RNG-Synchronisation) ---
            teilgenommene_angebote = []
            support_zeit_kosten = 0

            for angebot in support_list_all:
                ang_id = angebot["angebot_id"]
                typ = angebot["typ"]
                blocked = ang_id not in active_support_ids
                p = 0.0

                if rct_mode:
                    if typ == "fachlich":
                        rel_module = ang_to_mod.get(ang_id, [])
                        if any(m in rel_module for m in geplante_module):
                            p = rct_rates["fachlich"]
                    else:
                        p = rct_rates.get(typ, 0.02)
                else:
                    if typ == "fachlich":
                        rel_module = ang_to_mod.get(ang_id, [])
                        geplante_relevante = [m for m in geplante_module if m in rel_module]

                        if geplante_relevante:
                            p = p_base_fach + (studi.erwartete_note - 2.0) * p_slope_fach
                            for m in geplante_relevante:
                                if studi.modul_states[m].versuche > 0:
                                    p += p_rep_fach
                    elif typ == "ueberfachlich":
                        p = p_base_uebf + (0.5 - studi.motivation) * p_slope_uebf
                        if not disable_apathy and studi.motivation < apathy_thresh:
                            p *= (studi.motivation / apathy_thresh)
                    else:  # psychosozial
                        p = p_base_psych + (0.5 - studi.soziale_integration) * p_slope_psych
                        if not disable_apathy and studi.soziale_integration < apathy_thresh:
                            p *= (studi.soziale_integration / apathy_thresh)

                    if studi.erstakademiker and typ in ("fachlich", "psychosozial"):
                        p += p_erst_bonus
                    p = float(np.clip(p, 0.0, p_supp_max))

                # IMMER ziehen (Pad-Draw), auch bei blockierten Angeboten
                nutzt_support = rng_support.random() < p

                if nutzt_support and not blocked:
                    kost = angebot.get("kosten_h", 30) * kosten_faktor
                    if verfuegbare_zeit - support_zeit_kosten - kost >= 0 or rng_support.random() < p_zeit_ausnahme:
                        teilgenommene_angebote.append(ang_id)
                        support_zeit_kosten += kost
                        studi.support_teilnahmen.append({"semester_id": akt_sem_id, "angebot_id": ang_id})
                elif nutzt_support and blocked:
                    if verfuegbare_zeit - support_zeit_kosten - angebot.get("kosten_h", 30) < 0:
                        _ = rng_support.random()

            # --- Motivation/Integration Boost durch Support ---
            mult = cfg.get("support_effect_multiplier", CONFIG["support_effect_multiplier"])
            for ang_id in teilgenommene_angebote:
                ang = support_by_id.get(ang_id, {})
                if ang.get("typ") == "ueberfachlich":
                    studi.motivation = min(1.0, studi.motivation + boost_uebf_mot * mult)
                    studi.soziale_integration = min(1.0, studi.soziale_integration + boost_uebf_soz * mult)
                elif ang.get("typ") == "psychosozial":
                    studi.motivation = min(1.0, studi.motivation + boost_psych_mot * mult)
                    studi.soziale_integration = min(1.0, studi.soziale_integration + boost_psych_soz * mult)

            # --- Probabilistischer Modulabwurf ---
            geplanter_workload = sum(modul_data[m]["workload_h"] for m in geplante_module)
            ueberschuss = geplanter_workload + support_zeit_kosten - verfuegbare_zeit - studi.hidden_zeit_puffer
            if ueberschuss > 0 and len(geplante_module) > 1:
                tracker_overload_hits += 1
            while ueberschuss > 0 and len(geplante_module) > 1:
                p_drop = float(np.clip(ueberschuss / (ueberschuss + drop_sig_const), 0.0, 0.99))
                if rng_workload.random() < p_drop:
                    geplante_module.sort(key=lambda m: modul_data[m]["schwierigkeit"])
                    dropped = geplante_module.pop()
                    geplanter_workload -= modul_data[dropped]["workload_h"]
                    ueberschuss = geplanter_workload + support_zeit_kosten - verfuegbare_zeit - studi.hidden_zeit_puffer
                    tracker_modules_dropped += 1
                    studi.stat_modules_dropped += 1
                else:
                    break

            # Berechne verbleibenden Overload
            total_workload = geplanter_workload + support_zeit_kosten
            overload = max(0, total_workload - verfuegbare_zeit)
            overload_penalty_factor = cfg.get("overload_penalty_factor", CONFIG["overload_penalty_factor"])
            overload_penalty_raw = (overload / 100.0) * overload_penalty_factor
            overload_cap = cfg.get("overload_penalty_cap", CONFIG["overload_penalty_cap"])
            overload_penalty = float(min(overload_cap, overload_penalty_raw)) if overload_cap is not None else overload_penalty_raw

            # --- Prüfungen ablegen ---
            durchgefallen_dieses_sem = 0
            supp_deckel = cfg.get("support_deckel", CONFIG["support_deckel"])
            gew_supp_boost = cfg.get("gewicht_support_boost", CONFIG["gewicht_support_boost"])
            for m_id in geplante_module:
                m_state = studi.modul_states[m_id]
                m_state.versuche += 1

                # Fachlicher Support Boost MIT Carry-over
                current_boost_sum = sum(support_zuord_dict.get((m_id, ang_id), 0.0) for ang_id in teilgenommene_angebote)
                carryover_ids = bisherige_fach_supports - set(teilgenommene_angebote)
                carryover_boost_sum = sum(support_zuord_dict.get((m_id, ang_id), 0.0) for ang_id in carryover_ids)
                boost_sum = current_boost_sum + carryover_boost_sum * carryover_faktor
                raw_boost = boost_sum * gew_supp_boost * mult if boost_sum > 0.0 else 0.0
                boost = float(np.clip(raw_boost, 0.0, supp_deckel))

                # Deterministisches Prüfungsrauschen
                e_noise = get_exam_noise(base_seed, m_id, m_state.versuche, cfg)

                note, bestanden, note_cf = simuliere_pruefung(
                    schwierigkeit=modul_data[m_id]["schwierigkeit"],
                    erwartete_note=studi.erwartete_note,
                    motivation=studi.motivation,
                    soz_int=studi.soziale_integration,
                    fachlicher_boost=boost,
                    versuch=m_state.versuche,
                    overload_penalty=overload_penalty,
                    exam_noise=e_noise,
                    cfg=cfg
                )

                studi.pruefungen.append(PruefungsErgebnis(
                    semester_id=akt_sem_id, modul_id=m_id, versuch=m_state.versuche,
                    note=note, bestanden=bestanden, note_counterfactual=note_cf, support_genutzt=(boost > 0),
                    hidden_motivation=studi.motivation,
                    hidden_soziale_integration=studi.soziale_integration,
                    hidden_erwartete_note=studi.erwartete_note,
                    hidden_overload=overload,
                    hidden_zeit_puffer=studi.hidden_zeit_puffer,
                    hidden_penalty_capped=(overload_cap is not None and overload_penalty_raw > overload_cap),
                    hidden_support_capped=(raw_boost > supp_deckel)
                ))

                if bestanden:
                    m_state.status = "bestanden"
                    m_state.note = note
                else:
                    durchgefallen_dieses_sem += 1
                    if m_state.versuche >= 3:
                        m_state.status = "gescheitert"
                        if "bachelorarbeit" not in modul_data[m_id]["name"].lower():
                            studi.exmatrikuliert = True

            # --- Super-Klausur Motivationsboost & Dynamische Fähigkeiten ---
            sem_pruefungen = [p for p in studi.pruefungen if p.semester_id == akt_sem_id and p.bestanden]
            for p_erg in sem_pruefungen:
                grade_diff = studi.erwartete_note - p_erg.note
                if grade_diff >= super_diff_thresh:
                    super_boost = super_boost_base + super_boost_slope * (grade_diff - super_diff_thresh)
                    studi.motivation = min(1.0, studi.motivation + super_boost)

            # --- Dynamisches Update der erwarteten Note (Fähigkeiten-Gewinn) ---
            if sem_pruefungen:
                sem_gpa = sum(p.note for p in sem_pruefungen) / len(sem_pruefungen)
                if sem_gpa < studi.erwartete_note:
                    studi.erwartete_note = round(glaett_alt * studi.erwartete_note + glaett_neu * sem_gpa, 2)

            # --- Motivation/Integration nach Semesterergebnis ---
            if durchgefallen_dieses_sem > 0:
                studi.motivation = max(0.05, studi.motivation - mot_pen_fail * durchgefallen_dieses_sem)
            elif len(geplante_module) > 0:
                studi.motivation = min(1.0, studi.motivation + mot_bon_pass)

            # Soziale Integration Drift (rng_social)
            studi.soziale_integration = float(np.clip(studi.soziale_integration + rng_social.normal(0, soz_drift_std), 0.05, 1.0))

            # --- Abschluss / Dropout ---
            cp_bestanden = studi.cp_bestanden({m: modul_data[m]["cp"] for m in modul_data})
            if studi.alle_pflicht_bestanden(sg_pflicht_cache[studi.studiengang_id]):
                ba_module = sg_ba_cache[studi.studiengang_id]
                if not ba_module or studi.modul_states[ba_module[0]].status == "bestanden":
                    studi.abschluss_erreicht = True

            if not studi.abschluss_erreicht and not studi.exmatrikuliert:
                cp_soll = (fachsem / sg_info["regelstudienzeit"]) * sg_info["cp_gesamt"]
                cp_rueckstand = max(0.0, cp_soll - cp_bestanden)
                p_drop = berechne_dropout(studi.motivation, studi.soziale_integration, cp_rueckstand, durchgefallen_dieses_sem, fachsem, overload_penalty, cfg=cfg)
                if studi.anomalie_typ == "sehr_lang": p_drop *= 0.3
                if rng_dropout.random() < p_drop:
                    studi.abgebrochen = True

            # Carry-over fachlicher Supports für nächstes Semester merken
            for ang_id in teilgenommene_angebote:
                if support_by_id.get(ang_id, {}).get("typ") == "fachlich":
                    bisherige_fach_supports.add(ang_id)

            fachsem += 1
            chron_sem_idx += 1

    print(f"\n[V5 TRACKER] Zeitbudget-Analyse abgeschlossen:")
    print(f"[V5 TRACKER] In {tracker_overload_hits} Semestern wurde die Workload-Schranke gesprengt.")
    print(f"[V5 TRACKER] Insgesamt wurden {tracker_modules_dropped} Modulpruefungen wegen Ueberlast/Zeitbudget zurueck in die Warteschlange geworfen.")

    return studierende
