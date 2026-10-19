import math
import zlib
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from pathlib import Path
from models import Student, ModulState, PruefungsErgebnis
from config import CONFIG, MODULE_CURRICULA, STUDIENGAENGE, SUPPORT_ANGEBOTE, SUPPORT_KEYWORDS, HZB_TYPEN, HZB_GEWICHTE

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
    if cfg is None: cfg = CONFIG
    n = cfg["n_studierende"]
    studiengaenge = stammdaten["studiengaenge_df"]
    semester_df = stammdaten["semester_df"]
    kohorten_semester = semester_df[(semester_df["typ"] == "WS") & (semester_df["jahr"] >= cfg["start_jahr"]) & (semester_df["jahr"] <= cfg["end_jahr"])]["semester_id"].tolist()
    
    sg_gewichte = np.array([0.25, 0.28, 0.18, 0.17, 0.12])
    sg_gewichte = sg_gewichte / sg_gewichte.sum()

    studierende = []
    for i in range(n):
        sid = f"STUD{i + 1:06d}"
        sg_id = rng.choice(studiengaenge["studiengang_id"].values, p=sg_gewichte)
        koh = rng.choice(kohorten_semester)
        
        # Geschlecht, Alter, HZB
        geschlecht = rng.choice(["m", "w", "d"], p=[0.68, 0.30, 0.02]) if sg_id in ("SG01", "SG03") else rng.choice(["m", "w", "d"], p=[0.48, 0.50, 0.02])
        # Alter normalisiert: mean ~ (20.5-17)/(45-17) = 0.125, kappa=12.8 fuer std~2.49
        alter = int(17 + rng.beta(0.125 * 12.8, (1.0 - 0.125) * 12.8) * (45 - 17))
        # HZB normalisiert: mean ~ (2.4-1.0)/(4.0-1.0) = 0.466, kappa=6.5 fuer std~0.55
        hzb_note = round(float(1.0 + rng.beta(0.466 * 6.5, (1.0 - 0.466) * 6.5) * 3.0), 1)
        hzb_typ = rng.choice(HZB_TYPEN, p=HZB_GEWICHTE)
        
        # HZB-Typ Offsets & Spezialbehandlung
        hzb_offset = 0.0
        if hzb_typ == 'Allg. Hochschulreife':
            hzb_offset = -0.2
        elif hzb_typ in ('Fachgebundene HR', 'Berufl. Qualifikation'):
            hzb_offset = 0.2
            
        if hzb_typ == 'Berufl. Qualifikation':
            alter = max(alter, int(rng.uniform(24, 28)))
            
        erwartete_note = float(np.clip(hzb_note + hzb_offset, 1.0, 4.0))
        
        migration = bool(rng.random() < 0.22)
        erstakademiker = bool(rng.random() < 0.48)
        
        erwerb = int(np.clip(rng.choice([0, 5, 10, 15, 20, 25, 30], p=[0.25, 0.15, 0.20, 0.15, 0.13, 0.07, 0.05]), 0, 40))
        
        mean_mot = np.clip(cfg["motivation_startwert"] + (2.5 - hzb_note) * cfg["gewicht_motivation_hzb"] - erwerb * cfg["gewicht_motivation_erwerb"], 0.01, 0.99)
        motivation = float(rng.beta(mean_mot * 20.0, (1.0 - mean_mot) * 20.0))
        if hzb_typ == 'Berufl. Qualifikation':
            motivation = min(1.0, motivation + 0.10) # Motivationsboost wegen genauerer Zielvorstellung
            
        mean_soz = np.clip(cfg["integration_startwert"] - (cfg["gewicht_integration_erstakademiker"] if erstakademiker else 0) - (cfg["gewicht_integration_migration"] if migration else 0) - erwerb * cfg["gewicht_integration_erwerb"], 0.01, 0.99)
        soz_int = float(rng.beta(mean_soz * 20.0, (1.0 - mean_soz) * 20.0))

        # V4.1: Individueller Zeitpuffer als Beta-Verteilung (restauriert aus V3)
        # mean≈60h (0.33*180), kappa=8 → std≈26h, Wertebereich [0, 180]
        zeit_puffer = round(float(rng.beta(0.33 * 8.0, (1.0 - 0.33) * 8.0) * 180.0), 1)

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

def berechne_dropout(motivation: float, soz_int: float, cp_rueckstand: float, durchgefallen_aktuell: int, fachsemester: int, overload_penalty: float) -> float:
    """Dropout-Wahrscheinlichkeit. Erwerbstätigkeit wirkt hier NICHT direkt,
    sondern indirekt über das Zeitkontomodell (overload_penalty)."""
    p = 0.01 + max(0.0, (0.4 - motivation)) * 0.30 + max(0.0, (0.4 - soz_int)) * 0.20 + min(cp_rueckstand / 30.0, 1.0) * 0.15 + durchgefallen_aktuell * 0.04 + min(overload_penalty, 0.3) * 0.10
    if fachsemester == 1: p *= 1.4
    if fachsemester >= 5: p *= 0.6
    return float(np.clip(p * 0.5, 0.0, 0.45))

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
    
    # --- PRECOMPUTE PHASE (V4 PERFORMANCE OPTIMIZATION) ---
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

    # V4.1: Support-Zuordnungs-Dict für Carry-over (restauriert aus V3)
    support_zuord_dict = {(r["modul_id"], r["angebot_id"]): float(r["wirkungsstaerke"]) for _, r in support_zuord_df.iterrows()}
    support_by_id = {r["angebot_id"]: r for r in support_df.to_dict("records")}
    
    # V4.1: Vollständige Support-Liste (ALLE Angebote, auch blockierte) für Pad-Draws
    full_support_df = pd.DataFrame(SUPPORT_ANGEBOTE)
    support_list_all = full_support_df.to_dict("records")
    # Aktive Angebote (nach Universum-Filterung)
    active_support_ids = set(support_df["angebot_id"].tolist())

    # --- V4 Tracker ---
    tracker_modules_dropped = 0
    tracker_overload_hits = 0
    
    for idx, studi in enumerate(studierende):
        studi.stat_modules_dropped = 0 # Dynamisches Attribut fuer spuetere Auswertung
        if (idx + 1) % cfg.get("log_every_n_studis", 5000) == 0: print(f"Simuliert: {idx+1}/{len(studierende)}")
        
        # V4.1: Per-Student-Seeds (restauriert aus V3)
        base_seed = (zlib.crc32(studi.studierenden_id.encode('utf-8')) ^ population_seed) & 0xFFFFFFFF
        rng_support = np.random.default_rng((base_seed + 1) & 0xFFFFFFFF)
        rng_social  = np.random.default_rng((base_seed + 2) & 0xFFFFFFFF)
        rng_dropout = np.random.default_rng((base_seed + 3) & 0xFFFFFFFF)
        rng_anomalie = np.random.default_rng((base_seed + 4) & 0xFFFFFFFF)
        rng_workload = np.random.default_rng((base_seed + 5) & 0xFFFFFFFF)  # V4.1: Probabilistischer Modulabwurf
        
        # V4.1: Carry-over fachlicher Supports (restauriert aus V3)
        bisherige_fach_supports = set()
        
        sg_info = sg_infos[studi.studiengang_id]
        sg_module_list = sg_module_cache[studi.studiengang_id]
        
        koh_idx = semester_lookup[studi.kohorten_semester_id]
        
        fachsem = 1
        chron_sem_idx = koh_idx
        
        # Anomalie (V4.1: eigener RNG-Stream)
        anomalie_mask = rng_anomalie.random() < cfg.get("anomalie_quote", 0.05)
        if anomalie_mask:
            studi.anomalie_typ = rng_anomalie.choice(["super_schnell", "sehr_lang", "fruehabbruch", "plateau"], p=[0.20, 0.40, 0.25, 0.15])
            
        plateau_pausen = 0
        
        while chron_sem_idx < len(semester_order) and fachsem <= cfg.get("max_simulations_semester", 16) and not studi.abgebrochen and not studi.abschluss_erreicht and not studi.exmatrikuliert:
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
            # Budget: 900h für Vollzeit pro Semester. Abzug: Erwerb (z.B. 15h * 20 Wochen = 300h)
            verfuegbare_zeit = max(100, cfg.get("zeitkonto_budget_h", 900) - (studi.erwerbstaetigkeit_std * 20))
            
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
                    # Bachelorarbeit nur zulassen, wenn fast alles fertig (z.B. CP_ges - 18)
                    if "bachelorarbeit" in m_info["name"].lower():
                        cp_bestanden = studi.cp_bestanden({m: modul_data[m]["cp"] for m in modul_data})
                        if cp_bestanden < sg_info["cp_gesamt"] - 18:
                            continue
                    geplante_module.append(m_id)
            
            # Optional: Limit modules by Anomalie
            if studi.anomalie_typ == "super_schnell":
                # Versucht noch ein Modul aus dem nächsten Semester vorzuziehen
                voraus = [r["modul_id"] for r in sg_module_list if r["empfohlenes_fachsemester"] == fachsem + 1 and studi.modul_states[r["modul_id"]].status == "offen"]
                for v_m in voraus:
                    if modul_data[v_m]["turnus"] in ("beides", akt_sem_typ):
                        geplante_module.append(v_m)
                        break
            elif studi.anomalie_typ == "sehr_lang" and rng_anomalie.random() < 0.4:
                geplante_module = geplante_module[:max(1, len(geplante_module)-1)]
            
            # --- Reaktive Support-Nutzung simulieren (V4.1: Pad-Draws, rng_support) ---
            teilgenommene_angebote = []
            support_zeit_kosten = 0
            kosten_faktor = cfg.get("support_kosten_faktor", 1.0)  # V4.1: Faktor statt Override
            rct_mode = cfg.get("rct_support_uptake", False)
            
            # V4.1: Über ALLE Angebote iterieren (nicht vorfiltriert), Pad-Draws wie V3
            for angebot in support_list_all:
                ang_id = angebot["angebot_id"]
                typ = angebot["typ"]
                blocked = ang_id not in active_support_ids
                p = 0.0
                
                if rct_mode:
                    rct_rates = {"fachlich": 0.042, "ueberfachlich": 0.025, "psychosozial": 0.023}
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
                            p = 0.05 + (studi.erwartete_note - 2.0) * 0.05
                            for m in geplante_relevante:
                                if studi.modul_states[m].versuche > 0:
                                    p += 0.20
                    elif typ == "ueberfachlich":
                        p = 0.05 + (0.5 - studi.motivation) * 0.15
                        if not cfg.get("disable_apathy_dampening", False) and studi.motivation < 0.2:
                            p *= (studi.motivation / 0.2)
                    else: # psychosozial
                        p = 0.01 + (0.5 - studi.soziale_integration) * 0.12
                        if not cfg.get("disable_apathy_dampening", False) and studi.soziale_integration < 0.2:
                            p *= (studi.soziale_integration / 0.2)
                    
                    if studi.erstakademiker and typ in ("fachlich", "psychosozial"): p += 0.05
                    p = float(np.clip(p, 0.0, 0.9))
                
                # V4.1: IMMER ziehen (Pad-Draw), auch bei blockierten Angeboten
                nutzt_support = rng_support.random() < p
                
                if nutzt_support and not blocked:
                    # Zeitcheck mit stochastischem Puffer (wie V3)
                    kost = angebot.get("kosten_h", 30) * kosten_faktor
                    if verfuegbare_zeit - support_zeit_kosten - kost >= 0 or rng_support.random() < 0.2:
                        teilgenommene_angebote.append(ang_id)
                        support_zeit_kosten += kost
                        studi.support_teilnahmen.append({"semester_id": akt_sem_id, "angebot_id": ang_id})
                elif nutzt_support and blocked:
                    # V4.1: Pad-Draw für Zeitcheck (restauriert aus V3)
                    if verfuegbare_zeit - support_zeit_kosten - angebot.get("kosten_h", 30) < 0:
                        _ = rng_support.random()
            
            # --- Motivation/Integration Boost durch Support ---
            mult = cfg.get("support_effect_multiplier", 1.0)
            for ang_id in teilgenommene_angebote:
                ang = support_by_id.get(ang_id, {})
                if ang.get("typ") == "ueberfachlich":
                    studi.motivation = min(1.0, studi.motivation + 0.02 * mult)
                    studi.soziale_integration = min(1.0, studi.soziale_integration + 0.01 * mult)
                elif ang.get("typ") == "psychosozial":
                    studi.motivation = min(1.0, studi.motivation + 0.015 * mult)
                    studi.soziale_integration = min(1.0, studi.soziale_integration + 0.035 * mult)

            # --- V4.1: Probabilistischer Modulabwurf (statt deterministischem Schwellwert) ---
            # Wahrscheinlichkeit steigt mit Überschuss über individuellem Zeitpuffer
            geplanter_workload = sum(modul_data[m]["workload_h"] for m in geplante_module)
            ueberschuss = geplanter_workload + support_zeit_kosten - verfuegbare_zeit - studi.hidden_zeit_puffer
            if ueberschuss > 0 and len(geplante_module) > 1:
                tracker_overload_hits += 1
            while ueberschuss > 0 and len(geplante_module) > 1:
                # Sigmoid: bei 50h Überschuss ~50%, bei 150h ~75%, bei 300h ~86%
                p_drop = float(np.clip(ueberschuss / (ueberschuss + 50.0), 0.0, 0.99))
                if rng_workload.random() < p_drop:
                    geplante_module.sort(key=lambda m: modul_data[m]["schwierigkeit"])
                    dropped = geplante_module.pop()
                    geplanter_workload -= modul_data[dropped]["workload_h"]
                    ueberschuss = geplanter_workload + support_zeit_kosten - verfuegbare_zeit - studi.hidden_zeit_puffer
                    tracker_modules_dropped += 1
                    studi.stat_modules_dropped += 1
                else:
                    break  # Student behält Module trotz Überbelastung
            
            # Berechne verbleibenden Overload
            total_workload = geplanter_workload + support_zeit_kosten
            overload = max(0, total_workload - verfuegbare_zeit)
            # V4.1: Overload-Penalty, konfigurierbar. Optionales Cap via overload_penalty_cap (S14)
            overload_penalty_factor = cfg.get("overload_penalty_factor", 0.1)
            overload_penalty_raw = (overload / 100.0) * overload_penalty_factor
            overload_cap = cfg.get("overload_penalty_cap", None)
            overload_penalty = float(min(overload_cap, overload_penalty_raw)) if overload_cap is not None else overload_penalty_raw
            
            # --- Prüfungen ablegen ---
            durchgefallen_dieses_sem = 0
            for m_id in geplante_module:
                m_state = studi.modul_states[m_id]
                m_state.versuche += 1
                
                # V4.1: Fachlicher Support Boost MIT Carry-over (restauriert aus V3)
                current_boost_sum = sum(support_zuord_dict.get((m_id, ang_id), 0.0) for ang_id in teilgenommene_angebote)
                carryover_ids = bisherige_fach_supports - set(teilgenommene_angebote)
                carryover_boost_sum = sum(support_zuord_dict.get((m_id, ang_id), 0.0) for ang_id in carryover_ids)
                boost_sum = current_boost_sum + carryover_boost_sum * (2.0 / 3.0)
                raw_boost = boost_sum * cfg.get("gewicht_support_boost", 0.08) * cfg.get("support_effect_multiplier", 1.0) if boost_sum > 0.0 else 0.0
                boost = float(np.clip(raw_boost, 0.0, cfg.get("support_deckel", 1.0)))
                
                # V4.1: Deterministisches Prüfungsrauschen (restauriert aus V3)
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
                    hidden_overload=overload,                                                    # V4.1 restauriert
                    hidden_zeit_puffer=studi.hidden_zeit_puffer,                                  # V4.1 restauriert
                    hidden_penalty_capped=(overload_cap is not None and overload_penalty_raw > overload_cap),  # V4.1 restauriert
                    hidden_support_capped=(raw_boost > cfg.get("support_deckel", 1.0))            # V4.1 restauriert
                ))
                
                if bestanden:
                    m_state.status = "bestanden"
                    m_state.note = note
                else:
                    durchgefallen_dieses_sem += 1
                    if m_state.versuche >= 3:
                        m_state.status = "gescheitert"
                        if "bachelorarbeit" not in modul_data[m_id]["name"].lower():
                            studi.exmatrikuliert = True # Endgültig nicht bestanden
            
            # --- Super-Klausur Motivationsboost & Dynamische Fähigkeiten ---
            sem_pruefungen = [p for p in studi.pruefungen if p.semester_id == akt_sem_id and p.bestanden]
            for p_erg in sem_pruefungen:
                grade_diff = studi.erwartete_note - p_erg.note
                if grade_diff >= 0.5:
                    super_boost = 0.005 + 0.01 * (grade_diff - 0.5)
                    studi.motivation = min(1.0, studi.motivation + super_boost)

            # --- Dynamisches Update der erwarteten Note (Fähigkeiten-Gewinn) ---
            if sem_pruefungen:
                sem_gpa = sum(p.note for p in sem_pruefungen) / len(sem_pruefungen)
                if sem_gpa < studi.erwartete_note:
                    studi.erwartete_note = round(0.7 * studi.erwartete_note + 0.3 * sem_gpa, 2)

            # --- Motivation/Integration nach Semesterergebnis ---
            if durchgefallen_dieses_sem > 0:
                studi.motivation = max(0.05, studi.motivation - 0.05 * durchgefallen_dieses_sem)
            elif len(geplante_module) > 0:
                studi.motivation = min(1.0, studi.motivation + 0.02)
                
            # --- DEMOTIVATIONS-MECHANIK (Vorbereiteter inaktiver Code) ---
            # Demotivation durch Noten enttäuschung (schlechter als erwartete Note):
            # for p_erg in [p for p in studi.pruefungen if p.semester_id == akt_sem_id]:
            #     if p_erg.note - studi.erwartete_note >= 1.0:
            #         demotivation_penalty = 0.01 * (p_erg.note - studi.erwartete_note)
            #         studi.motivation = max(0.05, studi.motivation - demotivation_penalty)

            # V4.1: Soziale Integration Drift (rng_social, restauriert aus V3)
            studi.soziale_integration = float(np.clip(studi.soziale_integration + rng_social.normal(0, 0.05), 0.05, 1.0))
            
            # --- Abschluss / Dropout ---
            cp_bestanden = studi.cp_bestanden({m: modul_data[m]["cp"] for m in modul_data})
            if studi.alle_pflicht_bestanden(sg_pflicht_cache[studi.studiengang_id]):
                ba_module = sg_ba_cache[studi.studiengang_id]
                if not ba_module or studi.modul_states[ba_module[0]].status == "bestanden":
                    studi.abschluss_erreicht = True
            
            if not studi.abschluss_erreicht and not studi.exmatrikuliert:
                cp_soll = (fachsem / sg_info["regelstudienzeit"]) * sg_info["cp_gesamt"]
                cp_rueckstand = max(0.0, cp_soll - cp_bestanden)
                p_drop = berechne_dropout(studi.motivation, studi.soziale_integration, cp_rueckstand, durchgefallen_dieses_sem, fachsem, overload_penalty)
                if studi.anomalie_typ == "sehr_lang": p_drop *= 0.3
                # V4.1: rng_dropout statt globalem rng (restauriert aus V3)
                if rng_dropout.random() < p_drop:
                    studi.abgebrochen = True

            # V4.1: Carry-over fachlicher Supports für nächstes Semester merken (restauriert aus V3)
            for ang_id in teilgenommene_angebote:
                if support_by_id.get(ang_id, {}).get("typ") == "fachlich":
                    bisherige_fach_supports.add(ang_id)

            fachsem += 1
            chron_sem_idx += 1

    print(f"\n[V4 TRACKER] Zeitbudget-Analyse abgeschlossen:")
    print(f"[V4 TRACKER] In {tracker_overload_hits} Semestern wurde die Workload-Schranke gesprengt.")
    print(f"[V4 TRACKER] Insgesamt wurden {tracker_modules_dropped} Modulpruefungen wegen Ueberlast/Zeitbudget zurueck in die Warteschlange geworfen.")
    
    return studierende
