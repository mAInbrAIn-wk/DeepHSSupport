import math
import zlib
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from pathlib import Path
from models import Student, ModulState, PruefungsErgebnis
from config import CONFIG, MODULE_CURRICULA, STUDIENGAENGE, SUPPORT_ANGEBOTE, SUPPORT_KEYWORDS, HZB_TYPEN, HZB_GEWICHTE
from simulation_v2 import _erzeuge_semester_liste, generiere_stammdaten, simuliere_pruefung, berechne_dropout

def get_exam_noise(base_seed: int, modul_id: str, versuch: int) -> float:
    exam_seed = (base_seed ^ zlib.crc32(f"{modul_id}_{versuch}".encode('utf-8'))) & 0xFFFFFFFF
    return float(np.random.default_rng(exam_seed).normal(0, CONFIG["gewicht_rauschen"]))


def generiere_studierende_v3(stammdaten: Dict[str, pd.DataFrame], rng: np.random.Generator) -> List[Student]:
    n = CONFIG["n_studierende"]
    studiengaenge = stammdaten["studiengaenge_df"]
    semester_df = stammdaten["semester_df"]
    kohorten_semester = semester_df[(semester_df["typ"] == "WS") & (semester_df["jahr"] >= CONFIG["start_jahr"]) & (semester_df["jahr"] <= CONFIG["end_jahr"])]["semester_id"].tolist()
    
    sg_gewichte = np.array([0.25, 0.28, 0.18, 0.17, 0.12])
    sg_gewichte = sg_gewichte / sg_gewichte.sum()

    studierende = []
    for i in range(n):
        sid = f"STUD{i + 1:06d}"
        sg_id = rng.choice(studiengaenge["studiengang_id"].values, p=sg_gewichte)
        koh = rng.choice(kohorten_semester)
        
        # Geschlecht, Alter, HZB
        geschlecht = rng.choice(["m", "w", "d"], p=[0.68, 0.30, 0.02]) if sg_id in ("SG01", "SG03") else rng.choice(["m", "w", "d"], p=[0.48, 0.50, 0.02])
        alter = int(np.clip(rng.normal(20.5, 2.8), 17, 45))
        hzb_note = round(float(np.clip(rng.normal(2.4, 0.55), 1.0, 4.0)), 1)
        hzb_typ = rng.choice(HZB_TYPEN, p=HZB_GEWICHTE)
        
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
        
        motivation = float(np.clip(CONFIG["motivation_startwert"] + (2.5 - hzb_note) * CONFIG["gewicht_motivation_hzb"] - erwerb * CONFIG["gewicht_motivation_erwerb"] + rng.normal(0, CONFIG["gewicht_motivation_rauschen"]), 0.05, 1.0))
        if hzb_typ == 'Berufl. Qualifikation':
            motivation = min(1.0, motivation + 0.10)
            
        soz_int = float(np.clip(CONFIG["integration_startwert"] - (CONFIG["gewicht_integration_erstakademiker"] if erstakademiker else 0) - (CONFIG["gewicht_integration_migration"] if migration else 0) - erwerb * CONFIG["gewicht_integration_erwerb"] + rng.normal(0, CONFIG["gewicht_integration_rauschen"]), 0.05, 1.0))

        # SIMULATION V3: Stochastischer Puffer B_i ~ N(60, 30) clipped [0, 180]
        zeit_puffer = round(float(np.clip(rng.normal(60.0, 30.0), 0.0, 180.0)), 1)

        studi = Student(
            studierenden_id=sid, studiengang_id=sg_id, kohorten_semester_id=koh, geschlecht=geschlecht, alter_immatrikulation=alter,
            hzb_note=hzb_note, hzb_typ=hzb_typ, migrationshintergrund=migration, erstakademiker=erstakademiker, erwerbstaetigkeit_std=erwerb,
            motivation=round(motivation, 3), soziale_integration=round(soz_int, 3), motivation_initial=round(motivation, 3), soziale_integration_initial=round(soz_int, 3),
            erwartete_note=erwartete_note, erwartete_note_initial=erwartete_note, hidden_zeit_puffer=zeit_puffer
        )
        
        sg_module = stammdaten["modul_studiengang_df"]
        my_modules = sg_module[sg_module["studiengang_id"] == sg_id]["modul_id"].tolist()
        for m_id in my_modules:
            studi.modul_states[m_id] = ModulState(modul_id=m_id)
            
        studierende.append(studi)

    return studierende

def simuliere_verlaeufe_v3(
    studierende: List[Student],
    stammdaten: Dict[str, pd.DataFrame],
    block_fach: bool = False,
    block_uebf: bool = False,
    block_psych: bool = False,
    population_seed: int = 12345
) -> List[Student]:

    semester_df = stammdaten["semester_df"].sort_values("semester_nr").reset_index(drop=True)
    sg_module_df = stammdaten["modul_studiengang_df"]
    module_df = stammdaten["module_df"]
    support_df = stammdaten["support_angebote_df"]
    support_zuord_df = stammdaten["support_modul_zuordnung_df"]
    sg_info_dict = {r["studiengang_id"]: r for _, r in stammdaten["studiengaenge_df"].iterrows()}
    modul_data = {r["modul_id"]: r for _, r in module_df.iterrows()}

    semester_order = semester_df["semester_id"].tolist()
    semester_types = semester_df["typ"].tolist()
    semester_lookup = {sid: i for i, sid in enumerate(semester_order)}

    # Precompute dictionaries to avoid pandas dataframe filtering inside 800k loop iterations
    support_zuord_dict = {(r["modul_id"], r["angebot_id"]): float(r["wirkungsstaerke"]) for _, r in support_zuord_df.iterrows()}
    angebot_to_modules = support_zuord_df.groupby("angebot_id")["modul_id"].apply(set).to_dict()
    support_list = support_df.to_dict("records")
    support_by_id = {r["angebot_id"]: r for r in support_list}
    sg_module_dict = {sg_id: sub.to_dict("records") for sg_id, sub in sg_module_df.groupby("studiengang_id")}
    modul_cp_dict = {m: r["cp"] for m, r in modul_data.items()}

    for idx, studi in enumerate(studierende):
        base_seed = (zlib.crc32(studi.studierenden_id.encode('utf-8')) ^ population_seed) & 0xFFFFFFFF
        rng_init = np.random.default_rng(base_seed)
        rng_support = np.random.default_rng((base_seed + 1) & 0xFFFFFFFF)
        rng_social = np.random.default_rng((base_seed + 2) & 0xFFFFFFFF)
        rng_dropout = np.random.default_rng((base_seed + 3) & 0xFFFFFFFF)
        
        sg_info = sg_info_dict[studi.studiengang_id]
        sg_module_rows = sg_module_dict[studi.studiengang_id]
        
        koh_idx = semester_lookup[studi.kohorten_semester_id]
        
        fachsem = 1
        chron_sem_idx = koh_idx
        
        anomalie_mask = rng_init.random() < CONFIG["anomalie_quote"]
        if anomalie_mask:
            studi.anomalie_typ = rng_init.choice(["super_schnell", "sehr_lang", "fruehabbruch", "plateau"], p=[0.20, 0.40, 0.25, 0.15])
            
        plateau_pausen = 0
        bisherige_fach_supports = set()  # V3.2: Carry-over für fachliche Supports aus früheren Semestern
        
        while chron_sem_idx < len(semester_order) and fachsem <= CONFIG.get("max_simulations_semester", 16) and not studi.abschluss_erreicht and not studi.abgebrochen and not studi.exmatrikuliert:
            akt_sem_id = semester_order[chron_sem_idx]
            akt_sem_typ = semester_types[chron_sem_idx]
            
            if studi.anomalie_typ == "plateau" and fachsem in (3, 4) and plateau_pausen < 2:
                plateau_pausen += 1
                studi.motivation = max(0.05, studi.motivation - 0.08)
                chron_sem_idx += 1
                continue
            
            studi.einschreibungen.append({"semester_id": akt_sem_id, "fachsemester": fachsem, "status": "aktiv"})
            
            verfuegbare_zeit = max(100, CONFIG["zeitkonto_budget_h"] - (studi.erwerbstaetigkeit_std * 20))
            
            geplante_module = []
            for row in sg_module_rows:
                m_id = row["modul_id"]
                m_state = studi.modul_states[m_id]
                m_info = modul_data[m_id]
                
                if m_info["turnus"] not in ("beides", akt_sem_typ):
                    continue
                    
                if m_state.status == "offen" and (row["empfohlenes_fachsemester"] <= fachsem or m_state.versuche > 0):
                    if "bachelorarbeit" in m_info["name"].lower():
                        cp_bestanden = studi.cp_bestanden(modul_cp_dict)
                        if cp_bestanden < sg_info["cp_gesamt"] - 18:
                            continue
                    geplante_module.append(m_id)
            
            if studi.anomalie_typ == "super_schnell":
                for row in sg_module_rows:
                    if row["empfohlenes_fachsemester"] == fachsem + 1:
                        v_m = row["modul_id"]
                        if studi.modul_states[v_m].status == "offen" and modul_data[v_m]["turnus"] in ("beides", akt_sem_typ):
                            geplante_module.append(v_m)
                            break
            elif studi.anomalie_typ == "sehr_lang" and rng_init.random() < 0.4:
                geplante_module = geplante_module[:max(1, len(geplante_module)-1)]
            
            teilgenommene_angebote = []
            support_zeit_kosten = 0
            
            for angebot in support_list:
                ang_id = angebot["angebot_id"]
                p = 0.0
                
                if angebot["typ"] == "fachlich":
                    rel_modules = angebot_to_modules.get(ang_id, set())
                    geplante_relevante = [m for m in geplante_module if m in rel_modules]
                    
                    if geplante_relevante:
                        p = 0.05 + (studi.erwartete_note - 2.0) * 0.05
                        for m in geplante_relevante:
                            if studi.modul_states[m].versuche > 0:
                                p += 0.20
                elif angebot["typ"] == "ueberfachlich":
                    p = 0.05 + (0.5 - studi.motivation) * 0.15
                else: # psychosozial
                    p = 0.01 + (0.5 - studi.soziale_integration) * 0.12
                
                if studi.erstakademiker and angebot["typ"] in ("fachlich", "psychosozial"): p += 0.05
                p = float(np.clip(p, 0.0, 0.9))
                
                nutzt_support = rng_support.random() < p
                typ = angebot["typ"]
                blocked = (typ == "fachlich" and block_fach) or (typ == "ueberfachlich" and block_uebf) or (typ == "psychosozial" and block_psych)
                if nutzt_support and not blocked:
                    # Check Zeitkonto mit stochastischem Puffer
                    puffer = getattr(studi, 'hidden_zeit_puffer', 60.0)
                    if verfuegbare_zeit - support_zeit_kosten - angebot.get("kosten_h", 30) >= 0 or rng_support.random() < 0.2: 
                        teilgenommene_angebote.append(ang_id)
                        support_zeit_kosten += angebot.get("kosten_h", 30)
                        studi.support_teilnahmen.append({"semester_id": akt_sem_id, "angebot_id": ang_id})
                elif nutzt_support and blocked:
                    if verfuegbare_zeit - support_zeit_kosten - angebot.get("kosten_h", 30) < 0:
                        _ = rng_support.random()
            
            mult = CONFIG.get("support_effect_multiplier", 1.0)
            for ang_id in teilgenommene_angebote:
                ang = support_by_id[ang_id]
                if ang["typ"] == "ueberfachlich":
                    studi.motivation = min(1.0, studi.motivation + 0.02 * mult)
                    studi.soziale_integration = min(1.0, studi.soziale_integration + 0.01 * mult)
                elif ang["typ"] == "psychosozial":
                    studi.motivation = min(1.0, studi.motivation + 0.015 * mult)
                    studi.soziale_integration = min(1.0, studi.soziale_integration + 0.035 * mult)

            # SIMULATION V3.1: Modul-Abwurf basiert NUR auf geplantem Workload (Support-Zeit schlägt NICHT auf Abwurf durch)
            geplanter_workload = sum(modul_data[m]["workload_h"] for m in geplante_module)
            puffer = getattr(studi, 'hidden_zeit_puffer', 60.0)
            
            while geplanter_workload > verfuegbare_zeit + puffer and len(geplante_module) > 1:
                geplante_module.sort(key=lambda m: modul_data[m]["schwierigkeit"])
                dropped = geplante_module.pop()
                geplanter_workload -= modul_data[dropped]["workload_h"]
            
            # SIMULATION V3.1: Overload-Berechnung & Deckelung der overload_penalty (max 0.15)
            # Support-Zeitaufwand fließt hier weiterhin voll in den total_workload & overload ein
            total_workload = geplanter_workload + support_zeit_kosten
            overload = max(0.0, float(total_workload - verfuegbare_zeit))
            overload_penalty = float(min(0.15, (overload / 100.0) * 0.1))
            
            durchgefallen_dieses_sem = 0
            for m_id in geplante_module:
                m_state = studi.modul_states[m_id]
                m_state.versuche += 1
                
                # V3.2: Aktueller Boost (volle Wirkung) + Carry-over aus früheren Semestern (2/3 Wirkung)
                current_boost_sum = sum(support_zuord_dict.get((m_id, ang_id), 0.0) for ang_id in teilgenommene_angebote)
                carryover_ids = bisherige_fach_supports - set(teilgenommene_angebote)  # Keine Doppelzählung
                carryover_boost_sum = sum(support_zuord_dict.get((m_id, ang_id), 0.0) for ang_id in carryover_ids)
                boost_sum = current_boost_sum + carryover_boost_sum * (2.0 / 3.0)
                raw_boost = boost_sum * CONFIG["gewicht_support_boost"] * CONFIG.get("support_effect_multiplier", 1.0) if boost_sum > 0.0 else 0.0
                boost = float(np.clip(raw_boost, 0.0, CONFIG["support_deckel"]))
                support_capped = (raw_boost > CONFIG["support_deckel"])
                
                # V3.3: Deterministisches, positionsunabhängiges Prüfungsrauschen
                e_noise = get_exam_noise(base_seed, m_id, m_state.versuche)
                
                note, bestanden, note_cf = simuliere_pruefung(
                    schwierigkeit=modul_data[m_id]["schwierigkeit"],
                    erwartete_note=studi.erwartete_note,
                    motivation=studi.motivation,
                    soz_int=studi.soziale_integration,
                    fachlicher_boost=boost,
                    versuch=m_state.versuche,
                    overload_penalty=overload_penalty,
                    exam_noise=e_noise
                )
                
                # SIMULATION V3.2: Loggen von hidden_overload, hidden_zeit_puffer, hidden_penalty_capped & hidden_support_capped
                studi.pruefungen.append(PruefungsErgebnis(
                    semester_id=akt_sem_id, modul_id=m_id, versuch=m_state.versuche, 
                    note=note, bestanden=bestanden, note_counterfactual=note_cf, support_genutzt=(boost > 0),
                    hidden_motivation=studi.motivation,
                    hidden_soziale_integration=studi.soziale_integration,
                    hidden_erwartete_note=studi.erwartete_note,
                    hidden_overload=overload,
                    hidden_zeit_puffer=puffer,
                    hidden_penalty_capped=(overload_penalty >= 0.15),
                    hidden_support_capped=support_capped
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
            
            sem_pruefungen = [p for p in studi.pruefungen if p.semester_id == akt_sem_id and p.bestanden]
            for p_erg in sem_pruefungen:
                grade_diff = studi.erwartete_note - p_erg.note
                if grade_diff >= 0.5:
                    super_boost = 0.005 + 0.01 * (grade_diff - 0.5)
                    studi.motivation = min(1.0, studi.motivation + super_boost)

            if sem_pruefungen:
                sem_gpa = sum(p.note for p in sem_pruefungen) / len(sem_pruefungen)
                if sem_gpa < studi.erwartete_note:
                    studi.erwartete_note = round(0.7 * studi.erwartete_note + 0.3 * sem_gpa, 2)

            if durchgefallen_dieses_sem > 0:
                studi.motivation = max(0.05, studi.motivation - 0.05 * durchgefallen_dieses_sem)
            elif len(geplante_module) > 0:
                studi.motivation = min(1.0, studi.motivation + 0.02)
                
            studi.soziale_integration = float(np.clip(studi.soziale_integration + rng_social.normal(0, 0.05), 0.05, 1.0))
            
            cp_bestanden = studi.cp_bestanden({m: modul_data[m]["cp"] for m in modul_data})
            if studi.alle_pflicht_bestanden([r["modul_id"] for r in sg_module_rows if r["pflicht"]]):
                ba_module = [r["modul_id"] for r in sg_module_rows if "bachelorarbeit" in modul_data[r["modul_id"]]["name"].lower()]
                if not ba_module or studi.modul_states[ba_module[0]].status == "bestanden":
                    studi.abschluss_erreicht = True
            
            if not studi.abschluss_erreicht and not studi.exmatrikuliert:
                cp_soll = (fachsem / sg_info["regelstudienzeit"]) * sg_info["cp_gesamt"]
                cp_rueckstand = max(0.0, cp_soll - cp_bestanden)
                p_drop = berechne_dropout(studi.motivation, studi.soziale_integration, cp_rueckstand, durchgefallen_dieses_sem, fachsem, overload_penalty)
                if studi.anomalie_typ == "sehr_lang": p_drop *= 0.3
                if rng_dropout.random() < p_drop:
                    studi.abgebrochen = True

            # V3.2: Fachliche Supports dieses Semesters für Carry-over merken
            for ang_id in teilgenommene_angebote:
                if support_by_id.get(ang_id, {}).get("typ") == "fachlich":
                    bisherige_fach_supports.add(ang_id)
            
            fachsem += 1
            chron_sem_idx += 1

    return studierende


def main(population_seed: int = 12345, base_output_override: Path = None):
    import os, json, sys
    from pathlib import Path
    from export import as_dataframe, exportiere_csv
    from config import CONFIG
    print("Starte True Counterfactual Trajectory Simulator (Simulator v3) ...")
    print("  8 Parallele Universen mit per-Typ Support-Blockierung, Stochastischem Puffer & Gedeckeltem Overload")
    
    if base_output_override:
        base_output = base_output_override
    else:
        base_output = Path(CONFIG["output_dir"])
    os.makedirs(base_output / "metrics", exist_ok=True)
    
    stammdaten = generiere_stammdaten()
    
    UNIVERSES = {
        "A": {"label": "Alle Support-Typen erlaubt",       "block_fach": False, "block_uebf": False, "block_psych": False},
        "B": {"label": "Kein Support (komplett blockiert)",  "block_fach": True,  "block_uebf": True,  "block_psych": True},
        "C": {"label": "Kein fachlicher Support",           "block_fach": True,  "block_uebf": False, "block_psych": False},
        "D": {"label": "Kein ueberfachlicher Support",      "block_fach": False, "block_uebf": True,  "block_psych": False},
        "E": {"label": "Kein psychosozialer Support",       "block_fach": False, "block_uebf": False, "block_psych": True},
        "F": {"label": "Nur fachlicher Support",            "block_fach": False, "block_uebf": True,  "block_psych": True},
        "G": {"label": "Nur ueberfachlicher Support",       "block_fach": True,  "block_uebf": False, "block_psych": True},
        "H": {"label": "Nur psychosozialer Support",        "block_fach": True,  "block_uebf": True,  "block_psych": False},
    }
    
    POPULATION_SEED = population_seed
    results = {}
    
    for uni_key, uni_cfg in UNIVERSES.items():
        print(f"\\n  UNIVERSUM {uni_key}: {uni_cfg['label']}")
        rng = np.random.default_rng(POPULATION_SEED)
        studierende = generiere_studierende_v3(stammdaten, rng)
        
        simuliere_verlaeufe_v3(
            studierende, stammdaten,
            block_fach=uni_cfg["block_fach"],
            block_uebf=uni_cfg["block_uebf"],
            block_psych=uni_cfg["block_psych"],
            population_seed=POPULATION_SEED
        )
        
        dfs = stammdaten.copy()
        dfs.update(as_dataframe(studierende, stammdaten))
        
        if uni_key == "A":
            uni_dir = base_output
        else:
            uni_dir = base_output / f"universe_{uni_key}"
            uni_dir.mkdir(exist_ok=True, parents=True)
            
        exportiere_csv(dfs, uni_dir)
        try:
            from aggregate import aggregiere_daten
            aggregiere_daten(uni_dir)
        except ImportError:
            pass
        
        dropout_cnt = sum(1 for s in studierende if s.abgebrochen or s.exmatrikuliert or (not s.abschluss_erreicht and len(s.einschreibungen) >= 16))
        drop_rate = dropout_cnt / len(studierende)
        
        results[f"universe_{uni_key}"] = {
            "label": uni_cfg["label"],
            "dropout_rate": round(drop_rate, 5)
        }
        print(f"  Dropout-Rate Universum {uni_key}: {drop_rate*100:.2f}% ({dropout_cnt}/{len(studierende)})")

    base_rate_A = results["universe_A"]["dropout_rate"]
    base_rate_B = results.get("universe_B", {}).get("dropout_rate", 0)

    for u in ["B", "C", "D", "E"]:
        if f"universe_{u}" in results:
            u_rate = results[f"universe_{u}"]["dropout_rate"]
            diff = u_rate - base_rate_A
            rr = u_rate / base_rate_A if base_rate_A > 0 else 1.0
            results[f"universe_{u}"]["vs_A_absolute_diff"] = round(diff, 5)
            results[f"universe_{u}"]["vs_A_relative_risk"] = round(rr, 5)
            results[f"universe_{u}"]["vs_A_relative_reduction_pct"] = round((1.0 - rr) * 100, 5)

    for u in ["F", "G", "H"]:
        if f"universe_{u}" in results:
            u_rate = results[f"universe_{u}"]["dropout_rate"]
            rr_vs_B = u_rate / base_rate_B if base_rate_B > 0 else 1.0
            results[f"universe_{u}"]["vs_B_absolute_diff"] = round(u_rate - base_rate_B, 5)
            results[f"universe_{u}"]["vs_B_relative_risk"] = round(rr_vs_B, 5)
            results[f"universe_{u}"]["vs_B_relative_reduction_pct"] = round((1.0 - rr_vs_B) * 100, 5)

    with open(base_output / "metrics" / "true_macro_effects_v3.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\\nWahre Makro-Effekte Simulation V3 (Alle 8 Universen) erfolgreich gespeichert!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=12345, help="Population seed")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    import os, json, sys
    from pathlib import Path
    from export import as_dataframe, exportiere_csv
    from config import CONFIG
    
    out_dir = Path(args.output_dir) if args.output_dir else None
    main(population_seed=args.seed, base_output_override=out_dir)
