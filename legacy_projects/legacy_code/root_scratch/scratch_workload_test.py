import os
import sys
import numpy as np
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path('src').absolute()))
import simulation as sim
from config import CONFIG

def test_workload_drop():
    rng = np.random.default_rng(42)
    stammdaten = sim._erzeuge_semester_liste(2018, 2024)
    # Generate 5000 students
    studis = sim.generiere_studierende(stammdaten, rng)[:5000]
    
    # We will hijack the simulation loop to count drops
    dropped_due_to_support = 0
    total_semesters_simulated = 0
    support_booked_count = 0
    support_hours_total = 0
    
    modul_data = {row["modul_id"]: row for _, row in stammdaten["module_df"].iterrows()}
    support_df = stammdaten["support_angebote_df"]
    support_zuord_df = stammdaten["support_modul_zuordnung_df"]
    
    for studi in studis:
        sg_module = stammdaten["modul_studiengang_df"][stammdaten["modul_studiengang_df"]["studiengang_id"] == studi.studiengang_id]
        fachsem = 1
        akt_sem_id = 1
        
        while not studi.abschluss_erreicht and not studi.exmatrikuliert and not studi.abgebrochen and fachsem <= 12:
            total_semesters_simulated += 1
            verfuegbare_zeit = 900 - (studi.erwerbstaetigkeit_std * 24)
            geplante_module = [m for m, state in studi.modul_states.items() if state.status == "geplant" or state.status == "nicht_bestanden"]
            
            teilgenommene_angebote = []
            support_zeit_kosten = 0
            
            for _, angebot in support_df.iterrows():
                ang_id = angebot["angebot_id"]
                p = 0.0
                if angebot["typ"] == "fachlich":
                    rel_zuordnungen = support_zuord_df[support_zuord_df["angebot_id"] == ang_id]
                    rel_module = rel_zuordnungen["modul_id"].tolist()
                    geplante_relevante = [m for m in geplante_module if m in rel_module]
                    if geplante_relevante:
                        p = 0.15 + (studi.erwartete_note - 2.5) * 0.1
                        p += sum(0.15 for m in geplante_relevante if studi.modul_states[m].status == "nicht_bestanden")
                elif angebot["typ"] == "ueberfachlich":
                    p = 0.5 - studi.motivation
                elif angebot["typ"] == "psychosozial":
                    p = 0.5 - studi.soziale_integration
                    
                p = max(0.0, min(0.9, p))
                if rng.random() < p:
                    if verfuegbare_zeit - support_zeit_kosten - angebot.get("kosten_h", 30) >= 0 or rng.random() < 0.2:
                        teilgenommene_angebote.append(ang_id)
                        support_zeit_kosten += angebot.get("kosten_h", 30)
                        support_booked_count += 1
            
            support_hours_total += support_zeit_kosten
            
            geplanter_workload = sum(modul_data[m]["workload_h"] for m in geplante_module)
            dropped_this_sem = 0
            while geplanter_workload + support_zeit_kosten > verfuegbare_zeit + 150 and len(geplante_module) > 1:
                geplante_module.sort(key=lambda m: modul_data[m]["schwierigkeit"])
                dropped = geplante_module.pop()
                geplanter_workload -= modul_data[dropped]["workload_h"]
                dropped_this_sem += 1
                dropped_due_to_support += 1
                
            fachsem += 1
            studi.abschluss_erreicht = True # short circuit
            
    print(f"Total Semesters Simulated: {total_semesters_simulated}")
    print(f"Total Support Bookings: {support_booked_count}")
    print(f"Average Support Hours per Sem: {support_hours_total / total_semesters_simulated:.2f}")
    print(f"Modules Dropped due to Overload (incl. Support): {dropped_due_to_support}")
    print(f"Drop Rate (Drops per Semester): {dropped_due_to_support / total_semesters_simulated:.4f}")

if __name__ == '__main__':
    test_workload_drop()
