import re

with open('src/simulation_v4.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add Tracker Initialization
old_init = '''    modul_data = module_df.set_index("modul_id").to_dict("index")
    sg_infos = studiengaenge_df.set_index("studiengang_id").to_dict("index")
    
    for idx, studi in enumerate(studierende):'''
new_init = '''    modul_data = module_df.set_index("modul_id").to_dict("index")
    sg_infos = studiengaenge_df.set_index("studiengang_id").to_dict("index")
    
    # --- V4 Tracker ---
    tracker_modules_dropped = 0
    tracker_overload_hits = 0
    
    for idx, studi in enumerate(studierende):
        studi.stat_modules_dropped = 0 # Dynamisches Attribut fuer spuetere Auswertung'''
content = content.replace(old_init, new_init)

# Add Tracker Logic in while loop
old_while = '''            while geplanter_workload + support_zeit_kosten > verfuegbare_zeit + 150 and len(geplante_module) > 1:
                # Wirft das schwerste Modul ab
                geplante_module.sort(key=lambda m: modul_data[m]["schwierigkeit"])
                dropped = geplante_module.pop()
                geplanter_workload -= modul_data[dropped]["workload_h"]'''
new_while = '''            if geplanter_workload + support_zeit_kosten > verfuegbare_zeit + 150 and len(geplante_module) > 1:
                tracker_overload_hits += 1
                
            while geplanter_workload + support_zeit_kosten > verfuegbare_zeit + 150 and len(geplante_module) > 1:
                # Wirft das schwerste Modul ab
                geplante_module.sort(key=lambda m: modul_data[m]["schwierigkeit"])
                dropped = geplante_module.pop()
                geplanter_workload -= modul_data[dropped]["workload_h"]
                tracker_modules_dropped += 1
                studi.stat_modules_dropped += 1'''
content = content.replace(old_while, new_while)

# Add Tracker Print at the end
old_return = '''    return studierende'''
new_return = '''    print(f"\\n[V4 TRACKER] Zeitbudget-Analyse abgeschlossen:")
    print(f"[V4 TRACKER] In {tracker_overload_hits} Semestern wurde die Workload-Schranke gesprengt.")
    print(f"[V4 TRACKER] Insgesamt wurden {tracker_modules_dropped} Modulpruefungen wegen Ueberlast/Zeitbudget zurueck in die Warteschlange geworfen.")
    
    return studierende'''
content = content.replace(old_return, new_return)

with open('src/simulation_v4.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Simulation V4 patched with time budget trackers!")
