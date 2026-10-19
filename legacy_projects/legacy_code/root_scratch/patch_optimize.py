import re

with open('src/simulation_v4.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Define the new simuliere_verlaeufe function up to the student loop
new_precompute = '''def simuliere_verlaeufe(studierende: List[Student], stammdaten: Dict[str, pd.DataFrame], rng: np.random.Generator):
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

    # --- V4 Tracker ---
    tracker_modules_dropped = 0
    tracker_overload_hits = 0
    
    for idx, studi in enumerate(studierende):'''

# Find the old signature and replace until the student loop
content = re.sub(r'def simuliere_verlaeufe.*?for idx, studi in enumerate\(studierende\):', new_precompute, content, flags=re.DOTALL)

# Replace the inner loop lookups
content = content.replace('sg_module = modul_sg_df[modul_sg_df["studiengang_id"] == studi.studiengang_id]', 'sg_module_list = sg_module_cache[studi.studiengang_id]')

# Replace or _, row in sg_module.iterrows(): with or row in sg_module_list:
content = content.replace('for _, row in sg_module.iterrows():', 'for row in sg_module_list:')

# Replace boolean indexing for 'super_schnell'
old_voraus = '''                if studi.anomalie_typ == "super_schnell":
                    # Versucht noch ein Modul aus dem nchsten Semester vorzuziehen
                    voraus = sg_module[(sg_module["empfohlenes_fachsemester"] == fachsem + 1) & (sg_module["modul_id"].apply(lambda m: studi.modul_states[m].status == "offen"))]["modul_id"].tolist()'''
new_voraus = '''                if studi.anomalie_typ == "super_schnell":
                    voraus = [r["modul_id"] for r in sg_module_list if r["empfohlenes_fachsemester"] == fachsem + 1 and studi.modul_states[r["modul_id"]].status == "offen"]'''
content = content.replace(old_voraus, new_voraus)

# Replace or _, angebot in support_df.iterrows():
content = content.replace('for _, angebot in support_df.iterrows():', 'for angebot in support_list:')

# Replace 
el_zuordnungen = ...
old_rel = '''                    if angebot["typ"] == "fachlich":
                        # Hat der Student eines der relevanten Module in Planung?
                        rel_zuordnungen = support_zuord_df[support_zuord_df["angebot_id"] == ang_id]
                        rel_module = rel_zuordnungen["modul_id"].tolist()'''
new_rel = '''                    if angebot["typ"] == "fachlich":
                        rel_module = ang_to_mod.get(ang_id, [])'''
content = content.replace(old_rel, new_rel)

# Replace or _, ang in support_df[...].iterrows():
old_supp_effect = '''            for _, ang in support_df[support_df["angebot_id"].isin(teilgenommene_angebote)].iterrows():'''
new_supp_effect = '''            for ang in [a for a in support_list if a["angebot_id"] in teilgenommene_angebote]:'''
content = content.replace(old_supp_effect, new_supp_effect)

# Replace 
# el = support_zuord_df...
old_boost = '''                # Fachlicher Support Boost
                rel = support_zuord_df[(support_zuord_df["modul_id"] == m_id) & (support_zuord_df["angebot_id"].isin(teilgenommene_angebote))]
                boost = float(np.clip(rel["wirkungsstaerke"].sum() * CONFIG["gewicht_support_boost"] * CONFIG.get("support_effect_multiplier", 1.0), 0.0, CONFIG["support_deckel"])) if not rel.empty else 0.0'''
new_boost = '''                # Fachlicher Support Boost
                boost_sum = sum(mod_to_ang_boost.get(m_id, {}).get(a_id, 0.0) for a_id in teilgenommene_angebote)
                boost = float(np.clip(boost_sum * CONFIG["gewicht_support_boost"] * CONFIG.get("support_effect_multiplier", 1.0), 0.0, CONFIG["support_deckel"]))'''
content = content.replace(old_boost, new_boost)

# Replace abschluss check
old_abschluss = '''            if studi.alle_pflicht_bestanden([r["modul_id"] for _, r in sg_module.iterrows() if r["pflicht"]]):
                ba_module = [r["modul_id"] for _, r in sg_module.iterrows() if "bachelorarbeit" in modul_data[r["modul_id"]]["name"].lower()]'''
new_abschluss = '''            if studi.alle_pflicht_bestanden(sg_pflicht_cache[studi.studiengang_id]):
                ba_module = sg_ba_cache[studi.studiengang_id]'''
content = content.replace(old_abschluss, new_abschluss)

with open('src/simulation_v4.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("simulation_v4.py optimized successfully!")
