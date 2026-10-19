import re

with open('src/simulation.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove Zeitbudget-Check
old_loop = '''                if rng.random() < p:
                    # Check Zeitkonto
                    if verfuegbare_zeit - support_zeit_kosten - angebot.get("kosten_h", 30) >= 0 or rng.random() < 0.2: 
                        # Nimmt auch teil wenn es in Overload führt mit 20% Chance
                        teilgenommene_angebote.append(ang_id)
                        support_zeit_kosten += angebot.get("kosten_h", 30)
                        studi.support_teilnahmen.append({"semester_id": akt_sem_id, "angebot_id": ang_id})'''

new_loop = '''                if rng.random() < p:
                    teilgenommene_angebote.append(ang_id)
                    support_zeit_kosten += angebot.get("kosten_h", 30)
                    studi.support_teilnahmen.append({"semester_id": akt_sem_id, "angebot_id": ang_id})'''

content = content.replace(old_loop, new_loop)

# 2. Add Support Friction
old_uebf = '''                elif angebot["typ"] == "ueberfachlich":
                    p = 0.05 + (0.5 - studi.motivation) * 0.15
                else: # psychosozial
                    p = 0.01 + (0.5 - studi.soziale_integration) * 0.12'''
                    
new_uebf = '''                elif angebot["typ"] == "ueberfachlich":
                    p = 0.05 + (0.5 - studi.motivation) * 0.15
                    if studi.motivation < 0.2: p *= (studi.motivation / 0.2)
                else: # psychosozial
                    p = 0.01 + (0.5 - studi.soziale_integration) * 0.12
                    if studi.soziale_integration < 0.2: p *= (studi.soziale_integration / 0.2)'''

content = content.replace(old_uebf, new_uebf)

# 3. Beta Distribution for Motivation & Social Integration
# Original:
# motivation = float(np.clip(CONFIG["motivation_startwert"] + (2.5 - hzb_note) * CONFIG["gewicht_motivation_hzb"] - erwerb * CONFIG["gewicht_motivation_erwerb"] + rng.normal(0, CONFIG["gewicht_motivation_rauschen"]), 0.05, 1.0))
# soz_int = float(np.clip(CONFIG["integration_startwert"] - (CONFIG["gewicht_integration_erstakademiker"] if erstakademiker else 0) - (CONFIG["gewicht_integration_migration"] if migration else 0) - erwerb * CONFIG["gewicht_integration_erwerb"] + rng.normal(0, CONFIG["gewicht_integration_rauschen"]), 0.05, 1.0))

old_mot = '''        motivation = float(np.clip(CONFIG["motivation_startwert"] + (2.5 - hzb_note) * CONFIG["gewicht_motivation_hzb"] - erwerb * CONFIG["gewicht_motivation_erwerb"] + rng.normal(0, CONFIG["gewicht_motivation_rauschen"]), 0.05, 1.0))'''
new_mot = '''        mean_mot = np.clip(CONFIG["motivation_startwert"] + (2.5 - hzb_note) * CONFIG["gewicht_motivation_hzb"] - erwerb * CONFIG["gewicht_motivation_erwerb"], 0.01, 0.99)
        motivation = float(rng.beta(mean_mot * 20.0, (1.0 - mean_mot) * 20.0))'''
content = content.replace(old_mot, new_mot)

old_soz = '''        soz_int = float(np.clip(CONFIG["integration_startwert"] - (CONFIG["gewicht_integration_erstakademiker"] if erstakademiker else 0) - (CONFIG["gewicht_integration_migration"] if migration else 0) - erwerb * CONFIG["gewicht_integration_erwerb"] + rng.normal(0, CONFIG["gewicht_integration_rauschen"]), 0.05, 1.0))'''
new_soz = '''        mean_soz = np.clip(CONFIG["integration_startwert"] - (CONFIG["gewicht_integration_erstakademiker"] if erstakademiker else 0) - (CONFIG["gewicht_integration_migration"] if migration else 0) - erwerb * CONFIG["gewicht_integration_erwerb"], 0.01, 0.99)
        soz_int = float(rng.beta(mean_soz * 20.0, (1.0 - mean_soz) * 20.0))'''
content = content.replace(old_soz, new_soz)

# Random Walk Soziale Integration:
old_walk = '''            studi.soziale_integration = float(np.clip(studi.soziale_integration + rng.normal(0, 0.05), 0.05, 1.0))'''
new_walk = '''            mean_walk = np.clip(studi.soziale_integration, 0.01, 0.99)
            studi.soziale_integration = float(rng.beta(mean_walk * 40.0, (1.0 - mean_walk) * 40.0))'''
content = content.replace(old_walk, new_walk)

with open('src/simulation.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Simulation.py updated for V4!")
