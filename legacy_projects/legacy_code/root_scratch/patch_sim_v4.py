import re

with open('src/simulation.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Alter -> Beta
old_alter = '''        alter = int(np.clip(rng.normal(20.5, 2.8), 17, 45))'''
new_alter = '''        # Alter normalisiert: mean ~ (20.5-17)/(45-17) = 0.125
        alter = int(17 + rng.beta(0.125 * 20.0, (1.0 - 0.125) * 20.0) * (45 - 17))'''
content = content.replace(old_alter, new_alter)

# HZB Note -> Beta
old_hzb = '''        hzb_note = round(float(np.clip(rng.normal(2.4, 0.55), 1.0, 4.0)), 1)'''
new_hzb = '''        # HZB normalisiert: mean ~ (2.4-1.0)/(4.0-1.0) = 0.466
        hzb_note = round(float(1.0 + rng.beta(0.466 * 20.0, (1.0 - 0.466) * 20.0) * 3.0), 1)'''
content = content.replace(old_hzb, new_hzb)

with open('src/simulation_v4.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("simulation_v4.py created!")
