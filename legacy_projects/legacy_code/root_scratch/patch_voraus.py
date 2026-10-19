import re

with open('src/simulation_v4.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the specific pandas logic with list comprehension
# Old: voraus = sg_module[(sg_module["empfohlenes_fachsemester"] == fachsem + 1) & (sg_module["modul_id"].apply(lambda m: studi.modul_states[m].status == "offen"))]["modul_id"].tolist()
# New: voraus = [r["modul_id"] for r in sg_module_list if r["empfohlenes_fachsemester"] == fachsem + 1 and studi.modul_states[r["modul_id"]].status == "offen"]

content = re.sub(
    r'voraus = sg_module\[.*?\]\["modul_id"\]\.tolist\(\)',
    'voraus = [r["modul_id"] for r in sg_module_list if r["empfohlenes_fachsemester"] == fachsem + 1 and studi.modul_states[r["modul_id"]].status == "offen"]',
    content,
    flags=re.DOTALL
)

with open('src/simulation_v4.py', 'w', encoding='utf-8') as f:
    f.write(content)
