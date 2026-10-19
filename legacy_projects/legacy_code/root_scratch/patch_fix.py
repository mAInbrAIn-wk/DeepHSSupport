import re

with open('src/simulation_v4.py', 'r', encoding='utf-8') as f:
    content = f.read()

bad_snippet = '''        studierende.append(studi)
    print(f"\\n[V4 TRACKER] Zeitbudget-Analyse abgeschlossen:")
    print(f"[V4 TRACKER] In {tracker_overload_hits} Semestern wurde die Workload-Schranke gesprengt.")
    print(f"[V4 TRACKER] Insgesamt wurden {tracker_modules_dropped} Modulpruefungen wegen Ueberlast/Zeitbudget zurueck in die Warteschlange geworfen.")
    
    return studierende'''

good_snippet = '''        studierende.append(studi)
    return studierende'''

content = content.replace(bad_snippet, good_snippet)

with open('src/simulation_v4.py', 'w', encoding='utf-8') as f:
    f.write(content)
