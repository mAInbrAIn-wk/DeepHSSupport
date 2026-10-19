import sys

with open('src/simulation_v3.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.startswith('from models import Student'):
        new_lines.append('class BudgetTracker:\n')
        new_lines.append('    def __init__(self):\n')
        new_lines.append('        self.override_students = set()\n')
        new_lines.append('tracker = BudgetTracker()\n')
        new_lines.append(line)
    elif 'if verfuegbare_zeit - support_zeit_kosten - angebot.get("kosten_h", 30) >= 0 or rng_support.random() < 0.2:' in line:
        new_lines.append('                    has_time = verfuegbare_zeit - support_zeit_kosten - angebot.get("kosten_h", 30) >= 0\n')
        new_lines.append('                    override = False\n')
        new_lines.append('                    if not has_time and rng_support.random() < 0.2:\n')
        new_lines.append('                        override = True\n')
        new_lines.append('                        tracker.override_students.add(id(studi))\n')
        new_lines.append('                    if has_time or override:\n')
    elif 'drop_rate = dropout_cnt / len(studierende)' in line:
        new_lines.append('        drop_rate = dropout_cnt / len(studierende)\n')
        new_lines.append('        override_dropouts = sum(1 for s in studierende if id(s) in tracker.override_students and (s.abgebrochen or s.exmatrikuliert or (not s.abschluss_erreicht and len(s.einschreibungen) >= 16)))\n')
        new_lines.append('        override_total = sum(1 for s in studierende if id(s) in tracker.override_students)\n')
        new_lines.append('        normal_dropouts = dropout_cnt - override_dropouts\n')
        new_lines.append('        normal_total = len(studierende) - override_total\n')
        new_lines.append('        o_rate = override_dropouts / override_total if override_total > 0 else 0\n')
        new_lines.append('        n_rate = normal_dropouts / normal_total if normal_total > 0 else 0\n')
    elif 'results[f"universe_{uni_key}"] = {' in line:
        new_lines.append(line)
    elif '    "dropout_rate": round(drop_rate, 5)' in line:
        new_lines.append(line)
        new_lines.append('            ,"override_dropout_rate": round(o_rate, 5)\n')
        new_lines.append('            ,"normal_dropout_rate": round(n_rate, 5)\n')
    elif 'print(f"  Dropout-Rate Universum {uni_key}: {drop_rate*100:.2f}% ({dropout_cnt}/{len(studierende)})")' in line:
        new_lines.append(line)
        new_lines.append('        if override_total > 0:\n')
        new_lines.append('            print(f"    -> Studierende MIT 20%-Override: Dropout = {o_rate*100:.2f}% ({override_dropouts}/{override_total})")\n')
        new_lines.append('            print(f"    -> Studierende OHNE Override:    Dropout = {n_rate*100:.2f}% ({normal_dropouts}/{normal_total})")\n')
        new_lines.append('        tracker.override_students.clear()\n')
    else:
        new_lines.append(line)

with open('src/simulation_v3.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
