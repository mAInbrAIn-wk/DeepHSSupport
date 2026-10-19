import ast
import json

def extract_configs():
    with open('../Projekt_DA/GeneriereHSDS.py', 'r', encoding='utf-8') as f:
        source = f.read()

    tree = ast.parse(source)
    
    config_dict = {}
    module_curricula = {}
    studiengaenge = []
    support_angebote = []
    support_keywords = []
    
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            if name == 'CONFIG':
                config_dict = ast.literal_eval(node.value)
            elif name == 'MODULE_CURRICULA':
                module_curricula = ast.literal_eval(node.value)
            elif name == 'STUDIENGAENGE':
                studiengaenge = ast.literal_eval(node.value)
            elif name == 'SUPPORT_ANGEBOTE':
                support_angebote = ast.literal_eval(node.value)
            elif name == 'SUPPORT_KEYWORDS':
                support_keywords = ast.literal_eval(node.value)
    
    # Add turnus and work requirements
    for sg, modules in module_curricula.items():
        for m in modules:
            # Add Turnus (WS for odd fachsem, SS for even fachsem)
            if 'bachelorarbeit' in m['name'].lower() or 'kolloquium' in m['name'].lower() or 'projekt' in m['name'].lower():
                m['turnus'] = 'beides'
            else:
                m['turnus'] = 'WS' if m['fachsem'] % 2 != 0 else 'SS'
            # 1 CP = 30 hours workload
            m['workload_h'] = m['cp'] * 30
            
    for s in support_angebote:
        # Give support offers a time cost (e.g., 30 hours = 1 CP equivalent)
        s['kosten_h'] = 30
    
    # Adjust config for DL and Zeitkonto
    config_dict['zeitkonto_budget_h'] = 900 # 30 CP * 30h per semester for full-time
    config_dict['output_dir'] = '../output_dl'
    
    # Write to config.py
    import pprint
    out_lines = [
        "\"\"\"",
        "Konfiguration und Stammdaten für das Deep Learning Absolventenprojekt.",
        "Generiert aus dem Originalskript mit Erweiterungen (Turnus, Zeitkonto).",
        "\"\"\"",
        "",
        "CONFIG = " + pprint.pformat(config_dict, indent=4),
        "",
        "MODULE_CURRICULA = " + pprint.pformat(module_curricula, indent=4),
        "",
        "STUDIENGAENGE = " + pprint.pformat(studiengaenge, indent=4),
        "",
        "SUPPORT_ANGEBOTE = " + pprint.pformat(support_angebote, indent=4),
        "",
        "SUPPORT_KEYWORDS = " + pprint.pformat(support_keywords, indent=4),
        "",
        "HZB_TYPEN = ['Allg. Hochschulreife', 'Fachhochschulreife', 'Fachgebundene HR', 'Berufl. Qualifikation']",
        "HZB_GEWICHTE = [0.70, 0.20, 0.05, 0.05]",
        ""
    ]
    
    with open('config.py', 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines))
        
if __name__ == '__main__':
    extract_configs()
