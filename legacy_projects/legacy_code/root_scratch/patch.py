import re
with open('src/simulation_v3.py', 'r') as f:
    code = f.read()

# 1. Update simuliere_verlaeufe_v3 signature
code = code.replace(
    'def simuliere_verlaeufe_v3(\\n    studierende: List[Student],\\n    stammdaten: Dict[str, pd.DataFrame],\\n    block_fach: bool = False,\\n    block_uebf: bool = False,\\n    block_psych: bool = False\\n):',
    'def simuliere_verlaeufe_v3(\\n    studierende: List[Student],\\n    stammdaten: Dict[str, pd.DataFrame],\\n    block_fach: bool = False,\\n    block_uebf: bool = False,\\n    block_psych: bool = False,\\n    population_seed: int = 12345\\n):'
)

# 2. Update RNG seeds
code = code.replace(
    \"base_seed = zlib.crc32(studi.studierenden_id.encode('utf-8'))\",
    \"base_seed = (zlib.crc32(studi.studierenden_id.encode('utf-8')) ^ population_seed) & 0xFFFFFFFF\"
)
code = code.replace(
    \"rng_support = np.random.default_rng(base_seed + 1)\",
    \"rng_support = np.random.default_rng((base_seed + 1) & 0xFFFFFFFF)\"
)
code = code.replace(
    \"rng_social = np.random.default_rng(base_seed + 2)\",
    \"rng_social = np.random.default_rng((base_seed + 2) & 0xFFFFFFFF)\"
)
code = code.replace(
    \"rng_dropout = np.random.default_rng(base_seed + 3)\",
    \"rng_dropout = np.random.default_rng((base_seed + 3) & 0xFFFFFFFF)\"
)

# 3. Add main block and universes
new_main = '''
def main(population_seed: int = 12345, base_output_override=None):
    print(\"Starte True Counterfactual Trajectory Simulator (Simulator v3) ...\")
    print(\"  8 Parallele Universen mit per-Typ Support-Blockierung, Stochastischem Puffer & Gedeckeltem Overload\")
    
    if base_output_override:
        base_output = base_output_override
    else:
        base_output = Path(CONFIG[\"output_dir\"])
    os.makedirs(base_output / \"metrics\", exist_ok=True)
    
    stammdaten = generiere_stammdaten()
    
    UNIVERSES = {
        \"A\": {\"label\": \"Alle Support-Typen erlaubt\",       \"block_fach\": False, \"block_uebf\": False, \"block_psych\": False},
        \"B\": {\"label\": \"Kein Support (komplett blockiert)\",  \"block_fach\": True,  \"block_uebf\": True,  \"block_psych\": True},
        \"C\": {\"label\": \"Kein fachlicher Support\",           \"block_fach\": True,  \"block_uebf\": False, \"block_psych\": False},
        \"D\": {\"label\": \"Kein ueberfachlicher Support\",      \"block_fach\": False, \"block_uebf\": True,  \"block_psych\": False},
        \"E\": {\"label\": \"Kein psychosozialer Support\",       \"block_fach\": False, \"block_uebf\": False, \"block_psych\": True},
        \"F\": {\"label\": \"Nur fachlicher Support\",            \"block_fach\": False, \"block_uebf\": True,  \"block_psych\": True},
        \"G\": {\"label\": \"Nur ueberfachlicher Support\",       \"block_fach\": True,  \"block_uebf\": False, \"block_psych\": True},
        \"H\": {\"label\": \"Nur psychosozialer Support\",        \"block_fach\": True,  \"block_uebf\": True,  \"block_psych\": False},
    }
    
    POPULATION_SEED = population_seed
    results = {}
    
    for uni_key, uni_cfg in UNIVERSES.items():
        print(f\"\\n  UNIVERSUM {uni_key}: {uni_cfg['label']}\")
        rng = np.random.default_rng(POPULATION_SEED)
        studierende = generiere_studierende_v3(stammdaten, rng)
        
        simuliere_verlaeufe_v3(
            studierende, stammdaten,
            block_fach=uni_cfg[\"block_fach\"],
            block_uebf=uni_cfg[\"block_uebf\"],
            block_psych=uni_cfg[\"block_psych\"],
            population_seed=POPULATION_SEED
        )
        
        dfs = stammdaten.copy()
        dfs.update(as_dataframe(studierende, stammdaten))
        
        if uni_key == \"A\":
            uni_dir = base_output
        else:
            uni_dir = base_output / f\"universe_{uni_key}\"
            uni_dir.mkdir(exist_ok=True, parents=True)
            
        exportiere_csv(dfs, uni_dir)
        
        dropout_cnt = sum(1 for s in studierende if s.abgebrochen or s.exmatrikuliert or (not s.abschluss_erreicht and len(s.einschreibungen) >= 16))
        drop_rate = dropout_cnt / len(studierende)
        
        results[f\"universe_{uni_key}\"] = {
            \"label\": uni_cfg[\"label\"],
            \"dropout_rate\": round(drop_rate, 5)
        }
        print(f\"  Dropout-Rate Universum {uni_key}: {drop_rate*100:.2f}% ({dropout_cnt}/{len(studierende)})\")

    base_rate_A = results[\"universe_A\"][\"dropout_rate\"]
    base_rate_B = results.get(\"universe_B\", {}).get(\"dropout_rate\", 0)

    for u in [\"B\", \"C\", \"D\", \"E\"]:
        if f\"universe_{u}\" in results:
            u_rate = results[f\"universe_{u}\"][\"dropout_rate\"]
            diff = u_rate - base_rate_A
            rr = u_rate / base_rate_A if base_rate_A > 0 else 1.0
            results[f\"universe_{u}\"][\"vs_A_absolute_diff\"] = round(diff, 5)
            results[f\"universe_{u}\"][\"vs_A_relative_risk\"] = round(rr, 5)
            results[f\"universe_{u}\"][\"vs_A_relative_reduction_pct\"] = round((1.0 - rr) * 100, 5)

    for u in [\"F\", \"G\", \"H\"]:
        if f\"universe_{u}\" in results:
            u_rate = results[f\"universe_{u}\"][\"dropout_rate\"]
            rr_vs_B = u_rate / base_rate_B if base_rate_B > 0 else 1.0
            results[f\"universe_{u}\"][\"vs_B_absolute_diff\"] = round(u_rate - base_rate_B, 5)
            results[f\"universe_{u}\"][\"vs_B_relative_risk\"] = round(rr_vs_B, 5)
            results[f\"universe_{u}\"][\"vs_B_relative_reduction_pct\"] = round((1.0 - rr_vs_B) * 100, 5)

    with open(base_output / \"metrics\" / \"true_macro_effects_v3.json\", \"w\") as f:
        json.dump(results, f, indent=4)
        
    print(f\"\\nWahre Makro-Effekte Simulation V3 (Alle 8 Universen) erfolgreich gespeichert!\")

if __name__ == \"__main__\":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--seed\", type=int, default=12345, help=\"Population seed\")
    parser.add_argument(\"--output-dir\", type=str, default=None, help=\"Output directory\")
    args = parser.parse_args()

    import os, json, sys
    from pathlib import Path
    from export import as_dataframe, exportiere_csv
    
    out_dir = Path(args.output_dir) if args.output_dir else None
    main(population_seed=args.seed, base_output_override=out_dir)
'''

code = re.sub(r'if __name__ == \"__main__\":.*', new_main, code, flags=re.DOTALL)

with open('src/simulation_v3.py', 'w') as f:
    f.write(code)
print(\"Patched simulation_v3.py\")
