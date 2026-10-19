import os
import json
import sys
from pathlib import Path
import copy
import numpy as np
import time

sys.path.insert(0, str(Path('src').absolute()))
from config import CONFIG
from export import as_dataframe, exportiere_csv
# Wir nutzen die neue Engine v4
from simulation_v4 import generiere_stammdaten, generiere_studierende, simuliere_verlaeufe

def run_v4_universes(population_seed: int = 12345, base_output_override: Path = None):
    print("Starte True Counterfactual Trajectory Simulator (Simulator V4 Engine) ...")
    
    if base_output_override:
        base_output = base_output_override
    else:
        base_output = Path("src/output_v4_universes")
        
    os.makedirs(base_output / "metrics", exist_ok=True)
    
    # 1. Stammdaten (Gleich fr alle Universen)
    stammdaten = generiere_stammdaten()
    
    UNIVERSES = {
        "A": {"label": "Alle Support-Typen erlaubt",       "block_fach": False, "block_uebf": False, "block_psych": False},
        "B": {"label": "Kein Support (komplett blockiert)",  "block_fach": True,  "block_uebf": True,  "block_psych": True},
        "C": {"label": "Kein fachlicher Support",           "block_fach": True,  "block_uebf": False, "block_psych": False},
        "D": {"label": "Kein ueberfachlicher Support",      "block_fach": False, "block_uebf": True,  "block_psych": False},
        "E": {"label": "Kein psychosozialer Support",       "block_fach": False, "block_uebf": False, "block_psych": True},
        # Isolierte Welten (Nur 1 Support-Typ)
        "F": {"label": "Nur fachlicher Support",            "block_fach": False, "block_uebf": True,  "block_psych": True},
        "G": {"label": "Nur ueberfachlicher Support",       "block_fach": True,  "block_uebf": False, "block_psych": True},
        "H": {"label": "Nur psychosozialer Support",        "block_fach": True,  "block_uebf": True,  "block_psych": False}
    }
    
    results = {}
    
    # Studierende VOR der Universen-Schleife mit fixem Seed generieren (Identische Zuweisung)
    rng_init = np.random.default_rng(population_seed)
    base_studierende = generiere_studierende(stammdaten, rng_init)
    
    for uni_key, uni_cfg in UNIVERSES.items():
        print(f"\n=======================================================")
        print(f"Simuliere Universum {uni_key}: {uni_cfg['label']}")
        print(f"=======================================================")
        
        # 1. Blockierte Angebote in Stammdaten deaktivieren
        uni_stammdaten = copy.deepcopy(stammdaten)
        support_df = uni_stammdaten["support_angebote_df"]
        
        # Mapping der Boolean-Flags zu "typ"
        # Typen: 'fachlich', 'ueberfachlich', 'psychosozial'
        drop_types = []
        if uni_cfg["block_fach"]: drop_types.append("fachlich")
        if uni_cfg["block_uebf"]: drop_types.append("ueberfachlich")
        if uni_cfg["block_psych"]: drop_types.append("psychosozial")
        
        uni_stammdaten["support_angebote_df"] = support_df[~support_df["typ"].isin(drop_types)]
        
        # 2. Studierende klonen, damit ihr initialer Zustand in jedem Universum gleich ist
        studierende_klohn = copy.deepcopy(base_studierende)
        
        # 3. Kausaler Stochastik-Seed fr dieses Universum
        # Jedes Universum bekommt den Glichen Simulator-Seed (Salting fr exakte Varianzkontrolle)
        # So ist sichergestellt, dass Rauschen (Krankheit etc.) deterministisch bleibt
        rng_sim = np.random.default_rng(population_seed + 100)
        
        # 4. Simulation V4 ausfuhren
        start_t = time.time()
        studierende = simuliere_verlaeufe(studierende_klohn, uni_stammdaten, rng_sim, population_seed=population_seed)
        
        # 5. Export
        df_dict = uni_stammdaten.copy()
        df_dict.update(as_dataframe(studierende, uni_stammdaten))
        
        uni_out = base_output / f"universe_{uni_key}"
        uni_out.mkdir(parents=True, exist_ok=True)
        exportiere_csv(df_dict, uni_out)
        
        dropout_cnt = sum(1 for s in studierende if s.abgebrochen or s.exmatrikuliert or (not s.abschluss_erreicht and len(s.einschreibungen) >= 16))
        drop_rate = dropout_cnt / len(studierende)
        
        results[f"universe_{uni_key}"] = {
            "label": uni_cfg["label"],
            "dropout_rate": round(drop_rate, 5)
        }
        print(f"  Dropout-Rate Universum {uni_key}: {drop_rate*100:.2f}% ({dropout_cnt}/{len(studierende)})")
        print(f"  Time taken: {time.time() - start_t:.1f}s")
        
    base_rate_A = results["universe_A"]["dropout_rate"]
    base_rate_B = results.get("universe_B", {}).get("dropout_rate", 0)

    for u in ["B", "C", "D", "E"]:
        if f"universe_{u}" in results:
            u_rate = results[f"universe_{u}"]["dropout_rate"]
            diff = u_rate - base_rate_A
            rr = u_rate / base_rate_A if base_rate_A > 0 else 1.0
            results[f"universe_{u}"]["vs_A_absolute_diff"] = round(diff, 5)
            results[f"universe_{u}"]["vs_A_relative_risk"] = round(rr, 5)
            results[f"universe_{u}"]["vs_A_relative_reduction_pct"] = round((1.0 - rr) * 100, 5)

    for u in ["F", "G", "H"]:
        if f"universe_{u}" in results:
            u_rate = results[f"universe_{u}"]["dropout_rate"]
            rr_vs_B = u_rate / base_rate_B if base_rate_B > 0 else 1.0
            results[f"universe_{u}"]["vs_B_absolute_diff"] = round(u_rate - base_rate_B, 5)
            results[f"universe_{u}"]["vs_B_relative_risk"] = round(rr_vs_B, 5)
            results[f"universe_{u}"]["vs_B_relative_reduction_pct"] = round((1.0 - rr_vs_B) * 100, 5)

    with open(base_output / "metrics" / "true_macro_effects_v4.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nWahre Makro-Effekte Simulation V4 (Alle 8 Universen) erfolgreich gespeichert!")

if __name__ == "__main__":
    run_v4_universes(population_seed=99999)
