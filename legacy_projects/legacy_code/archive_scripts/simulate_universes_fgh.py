"""
Partielle Simulation der Universen F, G, H (Ground-Truth Isolations-Welten)
=============================================================================
Simuliert ausschließlich die neuen kontrafaktischen Universen F, G, H:
- Universum F: Nur fachlicher Support aktiv (uebf & psych blockiert)
- Universum G: Nur überfachlicher Support aktiv (fach & psych blockiert)
- Universum H: Nur psychosozialer Support aktiv (fach & uebf blockiert)

Speichert die Ergebnisse in:
  src/output_dl/universe_F/
  src/output_dl/universe_G/
  src/output_dl/universe_H/
und aktualisiert src/output_dl/metrics/true_macro_effects_v3.json mit:
  - Partiellen Relative Risks (vs. Baseline A): R_A / R_C, R_A / R_D, R_A / R_E
  - Isolierten Relative Risks (vs. Null-Support B): R_F / R_B, R_G / R_B, R_H / R_B
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np

# Projekt-Pfade einbinden
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.chdir(SRC_DIR)

from simulation_v3 import simuliere_verlaeufe_v3, generiere_studierende_v3, generiere_stammdaten
from config import CONFIG
from export import as_dataframe, exportiere_csv
from aggregate import aggregiere_daten

def simulate_universes_fgh():
    print("=" * 70)
    print("   PARTIELLE SIMULATION: UNIVERSEN F, G, H (ISOLIERTE GROUND TRUTH)")
    print(f"   Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    base_output = Path(CONFIG["output_dir"]) if Path(CONFIG["output_dir"]).exists() else Path("output_dl")
    os.makedirs(base_output / "metrics", exist_ok=True)
    
    stammdaten = generiere_stammdaten()
    
    UNIVERSES_NEW = {
        "F": {"label": "Nur fachlicher Support (uebf & psych blockiert)",       "block_fach": False, "block_uebf": True,  "block_psych": True},
        "G": {"label": "Nur überfachlicher Support (fach & psych blockiert)",   "block_fach": True,  "block_uebf": False, "block_psych": True},
        "H": {"label": "Nur psychosozialer Support (fach & uebf blockiert)",    "block_fach": True,  "block_uebf": True,  "block_psych": False},
    }
    
    POPULATION_SEED = 12345
    results = {}
    
    for uni_key, uni_cfg in UNIVERSES_NEW.items():
        print(f"\n>>> Simuliere Universum {uni_key}: {uni_cfg['label']} ...")
        t0 = time.time()
        
        rng = np.random.default_rng(POPULATION_SEED)
        studierende = generiere_studierende_v3(stammdaten, rng)
        
        simuliere_verlaeufe_v3(
            studierende, stammdaten,
            block_fach=uni_cfg["block_fach"],
            block_uebf=uni_cfg["block_uebf"],
            block_psych=uni_cfg["block_psych"]
        )
        
        n = len(studierende)
        dropouts = sum(1 for s in studierende if s.abgebrochen)
        rate = dropouts / n
        results[uni_key] = {"label": uni_cfg["label"], "dropout_rate": rate, "dropouts": dropouts, "n": n}
        
        elapsed = time.time() - t0
        print(f"    [OK] Universum {uni_key}: Dropout-Rate = {rate:.4%} ({dropouts}/{n}) [{elapsed/60:.1f} Min.]")
        
        uni_output = base_output / f"universe_{uni_key}"
        os.makedirs(uni_output, exist_ok=True)
        
        print(f"    Exportiere CSVs nach {uni_output} ...")
        df_dict = stammdaten.copy()
        df_dict.update(as_dataframe(studierende, stammdaten))
        exportiere_csv(df_dict, uni_output)
        
        print(f"    Aggregiere DataCube in {uni_output} ...")
        aggregiere_daten(uni_output)
        
    # Lade bestehende Makro-Effekte von A-E
    macro_file = base_output / "metrics" / "true_macro_effects_v3.json"
    if macro_file.exists():
        with open(macro_file, "r") as f:
            macro_data = json.load(f)
    else:
        macro_data = {}
        
    rate_A = macro_data.get("universe_A_baseline", {}).get("dropout_rate", 0.2737)
    rate_B = macro_data.get("universe_B", {}).get("dropout_rate", 0.32346)
    rate_C = macro_data.get("universe_C", {}).get("dropout_rate", 0.28572)
    rate_D = macro_data.get("universe_D", {}).get("dropout_rate", 0.29156)
    rate_E = macro_data.get("universe_E", {}).get("dropout_rate", 0.28768)
    
    rate_F = results["F"]["dropout_rate"]
    rate_G = results["G"]["dropout_rate"]
    rate_H = results["H"]["dropout_rate"]
    
    # Füge F, G, H hinzu
    for uni_key in ["F", "G", "H"]:
        rate_X = results[uni_key]["dropout_rate"]
        rr_vs_B = rate_X / rate_B if rate_B > 0 else 1.0
        macro_data[f"universe_{uni_key}"] = {
            "label": results[uni_key]["label"],
            "dropout_rate": rate_X,
            "vs_B_absolute_diff": rate_X - rate_B,
            "vs_B_relative_risk": rr_vs_B,
            "vs_B_relative_reduction_pct": (1.0 - rr_vs_B) * 100
        }
        
    # Dual-Teststrang Ground Truth Zusammenfassung:
    macro_data["ground_truth_summary"] = {
        "partial_vs_A": {
            "description": "Partieller Effekt des Wegnehmens (A vs. C/D/E): R_A / R_ohne",
            "RR_fachlich_partial": rate_A / rate_C,
            "RR_ueberfachlich_partial": rate_A / rate_D,
            "RR_psychosozial_partial": rate_A / rate_E,
            "RR_all_partial": rate_A / rate_B
        },
        "isolated_vs_B": {
            "description": "Isolierter Effekt des Alleinseins (B vs. F/G/H): R_nur / R_B",
            "RR_fachlich_isolated": rate_F / rate_B,
            "RR_ueberfachlich_isolated": rate_G / rate_B,
            "RR_psychosozial_isolated": rate_H / rate_B
        }
    }
    
    with open(macro_file, "w") as f:
        json.dump(macro_data, f, indent=4)
        
    print("\n" + "=" * 70)
    print("   GROUND TRUTH MAKRO-EFFEKTE (DUAL-STRANG)")
    print("=" * 70)
    print(f"  Baseline A (alle aktiv):      {rate_A:.4%}")
    print(f"  Null-Support B (alle blockiert): {rate_B:.4%}")
    print(f"  --------------------------------------------------")
    print(f"  PARTIELLER EFFEKT (Wegnehmen: R_A / R_ohne):")
    print(f"    • Fachlich:     RR = {rate_A / rate_C:.4f}  (1 - RR = {(1 - rate_A / rate_C)*100:.2f}%)")
    print(f"    • Überfachlich: RR = {rate_A / rate_D:.4f}  (1 - RR = {(1 - rate_A / rate_D)*100:.2f}%)")
    print(f"    • Psychosozial: RR = {rate_A / rate_E:.4f}  (1 - RR = {(1 - rate_A / rate_E)*100:.2f}%)")
    print(f"  --------------------------------------------------")
    print(f"  ISOLIERTER EFFEKT (Allein aktiv: R_nur / R_B):")
    print(f"    • Fachlich (F):     Dropout = {rate_F:.4%} -> RR = {rate_F / rate_B:.4f}  (1 - RR = {(1 - rate_F / rate_B)*100:.2f}%)")
    print(f"    • Überfachlich (G): Dropout = {rate_G:.4%} -> RR = {rate_G / rate_B:.4f}  (1 - RR = {(1 - rate_G / rate_B)*100:.2f}%)")
    print(f"    • Psychosozial (H): Dropout = {rate_H:.4%} -> RR = {rate_H / rate_B:.4f}  (1 - RR = {(1 - rate_H / rate_B)*100:.2f}%)")
    print("=" * 70)
    print(f"Makro-Metriken erfolgreich aktualisiert in: {macro_file}")

if __name__ == "__main__":
    simulate_universes_fgh()