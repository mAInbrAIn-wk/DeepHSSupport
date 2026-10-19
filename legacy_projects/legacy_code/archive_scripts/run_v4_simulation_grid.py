"""
V4 Simulation Sensitivity Grid Search Runner (Multiprocessing)
==============================================================
Führt eine systematische Sensitivitätsanalyse der V4-Simulations-Engine
über 12 Szenarien (inkl. 8 Universen A-H pro Szenario) durch.
"""

import os
import sys
import copy
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path('src').absolute()))
from config import CONFIG
from export import as_dataframe, exportiere_csv
from simulation_v4 import generiere_stammdaten, generiere_studierende, simuliere_verlaeufe


UNIVERSES = {
    "A": {"label": "Alle Support-Typen erlaubt",       "block_fach": False, "block_uebf": False, "block_psych": False},
    "B": {"label": "Kein Support (komplett blockiert)",  "block_fach": True,  "block_uebf": True,  "block_psych": True},
    "C": {"label": "Kein fachlicher Support",           "block_fach": True,  "block_uebf": False, "block_psych": False},
    "D": {"label": "Kein ueberfachlicher Support",      "block_fach": False, "block_uebf": True,  "block_psych": False},
    "E": {"label": "Kein psychosozialer Support",       "block_fach": False, "block_uebf": False, "block_psych": True},
    "F": {"label": "Nur fachlicher Support",            "block_fach": False, "block_uebf": True,  "block_psych": True},
    "G": {"label": "Nur ueberfachlicher Support",       "block_fach": True,  "block_uebf": False, "block_psych": True},
    "H": {"label": "Nur psychosozialer Support",        "block_fach": True,  "block_uebf": True,  "block_psych": False}
}

GRID_SCENARIOS = [
    # --- S01 ist implizit: Baseline ohne Override (wird separat via run_v4_universes.py erzeugt) ---
    # --- Support-Wirkung ---
    {
        "id": "S02_supp_half",
        "name": "Support-Wirkung Halbiert (2.5 = 0.5x Baseline 5.0)",
        "dim": "Support-Wirkung",
        "override": {"support_effect_multiplier": 2.5}
    },
    {
        "id": "S03_supp_double",
        "name": "Support-Wirkung Verdoppelt (10.0 = 2x Baseline 5.0)",
        "dim": "Support-Wirkung",
        "override": {"support_effect_multiplier": 10.0}
    },
    # --- Notenboost ---
    {
        "id": "S04_grade_half",
        "name": "Notenboost Halbiert (0.04 statt 0.08)",
        "dim": "Notenboost",
        "override": {"gewicht_support_boost": 0.04}
    },
    {
        "id": "S05_grade_double",
        "name": "Notenboost Verdoppelt (0.16 statt 0.08)",
        "dim": "Notenboost",
        "override": {"gewicht_support_boost": 0.16}
    },
    {
        "id": "S06_grade_quad",
        "name": "Notenboost Vervierfacht (0.32 statt 0.08)",
        "dim": "Notenboost",
        "override": {"gewicht_support_boost": 0.32}
    },
    # --- Rauschen ---
    {
        "id": "S07_noise_half",
        "name": "Rauschen Halbiert (0.09 statt 0.18)",
        "dim": "Rauschen",
        "override": {"gewicht_rauschen": 0.09}
    },
    {
        "id": "S08_noise_double",
        "name": "Rauschen Verdoppelt (0.36 statt 0.18)",
        "dim": "Rauschen",
        "override": {"gewicht_rauschen": 0.36}
    },
    # --- Zeitkosten ---
    {
        "id": "S09_cost_zero",
        "name": "Support Kostenlos (Faktor 0)",
        "dim": "Zeitkosten",
        "override": {"support_kosten_faktor": 0.0}
    },
    {
        "id": "S10_cost_double",
        "name": "Support-Kosten Verdoppelt (Faktor 2)",
        "dim": "Zeitkosten",
        "override": {"support_kosten_faktor": 2.0}
    },
    # --- Selektion ---
    {
        "id": "S11_rct_calibrated",
        "name": "RCT Kalibriert (Gleiches Volumen, zufaellige Zuordnung)",
        "dim": "Selektion",
        "override": {"rct_support_uptake": True}
    },
    # --- Overload-Penalty ---
    {
        "id": "S12_overload_half",
        "name": "Overload-Penalty Halbiert (0.05 statt 0.1)",
        "dim": "Overload-Penalty",
        "override": {"overload_penalty_factor": 0.05}
    },
    {
        "id": "S13_overload_double",
        "name": "Overload-Penalty Verdoppelt (0.2 statt 0.1)",
        "dim": "Overload-Penalty",
        "override": {"overload_penalty_factor": 0.2}
    },
    {
        "id": "S14_overload_cap",
        "name": "Overload-Penalty mit Cap (0.15, wie V3.6)",
        "dim": "Overload-Penalty",
        "override": {"overload_penalty_cap": 0.15}
    },
    # --- Kombi-Szenarien ---
    {
        "id": "S15_cost_effect_double",
        "name": "Kosten UND Wirkung Verdoppelt (Faktor 2 + Mult 10.0)",
        "dim": "Kombi",
        "override": {"support_kosten_faktor": 2.0, "support_effect_multiplier": 10.0}
    },
    # --- Selektionsmechanismen / Apathie-Ablation ---
    {
        "id": "S16_no_apathy_dampening",
        "name": "Ohne Apathie-Dämpfung (Lineare Support-Suche ohne Kollaps bei mot<0.2)",
        "dim": "Selektion",
        "override": {"disable_apathy_dampening": True}
    }
]


def _simulate_single_universe_worker(args: Tuple) -> Dict:
    """Worker-Funktion fuer Multiprocessing eines einzelnen Universums."""
    scenario_id, uni_key, uni_cfg, cfg_scenario, population_seed, save_csv, out_base_str = args
    out_base = Path(out_base_str)
    
    t0 = time.time()
    
    # 1. Stammdaten
    stammdaten = generiere_stammdaten()
    
    # 2. Blockierte Angebote filtern
    support_df = stammdaten["support_angebote_df"]
    drop_types = []
    if uni_cfg["block_fach"]: drop_types.append("fachlich")
    if uni_cfg["block_uebf"]: drop_types.append("ueberfachlich")
    if uni_cfg["block_psych"]: drop_types.append("psychosozial")
    stammdaten["support_angebote_df"] = support_df[~support_df["typ"].isin(drop_types)]
    
    # 3. Studierende mit fixem Population-Seed generieren
    rng_init = np.random.default_rng(population_seed)
    base_studierende = generiere_studierende(stammdaten, rng_init, cfg=cfg_scenario)
    
    # 4. Klonen und Simulation mit fixem Simulator-Seed
    rng_sim = np.random.default_rng(population_seed + 100)
    studierende = simuliere_verlaeufe(base_studierende, stammdaten, rng_sim, cfg=cfg_scenario, population_seed=population_seed)
    
    # 5. Metriken berechnen
    N = len(studierende)
    dropout_studis = [s for s in studierende if s.abgebrochen or s.exmatrikuliert or (not s.abschluss_erreicht and len(s.einschreibungen) >= 16)]
    grad_studis = [s for s in studierende if s.abschluss_erreicht]
    
    drop_rate = len(dropout_studis) / N
    grad_rate = len(grad_studis) / N
    
    # Subgruppen: First-Gen & Migration
    fg_studis = [s for s in studierende if s.erstakademiker]
    nfg_studis = [s for s in studierende if not s.erstakademiker]
    fg_drop = sum(1 for s in fg_studis if s in dropout_studis) / len(fg_studis) if fg_studis else 0.0
    nfg_drop = sum(1 for s in nfg_studis if s in dropout_studis) / len(nfg_studis) if nfg_studis else 0.0
    
    mig_studis = [s for s in studierende if s.migrationshintergrund]
    nmig_studis = [s for s in studierende if not s.migrationshintergrund]
    mig_drop = sum(1 for s in mig_studis if s in dropout_studis) / len(mig_studis) if mig_studis else 0.0
    nmig_drop = sum(1 for s in nmig_studis if s in dropout_studis) / len(nmig_studis) if nmig_studis else 0.0
    
    # Noten & Dauer
    gpa_list = []
    for s in grad_studis:
        passed = [p.note for p in s.pruefungen if p.bestanden]
        if passed:
            gpa_list.append(np.mean(passed))
    gpa_grad = float(np.mean(gpa_list)) if gpa_list else np.nan
    dauer_grad = float(np.mean([len(s.einschreibungen) for s in grad_studis])) if grad_studis else np.nan
    
    # Workload drops
    total_mod_drops = sum(s.stat_modules_dropped for s in studierende)
    
    # Optional CSV Export (nur wenn save_csv=True, spart I/O)
    if save_csv:
        df_dict = stammdaten.copy()
        df_dict.update(as_dataframe(studierende, stammdaten))
        uni_out = out_base / scenario_id / f"universe_{uni_key}"
        uni_out.mkdir(parents=True, exist_ok=True)
        exportiere_csv(df_dict, uni_out)
        
    duration = time.time() - t0
    
    return {
        "scenario_id": scenario_id,
        "universe_key": uni_key,
        "label": uni_cfg["label"],
        "n_studierende": N,
        "dropout_count": len(dropout_studis),
        "dropout_rate": round(drop_rate, 5),
        "grad_rate": round(grad_rate, 5),
        "first_gen_dropout": round(fg_drop, 5),
        "non_first_gen_dropout": round(nfg_drop, 5),
        "first_gen_gap": round(fg_drop - nfg_drop, 5),
        "migration_dropout": round(mig_drop, 5),
        "non_migration_dropout": round(nmig_drop, 5),
        "migration_gap": round(mig_drop - nmig_drop, 5),
        "gpa_graduation": round(gpa_grad, 3),
        "duration_graduation": round(dauer_grad, 2),
        "total_modules_dropped": total_mod_drops,
        "calc_duration_sec": round(duration, 2)
    }


def run_full_sensitivity_grid(
    n_studierende: int = 25000,
    population_seed: int = 99999,
    max_workers: int = 5,
    out_dir: Path = Path("src/output_v4_grid")
):
    print("=" * 95)
    print(f"V4 SIMULATION SENSITIVITY GRID SEARCH ({len(GRID_SCENARIOS)} SZENARIEN x 8 UNIVERSEN)")
    print(f"Cohort Size per Universe: N = {n_studierende:,} | Workers: {max_workers} | Seed: {population_seed}")
    print("=" * 95)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Aufgaben vorbereiten
    tasks = []
    for sc in GRID_SCENARIOS:
        sc_cfg = copy.deepcopy(CONFIG)
        sc_cfg["n_studierende"] = n_studierende
        sc_cfg.update(sc["override"])
        
        # Save CSV fuer alle Szenarien (fuer Detailanalysen auf Studierendenebene)
        save_csv = True
        
        for u_key, u_cfg in UNIVERSES.items():
            tasks.append((sc["id"], u_key, u_cfg, sc_cfg, population_seed, save_csv, str(out_dir)))
            
    print(f"\nGeneriert: {len(tasks)} Universe-Simulations-Tasks. Starte parallele Ausfuehrung...")
    start_total = time.time()
    
    results_raw = {}
    completed_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_simulate_single_universe_worker, t): t for t in tasks}
        
        for future in as_completed(futures):
            res = future.result()
            sc_id = res["scenario_id"]
            u_key = res["universe_key"]
            
            if sc_id not in results_raw:
                results_raw[sc_id] = {}
            results_raw[sc_id][u_key] = res
            
            completed_count += 1
            if completed_count % 8 == 0 or completed_count == len(tasks):
                elapsed = time.time() - start_total
                print(f" [{completed_count:2d}/{len(tasks)}] Fertig ({elapsed:.1f}s) -> Zuletzt: {sc_id} Uni {u_key} (Drop: {res['dropout_rate']*100:.2f}%)")

    print("\n" + "=" * 95)
    print("AGGREGIERE MAKRO-EFFEKTE & RELATIVE RISIKEN FUER ALLE SZENARIEN")
    print("=" * 95)
    
    grid_summary = []
    
    for sc in GRID_SCENARIOS:
        sc_id = sc["id"]
        unis = results_raw[sc_id]
        
        drop_A = unis["A"]["dropout_rate"]
        drop_B = unis["B"]["dropout_rate"]
        drop_C = unis["C"]["dropout_rate"]
        drop_D = unis["D"]["dropout_rate"]
        drop_E = unis["E"]["dropout_rate"]
        drop_F = unis["F"]["dropout_rate"]
        drop_G = unis["G"]["dropout_rate"]
        drop_H = unis["H"]["dropout_rate"]
        
        # Relative Risiken vs Universum A (Full Support)
        rr_B = drop_B / drop_A if drop_A > 0 else 1.0
        rr_C = drop_C / drop_A if drop_A > 0 else 1.0
        rr_D = drop_D / drop_A if drop_A > 0 else 1.0
        rr_E = drop_E / drop_A if drop_A > 0 else 1.0
        
        # Relative Risiken vs Universum B (No Support - Isolierte Effekte)
        rr_F_vs_B = drop_F / drop_B if drop_B > 0 else 1.0 # Nur Fachlich
        rr_G_vs_B = drop_G / drop_B if drop_B > 0 else 1.0 # Nur Ueberfachlich
        rr_H_vs_B = drop_H / drop_B if drop_B > 0 else 1.0 # Nur Psychosozial
        
        # Synergie / Superadditivitaet
        # Summe der isolierten Reduktionen vs gemeinsame Reduktion
        red_F = (1.0 - rr_F_vs_B) * 100 # % Schutz durch nur Fachlich
        red_G = (1.0 - rr_G_vs_B) * 100 # % Schutz durch nur Ueberfachlich
        red_H = (1.0 - rr_H_vs_B) * 100 # % Schutz durch nur Psychosozial
        red_all = (1.0 - (drop_A / drop_B)) * 100 # % Schutz durch alle 3 gemeinsam
        synergy_gap = red_all - (red_F + red_G + red_H) # Positiv = Superadditivitaet
        
        # Equalizer Effekt (First-Gen Schutz in A vs B)
        fg_gap_A = unis["A"]["first_gen_gap"]
        fg_gap_B = unis["B"]["first_gen_gap"]
        equalizer_gain = fg_gap_B - fg_gap_A # Positiv = Support verringert Bildungsungleichheit
        
        sc_res = {
            "scenario_id": sc_id,
            "name": sc["name"],
            "dimension": sc["dim"],
            "dropout_A": drop_A,
            "dropout_B": drop_B,
            "dropout_C": drop_C,
            "dropout_D": drop_D,
            "dropout_E": drop_E,
            "dropout_F": drop_F,
            "dropout_G": drop_G,
            "dropout_H": drop_H,
            "RR_B_vs_A": round(rr_B, 4),
            "RR_C_vs_A": round(rr_C, 4),
            "RR_D_vs_A": round(rr_D, 4),
            "RR_E_vs_A": round(rr_E, 4),
            "RR_F_vs_B_isol_fach": round(rr_F_vs_B, 4),
            "RR_G_vs_B_isol_uebf": round(rr_G_vs_B, 4),
            "RR_H_vs_B_isol_psych": round(rr_H_vs_B, 4),
            "protection_all_pct": round(red_all, 2),
            "protection_fach_pct": round(red_F, 2),
            "protection_uebf_pct": round(red_G, 2),
            "protection_psych_pct": round(red_H, 2),
            "synergy_gap_pct_pts": round(synergy_gap, 2),
            "first_gen_gap_A": fg_gap_A,
            "first_gen_gap_B": fg_gap_B,
            "equalizer_gain_pct_pts": round(equalizer_gain * 100, 2),
            "modules_dropped_A": unis["A"]["total_modules_dropped"],
            "modules_dropped_B": unis["B"]["total_modules_dropped"],
            "universes": unis
        }
        
        grid_summary.append(sc_res)
        
        # Speichern des Szenario-Einzelberichts
        sc_dir = out_dir / sc_id / "metrics"
        sc_dir.mkdir(parents=True, exist_ok=True)
        with open(sc_dir / "true_macro_effects.json", "w") as f:
            json.dump(sc_res, f, indent=2)

    # Gesamtergebnis speichern
    with open(metrics_dir / "full_sensitivity_grid_results.json", "w") as f:
        json.dump(grid_summary, f, indent=2)
        
    total_time = time.time() - start_total
    print(f"\n[OK] Gridsearch erfolgreich abgeschlossen in {total_time:.1f} Sekunden ({total_time/60:.2f} Minuten)!")
    print(f"[OK] Gespeichert unter {metrics_dir / 'full_sensitivity_grid_results.json'}")
    return grid_summary


if __name__ == "__main__":
    run_full_sensitivity_grid(n_studierende=25000, population_seed=99999, max_workers=5)
