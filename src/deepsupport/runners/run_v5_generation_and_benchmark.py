"""
DGP V5.0 Full Multi-Universe Generator & Master Model Benchmark Runner
======================================================================
1. Generiert alle 8 kontrafaktischen Parallelwelten (Universum A bis H) unter
   DGP Version 5.0 (N = 50.000, population_seed = 99999) in ein separates,
   isoliertes Verzeichnis: `data_v5_grid/S01_baseline/`.
2. Führt die DuckDB-Voraggregation (`agg_abschluesse.csv`, `agg_pruefungen.csv`)
   für die generierten Universen aus.
3. Trainiert und evaluiert anschließend alle 6 Modellfamilien (>25 Modelle in
   Keras 3 & PyTorch 2.x inkl. Kausal-Suite und Policy-Targeting-Showcase) auf
   den neuen V5.0-Daten mit Ausgabe nach `outputs/S01_baseline_v5/`.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deepsupport.data_engine.aggregate import aggregiere_daten
from deepsupport.data_engine.config import CONFIG
from deepsupport.runners.run_post_refactor_benchmark_s01 import run_benchmark
from deepsupport.simulation.sensitivity_grid import (
    UNIVERSES,
    _simulate_single_universe_worker,
)


def generate_v5_baseline_universes(
    out_grid_dir: Path,
    scenario_id: str = "S01_baseline",
    n_studierende: int = 50000,
    population_seed: int = 99999,
    max_workers: int = 4,
    force_regen: bool = False,
) -> Path:
    """
    Generiert die 8 Parallelwelten (A–H) für DGP V5.0 in `out_grid_dir / scenario_id`.
    Überspringt bereits vollständig generierte Universen, sofern `force_regen=False`.
    """
    out_grid_dir = Path(out_grid_dir).resolve()
    scenario_dir = out_grid_dir / scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)

    cfg_v5 = deepcopy(CONFIG)
    cfg_v5["n_studierende"] = n_studierende
    cfg_v5["log_every_n_studis"] = 10000

    tasks = []
    for u_key, u_cfg in UNIVERSES.items():
        u_dir = scenario_dir / f"universe_{u_key}"
        if not force_regen and (u_dir / "abschluesse.csv").exists() and (u_dir / "pruefungen.csv").exists():
            print(f"[SKIP-GEN] Universe {u_key} existiert bereits in {u_dir}")
            continue
        tasks.append(
            (
                scenario_id,
                u_key,
                u_cfg,
                cfg_v5,
                population_seed,
                True,
                str(out_grid_dir),
            )
        )

    if tasks:
        print("=" * 84)
        print(
            f"  [DGP V5.0 GENERATOR] Starte Generierung von {len(tasks)} Universen (N={n_studierende:,}, Seed={population_seed})"
        )
        print(f"  Zielverzeichnis: {scenario_dir}")
        print("=" * 84)
        t0 = time.time()
        summary_results = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_simulate_single_universe_worker, t): t[1]
                for t in tasks
            }
            for future in as_completed(futures):
                u_key = futures[future]
                res = future.result()
                summary_results[u_key] = res
                print(
                    f"  [OK] Universe {u_key} ({res['label']}) fertig in {res['calc_duration_sec']:.1f}s "
                    f"| Dropout: {res['dropout_rate']*100:.2f}% | Grad: {res['grad_rate']*100:.2f}%"
                )

        elapsed = time.time() - t0
        summary_path = scenario_dir / "v5_generation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dgp_version": cfg_v5.get("dgp_version", "5.0"),
                    "n_studierende": n_studierende,
                    "population_seed": population_seed,
                    "elapsed_seconds": round(elapsed, 2),
                    "universes": summary_results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[INFO] V5.0 Generierung abgeschlossen in {elapsed/60:.2f} Min. Summary: {summary_path}")

    # DuckDB Voraggregation für Universe A und Universe B sicherstellen
    for u_key in ["A", "B"]:
        u_dir = scenario_dir / f"universe_{u_key}"
        if not (u_dir / "agg_abschluesse.csv").exists():
            print(f"[AGG] Erzeuge DuckDB-Voraggregation für Universe {u_key} ({u_dir}) ...")
            aggregiere_daten(u_dir, backend="duckdb")

    return scenario_dir


def main():
    parser = argparse.ArgumentParser(
        description="DGP V5.0 Multi-Universe Generator & Full Model Benchmark Runner"
    )
    parser.add_argument(
        "--data_grid_dir",
        type=str,
        default="data_v5_grid",
        help="Basisverzeichnis für die V5.0-Rohdaten (Standard: data_v5_grid)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/S01_baseline_v5",
        help="Ausgabeverzeichnis für alle V5.0-Modellmetriken & Plots",
    )
    parser.add_argument("--n_studierende", type=int, default=50000)
    parser.add_argument("--population_seed", type=int, default=99999)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--skip_gen", action="store_true")
    parser.add_argument("--skip_models", action="store_true")
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=[
            "all",
            "family1_landmark",
            "family2_panel",
            "family3_sequence",
            "family4_autoregressive",
            "family6_causal",
        ],
    )
    args = parser.parse_args()

    out_grid_dir = PROJECT_ROOT / args.data_grid_dir
    scenario_dir = out_grid_dir / "S01_baseline"

    if not args.skip_gen:
        generate_v5_baseline_universes(
            out_grid_dir=out_grid_dir,
            scenario_id="S01_baseline",
            n_studierende=args.n_studierende,
            population_seed=args.population_seed,
            max_workers=args.max_workers,
        )

    if not args.skip_models:
        run_benchmark(
            data_dir=scenario_dir / "universe_A",
            output_dir=PROJECT_ROOT / args.output_dir,
            phase=args.phase,
            temporal="prev",
            seed=42,
        )


if __name__ == "__main__":
    main()
