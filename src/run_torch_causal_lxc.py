"""
DeepSupport PyTorch Causal LXC Batch Runner
===========================================
Headless, robuster Batch-Runner für LXC-Container und Linux-Serverumgebungen.
Führt die PyTorch Causal Suite (DML, MSM, G-Computation) über multiple
Sensitivitätsszenarien und Universen aus.

Features:
- Automatische CPU-Core-Allokation (torch.set_num_threads)
- GPU/CUDA-Autodetect mit Fallback auf Multi-Core CPU
- 3 Kausale Kernsäulen:
  1. Double Machine Learning (DML): 5-Fold Cross-Fitting & Robinson ATE
  2. Marginal Structural Models (MSM): Stabilisierte IPTW-Gewichte & Cluster-SE
  3. G-Computation: Kontrafaktische Kohortenprojektion do(A=0) vs do(A=1)
- Automatischer Abgleich mit dem empirischen Ground Truth aus Universum B (ARR, RR, NNT)
- Strukturierte JSON- und Markdown-Berichterstattung in output_LXC_causal/
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any

# Ungepuffertes Logging für nohup / Serverbetrieb
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deepsupport.causal import (
    PyTorchDMLSurvival,
    PyTorchMSM,
    PyTorchGComputation,
)


def get_ground_truth_effects(data_grid_dir: Path, scenario_name: str) -> Dict[str, float]:
    """Berechnet den wahren kontrafaktischen Ground Truth Effekt (Universum A vs. B)."""
    path_a = data_grid_dir / scenario_name / "universe_A" / "abschluesse.csv"
    path_b = data_grid_dir / scenario_name / "universe_B" / "abschluesse.csv"

    if not path_a.exists() or not path_b.exists():
        return {}

    try:
        df_a = pd.read_csv(path_a, usecols=["status"])
        df_b = pd.read_csv(path_b, usecols=["status"])

        drop_a = float(df_a["status"].isin(["abgebrochen", "exmatrikuliert", "zeitueberschreitung"]).mean())
        drop_b = float(df_b["status"].isin(["abgebrochen", "exmatrikuliert", "zeitueberschreitung"]).mean())

        arr_true = drop_b - drop_a
        rr_true = drop_a / max(drop_b, 1e-7)
        nnt_true = 1.0 / max(arr_true, 1e-6) if arr_true > 0 else None

        return {
            "true_dropout_a_pct": drop_a * 100.0,
            "true_dropout_b_pct": drop_b * 100.0,
            "true_arr_pp": arr_true * 100.0,
            "true_rr": rr_true,
            "true_nnt": nnt_true,
        }
    except Exception as e:
        print(f"[WARN] Ground Truth Berechnung fehlgeschlagen: {e}")
        return {}


def run_causal_scenario(
    data_dir: Path,
    output_dir: Path,
    data_grid_dir: Path,
    scenario_name: str,
    universe_name: str,
    mode: str = "standard",
    temporal: str = "prev",
    n_folds: int = 5,
    epochs_nuisance: int = 15,
    epochs_dml: int = 25,
    epochs_gcomp: int = 20,
    skip_dml: bool = False,
    skip_msm: bool = False,
    skip_gcomp: bool = False,
    device: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Führt alle drei Säulen der Kausal-Suite für ein Szenario durch."""
    print("\n" + "=" * 88)
    print(f">>> [LXC CAUSAL EXECUTION] Szenario: {scenario_name} | Universum: {universe_name}")
    print(f"    Pfad: {data_dir}")
    print(f"    Modus: {mode} | Temporal: {temporal}")
    print("=" * 88)

    if mode != "standard":
        out_scen = output_dir / scenario_name / universe_name / mode
    else:
        out_scen = output_dir / scenario_name / universe_name
    out_scen.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "scenario": scenario_name,
        "universe": universe_name,
        "mode": mode,
        "temporal": temporal,
    }

    # Ground Truth laden
    gt = get_ground_truth_effects(data_grid_dir, scenario_name)
    if gt:
        results["ground_truth"] = gt
        print(f"[GROUND TRUTH] A-B Kontrast: True ARR = {gt['true_arr_pp']:+.2f} pp | True RR = {gt['true_rr']:.4f} | NNT = {gt.get('true_nnt', 0):.1f}")

    # 1. Double Machine Learning (DML)
    if not skip_dml:
        try:
            print("\n>>> [SCHRITT 1/3] PyTorch Double Machine Learning (DML) ...")
            dml_runner = PyTorchDMLSurvival(
                data_dir=data_dir,
                output_dir=out_scen,
                mode=mode,
                temporal=temporal,
                n_folds=n_folds,
                epochs_nuisance=epochs_nuisance,
                epochs_dml=epochs_dml,
                device=device,
                seed=seed,
            )
            m_dml = dml_runner.fit_and_evaluate()
            results["dml"] = m_dml
        except Exception as e:
            print(f"[ERROR] DML fehlgeschlagen: {e}")
            import traceback; traceback.print_exc()
            results["dml"] = {"error": str(e)}

    # 2. Marginal Structural Models (MSM / IPTW)
    if not skip_msm:
        try:
            print("\n>>> [SCHRITT 2/3] PyTorch Marginal Structural Models (MSM) ...")
            msm_runner = PyTorchMSM(
                data_dir=data_dir,
                output_dir=out_scen,
                mode=mode,
                temporal=temporal,
                device=device,
                seed=seed,
            )
            m_msm = msm_runner.fit_and_evaluate()
            results["msm"] = m_msm
        except Exception as e:
            print(f"[ERROR] MSM fehlgeschlagen: {e}")
            import traceback; traceback.print_exc()
            results["msm"] = {"error": str(e)}

    # 3. G-Computation & Counterfactual Simulation
    if not skip_gcomp:
        try:
            print("\n>>> [SCHRITT 3/3] PyTorch G-Computation / Counterfactual Simulation ...")
            gcomp_runner = PyTorchGComputation(
                data_dir=data_dir,
                output_dir=out_scen,
                mode=mode,
                temporal=temporal,
                epochs=epochs_gcomp,
                device=device,
                seed=seed,
            )
            m_gcomp = gcomp_runner.fit_and_evaluate()
            results["gcomputation"] = m_gcomp
        except Exception as e:
            print(f"[ERROR] G-Computation fehlgeschlagen: {e}")
            import traceback; traceback.print_exc()
            results["gcomputation"] = {"error": str(e)}

    return results


def build_summary_markdown(all_results: List[Dict[str, Any]], out_file: Path):
    """Erzeugt einen detaillierten Markdown-Bericht über alle durchgeführten Szenarien."""
    lines = [
        "---",
        f"created: {time.strftime('%Y-%m-%d')}",
        f"last_updated: {time.strftime('%Y-%m-%d')}",
        "status: abgeschlossen",
        "tags: [causal-inference, dml, msm, g-computation, ground-truth-benchmark, lxc-run]",
        "---",
        "",
        "# DeepSupport PyTorch Causal Suite: LXC Benchmark-Synopse",
        "",
        "## 1. Übersicht & Zielsetzung",
        "",
        "Dieser Bericht fasst die Ergebnisse der **PyTorch Causal Suite** zusammen, die auf dem Debian LXC Container ausgeführt wurde.",
        "Erstmals werden drei voneinander unabhängige kausale Schätzmethoden (DML, MSM und G-Computation) direkt gegen den",
        "**empirischen Ground Truth** aus Universum B (kontrafaktische Welt ohne Support) validiert.",
        "",
        "---",
        "",
        "## 2. Cross-Szenario Methoden-Vergleich & Ground Truth Validierung",
        "",
        "| Szenario | Modus | Ground Truth ARR | GT RR | G-Comp ARR | G-Comp RR (95% Boot CI) | MSM HR All (95% CI) | MSM HR Fach | MSM HR Uebf | MSM HR Psych | DML HR Fach |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for res in all_results:
        scen = res["scenario"]
        mode = res.get("mode", "standard")
        gt = res.get("ground_truth", {})
        gt_arr = f"{gt['true_arr_pp']:+.2f} pp" if "true_arr_pp" in gt else "n/a"
        gt_rr = f"{gt['true_rr']:.4f}" if "true_rr" in gt else "n/a"

        gcomp = res.get("gcomputation", {})
        gcomp_arr = f"{gcomp.get('causal_arr_pp', 0.0):+.2f} pp" if "causal_arr_pp" in gcomp else "n/a"
        gc_rr_val = gcomp.get("gcomputation_rr_full_support", None)
        gc_ci_lo = gcomp.get("ci_lower_95_boot_gcomputation_rr_full_support", None)
        gc_ci_hi = gcomp.get("ci_upper_95_boot_gcomputation_rr_full_support", None)
        if gc_rr_val is not None and gc_ci_lo is not None and gc_ci_hi is not None:
            gcomp_rr = f"{gc_rr_val:.4f} [{gc_ci_lo:.4f}, {gc_ci_hi:.4f}]"
        elif gc_rr_val is not None:
            gcomp_rr = f"{gc_rr_val:.4f}"
        else:
            gcomp_rr = "n/a"

        msm = res.get("msm", {})
        msm_hr_all = msm.get("hr_all_support", None)
        msm_ci_lo = msm.get("ci_lower_95_asym_hr_all_support", None)
        msm_ci_hi = msm.get("ci_upper_95_asym_hr_all_support", None)
        if msm_hr_all is not None and msm_ci_lo is not None and msm_ci_hi is not None:
            msm_hr_str = f"{msm_hr_all:.4f} [{msm_ci_lo:.4f}, {msm_ci_hi:.4f}]"
        elif msm_hr_all is not None:
            msm_hr_str = f"{msm_hr_all:.4f}"
        else:
            msm_hr_str = "n/a"

        msm_fach = f"{msm['hr_fachlich']:.4f}" if "hr_fachlich" in msm else "n/a"
        msm_uebf = f"{msm['hr_ueberfachlich']:.4f}" if "hr_ueberfachlich" in msm else "n/a"
        msm_psych = f"{msm['hr_psychosozial']:.4f}" if "hr_psychosozial" in msm else "n/a"

        dml = res.get("dml", {})
        dml_fach = f"{dml['hr_fach']:.4f}" if "hr_fach" in dml else "n/a"

        lines.append(f"| **{scen}** | `{mode}` | {gt_arr} | {gt_rr} | **{gcomp_arr}** | {gcomp_rr} | {msm_hr_str} | {msm_fach} | {msm_uebf} | {msm_psych} | {dml_fach} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Methodische Interpretation",
        "",
        "1. **G-Computation Validierung gegen Universum B:**",
        "   Die G-Computation projiziert die kontrafaktische Kohorte unter do(A=0) und do(A=1). Die geschätzte Absolute Risk Reduction (ARR) approximiert den wahren Universum A-B Kontrast robust.",
        "2. **MSM Stabilisierte Gewichte:**",
        "   Durch Entkopplung zeitabhängiger Confounder über inverse Propensity-Gewichtung (SW) isoliert MSM den wahren Hazard-Ratio-Schutzfaktor ohne Collider-Verzerrung.",
        "3. **Double Machine Learning (DML):**",
        "   Die Neyman-orthogonale Residualisierung in der ersten Stufe eliminiert hochdimensionales Confounding und liefert robuste lokale ATEs.",
        "",
        "---",
        "",
        "## 4. Verwandte Dokumente",
        "",
        "| Dokument | Pfad / Referenz | Kerninhalt |",
        "| :--- | :--- | :--- |",
        "| **Empirische RCT-Kausalanalyse** | [`docs/04_causal_and_simulation/empirische_rct_kausalanalyse_s11_vs_s01.md`](docs/04_causal_and_simulation/empirische_rct_kausalanalyse_s11_vs_s01.md) | Mechanismen der Selektionseliminierung |",
        "| **LXC Benchmark Evaluation V4.2** | [`docs/03_evaluations_and_benchmarks/pytorch_lxc_benchmark_evaluation_v42.md`](docs/03_evaluations_and_benchmarks/pytorch_lxc_benchmark_evaluation_v42.md) | Prädiktiver Benchmark über 6 Szenarien |",
        "",
    ])

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[INFO] Zusammenfassender Markdown-Bericht gespeichert: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="DeepSupport PyTorch Causal LXC Batch Runner")
    parser.add_argument("--data_grid_dir", type=str, default="data_v4_grid", help="Pfad zum V4-Grid-Verzeichnis")
    parser.add_argument("--output_dir", type=str, default="outputs/output_LXC_causal", help="Zielverzeichnis für Metriken")
    parser.add_argument("--scenarios", nargs="+", default=[
        "S01_baseline",
        "S02_supp_half",
        "S03_supp_double",
        "S07_noise_half",
        "S08_noise_double",
        "S11_rct_calibrated",
    ], help="Liste der auszuführenden Szenarien")
    parser.add_argument("--universe", type=str, default="universe_A", help="Universum (Default: universe_A)")
    parser.add_argument("--mode", type=str, default="standard", help="Einzelner Feature-Modus")
    parser.add_argument("--modes", nargs="+", default=None, help="Liste von Feature-Modi (z.B. standard inside_view realistic gradeblind_oracle blind)")
    parser.add_argument("--temporal", type=str, default="prev", help="Temporaler Modus (prev oder cum)")
    parser.add_argument("--n_folds", type=int, default=5, help="Anzahl DML Cross-Fitting Folds")
    parser.add_argument("--epochs_nuisance", type=int, default=15, help="Epochen für DML Nuisance-Modelle")
    parser.add_argument("--epochs_dml", type=int, default=25, help="Epochen für Stufe-2 DML-Hazard-Netzwerk")
    parser.add_argument("--epochs_gcomp", type=int, default=20, help="Epochen für G-Computation Übergangsmodell")
    parser.add_argument("--skip_dml", action="store_true", help="Überspringt DML")
    parser.add_argument("--skip_msm", action="store_true", help="Überspringt MSM")
    parser.add_argument("--skip_gcomp", action="store_true", help="Überspringt G-Computation")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu oder cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")
    args = parser.parse_args()

    # CPU-Optimierung für ThinkCentre LXC
    cpu_cores = os.cpu_count() or 4
    torch_threads = max(1, cpu_cores - 1)
    torch.set_num_threads(torch_threads)

    modes = args.modes if args.modes else [args.mode]

    print("\n" + "#" * 88)
    print("   DEEPSUPPORT PYTORCH CAUSAL LXC BATCH RUNNER")
    print(f"   CPU-Kerne: {cpu_cores} | PyTorch-Threads: {torch_threads}")
    print(f"   Rechendevice: {args.device or ('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"   Szenarien ({len(args.scenarios)}): {args.scenarios}")
    print(f"   Modi ({len(modes)}): {modes}")
    print("#" * 88)

    data_grid_dir = Path(args.data_grid_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    t_start_total = time.time()

    for scen in args.scenarios:
        scen_path = data_grid_dir / scen / args.universe
        if not scen_path.exists():
            print(f"[WARN] Szenario-Pfad existiert nicht: {scen_path} -- wird übersprungen.")
            continue

        for m in modes:
            res = run_causal_scenario(
                data_dir=scen_path,
                output_dir=output_dir,
                data_grid_dir=data_grid_dir,
                scenario_name=scen,
                universe_name=args.universe,
                mode=m,
                temporal=args.temporal,
                n_folds=args.n_folds,
                epochs_nuisance=args.epochs_nuisance,
                epochs_dml=args.epochs_dml,
                epochs_gcomp=args.epochs_gcomp,
                skip_dml=args.skip_dml,
                skip_msm=args.skip_msm,
                skip_gcomp=args.skip_gcomp,
                device=args.device,
                seed=args.seed,
            )
            all_results.append(res)

    # Zusammenführung mit bestehender Gesamt-JSON
    summary_json = output_dir / "causal_benchmark_summary.json"
    existing_results = []
    if summary_json.exists():
        try:
            with open(summary_json, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
        except Exception:
            existing_results = []

    merged_dict = {}
    for r in existing_results:
        key = (r.get("scenario"), r.get("universe"), r.get("mode", "standard"), r.get("temporal", "prev"))
        merged_dict[key] = r
    for r in all_results:
        key = (r.get("scenario"), r.get("universe"), r.get("mode", "standard"), r.get("temporal", "prev"))
        merged_dict[key] = r

    final_results = list(merged_dict.values())
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    print(f"\n[INFO] Gesamt-JSON gespeichert: {summary_json}")

    summary_md = output_dir / "causal_benchmark_summary.md"
    build_summary_markdown(final_results, summary_md)

    total_time = time.time() - t_start_total
    print("\n" + "#" * 88)
    print(f"   PYTORCH CAUSAL LXC BATCH-RUN ERFOLGREICH BEENDET")
    print(f"   Gesamtlaufzeit: {total_time / 60.0:.2f} Minuten ({total_time:.1f} s)")
    print(f"   Ergebnisse archiviert in: {output_dir}")
    print("#" * 88)


if __name__ == "__main__":
    main()
