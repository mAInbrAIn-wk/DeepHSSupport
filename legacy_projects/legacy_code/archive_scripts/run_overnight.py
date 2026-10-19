"""
Master Orchestration: Vollständiger Nachtlauf Pipeline (V3.6)
=============================================================
Führt die gesamte Modell- und Analyselandschaft automatisiert und benchmark-getrackt aus:

1. Simulation V3 (5-8 Universen mit Clipping-Diagnostik & Seed-Salting)
2. 3-Way Backbone Aggregation (DuckDB / NumPy / Pandas)
3. Wahre Makro Ground-Truth Effekte
4. Modell-Trainings über alle 8 Modellklassen (25+ Modelle) via `feature_builder.py`
5. Kausale & Kontrafaktische Dual-Strang Evaluation (Partiell vs. Isoliert)
6. Feature-Grid Sweep (Standard, Gradeblind, Blind, Realistic, Oracle)
7. Synoptischer Gesamt-Report (JSON + Markdown) mit psutil CPU- & RAM-Tracking
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import time
import json
import argparse
from pathlib import Path
import psutil

# Projekt-Pfade einbinden
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.chdir(SRC_DIR)

# Metrics & Feature Builder
import feature_builder as fb
from metrics_logger import save_metrics


class PipelineBenchmarkTracker:
    """Trackt Laufzeiten, CPU-Auslastung und RAM-Verbrauch pro Pipeline-Schritt."""
    def __init__(self):
        self.steps = []
        self.process = psutil.Process(os.getpid())

    def run_step(self, step_name: str, func, *args, **kwargs):
        print("\n" + "=" * 80)
        print(f"   START: {step_name}")
        print(f"   Zeit: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Baseline Messung
        mem_start = self.process.memory_info().rss / (1024 * 1024)
        t0 = time.time()
        cpu_t0 = psutil.cpu_percent(interval=None)

        result = None
        status = "PASSED"
        err_msg = None

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - t0
            mem_end = self.process.memory_info().rss / (1024 * 1024)
            cpu_t1 = psutil.cpu_percent(interval=None)
            print(f"\n[OK] {step_name} ERFOLGREICH BEENDET ({elapsed/60:.2f} Min. / {elapsed:.1f}s | RAM: {mem_end:.1f}MB)")
        except Exception as e:
            elapsed = time.time() - t0
            mem_end = self.process.memory_info().rss / (1024 * 1024)
            status = "FAILED"
            err_msg = str(e)
            print(f"\n[FEHLER] {step_name} FEHLGESCHLAGEN nach {elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()

        self.steps.append({
            "step_name": step_name,
            "status": status,
            "duration_s": round(elapsed, 2),
            "ram_start_mb": round(mem_start, 1),
            "ram_end_mb": round(mem_end, 1),
            "ram_delta_mb": round(mem_end - mem_start, 1),
            "error": err_msg
        })
        return result

    def export_report(self, output_dir: Path):
        diag_dir = output_dir / "diagnostics"
        diag_dir.mkdir(exist_ok=True, parents=True)

        json_path = diag_dir / "pipeline_benchmark_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "total_duration_s": sum(s["duration_s"] for s in self.steps),
                "steps": self.steps
            }, f, indent=2)

        md_path = diag_dir / "pipeline_benchmark_report.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# Pipeline Benchmark & Execution Report (V3.6)\n\n")
            f.write(f"**Generiert am:** {time.strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**Gesamtlaufzeit:** {sum(s['duration_s'] for s in self.steps)/60:.2f} Minuten\n\n")
            f.write("| Schritt | Status | Dauer (s) | RAM Start (MB) | RAM Ende (MB) | RAM Delta (MB) |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
            for s in self.steps:
                f.write(f"| {s['step_name']} | {s['status']} | {s['duration_s']} | {s['ram_start_mb']} | {s['ram_end_mb']} | {s['ram_delta_mb']:+} |\n")

        print(f"\n[REPORT] Pipeline-Benchmark-Report gespeichert unter: {md_path}")


def run_master_overnight_pipeline(data_dir: Path = None,
                                  skip_sim: bool = False,
                                  temporal: str = 'prev',
                                  population_seed: int = 12345):
    if data_dir is None or str(data_dir) in ('src/output_dl', 'output_dl'):
        data_dir = Path('output_dl')
    else:
        data_dir = Path(data_dir)

    tracker = PipelineBenchmarkTracker()
    total_t0 = time.time()

    print("*" * 80)
    print("   MASTER NACHTLAUF PIPELINE V3.6 (DEEPSUPPORT)")
    print(f"   Start: {time.strftime('%Y-%m-%d %H:%M:%S')} | Temporal: {temporal} | Seed: {population_seed}")
    print(f"   Data Dir: {data_dir.resolve()}")
    print("*" * 80)

    # -------------------------------------------------------------------------
    # 0. SIMULATION V3 & AGGREGATION
    # -------------------------------------------------------------------------
    if not skip_sim:
        def step_sim():
            import simulation_v3
            simulation_v3.main(population_seed=population_seed, base_output_override=data_dir)
        tracker.run_step("0. Simulation V3 (8 Universen, Konsistent)", step_sim)
    else:
        print("\n[INFO] Simulation übersprungen (--skip_sim). Nutze bestehende Daten.")

    # -------------------------------------------------------------------------
    # 1. TRADITIONELLE SURVIVAL- UND HAZARD-MODELLE (KLASSE 5)
    # -------------------------------------------------------------------------
    from extended_cox_survival import train_extended_cox_model
    tracker.run_step("1. Extended Cox Proportional Hazards (Statsmodels PHReg)", train_extended_cox_model, data_dir, temporal, 'standard')

    from extended_deep_survival import train_extended_deep_survival
    tracker.run_step("2. Extended DeepSurv & Logistic Hazard (Panel Breslow)", train_extended_deep_survival, data_dir, temporal, 'standard')

    # -------------------------------------------------------------------------
    # 2. REKURRIERENDE SURVIVAL- UND COMPETING-RISKS-MODELLE (KLASSE 6)
    # -------------------------------------------------------------------------
    from recurrent_survival_model import train_recurrent_survival_model
    tracker.run_step("3. Recurrent Semester Survival GRU", train_recurrent_survival_model, data_dir, 16, temporal, 'standard')

    from dynamic_deephit_model import train_dynamic_deephit_model
    tracker.run_step("4. Dynamic DeepHit Competing Risks (Dropout & Abschluss)", train_dynamic_deephit_model, data_dir, 16, temporal, 'standard')

    from transformer_survival_model import train_transformer_survival
    tracker.run_step("5. Causal Semester Transformer Survival", train_transformer_survival, data_dir, 16, temporal, 'standard')

    # -------------------------------------------------------------------------
    # 3. EXAM-LEVEL SEQUENZ- UND SURVIVAL-MODELLE (KLASSE 7)
    # -------------------------------------------------------------------------
    from recurrent_exam_survival import train_recurrent_exam_survival_model
    tracker.run_step("6. Recurrent Exam Survival GRU", train_recurrent_exam_survival_model, data_dir, 40, temporal, 'standard')

    from transformer_exam_survival import train_transformer_exam_survival
    tracker.run_step("7. Causal Exam Transformer Survival", train_transformer_exam_survival, data_dir, 40, temporal, 'standard')

    # -------------------------------------------------------------------------
    # 4. LANDMARK BASELINES & REGRESSIONEN (KLASSE 1)
    # -------------------------------------------------------------------------
    from train_mlp_baseline import run_baseline_training
    tracker.run_step("8. Landmark Baseline Classifiers (RF, SVM, NaiveBayes, MLP)", run_baseline_training, data_dir, True, 'standard')

    from train_mlp_regression import run_regression_training
    tracker.run_step("9. Landmark Abschlussnoten-Regression (Ridge, SVR, RF, MLP)", run_regression_training, data_dir, True, 'standard')

    # -------------------------------------------------------------------------
    # 5. KAUSALE DML- UND TRANSFOMER-DML MODELLE (KLASSE 3)
    # -------------------------------------------------------------------------
    from dml_orthogonal_survival import train_dml_orthogonal_survival
    tracker.run_step("10. Double Machine Learning (DML Orthogonalized Survival)", train_dml_orthogonal_survival, data_dir, temporal, 'standard')

    from train_transformer_dml import train_transformer_dml
    tracker.run_step("11. Deep Transformer-DML Pipeline", train_transformer_dml, data_dir, temporal, 'standard')

    # -------------------------------------------------------------------------
    # 6. ZEITREIHEN-REGRESSIONEN (KLASSE 6 & 7 REGRESSION)
    # -------------------------------------------------------------------------
    from timeseries_semester import train_timeseries_semester
    tracker.run_step("12. Semester Timeseries LSTM GPA Regression", train_timeseries_semester, data_dir, 16, temporal, 'standard')

    from timeseries_semester_transformer import train_timeseries_semester_transformer
    tracker.run_step("13. Semester Timeseries Transformer Abschlussnoten-Regression", train_timeseries_semester_transformer, data_dir, 16, temporal, 'standard')

    from timeseries_exam import train_timeseries_exam
    tracker.run_step("14. Exam Timeseries GRU Grade Regression", train_timeseries_exam, data_dir, 40, temporal, 'standard')

    from timeseries_exam_transformer import train_timeseries_exam_transformer
    tracker.run_step("15. Exam Timeseries Transformer Grade Regression", train_timeseries_exam_transformer, data_dir, 40, temporal, 'standard')

    # -------------------------------------------------------------------------
    # 7. ORACLE & DSGVO REALISTIC BENCHMARKS (KLASSE 2)
    # -------------------------------------------------------------------------
    from train_oracle_models import train_oracle_models
    tracker.run_step("16. Oracle Models (Theoretischer Maximum Lift)", train_oracle_models, data_dir, temporal)

    from train_erwerb_blind_models import train_erwerb_blind_models
    tracker.run_step("17. DSGVO Realistic Models (Feature Blindness Analysis)", train_erwerb_blind_models, data_dir, temporal)

    # -------------------------------------------------------------------------
    # 8. DEEP TRANSFORMER SUITE (KLASSE 8)
    # -------------------------------------------------------------------------
    from deep_transformer_regression import train_deep_transformer_models
    tracker.run_step("18. Deep Transformer Suite (Enlarged Capacity)", train_deep_transformer_models, data_dir, temporal, 'standard')

    # -------------------------------------------------------------------------
    # 9. AUTOREGRESSIVE NEXT-EXAM & STRUCTURAL MEDIATION (KLASSE 8B / AP7 / AP8)
    # -------------------------------------------------------------------------
    from autoregressive_next_exam import train_autoregressive_next_exam
    tracker.run_step("19. Autoregressive Next-Exam Prediction (Dual-Head Multi-Task)", train_autoregressive_next_exam, data_dir)

    from structural_mediation_analysis import run_structural_mediation_analysis
    tracker.run_step("20. Strukturelle Mediationsanalyse (Imai / Pearl Framework)", run_structural_mediation_analysis, data_dir)

    # Benchmark-Report exportieren
    tracker.export_report(data_dir)

    total_elapsed = time.time() - total_t0
    print("\n" + "=" * 80)
    print(f"   MASTER NACHTLAUF ERFOLGREICH BEENDET ({total_elapsed/60:.2f} Minuten)")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Master Overnight Pipeline V3.6")
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--skip_sim', action='store_true', default=False)
    parser.add_argument('--temporal', type=str, default='prev', choices=['prev', 'cum'])
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    run_master_overnight_pipeline(
        data_dir=Path(args.data_dir) if args.data_dir else None,
        skip_sim=args.skip_sim,
        temporal=args.temporal,
        population_seed=args.seed
    )
