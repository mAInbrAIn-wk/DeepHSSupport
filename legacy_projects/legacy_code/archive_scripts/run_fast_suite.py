"""
Fast Core Suite Runner (V4.1)
=============================
Fuehrt alle leichten und mittelschweren Modellklassen sowie saemtliche
Kausale Inferenz- und Kontrafaktik-Skripte schnell und automatisiert aus.
Laufzeit pro Szenario: ca. 15-25 Minuten.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import time
import json
import argparse
from pathlib import Path
import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.chdir(SRC_DIR)

class PipelineBenchmarkTracker:
    def __init__(self):
        self.steps = []
        self.process = psutil.Process(os.getpid())

    def run_step(self, step_name: str, func, *args, **kwargs):
        print("\n" + "=" * 80)
        print(f"   START: {step_name}")
        print(f"   Zeit: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        mem_start = self.process.memory_info().rss / (1024 * 1024)
        t0 = time.time()

        result = None
        status = "PASSED"
        err_msg = None

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - t0
            mem_end = self.process.memory_info().rss / (1024 * 1024)
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

        json_path = diag_dir / "fast_suite_benchmark_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "total_duration_s": sum(s["duration_s"] for s in self.steps),
                "steps": self.steps
            }, f, indent=2)

        md_path = diag_dir / "fast_suite_benchmark_report.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# Fast Core Suite Benchmark Report\n\n")
            f.write(f"**Generiert am:** {time.strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**Gesamtlaufzeit:** {sum(s['duration_s'] for s in self.steps)/60:.2f} Minuten\n\n")
            f.write("| Schritt | Status | Dauer (s) | RAM Start (MB) | RAM Ende (MB) | RAM Delta (MB) |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
            for s in self.steps:
                f.write(f"| {s['step_name']} | {s['status']} | {s['duration_s']} | {s['ram_start_mb']} | {s['ram_end_mb']} | {s['ram_delta_mb']:+} |\n")

        print(f"\n[REPORT] Fast Suite Benchmark gespeichert unter: {md_path}")


def run_fast_suite(data_dir: Path, temporal: str = 'prev', modes: list = None, population_seed: int = 42):
    if modes is None:
        modes = ['standard', 'gradeblind']
        
    os.environ['DATA_DIR'] = str(data_dir)
    tracker = PipelineBenchmarkTracker()
    total_t0 = time.time()

    print("*" * 80)
    print("   FAST CORE SUITE RUNNER (V4.1)")
    print(f"   Start: {time.strftime('%Y-%m-%d %H:%M:%S')} | Temporal: {temporal} | Seed: {population_seed}")
    print(f"   Data Dir: {data_dir.resolve()}")
    print(f"   Aktive Modi: {modes}")
    print("*" * 80)

    # 1. MODUS-ABHAENGIGE SCHNELLE MODELLE
    for mode in modes:
        print(f"\n{'='*80}\n   FAST SUITE -> MODUS: {mode.upper()}\n{'='*80}")

        tracker.run_step(f"Extended Cox [{mode}]", lambda m=mode: __import__('extended_cox_survival').train_extended_cox_model(data_dir, temporal, m))
        tracker.run_step(f"Extended DeepSurv [{mode}]", lambda m=mode: __import__('extended_deep_survival').train_extended_deep_survival(data_dir, temporal, m))
        tracker.run_step(f"Recurrent Survival GRU [{mode}]", lambda m=mode: __import__('recurrent_survival_model').train_recurrent_survival_model(data_dir, 16, temporal, m))
        tracker.run_step(f"Dynamic DeepHit Competing Risks [{mode}]", lambda m=mode: __import__('dynamic_deephit_model').train_dynamic_deephit_model(data_dir, 16, temporal, m))
        tracker.run_step(f"Transformer Survival [{mode}]", lambda m=mode: __import__('transformer_survival_model').train_transformer_survival(data_dir, 16, temporal, m))
        tracker.run_step(f"Recurrent Exam Survival GRU [{mode}]", lambda m=mode: __import__('recurrent_exam_survival').train_recurrent_exam_survival_model(data_dir, 40, temporal, m))
        tracker.run_step(f"Transformer Exam Survival [{mode}]", lambda m=mode: __import__('transformer_exam_survival').train_transformer_exam_survival(data_dir, 40, temporal, m))
        tracker.run_step(f"Landmark Baseline Classifiers [{mode}]", lambda m=mode: __import__('train_mlp_baseline').run_baseline_training(data_dir, True, m))
        tracker.run_step(f"Landmark Regression [{mode}]", lambda m=mode: __import__('train_mlp_regression').run_regression_training(data_dir, True, m))
        tracker.run_step(f"DML Orthogonal Survival [{mode}]", lambda m=mode: __import__('dml_orthogonal_survival').train_dml_orthogonal_survival(data_dir, temporal, m))
        tracker.run_step(f"Transformer DML [{mode}]", lambda m=mode: __import__('train_transformer_dml').train_transformer_dml(data_dir, temporal, m))
        tracker.run_step(f"Timeseries Semester LSTM [{mode}]", lambda m=mode: __import__('timeseries_semester').train_timeseries_semester(data_dir, 16, temporal, m))
        tracker.run_step(f"Timeseries Semester Transformer [{mode}]", lambda m=mode: __import__('timeseries_semester_transformer').train_timeseries_semester_transformer(data_dir, 16, temporal, m))
        tracker.run_step(f"Timeseries Exam GRU [{mode}]", lambda m=mode: __import__('timeseries_exam').train_timeseries_exam(data_dir, 40, temporal, m))
        tracker.run_step(f"Timeseries Exam Transformer [{mode}]", lambda m=mode: __import__('timeseries_exam_transformer').train_timeseries_exam_transformer(data_dir, 40, temporal, m))

    # 2. SCHNELLE SPEZIAL- & DIAGNOSE-MODELLE
    print(f"\n{'='*80}\n   FAST SUITE -> SPEZIAL- UND STATISTISCHE DIAGNOSEMODELLE\n{'='*80}")
    tracker.run_step("Oracle Models (Lift Analysis)", lambda: __import__('train_oracle_models').train_oracle_models(data_dir, temporal))
    tracker.run_step("DSGVO Realistic Models", lambda: __import__('train_erwerb_blind_models').train_erwerb_blind_models(data_dir, temporal))
    tracker.run_step("Strukturelle Mediationsanalyse", lambda: __import__('structural_mediation_analysis').run_structural_mediation_analysis(data_dir))
    tracker.run_step("Deep Survival Landmark (LH & DS)", lambda: __import__('deep_survival').train_deep_survival(data_dir=data_dir))
    tracker.run_step("Plot Calibration Curves", lambda: __import__('plot_calibration_curves').main(data_dir=data_dir))
    tracker.run_step("Feature Grid Experiments (Cross-Mode)", lambda: __import__('run_feature_grid_experiments').main(data_dir=data_dir))

    # 3. KONTRAFAKTISCHE INFERENZ-SUITE (VOLLSTAENDIG & AUTARK)
    print(f"\n{'='*80}\n   FAST SUITE -> KONTRAFAKTISCHE INFERENZ-SUITE\n{'='*80}")
    tracker.run_step("Counterfactual HR Analyzer (Extended Cox/Panel)", lambda: __import__('counterfactual_hr_analyzer').analyze_counterfactual_hr(data_dir))
    tracker.run_step("Counterfactual DeepHit Competing Risks", lambda: __import__('counterfactual_deephit_fixed').main(data_dir=data_dir))
    tracker.run_step("Counterfactual Grade Transformer", lambda: __import__('counterfactual_grade_transformer').main(data_dir=data_dir))
    tracker.run_step("Counterfactual Oracle Logistic Hazard", lambda: __import__('counterfactual_oracle_logistic_hazard').main(data_dir=data_dir))

    tracker.export_report(data_dir)
    total_elapsed = time.time() - total_t0
    print("\n" + "=" * 80)
    print(f"   FAST CORE SUITE ERFOLGREICH BEENDET ({total_elapsed/60:.2f} Minuten)")
    print("=" * 80)
    return tracker

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fast Core Suite Runner V4.1")
    parser.add_argument('--data_dir', type=str, default="output_v4_grid_v41/S01_baseline/universe_A")
    parser.add_argument('--temporal', type=str, default='prev', choices=['prev', 'cum'])
    parser.add_argument('--modes', type=str, default='standard,gradeblind', help="Kommagetrennte Liste der Modi")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    mode_list = [m.strip() for m in args.modes.split(',') if m.strip()]
    run_fast_suite(
        data_dir=Path(args.data_dir),
        temporal=args.temporal,
        modes=mode_list,
        population_seed=args.seed
    )
