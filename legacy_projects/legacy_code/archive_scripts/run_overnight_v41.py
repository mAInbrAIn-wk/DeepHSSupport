"""
Master Orchestration: Extended Overnight Runner Pipeline (V4.1)
===============================================================
Führt die gesamte Modell- und Analyselandschaft automatisiert und benchmark-getrackt aus.
Erweitert um alle Modelle auf V4.1 Daten mit allen 5 Feature-Modi (Standard, Gradeblind, Blind, Oracle, Realistic).
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

        json_path = diag_dir / "pipeline_benchmark_report_v41.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "total_duration_s": sum(s["duration_s"] for s in self.steps),
                "steps": self.steps
            }, f, indent=2)

        md_path = diag_dir / "pipeline_benchmark_report_v41.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# Pipeline Benchmark & Execution Report (V4.1)\n\n")
            f.write(f"**Generiert am:** {time.strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**Gesamtlaufzeit:** {sum(s['duration_s'] for s in self.steps)/60:.2f} Minuten\n\n")
            f.write("| Schritt | Status | Dauer (s) | RAM Start (MB) | RAM Ende (MB) | RAM Delta (MB) |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
            for s in self.steps:
                f.write(f"| {s['step_name']} | {s['status']} | {s['duration_s']} | {s['ram_start_mb']} | {s['ram_end_mb']} | {s['ram_delta_mb']:+} |\n")

        print(f"\n[REPORT] Pipeline-Benchmark-Report gespeichert unter: {md_path}")


def run_master_overnight_pipeline_v41(data_dir: Path, temporal: str, population_seed: int):
    # Setup Env
    os.environ['DATA_DIR'] = str(data_dir)
    
    tracker = PipelineBenchmarkTracker()
    total_t0 = time.time()

    print("*" * 80)
    print("   MASTER NACHTLAUF PIPELINE V4.1 (EXTENDED)")
    print(f"   Start: {time.strftime('%Y-%m-%d %H:%M:%S')} | Temporal: {temporal} | Seed: {population_seed}")
    print(f"   Data Dir: {data_dir.resolve()}")
    print("*" * 80)

    MODES = ['standard', 'gradeblind', 'blind', 'oracle', 'realistic']

    # -------------------------------------------------------------------------
    # MODES LOOP (1-15 & 18 von run_overnight.py, die 'mode' unterstützen)
    # -------------------------------------------------------------------------
    for mode in MODES:
        print(f"\n{'='*80}\n   STARTE MODUS: {mode.upper()}\n{'='*80}")
        
        def step_1(m=mode):
            from extended_cox_survival import train_extended_cox_model
            return train_extended_cox_model(data_dir, temporal, m)
        tracker.run_step(f"1. Extended Cox [{mode}]", step_1)

        def step_2(m=mode):
            from extended_deep_survival import train_extended_deep_survival
            return train_extended_deep_survival(data_dir, temporal, m)
        tracker.run_step(f"2. Extended DeepSurv [{mode}]", step_2)

        def step_3(m=mode):
            from recurrent_survival_model import train_recurrent_survival_model
            return train_recurrent_survival_model(data_dir, 16, temporal, m)
        tracker.run_step(f"3. Recurrent Survival GRU [{mode}]", step_3)

        def step_4(m=mode):
            from dynamic_deephit_model import train_dynamic_deephit_model
            return train_dynamic_deephit_model(data_dir, 16, temporal, m)
        tracker.run_step(f"4. Dynamic DeepHit Competing Risks [{mode}]", step_4)

        def step_5(m=mode):
            from transformer_survival_model import train_transformer_survival
            return train_transformer_survival(data_dir, 16, temporal, m)
        tracker.run_step(f"5. Transformer Survival [{mode}]", step_5)

        def step_6(m=mode):
            from recurrent_exam_survival import train_recurrent_exam_survival_model
            return train_recurrent_exam_survival_model(data_dir, 40, temporal, m)
        tracker.run_step(f"6. Recurrent Exam Survival GRU [{mode}]", step_6)

        def step_7(m=mode):
            from transformer_exam_survival import train_transformer_exam_survival
            return train_transformer_exam_survival(data_dir, 40, temporal, m)
        tracker.run_step(f"7. Transformer Exam Survival [{mode}]", step_7)

        def step_8(m=mode):
            from train_mlp_baseline import run_baseline_training
            return run_baseline_training(data_dir, True, m)
        tracker.run_step(f"8. Landmark Baseline [{mode}]", step_8)

        def step_9(m=mode):
            from train_mlp_regression import run_regression_training
            return run_regression_training(data_dir, True, m)
        tracker.run_step(f"9. Landmark Regression [{mode}]", step_9)

        def step_10(m=mode):
            from dml_orthogonal_survival import train_dml_orthogonal_survival
            return train_dml_orthogonal_survival(data_dir, temporal, m)
        tracker.run_step(f"10. DML Orthogonal Survival [{mode}]", step_10)

        def step_11(m=mode):
            from train_transformer_dml import train_transformer_dml
            return train_transformer_dml(data_dir, temporal, m)
        tracker.run_step(f"11. Transformer DML [{mode}]", step_11)

        def step_12(m=mode):
            from timeseries_semester import train_timeseries_semester
            return train_timeseries_semester(data_dir, 16, temporal, m)
        tracker.run_step(f"12. Timeseries Semester LSTM [{mode}]", step_12)

        def step_13(m=mode):
            from timeseries_semester_transformer import train_timeseries_semester_transformer
            return train_timeseries_semester_transformer(data_dir, 16, temporal, m)
        tracker.run_step(f"13. Timeseries Semester Transformer [{mode}]", step_13)

        def step_14(m=mode):
            from timeseries_exam import train_timeseries_exam
            return train_timeseries_exam(data_dir, 40, temporal, m)
        tracker.run_step(f"14. Timeseries Exam GRU [{mode}]", step_14)

        def step_15(m=mode):
            from timeseries_exam_transformer import train_timeseries_exam_transformer
            return train_timeseries_exam_transformer(data_dir, 40, temporal, m)
        tracker.run_step(f"15. Timeseries Exam Transformer [{mode}]", step_15)

        def step_18(m=mode):
            from deep_transformer_regression import train_deep_transformer_models
            return train_deep_transformer_models(data_dir, temporal, m)
        tracker.run_step(f"18. Deep Transformer Suite [{mode}]", step_18)


    # -------------------------------------------------------------------------
    # MODE-INDEPENDENT MODELLE (Rest von run_overnight.py)
    # -------------------------------------------------------------------------
    print(f"\n{'='*80}\n   STARTE MODE-INDEPENDENT MODELLE\n{'='*80}")

    def step_16():
        from train_oracle_models import train_oracle_models
        return train_oracle_models(data_dir, temporal)
    tracker.run_step("16. Oracle Models", step_16)

    def step_17():
        from train_erwerb_blind_models import train_erwerb_blind_models
        return train_erwerb_blind_models(data_dir, temporal)
    tracker.run_step("17. Erwerb Blind Models", step_17)

    def step_19():
        from autoregressive_next_exam import train_autoregressive_next_exam
        return train_autoregressive_next_exam(data_dir)
    tracker.run_step("19. Autoregressive Next-Exam", step_19)

    def step_20():
        from structural_mediation_analysis import run_structural_mediation_analysis
        return run_structural_mediation_analysis(data_dir)
    tracker.run_step("20. Structural Mediation Analysis", step_20)

    # -------------------------------------------------------------------------
    # NEUE MODELLE (VON run_all_experiments.py & Prompt)
    # -------------------------------------------------------------------------
    def step_21():
        from deep_survival import train_deep_survival
        return train_deep_survival(data_dir=data_dir)
    tracker.run_step("21. Deep Survival", step_21)

    def step_22():
        from dynamic_deephit_delta_model import train_dynamic_deephit_delta_model
        return train_dynamic_deephit_delta_model(data_dir, temporal=temporal)
    tracker.run_step("22. Dynamic DeepHit Delta Model", step_22)

    def step_23():
        from extended_deep_survival_delta import train_extended_deep_survival_delta
        return train_extended_deep_survival_delta(data_dir, temporal=temporal)
    tracker.run_step("23. Extended Deep Survival Delta", step_23)

    def step_24():
        from extended_exam_survival import train_extended_exam_survival
        return train_extended_exam_survival(data_dir, temporal=temporal)
    tracker.run_step("24. Extended Exam Survival", step_24)

    def step_25():
        from extended_cox_delta import build_delta_panel, fit_extended_cox_delta
        panel = build_delta_panel(data_dir)
        return fit_extended_cox_delta(panel, data_dir)
    tracker.run_step("25. Extended Cox Delta", step_25)

    def step_26():
        from recurrent_survival_model_delta import train_recurrent_survival_model_delta
        return train_recurrent_survival_model_delta(data_dir, temporal=temporal)
    tracker.run_step("26. Recurrent Survival Model Delta", step_26)

    def step_27():
        from recurrent_exam_survival_delta import train_recurrent_exam_survival_delta
        return train_recurrent_exam_survival_delta(data_dir, temporal=temporal)
    tracker.run_step("27. Recurrent Exam Survival Delta", step_27)

    def step_28():
        from recurrent_exam_survival_v2 import train_recurrent_exam_survival_v2
        return train_recurrent_exam_survival_v2(data_dir, temporal=temporal)
    tracker.run_step("28. Recurrent Exam Survival V2", step_28)

    def step_29():
        from autoregressive_deep_transformer import train_autoregressive_deep_transformer
        return train_autoregressive_deep_transformer(data_dir)
    tracker.run_step("29. Autoregressive Deep Transformer", step_29)

    def step_30():
        from landmark_prediction import main as run_landmark
        return run_landmark(data_dir=data_dir)
    tracker.run_step("30. Landmark Prediction", step_30)

    def step_31():
        from plot_calibration_curves import main as plot_calibration_curves
        return plot_calibration_curves(data_dir=data_dir)
    tracker.run_step("31. Plot Calibration Curves", step_31)

    # -------------------------------------------------------------------------
    # FEATURE GRID & COUNTERFACTUAL SUITE
    # -------------------------------------------------------------------------
    def step_32():
        from run_feature_grid_experiments import main as run_feature_grid
        return run_feature_grid(data_dir=data_dir)
    tracker.run_step("32. Feature Grid Experiments", step_32)

    def step_33():
        from counterfactual_hr_delta import analyze_counterfactual_hr_delta
        return analyze_counterfactual_hr_delta(data_dir)
    tracker.run_step("33. Counterfactual HR Delta", step_33)

    def step_34():
        from counterfactual_rr_logistic_hazard_delta import analyze_counterfactual_rr_logistic_hazard_delta
        return analyze_counterfactual_rr_logistic_hazard_delta(data_dir)
    tracker.run_step("34. Counterfactual RR Logistic Hazard Delta", step_34)

    def step_35():
        from counterfactual_rr_deephit_delta import main as run_cf_rr_deephit
        return run_cf_rr_deephit(data_dir=data_dir)
    tracker.run_step("35. Counterfactual RR DeepHit Delta", step_35)

    def step_36():
        from counterfactual_inference_semester_transformer import run_counterfactual_transformer
        return run_counterfactual_transformer(data_dir=data_dir)
    tracker.run_step("36. Counterfactual Inference Semester Transformer", step_36)

    def step_37():
        from counterfactual_rr_exam_rnn_delta import main as run_cf_exam_rnn_delta
        return run_cf_exam_rnn_delta(data_dir=data_dir)
    tracker.run_step("37. Counterfactual RR Exam RNN Delta", step_37)


    # Benchmark-Report exportieren
    tracker.export_report(data_dir)

    total_elapsed = time.time() - total_t0
    print("\n" + "=" * 80)
    print(f"   MASTER NACHTLAUF V4.1 ERFOLGREICH BEENDET ({total_elapsed/60:.2f} Minuten)")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Master Overnight Pipeline V4.1")
    parser.add_argument('--data_dir', type=str, default="output_v4_grid_v41/S01_baseline/universe_A")
    parser.add_argument('--skip_sim', action='store_true', default=True, help="Immer True fuer V4.1")
    parser.add_argument('--temporal', type=str, default='prev', choices=['prev', 'cum'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_master_overnight_pipeline_v41(
        data_dir=Path(args.data_dir),
        temporal=args.temporal,
        population_seed=args.seed
    )
