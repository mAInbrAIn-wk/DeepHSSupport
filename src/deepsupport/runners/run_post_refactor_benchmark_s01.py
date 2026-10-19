"""
Post-Refactor Master Benchmark Runner (V4.2.5)
==============================================
Führt alle 6 refaktorierten Modellfamilien (>25 Architekturen in Keras 3 & PyTorch)
auf dem Referenz-Szenario `data_v4_grid/S01_baseline/universe_A` (N=50.000) aus
und speichert sämtliche standardisierten OOP-Evaluator-Artefakte in einem sauberen
Benchmark-Verzeichnis (`outputs/S01_baseline_post_refactor/`).
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import time
import json
import shutil
import argparse
from pathlib import Path
from typing import Optional
import psutil
import torch

# Ungepuffertes Logging erzwingen
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deepsupport.data_engine.aggregate import aggregiere_daten
from deepsupport.models.baseline_classifiers import run_baseline_training
from deepsupport.models.baseline_regressors import run_regression_training
from deepsupport.models.deep_survival import train_deep_survival
from deepsupport.models.extended_cox import train_extended_cox_model
from deepsupport.models.extended_deepsurv import train_extended_deep_survival
from deepsupport.models.dynamic_deephit import train_dynamic_deephit_model
from deepsupport.models.semester_gru import train_recurrent_survival_model
from deepsupport.models.semester_transformer import train_transformer_survival
from deepsupport.models.exam_gru import train_recurrent_exam_survival_model
from deepsupport.models.exam_transformer import train_transformer_exam_survival
from deepsupport.models.autoregressive_gru import train_autoregressive_next_exam
from deepsupport.models.autoregressive_transformer import train_autoregressive_deep_transformer
from deepsupport.models.dml_orthogonal import train_dml_orthogonal_survival
from deepsupport.models.dml_transformer import train_transformer_dml
from deepsupport.models.torch.autoregressive import train_autoregressive_dual_head_model
from deepsupport.models.torch.sequence import train_torch_sequence_survival_model
from deepsupport.evaluation.causal_ground_truth_benchmark import main as run_causal_ground_truth_benchmark

import deep_transformer_regression
import eval_autoregressive_fail
import landmark_prediction
from run_torch_lxc import run_scenario_universe
from run_torch_causal_lxc import run_causal_scenario


class PostRefactorTracker:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.steps = []
        self.process = psutil.Process(os.getpid())
        self.diag_dir = self.output_dir / "diagnostics"
        self.diag_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.diag_dir / "post_refactor_benchmark_report.json"
        if self.json_path.exists():
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.steps = data.get("steps", [])
            except Exception:
                self.steps = []

    def run_step(self, step_name: str, func, *args, **kwargs):
        for existing in self.steps:
            if existing.get("step_name") == step_name and existing.get("status") == "PASSED":
                print(f"[SKIP] {step_name} bereits PASSED ({existing.get('duration_s')}s) — überspringe.")
                return None

        print("\n" + "=" * 84)
        print(f"   [START] {step_name}")
        print(f"   Zeit : {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 84)

        mem_start = self.process.memory_info().rss / (1024 * 1024)
        t0 = time.time()
        status = "PASSED"
        err_msg = None
        result = None

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - t0
            mem_end = self.process.memory_info().rss / (1024 * 1024)
            print(f"\n[OK] {step_name} BEENDET ({elapsed/60:.2f} Min. / {elapsed:.1f}s | RAM: {mem_end:.1f} MB)")
        except Exception as e:
            elapsed = time.time() - t0
            mem_end = self.process.memory_info().rss / (1024 * 1024)
            status = "FAILED"
            err_msg = str(e)
            print(f"\n[FEHLER] {step_name} FEHLGESCHLAGEN nach {elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()

        entry = {
            "step_name": step_name,
            "status": status,
            "duration_s": round(elapsed, 2),
            "ram_start_mb": round(mem_start, 1),
            "ram_end_mb": round(mem_end, 1),
            "ram_delta_mb": round(mem_end - mem_start, 1),
            "error": err_msg,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        replaced = False
        for i, s in enumerate(self.steps):
            if s.get("step_name") == step_name:
                self.steps[i] = entry
                replaced = True
                break
        if not replaced:
            self.steps.append(entry)
        self.export_report()
        return result

    def export_report(self):
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_duration_s": round(sum(s["duration_s"] for s in self.steps), 2),
                "steps": self.steps,
            }, f, indent=2, ensure_ascii=False)

        md_path = self.diag_dir / "post_refactor_benchmark_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Post-Refactor Master Benchmark Report (V4.2.5)\n\n")
            f.write(f"**Aktualisiert am:** {time.strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**Kumulierte Laufzeit:** {sum(s['duration_s'] for s in self.steps)/60:.2f} Minuten\n\n")
            f.write("| Schritt | Status | Dauer (s) | RAM Start (MB) | RAM Ende (MB) | RAM Delta (MB) |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
            for s in self.steps:
                f.write(f"| {s['step_name']} | {s['status']} | {s['duration_s']} | {s['ram_start_mb']} | {s['ram_end_mb']} | {s['ram_delta_mb']:+} |\n")


def _sync_flat_metrics_from_nested(target_out: Path, nested_dir: Path):
    """Kopiert verschachtelte LXC-Metriken zusätzlich in das zentrale metrics/-Verzeichnis."""
    src_metrics = nested_dir / "metrics"
    dst_metrics = target_out / "metrics"
    dst_metrics.mkdir(parents=True, exist_ok=True)
    if src_metrics.exists():
        for jf in src_metrics.glob("*.json"):
            shutil.copy2(jf, dst_metrics / jf.name)


def run_benchmark(
    data_dir: Path,
    output_dir: Path,
    phase: str = "all",
    temporal: str = "prev",
    seed: int = 42,
):
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["DATA_DIR"] = str(data_dir)
    os.environ["OUTPUT_DIR"] = str(output_dir)

    num_cores = os.cpu_count() or 4
    torch.set_num_threads(max(1, num_cores - 1))

    if not (data_dir / "agg_abschluesse.csv").exists():
        print(f"[INFO] Aggregierte Daten fehlen in {data_dir}. Starte DuckDB Aggregation...")
        aggregiere_daten(data_dir, backend="duckdb")

    tracker = PostRefactorTracker(output_dir)

    # =========================================================================
    # FAMILIE 1: STATISCHE LANDMARK-MODELLE (ENDE SEMESTER 2, T0 = 2)
    # =========================================================================
    if phase in ("all", "family1_landmark"):
        tracker.run_step(
            "F1.1: Landmark Baseline Classifiers (4-Class & Binary | standard)",
            lambda: run_baseline_training(data_dir=data_dir, mode="standard"),
        )
        tracker.run_step(
            "F1.2: Landmark Baseline Classifiers (4-Class & Binary | gradeblind)",
            lambda: run_baseline_training(data_dir=data_dir, mode="gradeblind"),
        )
        tracker.run_step(
            "F1.3: Landmark Baseline Regressors (Hurdle Stage 2 | standard)",
            lambda: run_regression_training(data_dir=data_dir, mode="standard"),
        )
        tracker.run_step(
            "F1.4: Landmark Baseline Regressors (Hurdle Stage 2 | gradeblind)",
            lambda: run_regression_training(data_dir=data_dir, mode="gradeblind"),
        )
        tracker.run_step(
            "F1.5: Deep Survival Landmark (Logistic Hazard & Breslow DeepSurv | standard)",
            lambda: train_deep_survival(data_dir=data_dir, mode="standard"),
        )

    # =========================================================================
    # FAMILIE 2: COUNTING-PROCESS PANEL & COMPETING RISKS (SEMESTER-EBENE)
    # =========================================================================
    if phase in ("all", "family2_panel"):
        for m in ("standard", "oracle", "gradeblind"):
            tracker.run_step(
                f"F2.1: Extended Cox Counting-Process [{m}]",
                lambda mode=m: train_extended_cox_model(data_dir=data_dir, temporal=temporal, mode=mode),
            )
        for m in ("standard", "oracle"):
            tracker.run_step(
                f"F2.2: Keras Extended DeepSurv & Logistic Hazard [{m}]",
                lambda mode=m: train_extended_deep_survival(
                    data_dir=data_dir, temporal=temporal, mode=mode, epochs_ds=60, epochs_lh=30
                ),
            )
        tracker.run_step(
            "F2.3: Keras Dynamic DeepHit 3-State Softmax Competing Risks [standard]",
            lambda: train_dynamic_deephit_model(
                data_dir=data_dir, max_semesters=16, temporal=temporal, mode="standard", epochs=25, batch_size=256
            ),
        )
        for m in ("standard", "oracle"):
            tracker.run_step(
                f"F2.4: PyTorch Panel Survival Suite (LH, DeepHit, CoxPH, CoxTime, DeepHit-CR) [{m}]",
                lambda mode=m: run_scenario_universe(
                    data_dir=data_dir,
                    output_dir=output_dir / "torch_panel",
                    scenario_name="S01_baseline",
                    universe_name="universe_A",
                    mode_surv=mode,
                    mode_reg="gradeblind",
                    temporal=temporal,
                    batch_size=2048,
                    epochs_lh=12,
                    epochs_dh=12,
                    epochs_cox=8,
                    epochs_ct=8,
                    epochs_dh_cr=12,
                    skip_trans=True,
                    skip_surv=False,
                    include_ar=False,
                    include_seq=False,
                    seed=seed,
                ),
            )
            _sync_flat_metrics_from_nested(
                output_dir, output_dir / "torch_panel" / "S01_baseline" / "universe_A"
            )

    # =========================================================================
    # FAMILIE 3: SEQUENZIELLE SURVIVAL- & REGRESSIONS-MODELLE (3D-TENSOREN)
    # =========================================================================
    if phase in ("all", "family3_sequence"):
        for m in ("standard", "oracle"):
            tracker.run_step(
                f"F3.1: Keras Semester Survival GRU [{m}]",
                lambda mode=m: train_recurrent_survival_model(
                    data_dir=data_dir, max_semesters=16, temporal=temporal, mode=mode, epochs=20, batch_size=512
                ),
            )
        tracker.run_step(
            "F3.2: Keras Semester Causal Transformer Survival [standard]",
            lambda: train_transformer_survival(
                data_dir=data_dir, max_semesters=16, temporal=temporal, mode="standard", epochs=15, batch_size=512
            ),
        )
        tracker.run_step(
            "F3.3: Keras Exam Survival GRU [standard]",
            lambda: train_recurrent_exam_survival_model(
                data_dir=data_dir, max_exams=40, temporal=temporal, mode="standard", epochs=12, batch_size=512
            ),
        )
        tracker.run_step(
            "F3.4: Keras Exam Causal Transformer Survival [standard]",
            lambda: train_transformer_exam_survival(
                data_dir=data_dir, max_exams=40, temporal=temporal, mode="standard", epochs=10, batch_size=512
            ),
        )
        tracker.run_step(
            "F3.5: Keras Modern Deep Transformer Suite (Sem-Reg, Exam-Reg, Exam-Surv | gradeblind)",
            lambda: deep_transformer_regression.train_deep_transformer_models(
                data_dir=data_dir, output_dir=output_dir, temporal=temporal, mode="gradeblind", epochs=12, batch_size=512
            ),
        )
        tracker.run_step(
            "F3.6: PyTorch Sequence Survival & Transformer Suite (Sem-GRU, Sem-Trans, Exam-GRU, Dyn-DeepHit, Exam-Reg, Exam-Surv)",
            lambda: run_scenario_universe(
                data_dir=data_dir,
                output_dir=output_dir / "torch_seq",
                scenario_name="S01_baseline",
                universe_name="universe_A",
                mode_surv="standard",
                mode_reg="gradeblind",
                temporal=temporal,
                batch_size=2048,
                epochs_trans=10,
                epochs_seq=10,
                skip_surv=True,
                skip_trans=False,
                include_ar=False,
                include_seq=True,
                seed=seed,
            ),
        )
        _sync_flat_metrics_from_nested(
            output_dir, output_dir / "torch_seq" / "S01_baseline" / "universe_A"
        )

    # =========================================================================
    # FAMILIE 4 & 5: DUAL-HEAD NEXT-EXAM AUTOREGRESSOREN & REPRESENTATION LEARNING
    # =========================================================================
    if phase in ("all", "family4_autoregressive"):
        tracker.run_step(
            "F3.7: PyTorch Dynamic DeepHit (Causal Step-Hazard Fix | standard)",
            lambda: train_torch_sequence_survival_model(
                model_type="dynamic_deephit",
                data_dir=data_dir,
                output_dir=output_dir,
                mode="standard",
                temporal=temporal,
                epochs=10,
                batch_size=512,
                seed=seed,
            ),
        )
        tracker.run_step(
            "F4.1: Keras Autoregressive Next-Exam Dual-Head GRU (1/K_i weighted | standard)",
            lambda: train_autoregressive_next_exam(
                data_dir=data_dir, output_dir=output_dir, mode="standard", epochs=8, batch_size=1024
            ),
        )
        tracker.run_step(
            "F4.2: Evaluation Autoregressive Next-Exam Fail PR-AUC",
            lambda: eval_autoregressive_fail.main(data_dir=data_dir, output_dir=output_dir),
        )
        tracker.run_step(
            "F4.3: Keras Autoregressive Next-Exam Deep Transformer (Sin/Cos PE, 1/K_i weighted | standard)",
            lambda: train_autoregressive_deep_transformer(
                data_dir=data_dir, output_dir=output_dir, mode="standard", epochs=5, batch_size=1024
            ),
        )
        tracker.run_step(
            "F5.1: Landmark Representation Learning (Headless Transformer -> XGBoost 4-Class & Hurdle Reg)",
            lambda: landmark_prediction.main(data_dir=data_dir, output_dir=output_dir),
        )
        tracker.run_step(
            "F4.4: PyTorch Autoregressive Next-Exam Dual-Head GRU (1/K_i weighted | standard)",
            lambda: train_autoregressive_dual_head_model(
                model_type="gru", data_dir=data_dir, output_dir=output_dir, epochs=6, batch_size=1024, seed=seed
            ),
        )
        tracker.run_step(
            "F4.5: PyTorch Autoregressive Next-Exam Dual-Head Transformer (1/K_i weighted | standard)",
            lambda: train_autoregressive_dual_head_model(
                model_type="transformer", data_dir=data_dir, output_dir=output_dir, epochs=6, batch_size=1024, seed=seed
            ),
        )

    # =========================================================================
    # FAMILIE 6: KAUSALE INFERENZ-SUITE (DML, TRANSFORMER-DML, MSM, G-COMP)
    # =========================================================================
    if phase in ("all", "family6_causal"):
        for m in ("standard", "oracle"):
            tracker.run_step(
                f"F6.1: Keras Orthogonal DML Survival (5-Fold GroupKFold + Multivariate Robinson) [{m}]",
                lambda mode=m: train_dml_orthogonal_survival(
                    data_dir=data_dir, output_dir=output_dir, temporal=temporal, mode=mode, epochs=20
                ),
            )
            tracker.run_step(
                f"F6.2: Keras Deep Transformer-DML (64-D Causal Embedding + 5-Fold GroupKFold) [{m}]",
                lambda mode=m: train_transformer_dml(
                    data_dir=data_dir, temporal=temporal, mode=mode, epochs_pretrain=8, epochs_dml=15
                ),
            )
            tracker.run_step(
                f"F6.3: PyTorch Causal Suite (DML, MSM cloglog/logit, G-Computation) [{m}]",
                lambda mode=m: run_causal_scenario(
                    data_dir=data_dir,
                    output_dir=output_dir / "torch_causal",
                    data_grid_dir=data_dir.parents[1],
                    scenario_name="S01_baseline",
                    universe_name="universe_A",
                    mode=mode,
                    temporal=temporal,
                    n_folds=5,
                    epochs_nuisance=10,
                    epochs_dml=15,
                    epochs_gcomp=12,
                    seed=seed,
                ),
            )
            nested_causal = (
                output_dir / "torch_causal" / "S01_baseline" / "universe_A"
                if m == "standard"
                else output_dir / "torch_causal" / "S01_baseline" / "universe_A" / m
            )
            _sync_flat_metrics_from_nested(output_dir, nested_causal)
            if "data_v4_grid" in str(data_dir):
                lxc_causal_target = (
                    PROJECT_ROOT / "outputs" / "output_LXC_causal" / "S01_baseline" / "universe_A"
                    if m == "standard"
                    else PROJECT_ROOT / "outputs" / "output_LXC_causal" / "S01_baseline" / "universe_A" / m
                )
                _sync_flat_metrics_from_nested(lxc_causal_target, nested_causal)

        tracker.run_step(
            "F6.4: Unified Ground-Truth Benchmark & KI-Policy-Targeting Uplift Showcase",
            lambda: run_causal_ground_truth_benchmark(
                scenario_dir=data_dir.parent,
                out_json=output_dir / "metrics" / "causal_ground_truth_benchmark.json",
            ),
        )

    print("\n" + "=" * 84)
    print(f"   POST-REFACTOR BENCHMARK ({phase.upper()}) ABGESCHLOSSEN!")
    print(f"   Report: {tracker.diag_dir / 'post_refactor_benchmark_report.md'}")
    print("=" * 84)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-Refactor Master Benchmark Runner (V4.2.5)")
    parser.add_argument("--data_dir", type=str, default="data_v4_grid/S01_baseline/universe_A")
    parser.add_argument("--output_dir", type=str, default="outputs/S01_baseline_post_refactor")
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["all", "family1_landmark", "family2_panel", "family3_sequence", "family4_autoregressive", "family6_causal"],
    )
    parser.add_argument("--temporal", type=str, default="prev")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_benchmark(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        phase=args.phase,
        temporal=args.temporal,
        seed=args.seed,
    )
