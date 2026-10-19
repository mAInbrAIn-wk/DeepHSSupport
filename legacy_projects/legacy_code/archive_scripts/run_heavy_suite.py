"""
Heavy Deep Suite Runner (V4.1)
==============================
Fuehrt die rechenintensiven Deep Transformer- und Autoregressor-Architekturen
sowie das Repraesentationslernen gezielt auf Baseline-Daten aus.
Laufzeit pro Datensatz: ca. 2 bis 3 Stunden.
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

        json_path = diag_dir / "heavy_suite_benchmark_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "total_duration_s": sum(s["duration_s"] for s in self.steps),
                "steps": self.steps
            }, f, indent=2)

        md_path = diag_dir / "heavy_suite_benchmark_report.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# Heavy Deep Suite Benchmark Report\n\n")
            f.write(f"**Generiert am:** {time.strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**Gesamtlaufzeit:** {sum(s['duration_s'] for s in self.steps)/60:.2f} Minuten\n\n")
            f.write("| Schritt | Status | Dauer (s) | RAM Start (MB) | RAM Ende (MB) | RAM Delta (MB) |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
            for s in self.steps:
                f.write(f"| {s['step_name']} | {s['status']} | {s['duration_s']} | {s['ram_start_mb']} | {s['ram_end_mb']} | {s['ram_delta_mb']:+} |\n")

        print(f"\n[REPORT] Heavy Suite Benchmark gespeichert unter: {md_path}")


def run_heavy_suite(data_dir: Path, temporal: str = 'prev', modes: list = None, population_seed: int = 42, include_deep_transformers: bool = False):
    if modes is None:
        modes = ['standard', 'gradeblind']

    os.environ['DATA_DIR'] = str(data_dir)
    tracker = PipelineBenchmarkTracker()
    total_t0 = time.time()

    print("*" * 80)
    print("   HEAVY DEEP SUITE RUNNER (V4.1)")
    print(f"   Start: {time.strftime('%Y-%m-%d %H:%M:%S')} | Temporal: {temporal} | Seed: {population_seed}")
    print(f"   Data Dir: {data_dir.resolve()}")
    print(f"   Aktive Modi: {modes}")
    print(f"   Deep Transformer Suite (d=128): {'AKTIVIERT' if include_deep_transformers else 'DEAKTIVIERT (UNDER REVISION)'}")
    print("*" * 80)

    # 1. DEEP TRANSFORMER SUITE (4 SUB-MODELLE) - UNDER REVISION
    if include_deep_transformers:
        for mode in modes:
            tracker.run_step(
                f"Deep Transformer Suite (4 Sub-Modelle) [{mode}]",
                lambda m=mode: __import__('deep_transformer_regression').train_deep_transformer_models(data_dir, temporal, m)
            )
    else:
        print("\n[INFO] Deep Transformer Suite (d=128) ist temporär deaktiviert (Under Revision: Positional Encoding & Regularisierung).")

    # 2. AUTOREGRESSIVE NEXT-EXAM MULTI-TASK
    tracker.run_step(
        "Autoregressive Next-Exam Prediction (Dual-Head Multi-Task)",
        lambda: __import__('autoregressive_next_exam').train_autoregressive_next_exam(data_dir)
    )

    tracker.run_step(
        "Evaluation Autoregressive Fail/Grade",
        lambda: __import__('eval_autoregressive_fail').main() if hasattr(__import__('eval_autoregressive_fail'), 'main') else None
    )

    # 3. DEEP TRANSFORMER AUTOREGRESSOR
    tracker.run_step(
        "Autoregressive Deep Transformer (Pruefungs-Ebene)",
        lambda: __import__('autoregressive_deep_transformer').train_autoregressive_deep_transformer(data_dir)
    )

    # 4. LANDMARK REPRESENTATION LEARNING (GEKOEPFTER TRANSFORMER)
    tracker.run_step(
        "Landmark Representation Learning (Gekoepfter Transformer)",
        lambda: __import__('landmark_prediction').main(data_dir=data_dir)
    )

    tracker.export_report(data_dir)
    total_elapsed = time.time() - total_t0
    print("\n" + "=" * 80)
    print(f"   HEAVY DEEP SUITE ERFOLGREICH BEENDET ({total_elapsed/60:.2f} Minuten)")
    print("=" * 80)
    return tracker

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Heavy Deep Suite Runner V4.1")
    parser.add_argument('--data_dir', type=str, default="output_v4_grid_v41/S01_baseline/universe_A")
    parser.add_argument('--temporal', type=str, default='prev', choices=['prev', 'cum'])
    parser.add_argument('--modes', type=str, default='standard,gradeblind')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--include_deep_transformers', action='store_true', default=False, help="Aktiviere experimentelle Deep Transformer Suite (d=128, Under Revision)")
    args = parser.parse_args()

    mode_list = [m.strip() for m in args.modes.split(',') if m.strip()]
    run_heavy_suite(
        data_dir=Path(args.data_dir),
        temporal=args.temporal,
        modes=mode_list,
        population_seed=args.seed,
        include_deep_transformers=args.include_deep_transformers
    )
