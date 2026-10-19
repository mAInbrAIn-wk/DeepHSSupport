"""
Heavy Deep Suite Runner (V4.2 Refactored)
=========================================
Führt die rechenintensiven autoregressiven Next-Exam-Modelle,
den Deep Transformer Autoregressor (mit Sin/Cos Positional Encoding)
und das Landmark Representation Learning aus.
Unterstützt selektive Szenarien (z.B. S01_baseline, S07_noise_half, S08_noise_double).
Strikte Trennung von Input-Daten (data_root) und Output-Artefakten (output_root).
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Optional, Union
import psutil

# Korrektes Projekt-Root (3 Ebenen über src/deepsupport/runners/heavy_suite.py)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deepsupport.data_engine.aggregate import aggregiere_daten
from deepsupport.models.autoregressive_gru import train_autoregressive_next_exam
from deepsupport.models.autoregressive_transformer import train_autoregressive_deep_transformer
import eval_autoregressive_fail
import landmark_prediction


class PipelineBenchmarkTracker:
    def __init__(self):
        self.steps = []
        self.process = psutil.Process(os.getpid())

    def run_step(self, step_name: str, func, *args, **kwargs):
        print("\n" + "=" * 80)
        print(f"   START: {step_name}")
        print(f"   Zeit:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
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



def run_heavy_suite(data_dir: Path, output_dir: Optional[Path] = None, temporal: str = 'prev', modes: list = None, population_seed: int = 42, epochs_gru: int = 20, batch_size: int = 256):
    """Kompatibilitäts-Funktion für master_suite.py."""
    data_dir = Path(data_dir)
    if output_dir is None:
        scenario_name = data_dir.parent.name if data_dir.name == 'universe_A' else data_dir.name
        output_dir = Path("outputs/output_v4_heavy") / scenario_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return run_heavy_suite_for_scenario(data_dir, output_dir, epochs_gru=epochs_gru, batch_size=batch_size)


def run_heavy_suite_for_scenario(uni_dir: Path, scenario_out: Path, epochs_gru: int = 20, batch_size: int = 256):
    """Führt die Heavy Suite für ein einzelnes Szenario (universe_A) aus."""
    tracker = PipelineBenchmarkTracker()
    total_t0 = time.time()

    print("\n" + "#" * 80)
    print(f"   HEAVY DEEP SUITE: {uni_dir.parent.name}")
    print(f"   Input:  {uni_dir}")
    print(f"   Output: {scenario_out}")
    print("#" * 80)

    # 1. Sicherstellen, dass aggregierte Daten da sind (DuckDB)
    if not (uni_dir / 'agg_abschluesse.csv').exists():
        print(f"Aggregierte Daten fehlen in {uni_dir}. Starte DuckDB Aggregation...")
        aggregiere_daten(uni_dir, backend='duckdb')

    # 2. AUTOREGRESSIVE NEXT-EXAM PREDICTION (Dual-Head GRU Multi-Task)
    tracker.run_step(
        "1. Autoregressive Next-Exam Prediction (Dual-Head GRU)",
        lambda: train_autoregressive_next_exam(data_dir=uni_dir, output_dir=scenario_out, epochs=epochs_gru, batch_size=batch_size)
    )

    # 3. EVALUATION AUTOREGRESSIVE FAIL / PR-AUC
    tracker.run_step(
        "2. Evaluation Autoregressive Next-Exam Fail PR-AUC",
        lambda: eval_autoregressive_fail.main(data_dir=uni_dir, output_dir=scenario_out)
    )

    # 4. DEEP TRANSFORMER AUTOREGRESSOR (Prüfungs-Ebene mit Sin/Cos PosEnc)
    tracker.run_step(
        "3. Autoregressive Deep Transformer (Sin/Cos PosEnc)",
        lambda: train_autoregressive_deep_transformer(data_dir=uni_dir, output_dir=scenario_out)
    )

    # 5. LANDMARK REPRESENTATION LEARNING (Geköpfter Transformer -> XGBoost)
    tracker.run_step(
        "4. Landmark Representation Learning (Ende Sem 2)",
        lambda: landmark_prediction.main(data_dir=uni_dir, output_dir=scenario_out)
    )

    tracker.export_report(scenario_out)
    elapsed = time.time() - total_t0
    print(f"\n[OK] Szenario {uni_dir.parent.name} abgeschlossen in {elapsed/60:.2f} Minuten.")
    return tracker


def main(data_root: Optional[Union[str, Path]] = None,
         output_root: Optional[Union[str, Path]] = None,
         scenarios: Optional[List[str]] = None,
         epochs_gru: int = 20,
         batch_size: int = 256):
    
    data_root = Path(data_root) if data_root else Path('data_v4_grid')
    output_root = Path(output_root) if output_root else Path('outputs/output_v4_heavy')
    
    if scenarios is None:
        scenarios = ['S01_baseline', 'S07_noise_half', 'S08_noise_double']
        
    print("=" * 80)
    print("   HEAVY DEEP SUITE ORCHESTRATION")
    print(f"   Data Root:   {data_root.resolve()}")
    print(f"   Output Root: {output_root.resolve()}")
    print(f"   Szenarien:   {scenarios}")
    print("=" * 80)

    for sc_name in scenarios:
        uni_dir = data_root / sc_name / 'universe_A'
        if not uni_dir.exists():
            print(f"\n[WARNUNG] Szenario-Verzeichnis {uni_dir} existiert nicht. Überspringe...")
            continue

        scenario_out = output_root / sc_name
        scenario_out.mkdir(parents=True, exist_ok=True)
        run_heavy_suite_for_scenario(uni_dir, scenario_out, epochs_gru=epochs_gru, batch_size=batch_size)

    print("\n" + "=" * 80)
    print("   ALLE HEAVY DEEP SUITE SZENARIEN ERFOLGREICH BEENDET!")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Heavy Deep Suite Runner V4.2")
    parser.add_argument('--data_root', type=str, default='data_v4_grid')
    parser.add_argument('--output_root', type=str, default='outputs/output_v4_heavy')
    parser.add_argument('--scenarios', type=str, default='S01_baseline,S07_noise_half,S08_noise_double')
    parser.add_argument('--epochs_gru', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    sc_list = [s.strip() for s in args.scenarios.split(',') if s.strip()]
    main(
        data_root=Path(args.data_root),
        output_root=Path(args.output_root),
        scenarios=sc_list,
        epochs_gru=args.epochs_gru,
        batch_size=args.batch_size
    )
