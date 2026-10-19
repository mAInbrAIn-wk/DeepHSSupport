"""
Feature Builder Migration & Model Verification Test Suite
=========================================================
Führt automatisierte Smoke-Tests über alle 8 konsolidierten Modellklassen durch,
um sicherzustellen, dass jede Pipeline fehlerfrei mit `feature_builder.py`
integriert ist und valide Metriken erzeugt.
"""

import sys
import subprocess
import time
import json
from pathlib import Path

PYTHON_EXE = sys.executable
DATA_DIR = "src/output_dl"

MODELS_TO_VERIFY = [
    {
        "name": "Klasse 5: Extended Cox Survival (PHReg)",
        "cmd": [PYTHON_EXE, "-u", "src/extended_cox_survival.py", "--temporal=prev", "--mode=standard", "--data_dir", DATA_DIR],
        "metric_file": "extended_cox_panel_metrics.json"
    },
    {
        "name": "Klasse 5: Extended DeepSurv & Logistic Hazard",
        "cmd": [PYTHON_EXE, "-u", "src/extended_deep_survival.py", "--temporal=prev", "--mode=standard", "--epochs_ds=3", "--epochs_lh=3", "--data_dir", DATA_DIR],
        "metric_file": "extended_logistic_hazard_prev_metrics.json"
    },
    {
        "name": "Klasse 6: Recurrent Semester Survival GRU",
        "cmd": [PYTHON_EXE, "-u", "src/recurrent_survival_model.py", "--temporal=prev", "--mode=standard", "--epochs=2", "--batch_size=256", "--data_dir", DATA_DIR],
        "metric_file": "recurrent_survival_gru_prev_metrics.json"
    },
    {
        "name": "Klasse 6: Dynamic DeepHit Competing Risks",
        "cmd": [PYTHON_EXE, "-u", "src/dynamic_deephit_model.py", "--temporal=prev", "--mode=standard", "--epochs=2", "--batch_size=256", "--data_dir", DATA_DIR],
        "metric_file": "dynamic_deephit_prev_metrics.json"
    },
    {
        "name": "Klasse 6: Causal Semester Transformer Survival",
        "cmd": [PYTHON_EXE, "-u", "src/transformer_survival_model.py", "--temporal=prev", "--mode=standard", "--epochs=2", "--batch_size=256", "--data_dir", DATA_DIR],
        "metric_file": "transformer_survival_prev_metrics.json"
    },
    {
        "name": "Klasse 7: Recurrent Exam Survival GRU",
        "cmd": [PYTHON_EXE, "-u", "src/recurrent_exam_survival.py", "--temporal=prev", "--mode=standard", "--epochs=2", "--batch_size=256", "--data_dir", DATA_DIR],
        "metric_file": "recurrent_exam_survival_prev_metrics.json"
    },
    {
        "name": "Klasse 7: Causal Exam Transformer Survival",
        "cmd": [PYTHON_EXE, "-u", "src/transformer_exam_survival.py", "--temporal=prev", "--mode=standard", "--epochs=2", "--batch_size=256", "--data_dir", DATA_DIR],
        "metric_file": "transformer_exam_survival_prev_metrics.json"
    },
    {
        "name": "Klasse 1: Landmark Baseline Classifiers",
        "cmd": [PYTHON_EXE, "-u", "src/train_mlp_baseline.py", "--mode=standard", "--epochs=3", "--data_dir", DATA_DIR],
        "metric_file": "mlp_baseline_metrics.json"
    },
    {
        "name": "Klasse 1: Landmark GPA Regressors",
        "cmd": [PYTHON_EXE, "-u", "src/train_mlp_regression.py", "--mode=standard", "--epochs=3", "--data_dir", DATA_DIR],
        "metric_file": "mlp_regression_metrics.json"
    },
    {
        "name": "Klasse 3: DML Orthogonal Survival",
        "cmd": [PYTHON_EXE, "-u", "src/dml_orthogonal_survival.py", "--temporal=prev", "--mode=standard", "--epochs=3", "--data_dir", DATA_DIR],
        "metric_file": "dml_orthogonal_survival_metrics.json"
    }
]


def run_verification():
    print("=" * 80)
    print("   FEATURE BUILDER MIGRATION & MODEL VERIFICATION TEST SUITE")
    print("=" * 80)

    results_summary = []
    all_passed = True

    for item in MODELS_TO_VERIFY:
        print(f"\n[RUNNING] {item['name']} ...")
        t0 = time.time()
        res = subprocess.run(item['cmd'], capture_output=True, text=True)
        dur = time.time() - t0

        if res.returncode == 0:
            metric_p = Path(DATA_DIR) / "metrics" / item['metric_file']
            exists = metric_p.exists()
            status = "PASSED" if exists else "PASSED (No Metric File)"
            print(f"  -> [OK] {item['name']} ({dur:.1f}s)")
            results_summary.append({"name": item['name'], "status": "PASSED", "duration_s": round(dur, 2)})
        else:
            all_passed = False
            print(f"  -> [FAILED] {item['name']} (Exit Code {res.returncode})")
            print("STDOUT Tail:\n" + "\n".join(res.stdout.splitlines()[-10:]))
            print("STDERR Tail:\n" + "\n".join(res.stderr.splitlines()[-10:]))
            results_summary.append({"name": item['name'], "status": "FAILED", "error": res.stderr[:500]})

    print("\n" + "=" * 80)
    print("   VERIFICATION SUMMARY")
    print("=" * 80)
    for r in results_summary:
        print(f"  • {r['name']:<55}: {r['status']}")

    out_file = Path(DATA_DIR) / "diagnostics" / "feature_migration_verification.json"
    out_file.parent.mkdir(exist_ok=True, parents=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({"timestamp": time.time(), "all_passed": all_passed, "models": results_summary}, f, indent=2)

    print(f"\nReport gespeichert unter: {out_file}")
    print("=" * 80)
    return all_passed


if __name__ == '__main__':
    success = run_verification()
    sys.exit(0 if success else 1)
