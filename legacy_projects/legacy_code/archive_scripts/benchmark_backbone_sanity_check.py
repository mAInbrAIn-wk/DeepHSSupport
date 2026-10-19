"""
3-Way Backbone Sanity-Check & Performance Benchmark (Pandas vs. DuckDB vs. NumPy)
==================================================================================
Überprüft bit-identische Äquivalenz und misst Laufzeit sowie RAM-Nutzung:
1. Datenaggregation (aggregate.py): Pandas vs. DuckDB vs. NumPy
2. Feature-Engine (feature_builder.py): Alle 5 Datenstrukturen
3. Generiert JSON-Metriken und Markdown-Bericht.
"""

import sys
import time
import json
from pathlib import Path
import psutil
import pandas as pd
import numpy as np

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

import aggregate
import feature_builder as fb


def measure_peak_memory_mb():
    """Gibt den aktuellen RSS-Speicherverbrauch des Prozesses in MB zurück."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def run_sanity_check(data_dir: Path = Path('src/output_dl')) -> dict:
    print("=" * 70)
    print(" 3-WAY BACKBONE SANITY-CHECK & PERFORMANCE BENCHMARK ")
    print("=" * 70)
    print(f"Datensatz: {data_dir.resolve()}\n")

    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'data_dir': str(data_dir.resolve()),
        'aggregate_benchmarks': {},
        'equivalence_checks': {},
        'feature_builder_benchmarks': {}
    }

    # -------------------------------------------------------------
    # 1. Benchmark: aggregate.py (Pandas vs. DuckDB vs. NumPy)
    # -------------------------------------------------------------
    print("[1/2] Benchmarke aggregate.py Backends...")
    
    # Pandas
    m0 = measure_peak_memory_mb()
    t0 = time.perf_counter()
    pr_pd, abs_pd = aggregate._aggregiere_daten_pandas(data_dir)
    t_pd = time.perf_counter() - t0
    m_pd = measure_peak_memory_mb() - m0
    results['aggregate_benchmarks']['pandas'] = {'duration_sec': round(t_pd, 3), 'mem_mb': round(m_pd, 2)}
    print(f"  - Pandas:  {t_pd:.3f} s, +{m_pd:.1f} MB RAM")

    # DuckDB
    m0 = measure_peak_memory_mb()
    t0 = time.perf_counter()
    pr_duck, abs_duck = aggregate._aggregiere_daten_duckdb(data_dir)
    t_duck = time.perf_counter() - t0
    m_duck = measure_peak_memory_mb() - m0
    results['aggregate_benchmarks']['duckdb'] = {'duration_sec': round(t_duck, 3), 'mem_mb': round(m_duck, 2)}
    print(f"  - DuckDB:  {t_duck:.3f} s, +{m_duck:.1f} MB RAM (Speedup: {t_pd / t_duck:.2f}x)")

    # NumPy
    m0 = measure_peak_memory_mb()
    t0 = time.perf_counter()
    pr_np, abs_np = aggregate._aggregiere_daten_numpy(data_dir)
    t_np = time.perf_counter() - t0
    m_np = measure_peak_memory_mb() - m0
    results['aggregate_benchmarks']['numpy'] = {'duration_sec': round(t_np, 3), 'mem_mb': round(m_np, 2)}
    print(f"  - NumPy:   {t_np:.3f} s, +{m_np:.1f} MB RAM (Speedup: {t_pd / t_np:.2f}x)")

    # -------------------------------------------------------------
    # Äquivalenz-Prüfung der Aggregation
    # -------------------------------------------------------------
    print("\nPrüfe Bit-Äquivalenz der aggregierten Merkmale...")
    sup_cols = [
        'support_vorher_fachlich', 'support_vorher_ueberfachlich', 'support_vorher_psychosozial',
        'support_glz_fachlich', 'support_glz_ueberfachlich', 'support_glz_psychosozial'
    ]
    
    equiv_report = {}
    for sc in sup_cols:
        diff_duck_np = float(np.abs(pr_duck[sc].values - pr_np[sc].values).sum())
        diff_pd_np = float(np.abs(pr_pd[sc].values - pr_np[sc].values).sum())
        equiv_report[sc] = {
            'diff_duckdb_vs_numpy': diff_duck_np,
            'diff_pandas_vs_numpy': diff_pd_np,
            'is_identical': (diff_duck_np == 0.0 and diff_pd_np == 0.0)
        }
        print(f"  - {sc:28s}: Diff(DuckDB, NumPy)={diff_duck_np:4.0f}, Diff(Pandas, NumPy)={diff_pd_np:4.0f} -> {'[OK]' if diff_duck_np == 0 and diff_pd_np == 0 else '[CHECK]'}")

    # cp_attempted Check
    diff_cp = float(np.abs(pr_duck['cp_attempted'].values - pr_pd['cp_attempted'].values).sum())
    equiv_report['cp_attempted'] = {'diff_duckdb_vs_pandas': diff_cp, 'is_identical': (diff_cp == 0.0)}
    print(f"  - {'cp_attempted':28s}: Diff(DuckDB, Pandas)={diff_cp:4.0f} -> {'[OK]' if diff_cp == 0 else '[CHECK]'}")

    results['equivalence_checks'] = equiv_report

    # -------------------------------------------------------------
    # 2. Benchmark: feature_builder.py Funktionen
    # -------------------------------------------------------------
    print("\n[2/2] Benchmarke feature_builder.py Funktionen...")
    fb_bench = {}

    # 1. build_semester_sequence_tensor
    t0 = time.perf_counter()
    _, X_sem, y_sem, _, feats_sem, _ = fb.build_semester_sequence_tensor(data_dir, max_semesters=16, temporal='prev')
    t_sem = time.perf_counter() - t0
    fb_bench['build_semester_sequence_tensor_prev'] = {
        'duration_sec': round(t_sem, 3), 'shape_X': list(X_sem.shape), 'shape_y': list(y_sem.shape), 'n_feats': len(feats_sem)
    }
    print(f"  1. build_semester_sequence_tensor (prev):   {t_sem:.3f} s (Shape: {X_sem.shape})")

    # 2. build_semester_sequence_tensor (competing risks)
    t0 = time.perf_counter()
    _, X_cr, y_cr, _, feats_cr, _ = fb.build_semester_sequence_tensor(data_dir, max_semesters=16, temporal='cum', target_type='competing_risks')
    t_cr = time.perf_counter() - t0
    fb_bench['build_semester_sequence_tensor_cr_cum'] = {
        'duration_sec': round(t_cr, 3), 'shape_X': list(X_cr.shape), 'shape_y': list(y_cr.shape), 'n_feats': len(feats_cr)
    }
    print(f"  2. build_semester_sequence_tensor (cr, cum): {t_cr:.3f} s (Shape: {X_cr.shape})")

    # 3. build_exam_sequence_tensor
    t0 = time.perf_counter()
    _, X_ex, y_ex, _, feats_ex, _ = fb.build_exam_sequence_tensor(data_dir, max_exams=40, temporal='prev')
    t_ex = time.perf_counter() - t0
    fb_bench['build_exam_sequence_tensor_prev'] = {
        'duration_sec': round(t_ex, 3), 'shape_X': list(X_ex.shape), 'shape_y': list(y_ex.shape), 'n_feats': len(feats_ex)
    }
    print(f"  3. build_exam_sequence_tensor (prev):       {t_ex:.3f} s (Shape: {X_ex.shape})")

    # 4. build_semester_panel_df
    t0 = time.perf_counter()
    df_sem_pan, cols_sem, _, _ = fb.build_semester_panel_df(data_dir, temporal='prev')
    t_span = time.perf_counter() - t0
    fb_bench['build_semester_panel_df_prev'] = {
        'duration_sec': round(t_span, 3), 'shape': list(df_sem_pan.shape), 'n_feats': len(cols_sem)
    }
    print(f"  4. build_semester_panel_df (prev):          {t_span:.3f} s (Shape: {df_sem_pan.shape})")

    # 5. build_exam_panel_df
    t0 = time.perf_counter()
    df_ex_pan, cols_ex, _, _ = fb.build_exam_panel_df(data_dir, temporal='prev')
    t_expan = time.perf_counter() - t0
    fb_bench['build_exam_panel_df_prev'] = {
        'duration_sec': round(t_expan, 3), 'shape': list(df_ex_pan.shape), 'n_feats': len(cols_ex)
    }
    print(f"  5. build_exam_panel_df (prev):              {t_expan:.3f} s (Shape: {df_ex_pan.shape})")

    # 6. build_landmark_dataset
    t0 = time.perf_counter()
    df_lm, cols_lm, tgt_lm, _ = fb.build_landmark_dataset(data_dir, t0=2, target='abschlussnote', graduates_only=True)
    t_lm = time.perf_counter() - t0
    fb_bench['build_landmark_dataset_graduates'] = {
        'duration_sec': round(t_lm, 3), 'shape': list(df_lm.shape), 'target': tgt_lm, 'n_feats': len(cols_lm)
    }
    print(f"  6. build_landmark_dataset (graduates):      {t_lm:.3f} s (Shape: {df_lm.shape})")

    results['feature_builder_benchmarks'] = fb_bench

    # Output speichern
    diag_dir = data_dir / 'diagnostics'
    diag_dir.mkdir(parents=True, exist_ok=True)
    json_path = diag_dir / 'backbone_sanity_check.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\n[OK] Ergebnis gespeichert in: {json_path}")
    print("=" * 70)
    return results


if __name__ == '__main__':
    run_sanity_check()
