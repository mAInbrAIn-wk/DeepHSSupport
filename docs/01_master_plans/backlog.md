---
created: 2026-08-23
last_updated: 2026-10-18
status: aktiv
tags: [backlog, master-plan, roadmap, deep-transformer, pytorch, v5, unit-tests]
---

# DeepSupport Backlog & Forschungs-Roadmap (Stand: 18. Oktober 2026 / V4.2.5 & Simulator V5.0)

Dieses Dokument bündelt alle aktiven, geplanten und zurückgestellten Vorhaben des DeepSupport-Projekts. Es dient als strategische Steuerungsbasis für Modellentwicklungen, biostatistische Kausalanalysen und den DGP-Version-5-Rollout.

---

## Prioritaet 1 — Abgeschlossene Kern-Meilensteine (DGP V5.0 Vollgenerierung & Modell-Benchmark)

| # | Thema | Referenz / Modul | Status | Geplante Umsetzung |
|:--|:------|:-----------------|:-------|:-------------------|
| **V5.4** | **DGP V5.0 Vollgenerierung ($N=50.000$) & Komplett-Training** | `docs/03_evaluations_and_benchmarks/modell_gegenueberstellung_v41_vs_v50.md` | **Abgeschlossen (2026-10-18)** | Generierung aller 8 Parallelwelten (A–H) unter DGP V5.0 in `data_v5_grid/S01_baseline` (`75,84 Min.`) und 34/34-Schritte-Training aller >25 Modelle (`outputs/S01_baseline_v5/`, `220,28 Min.`). |
| **R1** | **Post-Refactor Gesamt-Benchmark V4.2.5 & 5-AP-Härtung** | `docs/03_evaluations_and_benchmarks/modell_gegenueberstellung_gesamt_v425.md` | **Abgeschlossen (2026-10-17)** | Breslow Float32 LSE-Fix (`ROC-AUC +22.3pp`), Counting-Process Left-Truncation, 3D-Skalierung, Student-Clustered Liang-Zeger SEs & multinomialer Bootstrap, Hurdle-Regression (`R²=0.8695`), 34/34 Benchmark-Schritte grün. |
| **R2** | **Kausaler Ground-Truth Benchmark & KI-Policy-Targeting** | `docs/03_evaluations_and_benchmarks/causal_ground_truth_benchmark.md` | **Abgeschlossen (2026-10-17)** | Direkter Abgleich aller 6 Kausalschätzer gegen Parallelwelten A–H; KI-Targeting verhindert bei gleichem Budget $2{,}39\times$ mehr Studienabbrüche als Gießkannen-Allokation. |
| **T1** | **Deep Transformer Modernisierung & Regularisierungs-Benchmark** | `src/deep_transformer_regression.py` | **Abgeschlossen (2026-09-10)** | Schlanker $d=64$, 4 Heads, 2 Blocks Backbone mit `SinCosPositionalEncoding` (inkl. Mask-Propagation), AttentionPooling, L2-Regularisierung. |

---

## Prioritaet 2 — Abgeschlossene PyTorch-Portierung & Kausal-Evaluation

| # | Thema | Referenz | Status | Geplante Umsetzung |
|:--|:------|:---------|:-------|:-------------------|
| **P1** | **PyTorch / PyCox Fork & LXC Benchmark** | `docs/03_evaluations_and_benchmarks/pytorch_lxc_benchmark_evaluation_v42.md` | **Abgeschlossen (2026-09-11)** | Parallele Modell-Suite in PyTorch 2.x & PyCox über 6 Kernszenarien (S01, S02, S03, S07, S08, S11) und 4 Informationsmodi evaluiert. |
| **P2** | **$2^4$-Faktorielle Architektur-Ablation Keras vs. PyTorch** | `docs/03_evaluations_and_benchmarks/ablationsstudie_2x4_keras_vs_pytorch_ergebnisse.md` | **Abgeschlossen (2026-09-12)** | Alle 16 Faktorkombinationen (Recurrent State Decay, Pre-LN, Logits-BCE, AdamW) empirisch ausgewertet. |
| **P3** | **Submodul-Review `DeepLearning/README.md`** | `ToDo.md` | **Abgeschlossen (2026-09-11)** | Bereinigung veralteter Pfad- und Leakage-Hinweise im Submodul. |

---

## Prioritaet 3 — Version 5 Roadmap (DGP-Rekalibrierung & Unit-Tests)

| Phase | Thema | Referenz | Status | Geplante Umsetzung |
|:---|:------|:---------|:-------|:-------------------|
| **V5.1** | **Config-Bereinigung** | `docs/04_causal_and_simulation/config_audit_und_v5_roadmap.md` | **Abgeschlossen (2026-10-17)** | Entfernung der 3 Zombie-Parameter; Explizierung aller 5 Szenario-Schalter in `CONFIG`. |
| **V5.2** | **Engine-Parametrisierung** | `config_audit_und_v5_roadmap.md` | **Abgeschlossen (2026-10-17)** | Auslagerung aller 25+ hartcodierten Konstanten (*Magic Numbers*) aus `engine.py` in `CONFIG`, inkl. `get_v41_legacy_config()`. |
| **V5.3** | **Empirische Startverteilungen & Unit-Tests** | `config_audit_und_v5_roadmap.md`, `tests/test_simulation_v5.py` | **Abgeschlossen (2026-10-17)** | Motivation/Integration $\kappa=9{,}0$ ($\sigma \approx 0{,}164$), Zero-Inflated Beta Erwerb ($37{,}3\,\%$ bei $0\,\text{h}$, Median $15{,}0\,\text{h}$), $28\,\%$ Migration, Destatis-Frauenanteile; 23/23 Unit-Tests grün. |
| **V5.4** | **V5-Validierungslauf ($N=50.000$)** | `modell_gegenueberstellung_v41_vs_v50.md` | **Abgeschlossen (2026-10-18)** | $N=50.000$ Simulationslauf über 8 Universen (`data_v5_grid/S01_baseline`) und Re-Benchmarking aller 6 Modellfamilien (`34/34 PASSED`). |

---

## Prioritaet 4 — Langfristig / Nice-to-Have

| # | Thema | Quelle | Status | Notizen |
|:--|:------|:-------|:-------|:--------|
| **T4** | **Gradeblind Oracle Exploration** | `src/deepsupport/data_engine/feature_builder.py` | Konzept | Theoretische Obergrenze der GPA-Vorhersage rein aus latenten DGP-Ressourcen ohne Noten. |
| **C1** | **Dynamische Motivations-Trajektorien** | `config_audit_und_v5_roadmap.md` | Konzept | Akkumulierter Entfremdungsprozess nach Eccles & Wigfield (Erwartungs-Wert-Theorie; Heublein et al.). |
| **C2** | **Realism-Mode: Tinto-Peer-Netzwerk** | `LIMITATIONEN_FUTURE_WORK.md` | Konzept | Informelle Lerngruppen und Peer-Netzwerkbildung zur Workload-Pufferung. |
| **C3** | **Marginal Structural Models (MSM)** | `marginal_structural_models_v42.md` | **Abgeschlossen** | IPTW-Schätzung für zeitvariierende Confounder mit Huber-White Cluster-SEs implementiert & validiert. |
| **C4** | **MoE / Stacking Router & Parquet-Layout** | `duckdb_architecture_analysis.md` | Konzept | Ensembling kontrafaktischer Universen & Multi-Universe Parquet-Layout (`universe=*/exams.parquet`). |

---

## Changelog

| Datum | Aenderung |
|:------|:---------|
| 23. Aug 2026 | Initiale Erstellung des Backlogs (Stand V3.6) |
| 09. Sep 2026 | Vollständige Modernisierung auf V4.2.5; Übernahme aller Meilensteine (Grid, OOP Evaluatoren, ERD, Survival-Grundlagen) |
| 17. Okt 2026 | Abschluss des 5-AP-Refactorings, Gesamt-Benchmark V4.2.5 (>25 Modelle), Kausales Policy-Targeting, DGP V5.0 Rollout (`config.py`, `engine.py`) und 23 Unit-Tests (`tests/`) |
| 18. Okt 2026 | Abschluss von Phase V5.4: Generierung aller 8 Universen ($N=50.000$) unter DGP V5.0 (`data_v5_grid/S01_baseline`) und 34/34-Schritte-Benchmark aller 6 Modellfamilien (`modell_gegenueberstellung_v41_vs_v50.md`) |

