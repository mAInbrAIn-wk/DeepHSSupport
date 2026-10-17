---
created: 2026-09-02
last_updated: 2026-10-17
status: abgeschlossen
tags: [dokumentation, index, navigation]
---

# DeepSupport: Dokumentations-Index & Wissensbasis

Dieses Verzeichnis bündelt alle architektonischen Entscheidungen (ADRs), Kausalsynthesen, Evaluierungsberichte, Master-Pläne und Entwicklungs-Protokolle des DeepSupport-Projekts.

[**Zurueck zur Projektuebersicht / Portfolio Showcase**](../README.md)

---

## Inhaltsverzeichnis

### 1. [01_master_plans/](01_master_plans/) — Strategie & Roadmaps
- [`02_Methodische_Evolution_und_Synthese.md`](01_master_plans/02_Methodische_Evolution_und_Synthese.md): Historischer und methodischer Reifegrad (Selection Bias → Causal Panels → Competing Risks → Attention).
- [`deepsupport_master_topology_and_roadmap.md`](01_master_plans/deepsupport_master_topology_and_roadmap.md): Master-Topologie des Frameworks.
- [`refactoring_plan_evaluation_pipeline1.md`](01_master_plans/refactoring_plan_evaluation_pipeline1.md): Masterplan zur Vereinheitlichung der 5 Evaluator-Klassen.
- [`pytorch_pycox_port_plan.md`](01_master_plans/pytorch_pycox_port_plan.md): Portierungs- und Migrationsstrategie für PyTorch & PyCox.

---

### 2. [02_architectures_and_models/](02_architectures_and_models/) — Modellierung & Data Backbone
- [`feature_builder_map.md`](02_architectures_and_models/feature_builder_map.md): Architektur des zentralen Feature Builders (5 Modi: `standard`, `gradeblind`, `blind`, `oracle`, `realistic`).
- [`duckdb_architecture_analysis.md`](02_architectures_and_models/duckdb_architecture_analysis.md): Hochperformante Voraggregation via DuckDB.
- [`model_architectures.md`](02_architectures_and_models/model_architectures.md): Übersicht aller Keras- und Survival-Architekturen.
- [`script_registry.md`](02_architectures_and_models/script_registry.md): Vollständiges Inventar aller Modell- und Trainingsskripte.

---

### 3. [03_evaluations_and_benchmarks/](03_evaluations_and_benchmarks/) — Synopsen & Benchmarks (V4.2 / V4.2.5)
- **Gesamtsynthesen & Modell-Gegenüberstellungen:**
  - [`modell_gegenueberstellung_gesamt_v425.md`](03_evaluations_and_benchmarks/modell_gegenueberstellung_gesamt_v425.md): **Vollständige Side-by-Side-Gegenüberstellung aller >25 Modelle aus dem V4.2.5 Post-Refactor Benchmark** (alle 6 Modellfamilien, Pre- vs. Post-Refactor, Keras 3 vs. PyTorch 2.x Paritätsanalyse).
  - [`master_synopse_v4_gesamt.md`](03_evaluations_and_benchmarks/master_synopse_v4_gesamt.md): **Master-Synthese über alle 15 Szenarien und 225 DL-Modelle** des Feature Grids.
  - [`causal_ground_truth_benchmark.md`](03_evaluations_and_benchmarks/causal_ground_truth_benchmark.md): **Vereinheitlichter Kausaler Ground-Truth Benchmark & KI-Policy-Targeting Showcase** (8 Parallelwelten vs. Cox/MSM/DML/G-Computation & 20%-Budget-Allokation).
  - [`synopse_heavy_suite_s01_s07_s08.md`](03_evaluations_and_benchmarks/synopse_heavy_suite_s01_s07_s08.md): **Heavy Deep Suite Gesamtauswertung** (Dual-Head GRU vs. Deep Transformer, Fail PR-AUC, Landmark Representation Learning).
- **Isolierte Sensitivitäts-Synopsen:**
  - [`synopse_supportwirkung_s01_s02_s03.md`](03_evaluations_and_benchmarks/synopse_supportwirkung_s01_s02_s03.md): Variation der Supportwirkung (0.5× bis 2.0×).
  - [`synopse_notenboost_s01_s04_s05_s06.md`](03_evaluations_and_benchmarks/synopse_notenboost_s01_s04_s05_s06.md): Isolierte Analyse des Notenboosts auf den fachlichen Support.
  - [`synopse_rauschen_s01_s07_s08.md`](03_evaluations_and_benchmarks/synopse_rauschen_s01_s07_s08.md): Rausch-Resilienz und Brier-Score-Analyse.
  - [`synopse_zeitkosten_s01_s09_s10.md`](03_evaluations_and_benchmarks/synopse_zeitkosten_s01_s09_s10.md): Zeitkosten-Dimension und Modulabwurf-Effekte.
  - [`synopse_rct_selektion_s01_s11.md`](03_evaluations_and_benchmarks/synopse_rct_selektion_s01_s11.md): Das RCT-Selektionsparadoxon (Zufall vs. Bedarfsallokation).
  - [`synopse_overload_s01_s12_s13_s14.md`](03_evaluations_and_benchmarks/synopse_overload_s01_s12_s13_s14.md): Überlastungs-Penalty und Kalibrierungs-Robustheit.
  - [`synopse_kombination_s01_s15.md`](03_evaluations_and_benchmarks/synopse_kombination_s01_s15.md): Resilienz im Extrem-Szenario (Kombination von Hoch-Effekt und Doppel-Kosten).

---

### 4. [04_causal_and_simulation/](04_causal_and_simulation/) — Kausale Inferenz & Data-Generating Process
- [`datenarchitektur_und_eda_v4.md`](04_causal_and_simulation/datenarchitektur_und_eda_v4.md): **Datenarchitektur, Relationales Schema & EDA** (Vollständiges Mermaid-ERD aller 11 Tabellen, Curriculares Noten- & Support-Gefüge via Sunburst I & II, mathematische DGP-Dekonstruktion von Overload/Dropouts & biostatistische Auflösung des Immortal Time Bias).
- [`grundlagen_survival_analyse_und_zensierung.md`](04_causal_and_simulation/grundlagen_survival_analyse_und_zensierung.md): **Methodische Grundlagen der Survival-Analyse & Zensierungsmathematik** (Formales Time-to-Event-Framework, Greenwood-Formel, Auflösung des Verweildauer-Missverständnisses, Competing Risks CIF vs. Absolventen-Zensierung, Cox & Dynamic DeepHit).
- [`visuelle_datenexploration_v4.md`](04_causal_and_simulation/visuelle_datenexploration_v4.md): Visuelle Datenexploration & Kausale Evidenz (6 Publikationsgrafiken, Kohorten-Sunburst, Modul-Treemap, KDE-Varianzanalyse, Kausaler Forest Plot & Kaplan-Meier-Dynamik).
- [`04_Kausale_Vergleichsanalyse.md`](04_causal_and_simulation/04_Kausale_Vergleichsanalyse.md): Empirische Gesamtauswertung der 8 Parallelwelten (A–H) vs. Kausal-Schätzer.
- [`03_Uebersicht_Kausale_Ansaetze.md`](04_causal_and_simulation/03_Uebersicht_Kausale_Ansaetze.md): Methodischer Vergleich von Naive vs. FWL-Partialling vs. DML vs. Oracle-Mediation.
- [`systematische_verteilungsanalyse_v36_vs_v41.md`](04_causal_and_simulation/systematische_verteilungsanalyse_v36_vs_v41.md): Empirische Verteilungsanalyse aller Merkmale ($N=50.000$) und Aufdeckung des selektiven Motivations-Varianzkollapses ($\kappa=20{,}0$).
- [`config_audit_und_v5_roadmap.md`](04_causal_and_simulation/config_audit_und_v5_roadmap.md): **DGP V5.0 Spezifikation, Config-Audit & Empirische Verifikation** (Zombie-Bereinigung, Parametrisierung aller Magic Numbers, Rekalibrierung $\kappa=9{,}0$, Zero-Inflated Beta Erwerb, Destatis-Quoten & Spätsemester-Apathie-Analyse).
- [`datenprovenienz_und_generator_zuordnung_v36_v41.md`](04_causal_and_simulation/datenprovenienz_und_generator_zuordnung_v36_v41.md): Lückenlose Dokumentation der Generator-Versionen und Daten-Herkunft (V3.1 bis V4.1).

---

### 5. [06_misc/](06_misc/) — Infrastruktur & Hardware
- [`system_and_hardware_stack.md`](06_misc/system_and_hardware_stack.md): Umfassende Dokumentation der Recheninfrastruktur (HP EliteDesk G5 Workstation vs. Lenovo ThinkCentre M70q Homeserver Cluster Node unter Proxmox VE / Debian LXC).
- [`docker_proxmox_deployment_guide.md`](06_misc/docker_proxmox_deployment_guide.md): Containerisierungs- und Proxmox-Deployment-Leitfaden.

---

### 6. [07_conversation_logs/](07_conversation_logs/) — Entwicklungs-Historie & User-Annotationen
- [`00_Historisches_Gesamtprotokoll.md`](07_conversation_logs/00_Historisches_Gesamtprotokoll.md): Chronologisches Archiv der bisherigen 305 Iterationen.
- [`01_History_Selection_Bias_and_Confounding.md`](07_conversation_logs/01_History_Selection_Bias_and_Confounding.md): Historischer Diskurs zur Genese der Parallelwelten und des Dropout-Paradoxons.
- [`2026_09_02_Master_Refactoring.md`](07_conversation_logs/2026_09_02_Master_Refactoring.md): Protokoll des großen Code- und Archiv-Refactorings (V3.6 → V4).
- [`2026_09_02_Submodules_and_Synthesis.md`](07_conversation_logs/2026_09_02_Submodules_and_Synthesis.md): Portfolio-Architektur via Git Submodules und LFS-Force-Push.
- [`2026_09_04_Full_Grid_Run_and_Heavy_Suite.md`](07_conversation_logs/2026_09_04_Full_Grid_Run_and_Heavy_Suite.md): Vollendung des 225-Modelle-Grid-Runs, ThinkCentre Cluster Execution & Heavy Suite Synopse.
- [`2026_09_06_Evaluation_Architecture_and_Doc_Linking.md`](07_conversation_logs/2026_09_06_Evaluation_Architecture_and_Doc_Linking.md): 5 OOP Evaluator-Klassen, Rollout auf alle 14 Modelle, Markdown-Cross-Linking (Tier 1–3) & Windows venv Härtung.
- [`2026_09_12_PyTorch_Suite_Causal_LXC_and_2x4_Ablation.md`](07_conversation_logs/2026_09_12_PyTorch_Suite_Causal_LXC_and_2x4_Ablation.md): PyTorch/PyCox Modeling Suite, LXC Multi-Mode Causal Benchmark, $2^4$-Architektur-Ablation & DGP-Audit.
- [`2026_10_17_Post_Refactor_Benchmark_V425_and_DGP_V5.md`](07_conversation_logs/2026_10_17_Post_Refactor_Benchmark_V425_and_DGP_V5.md): Post-Refactor Gesamt-Benchmark (V4.2.5), Kausales Policy-Targeting, DGP V5.0 Rollout & 23 Unit-Tests.

---

### 7. [08_project_evolution/](08_project_evolution/) — Projektentwicklung & Kritische Bewertung
- [`DeepSupport_Projektentwicklung.md`](08_project_evolution/DeepSupport_Projektentwicklung.md): Chronologische Rekonstruktion der intellektuellen und technischen Reise — von Data Engineering über Data Analysis und Deep Learning bis zu V4.2. Inkl. Legacy-Submodule.
- [`DeepSupport_Kritische_Bewertung.md`](08_project_evolution/DeepSupport_Kritische_Bewertung.md): Externe, unparteiische Bewertung von Stärken und Schwächen des Gesamtprojekts (synthetische Datenbasis, Bias-Kontrolle, Modell-Ensemble).