---
created: 2026-09-02
last_updated: 2026-10-17
status: aktiv
tags: [todo, roadmap, tasks, v5-roadmap, deep-transformer, pytorch, unit-tests]
---

# Aktuelle Baustellen (DeepSupport)

## 🔄 Laufende & Nächste Schritte (Akut)
- [x] **Cluster Grid Run (V4.2 Master Sensitivity Grid):** Alle 15 Szenarien (S01–S15) × 15 Modelle = 225 DL-Modelle erfolgreich trainiert und evaluiert (N=50.000, Seed 99999).
- [x] **Cross-Szenario-Synthese (S01–S15):** Vollständige metrische Synopse über alle 15 Szenarien und 225 Modelle generiert ([`master_synopse_v4_gesamt.md`](docs/03_evaluations_and_benchmarks/master_synopse_v4_gesamt.md)).
- [x] **Heavy Deep Suite (S01, S07, S08):** Vollständige Ausführung (GRU, Deep Transformer, Fail PR-AUC, Landmark Representation Learning) und Synopse ([`synopse_heavy_suite_s01_s07_s08.md`](docs/03_evaluations_and_benchmarks/synopse_heavy_suite_s01_s07_s08.md)).
- [x] **Klassenspezifische Evaluatoren implementieren (V4.2.2 / V4.2.5):** Vollständige Ablösung des manuellen Logging-Boilerplates durch 5 modulare OOP-Evaluator-Klassen (`SurvivalEvaluator`, `RegressionEvaluator`, `MulticlassEvaluator`, `CausalEvaluator`, `DualHeadEvaluator`), PR-AUC für alle Klassen, Skalen-getreue Dual-CIs (Delta-Methode für `HR`/`RR`, lineare CIs für `ATE`/`ARR`).
- [x] **Methoden-Härtung & Post-Refactor Gesamt-Benchmark (V4.2.5):** Behebung der 5 numerischen/methodischen Kernbugs (Breslow Float32 LSE-Underflow, Counting-Process Left-Truncation, 3D-Sequenzskalierung, Student-Clustered Liang-Zeger SEs & multinomialer Bootstrap, Hurdle-Regression) und vollständiger Neu-Lauf aller >25 Modelle ([`modell_gegenueberstellung_gesamt_v425.md`](docs/03_evaluations_and_benchmarks/modell_gegenueberstellung_gesamt_v425.md)).
- [x] **Automatisierte Unit-Test-Suite (`tests/`, 23 Tests):** 100% grüne Testabdeckung (`Ran 23 tests in 14.05s — OK`) über Survival-Losses, Kausalschätzer, Splits/Skalierung, Evaluatoren und den DGP-V5.0-Simulator.
- [ ] **Phase 4 (DGP V5.0 Datengenerierung & Komplett-Benchmark):** Generierung eines vollständigen V5.0-Referenzdatensatzes ($N=50.000$, 8 Universen A–H in separatem Verzeichnis `data_v5_grid/S01_baseline`) und Training/Evaluation aller 6 Modellfamilien auf den neuen V5.0-Daten.

---

## 🔬 Kausalinferenz & Modell-Vergleiche

- [x] **PyTorch / PyCox Fork & LXC Benchmark:** Aufbau der parallelen PyTorch-Suite und vollständiger LXC-Lauf über 6 Szenarien ([`pytorch_lxc_benchmark_evaluation_v42.md`](docs/03_evaluations_and_benchmarks/pytorch_lxc_benchmark_evaluation_v42.md)).
- [x] **LXC Causal 6-Szenarien-Basislauf & Multi-Mode Nachtlauf:** Vollständige Ausführung über 6 Szenarien (S01, S02, S03, S07, S08, S11) und 4 Informationsmodi (`inside_view`, `realistic`, `gradeblind_oracle`, `blind`) ([`pytorch_lxc_causal_multi_mode_evaluation_v42.md`](docs/03_evaluations_and_benchmarks/pytorch_lxc_causal_multi_mode_evaluation_v42.md)).
- [x] **$2^4$-Ablationsstudie Keras vs. PyTorch:** Empirische Prüfung aller 16 Faktorkombinationen abgeschlossen ([`ablationsstudie_2x4_keras_vs_pytorch_ergebnisse.md`](docs/03_evaluations_and_benchmarks/ablationsstudie_2x4_keras_vs_pytorch_ergebnisse.md)).
- [x] **Vereinheitlichter Kausaler Ground-Truth Benchmark & KI-Policy-Targeting:** Direkter Abgleich aller 6 Kausalschätzer gegen die Parallelwelten A–H sowie Nachweis des $2{,}39\times$-Effizienzgewinns durch KI-gestütztes Targeting ([`causal_ground_truth_benchmark.md`](docs/03_evaluations_and_benchmarks/causal_ground_truth_benchmark.md)).
- [x] **Re-Run Kausale Mediation auf V4-Daten:** 4-Stufen-Prüfplan der Imai/Pearl-Mediationsanalyse ([`kausale_mediationsanalyse_v42.md`](docs/04_causal_and_simulation/kausale_mediationsanalyse_v42.md)).
- [x] **Submodul DeepLearning README geprüft:** GitHub-Link auf `DeepHSSurvial` korrigiert, historische Effekte ($HR \approx 0{,}37$) kontextualisiert und Emojis bereinigt.

---

## 📦 Infrastruktur, Containerisierung & Deployment

- [x] **Docker-Kapselung & Proxmox-Deployment:**
  - `Dockerfile` auf Basis von `python:3.12-slim` mit C++ Build-Tools, OpenMP und PyTorch CPU Wheels erstellt.
  - `docker-compose.yml` und schlanke `.dockerignore` (Ausschluss der 25+ GB Daten aus Build-Kontext) konfiguriert.
  - Vollständiger Deployment-Leitfaden für Proxmox VE (`features: nesting=1,keyctl=1`) in [`docs/06_misc/docker_proxmox_deployment_guide.md`](docs/06_misc/docker_proxmox_deployment_guide.md) dokumentiert.

---

## 🚀 Version 5 Roadmap & Nice-to-Have Backlog
- [x] **Audit & Spezifikation:** Dokumentation aller Leerstellen, Zombies und empirischen Kalibrierungsziele erstellt ([`config_audit_und_v5_roadmap.md`](docs/04_causal_and_simulation/config_audit_und_v5_roadmap.md)).
- [x] **Phase 1 (Config-Bereinigung):** Zombies entfernt (`gewicht_erwerb`, `gewicht_motivation_rauschen`, `gewicht_integration_rauschen`), alle 5 Szenario-Schalter (`overload_penalty_factor`, `support_kosten_faktor`, etc.) explizit in `CONFIG` aufgenommen.
- [x] **Phase 2 (Engine-Parametrisierung):** Alle 25+ vormals hartcodierten Konstanten aus `engine.py` in `CONFIG` parametrisiert, inklusive `get_v41_legacy_config()` für direkte Replikationsvergleiche.
- [x] **Phase 3 (Empirische Startverteilungen & Unit-Tests):**
  - Motivation & Soziale Integration: $\kappa = 9{,}0$ ($\sigma_{\text{total}} = 0{,}1637$, Behebung des V4.1-Varianzkollapses).
  - HZB-Note: $\kappa = 6{,}5$ als konfigurierbarer Parameter ($\mu = 2{,}40, \sigma = 0{,}55$).
  - Erwerbstätigkeit: Zero-Inflated Beta-Verteilung ($37{,}3\,\%$ bei $0\,\text{h}$, Median $15{,}0\,\text{h}$, $14{,}0\,\%$ bei $\ge 20\,\text{h}$) nach 22. DSW-Sozialerhebung.
  - Migrationshintergrund: Anpassung auf bundesweite DSW-Quote ($28\,\%$).
  - Geschlechterverteilung: Fachbereichsgenaue Matrizen pro Studiengang aus Destatis Fachserie 11 (`SG01`: $24\,\%$ w, `SG02`: $48\,\%$ w, `SG03`: $23\,\%$ w, `SG04`: $78\,\%$ w, `SG05`: $76\,\%$ w).
  - Abgesichert durch 5 dedizierte Unit-Tests in [`tests/test_simulation_v5.py`](tests/test_simulation_v5.py).
- [ ] **Phase 4 (V5-Generierung $N=50.000$ & Komplett-Benchmark aller Modelle):** Vollständiger Simulationslauf in separatem Verzeichnis (`data_v5_grid/S01_baseline`) und Re-Benchmarking aller >25 Kausal-, Survival-, Sequenz- und Regressionsmodelle (`outputs/S01_baseline_v5/`).
- [ ] **Nice-to-Have / Backlog (Gradeblind Oracle Exploration):** Theoretische Obergrenze der GPA-Vorhersage rein aus latenten DGP-Ressourcen ohne Noten.
- [ ] **Nice-to-Have / Backlog (Dynamische Trajektorien):** Modellierung des Studienverlaufs als Desillusions- und Entfremdungsprozess (Eccles & Wigfield; Heublein et al. 2017/2022).
- [ ] **Nice-to-Have / Backlog (Realism-Mode):** Stochastischer Peer-Graph pro Kohorte zur Abbildung von Lerngruppen-Synergien und Isolationsrisiken (Tinto-Netzwerkmodell).
- [ ] **Nice-to-Have / Backlog (MoE / Stacking Router & Parquet-Layout):** Optionales Ensembling kontrafaktischer Universen sowie Multi-Universe Parquet-Partitionierung.

