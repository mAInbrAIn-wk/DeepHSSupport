---
created: 2026-09-02
last_updated: 2026-10-18
status: abgeschlossen
tags: [done, changelog, meilensteine, v4.2.5, v5.0]
---

# ✅ Abgeschlossene Aufgaben

## 2026-10-18: DGP V5.0 8-Universen-Generierung ($N=50.000$), 34-Schritte-Modell-Benchmark & V4.1/V5.0-Gegenüberstellung
- [x] **Vollständige DGP-V5.0-Generierung ($N = 50.000$, Universen A–H, `data_v5_grid/S01_baseline/`):**
  - Generierung aller 8 kontrafaktischen Parallelwelten (`seed=99999`) und DuckDB-Voraggregation (`run_v5_generation_and_benchmark.py`).
  - Exakte Ground Truth V5.0: Universum A $= \mathbf{27{,}82\,\%}$ Dropout vs. Universum B $= \mathbf{35{,}47\,\%}$ Dropout ($\text{ARR}_{A \text{ vs. } B} = \mathbf{+7{,}65\,\text{pp}}$, $\text{RR}_{A/B} = \mathbf{0{,}7843}$, $\text{NNT} = 13{,}1$).
- [x] **34-Schritte-Komplett-Benchmark auf DGP V5.0 (`outputs/S01_baseline_v5/`, `34/34 PASSED`):**
  - Training und Evaluierung aller >25 Modelle über sämtliche 6 Modellfamilien (Keras 3 & PyTorch 2.x + Kausaler Ground-Truth & Policy-Targeting Benchmark) in `220,28 Min.` (Peak RAM `2,83 GB`).
  - Dokumentation der vollständigen Gegenüberstellung **DGP V4.1 vs. DGP V5.0** in [`modell_gegenueberstellung_v41_vs_v50.md`](docs/03_evaluations_and_benchmarks/modell_gegenueberstellung_v41_vs_v50.md).

## 2026-10-17: Post-Refactor Gesamt-Benchmark (V4.2.5), Kausales Policy-Targeting, DGP V5.0 Rollout & 23 Unit-Tests
- [x] **Methoden- & Numerik-Refactoring (5 Arbeitspakete AP 1 – AP 5):**
  - **AP 1 (Survival-Mathematik):** Breslow Log-Sum-Exp Trick + Counting-Process Left-Truncation ($t_{\text{start},j} < t_i \le t_{\text{stop},j}$) in `extended_deepsurv.py`, `deep_survival.py` und `survival.py` (behebt Float32-Underflow: Keras Extended DeepSurv springt von `ROC-AUC = 0.5621` auf `0.7852` Standard / `0.8692` Oracle, `C-Index = 0.9039`). 3-Zustands-Softmax (`Survive`, `Dropout`, `Graduation`) und Dropout-CIF $F_1(\tau)$ in `dynamic_deephit.py` (`ROC-AUC: 0.5924 -> 0.8920`).
  - **AP 2 (Kausalinferenz & Cluster-Inferenz):** Gemeinsamer multivariater Robinson-Schätzer $\hat{\theta} = (\tilde{A}^\top \tilde{A})^{-1} \tilde{A}^\top \tilde{Y}$, student-geclusterte Liang-Zeger Sandwich-Standardfehler und multinomialer Cluster-Multiplizitäts-Bootstrap (`rng.multinomial` statt deduplizierendem `.isin()`).
  - **AP 3 (Sequenzskalierung & Hurdle-Regression):** `scale_3d_sequences` für alle Keras-3D-Tensormodelle aktiviert; `SinCosPositionalEncoding` propagiert `_keras_mask`; zweistufige Hurdle-Architektur (`graduates_only=True`) für die Abschlussnoten-Regression (`MLP-Regressor R²: 0.5223 -> 0.8695`).
  - **AP 4 (Evaluatoren):** Skalen-getreue Konfidenzintervalle im `CausalEvaluator` (Log-Skala für `HR`/`RR` vs. lineare Skala für `ATE`/`ARR`).
  - **AP 5 (Kausaler Ground-Truth Benchmark & KI-Policy-Targeting):** `causal_ground_truth_benchmark.py` und Bericht `causal_ground_truth_benchmark.md` (Nachweis von $2{,}39\times$ mehr verhinderten Studienabbrüchen durch KI-gestütztes Targeting gegenüber Gießkannen-Allokation).
- [x] **Vollständiger Post-Refactor Gesamt-Benchmark (`V4.2.5`, 34 Schritte):**
  - Alle >25 Modelle über sämtliche 6 Modellfamilien auf `S01_baseline` neu trainiert und in [`modell_gegenueberstellung_gesamt_v425.md`](docs/03_evaluations_and_benchmarks/modell_gegenueberstellung_gesamt_v425.md) dokumentiert.
- [x] **Upgrade des Simulators auf DGP Version 5.0 (`config.py`, `engine.py`):**
  - Entfernung aller 3 historischen Zombie-Parameter, Explizierung aller 5 Szenario-Schalter und Parametrisierung aller >25 *Magic Numbers*.
  - Empirische Rekalibrierung: $\kappa_{\text{Motivation}} = 9{,}0$ ($\sigma_{\text{total}} = 0{,}1637$), Zero-Inflated Beta Erwerbstätigkeit nach 22. DSW-Sozialerhebung ($37{,}3\,\%$ bei $0\,\text{h}$, Median $15{,}0\,\text{h}$), $28\,\%$ Migrationsquote und Destatis-Frauenanteile pro Studiengang.
- [x] **Automatisierte Unit-Test-Suite (`tests/`, 23 Tests):**
  - 23 Unit-Tests über alle 5 Testmodule (`test_survival_losses.py`, `test_causal_estimators.py`, `test_data_splits_and_scaling.py`, `test_metrics_logger.py`, `test_simulation_v5.py`) laufen in `14,05s` fehlerfrei durch (`OK`).

## 2026-09-12: $2^4$-Architektur-Ablationsstudie & Multi-Mode LXC Kausal-Benchmark
- [x] **$2^4$-Faktorielle Architektur-Ablation Keras 3 vs. PyTorch 2.x (`ablationsstudie_2x4_keras_vs_pytorch_ergebnisse.md`):**
  - Empirische Dekonstruktion aller 16 Faktorkombinationen (Recurrent State Decay, Pre-LayerNorm, Logits-Loss, AdamW).
- [x] **Multi-Mode LXC Kausal-Benchmark (`pytorch_lxc_causal_multi_mode_evaluation_v42.md`):**
  - Vollständige Auswertung über 4 Informationsmodi (`inside_view`, `realistic`, `gradeblind_oracle`, `blind`) und 6 Szenarien.

## 2026-09-11: PyTorch & PyCox Modeling Suite, LXC Multiscenario Benchmark & Pipeline Audit
- [x] **PyTorch 2.x & PyCox Modeling Suite (`src/deepsupport/models/torch/`):**
  - Vollständige Portierung von 4 Modellfamilien:
    1. Panel-Survival: `LogisticHazard`, `DeepHit` (Single Event), `CoxPH` (Extended DeepSurv mit Breslow-Kalibrierung), `CoxTime` (Nicht-proportionale Hazards), `DeepHitCompetingRisks` (Multivariate PMF über Dropout vs. Abschluss).
    2. Transformer Suite: `ExamTransformerRegressor` (Gradeblind Standard auf Absolventen), `CausalExamTransformerSurvival` (schrittweise Hazard-Prognose).
    3. Hybride Autoregressoren (Dual-Head Multi-Task): `NextExamGRU` und `NextExamTransformer` über 802.000 Prüfungshistorien mit Noten-MSE und Logits-BCE.
    4. Sequentielle Verlaufsmodelle: `SemesterGRU` und `SemesterTransformer` mit TimeDistributed Hazard-Heads.
  - Pre-LayerNorm (`norm_first=True`), FlashAttention-2 und numerisch stabiler Logits-Loss etabliert.
- [x] **Headless Turnkey LXC-Runner (`src/run_torch_lxc.py`):**
  - Automatisierte Multi-Thread-Allokation (7 Threads auf 8 vCPUs) auf dem Debian-LXC-Cluster-Node.
  - Multi-Szenario Batch-Lauf über 6 Kernszenarien (`S01`, `S02`, `S03`, `S07`, `S08`, `S11`) fehlerfrei abgeschlossen.
- [x] **Benchmark-Report & Keras-Vergleich (`pytorch_lxc_benchmark_evaluation_v42.md`):**
  - Vollständige Gegenüberstellung aller Metriken gegen die Keras-Baselines.
  - Auswertung der Rauschachse ($S07 \to S01 \to S08$) und der Confounding-Freiheit unter RCT ($S11$).
- [x] **19-Punkte Verkabelungs- & Datenintegritäts-Audit (`scratch/audit_full_pipeline_wiring.py`):**
  - Student-Leakage über alle 5 Pipelines formal widerlegt ($\text{Overlap} = 0$).
  - Strikte Einhaltung der Gradeblind-Policy und zeitlichen Kausalität (`shift(1)`) verifiziert.
  - Dynamische Entkopplung von `Pass (y=1)` (Mehrheit) und `Fail (y=0)` (Frühwarn-Minderheit) im `SurvivalEvaluator`.
- [x] **Methodenvergleich LogisticHazard Keras vs. PyCox (`methodenvergleich_logistic_hazard_keras_vs_pycox.md`):**
  - Mathematische Herleitung der Diskrepanz: Gepoolte logistische Regression (momentaner Hazard $h_t$) vs. diskrete 16-Kanal PyCox Intervall-PMF ($1 - S(t)$).
  - Beweis des Phasenversatzes (Phase-Lead Bias bei Evaluation von $F(t)$ gegen Einzelzeilen-Target $Y_{it}$).
- [x] **Empirische RCT-Kausalanalyse S11 vs. S01 (`empirische_rct_kausalanalyse_s11_vs_s01.md`):**
  - Vollständige quantitative Analyse auf $N=50.000$ Datensätzen: Aufhebung der Indikationsselektion ($r = +0{,}1189 \to -0{,}0538$).
  - Mathematische Erklärung des prädiktiven Signalgewinns (stärkere Merkmals-Monotonie, kein Cherry-Picking).
- [x] **Ablationsplan Keras vs. PyTorch Architekturhypothesen (`ablationsplan_keras_vs_pytorch_architekturhypothesen.md`):**
  - Strukturiertes $2^4$-Ablationsdesign zu Recurrent State Decay, Pre-LN, Logits-Loss und AdamW.
- [x] **PyTorch Causal Suite (`src/deepsupport/causal/`):**
  - `PyTorchDMLSurvival`: 5-Fold Student-Cluster Cross-Fitting, Robinson-Schätzer, neuronales Stufe-2 Hazard-Netzwerk, Student-Bootstrap.
  - `PyTorchMSM`: Stabilisierte IPTW-Gewichte, Perzentil-Trimming, gewichtete BCE-Hazard-Regression mit Huber-White Cluster-Sandwich Standardfehlern.
  - `PyTorchGComputation`: Vektorisierte Monte-Carlo G-Formel Kontrafaktik-Simulation unter $\text{do}(A=0)$ vs $\text{do}(A=1)$ mit ARR, RR, NNT und Ground-Truth-Validierung gegen Universum B.
- [x] **LXC Causal Batch-Runner (`src/run_torch_causal_lxc.py`):**
  - Headless Turnkey Batch-Runner zur vollautomatisierten Kausalanalyse über alle Szenarien auf dem Debian ThinkCentre LXC Node.
  - Lokaler Smoke-Test fehlerfrei bestanden.

## 2026-09-10: Deep Transformer Modernisierung & Kausale Mediation V4.2
- [x] **Deep Transformer Modernisierung (`src/deep_transformer_regression.py`):**
  - Schlanker $d=64$, 4 Heads, 2 Blocks Backbone mit `SinCosPositionalEncoding`, AttentionPooling und L2-Regularisierung.
  - Gradeblind $R^2 = 0{,}7850$, Standard $R^2 = 0{,}9885$, Survival Step ROC-AUC $= 0{,}8890$.
- [x] **Sideproject A (Regularisierungs-Benchmark):**
  - 9-Run-Benchmark (7.85h): Hybrid-Regularisierung konvergiert 3x schneller als unregularisiert (26.8m vs 77.2m). Overfitting auf $N=50.000$ vollständig beseitigt.
- [x] **Sideproject B (Focal Loss Grid):**
  - Vergleich von BCE vs. Focal Loss; Bestätigung, dass BCE saubere Wahrscheinlichkeitskalibrierung für Brier Scores liefert.
- [x] **Re-Run Kausale Mediation auf V4-Daten (`kausale_mediationsanalyse_v42.md`):**
  - 4-Stufen-Prüfplan durchgeführt: Selektions-Audit ($d = -0{,}945$), Realistische Mediation ($OR = 1{,}195$), Oracle-Entzauberung ($OR \le 0{,}999$).

## 2026-09-09: DGP-Audit, 11-Tabellen ERD, Survival-Grundlagen & Backlog-Neustrukturierung
- [x] **Relationales 11-Tabellen Mermaid ERD (`datenarchitektur_und_eda_v4.md`):**
  - Vollständiges, formal ACID-getreues Entity-Relationship-Diagramm der V4-Architektur inklusive Primär- und Fremdschlüsseln, Typisierung und Kardinalitäten.
  - Mathematische DGP-Dekonstruktion des Overload-Zeitmangels und der Zwangsexmatrikulations-Phasenverschiebung.
- [x] **Interaktive EDA & Sunburst I & II Re-Run (`visuelle_datenexploration_v4.md`):**
  - Wiederherstellung der Original-Plots (`legacy_projects/DataAnalysis/EDA.ipynb`) auf V4-Daten: Sunburst I (Curriculares Notengefüge aller 89 Module, Farbverlauf `RdYlGn_r`), Sunburst II (Dreischalige Support-Allokation).
  - 6 publikationsfähige Visualisierungen generiert (KDE-Verteilungen, Kausaler Forest Plot, Kaplan-Meier Dynamik).
- [x] **Methodische Grundlagen der Survival-Analyse (`grundlagen_survival_analyse_und_zensierung.md`):**
  - Formale mathematische Fundierung von Ereigniszeiten, Hazard-Raten und Zensierungsmechanismen.
  - Greenwood-Varianzherleitung und vollständige Auflösung des Verweildauer-Missverständnisses (warum KM im zensierungsfreien Grenzfall zur Gegen-ECDF wird und die Greenwood-Formel zur Binomial-Varianz kollabiert).
  - Analyse des Software-Absturzes von `statsmodels.duration.survfunc.SurvfuncRight` bei $n_K = d_K$ (Division durch Null).
  - Biostatistischer Beweis, warum Absolventen im Competing-Risks-Fall niemals naiv rechtszensiert werden dürfen, sowie Abgrenzung von Cause-Specific Cox vs. Fine-Gray / Dynamic DeepHit.
- [x] **Config-Audit & V5-Roadmap (`config_audit_und_v5_roadmap.md`):**
  - Vollständiger Codeabgleich aller 32 `CONFIG`-Einträge gegen `engine.py`.
  - Aufdeckung von 3 Zombie-Parametern, 5 heimlichen Defaults und 25+ Magic Numbers.
  - Empirischer 4-Phasen-Kalibrierungsplan nach DZHW und 22. DSW-Sozialerhebung.
- [x] **Systematische Verteilungsanalyse & S16 (`systematische_verteilungsanalyse_v36_vs_v41.md`):**
  - Falsifikation der Apathie-Hypothese auf $N=20.000$ (identischer Seed).
  - Aufdeckung des selektiven Motivations-Varianzkollapses ($\kappa=20{,}0$, $\sigma=0{,}12$ statt $0{,}24$).
- [x] **Backlog-Neustrukturierung (`docs/01_master_plans/backlog.md`):**
  - Stand 23.08. -> 09.09.2026 gehoben; Priorisierung von Deep Transformer Modernisierung, PyTorch Fork und V5-DGP.

## 2026-09-06: Evaluierungsarchitektur V4.2.2, 14-Modell-Rollout & Wissensnetz
- [x] **5 OOP Evaluator-Klassen (`metrics_logger.py`):**
  - Vollständige Implementierung von `SurvivalEvaluator`, `RegressionEvaluator`, `MulticlassEvaluator`, `CausalEvaluator` und `DualHeadEvaluator`.
  - PR-AUC für alle Klassen (Dropout $y=1$ und Non-Dropout $y=0$) inklusive Baseline-Prevalence $\pi_0$.
  - Dual-CI-Berechnung im `CausalEvaluator`: Asymptotische Delta-Methode **und** empirisches Bootstrapping für direkten methodischen Vergleich.
  - 100%ige Abwärtskompatibilität bestehender Logging-Funktionen gesichert.
- [x] **Rollout auf alle 14 Modellskripte (`src/deepsupport/models/`):**
  - Alle 14 Architekturen auf die typisierten Evaluatoren migriert (Commit `f299539`).
- [x] **Smoke-Test-Suite & Terminal-Härtung:**
  - `scratch/smoke_test_evaluators.py`: **5/5 Tests PASSED (Exit-Code 0)** für alle 5 Klassen.
  - Unicode-Bereinigung in den Print-Zusammenfassungen für Windows PowerShell CP1252 Terminal-Kompatibilität (Commit `1d2bdd6`).
- [x] **Windows Application Control Policy & Whitelisted Venv:**
  - Fehlerursache nativer SciPy/C++-DLL-Blocks im System-Python analysiert; permanenter Wechsel auf das whitelisted `C:\GitHub_public\.venv`.
  - Verbindliche Regeln in [`AGENTS.md`](AGENTS.md), [`GEMINI.md`](GEMINI.md) und [`.agent/rules/python_environment.md`](.agent/rules/python_environment.md) hinterlegt.
  - Hardware- und Systemdokumentation aktualisiert: [`docs/06_misc/system_and_hardware_stack.md`](docs/06_misc/system_and_hardware_stack.md).
- [x] **Vollständiges Dokumenten-Cross-Linking & Frontmatter (Tier 1–3):**
  - 59 Markdown-Dokumente mit relativen GitHub-Links, Git-Zeitstempeln und `## Verwandte Dokumente`-Tabellen ausgestattet (Commit `6072107`).
  - Master-Synopse mit vollständiger Navigationstabelle zu allen 8 Szenario-Synopsen versehen.
- [x] **Projektentwicklung & Kritische Gesamtevaluation:**
  - [`DeepSupport_Projektentwicklung.md`](docs/08_project_evolution/DeepSupport_Projektentwicklung.md): Rekonstruktion aller 4 Phasen (DE → DA → DL → V4.2).
  - [`DeepSupport_Kritische_Bewertung.md`](docs/08_project_evolution/DeepSupport_Kritische_Bewertung.md): Externe wissenschaftliche Bewertung (Note 8/10).

## 2026-09-04: Master Grid Run Vollendung (225 Modelle) & Heavy Suite Härtung
- [x] **Vollendung V4.2 Master Sensitivity Grid (S01–S15):**
  - Alle 15 Szenarien (S01 bis S15) × 3 Architekturen (`grid_semester_gru`, `grid_semester_transformer`, `grid_exam_gru`) × 5 Modi (`standard`, `gradeblind`, `blind`, `oracle`, `realistic`) = **225 DL-Modelle** erfolgreich trainiert, evaluiert und in `output_v4_models/` persistiert (Gesamtrechenzeit: ~23,5 Stunden, Exit-Code 0).
  - Volle Metriken (PR-AUC, ROC-AUC, Brier Score, partielle & isolierte Counterfactual Relative Risks) für alle 225 Modelle vorliegend.
- [x] **Heavy Suite Refactoring (`heavy_suite.py`):**
  - Robuste Pfadauflösung (Repo-Root 3 Ebenen up), I/O-Trennung (`output_v4_heavy/`), automatische DuckDB-Voraggregation und CLI-Szenarienauswahl (`--scenarios`).
  - Einbindung der modularen Architekturen (`autoregressive_gru`, `autoregressive_transformer`, `landmark_prediction`).
  - Getestet, synchronisiert und auf GitHub gepusht.
- [x] **Heavy Deep Suite & Synopse (S01, S07, S08):**
  - Autonome Ausführung der rechenintensiven Exam-Level-Pipelines auf dem Homeserver LXC-Node (ThinkCentre M70q) und Re-Evaluation von Step 2 & 4 auf der Workstation.
  - Next-Exam Dual-Head GRU vs. Deep Transformer mit Sin/Cos Positional Encoding evaluiert: Transformer übertrifft GRU konsistent um $+0.08$ bis $+0.25$ $R^2$ in der Notenvorhersage.
  - Fail-Focus PR-AUC (Minderheitenklasse Nicht-Bestehen) und Landmark Representation Learning (Ende Sem 2: 79.5% 4-Klassen Status-Acc, $R^2 = 0.76$ auf die finale Abschlussnote) vollständig berechnet.
  - Synoptischer Gesamtbericht erstellt: [`docs/03_evaluations_and_benchmarks/synopse_heavy_suite_s01_s07_s08.md`](docs/03_evaluations_and_benchmarks/synopse_heavy_suite_s01_s07_s08.md).
- [x] **Vollständige Cross-Szenario-Synopsen V4.2 (S01–S15):**
  - Alle verbleibenden Dimensionen vollständig ausgewertet und dokumentiert:
    - Zeitkosten (S01, S09, S10): [`synopse_zeitkosten_s01_s09_s10.md`](docs/03_evaluations_and_benchmarks/synopse_zeitkosten_s01_s09_s10.md)
    - RCT-Selektionsparadoxon (S01, S11): [`synopse_rct_selektion_s01_s11.md`](docs/03_evaluations_and_benchmarks/synopse_rct_selektion_s01_s11.md)
    - Overload-Penalty Kalibrierung (S01, S12, S13, S14): [`synopse_overload_s01_s12_s13_s14.md`](docs/03_evaluations_and_benchmarks/synopse_overload_s01_s12_s13_s14.md)
    - Kombi-Effekt-Resilienz (S01, S15): [`synopse_kombination_s01_s15.md`](docs/03_evaluations_and_benchmarks/synopse_kombination_s01_s15.md)
    - Master-Synopse über alle 225 Modelle: [`master_synopse_v4_gesamt.md`](docs/03_evaluations_and_benchmarks/master_synopse_v4_gesamt.md)
- [x] **Infrastruktur-Dokumentation:**
  - `docs/06_misc/system_and_hardware_stack.md` erstellt (HP EliteDesk G5 Workstation vs. Lenovo ThinkCentre M70q LXC Debian).

## 2026-09-03: Kausal-Synthese & V4 Grid Zwischenauswertungen
- [x] **Kausale & Historische Dokumentation:**
  - `01_History_Selection_Bias_and_Confounding.md`: Vollständige Aufarbeitung des Dropout-Paradoxons, Immortal-Time Bias und der Genese der Paralleluniversen.
  - `03_Uebersicht_Kausale_Ansaetze.md`: Differenzierung zwischen Sandbox (A-H), Imai/Pearl Struktureller Mediation und Oracle-Diagnose.
  - `04_Kausale_Vergleichsanalyse.md`: Empirisch verifizierte Gesamtauswertung aus den V3.6 JSONs/CSVs (Dropout RRs, kausaler Notenboost von -0.09 GPA vs. +0.22 naivem Confounding, Oracle-Mediation).
- [x] **Synoptische Zwischenberichte (S01–S08):**
  - `synopse_supportwirkung_s01_s02_s03.md`: Auswertung der Supportwirkung (0.5× vs. 1.0× vs. 2.0×).
  - `synopse_notenboost_s01_s04_s05_s06.md`: Verifikation der selektiven Notenboost-Wirkung auf den Fachsupport.
  - `synopse_rauschen_s01_s07_s08.md`: Rausch-Resilienz und Modellstabilität über alle Zielgrößen.
- [x] **V4 Grid Runner:** S01 bis S09 zu 100% fehlerfrei trainiert und persistiert (S10 läuft).

## 2026-09-02: V4 Master Refactoring & Portfolio
- [x] **Submodule Portfolio:** Archive, DataAnalysis, DataEngineering und DeepLearning als saubere Git-Submodule integriert.
- [x] **Git LFS Architektur:** 25 GB Rohdaten komplett aus dem Code-Repo extrahiert und in LFS-gestütztes Archiv migriert.
- [x] **Requirements Update:** `requirements.txt` exakt an die Cluster-Umgebung (inkl. scikit-survival, DuckDB, Tensorflow 2.21) angepasst.
- [x] **Orchestrierung:** `grid_runner.py` I/O-sicher gemacht (Trennung von data_root und output_root).
- [x] **Doku-Struktur:** Markdown-Dateien systematisch in docs/ einsortiert und Historisches Protokoll (299 Iterationen) generiert.

## Frühere Meilensteine
- [x] Leakage-Fix in den 5 ML-Skripten (Student-Level Split implementiert)
- [x] Future Leakage Fix: cp_rueckstand nutzt nun cp_cum_prev in Exam-Formaten
- [x] Oracle Feature Extension: hidden_overload und hidden_zeit_puffer in Feature Builder ergänzt
- [x] Feature Counts in README.md & Co. aktualisieren
- [x] SQL Backend implementieren (DuckDB & Arrow Ansatz evaluiert und in Architektur integriert)
