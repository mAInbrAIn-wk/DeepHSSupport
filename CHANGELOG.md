# Changelog

Alle nennenswerten Änderungen an diesem Projekt werden in dieser Datei dokumentiert.
Das Format basiert auf [Keep a Changelog](https://keepachangelog.com/de/1.0.0/).

## [V5.0.0 & V4.2.5 Post-Refactor Gesamt-Benchmark, Kausales Policy-Targeting & DGP Version 5] - 2026-10-17
### Added
- **Vollständige Modell-Gegenüberstellung V4.2.5 (`docs/03_evaluations_and_benchmarks/modell_gegenueberstellung_gesamt_v425.md`):**
  - Kompletter Neu-Lauf aller 34 Benchmark-Schritte (>25 Modelle über alle 6 Familien in Keras 3 & PyTorch 2.x) via `src/deepsupport/runners/run_post_refactor_benchmark_s01.py`.
  - Side-by-Side-Abgleich Pre-Refactor (V4.2) vs. Post-Refactor (V4.2.5) sowie Dual-Framework-Paritätsanalyse Keras 3 vs. PyTorch 2.x.
- **Vereinheitlichter Kausaler Ground-Truth & KI-Policy-Targeting Benchmark (`src/deepsupport/evaluation/causal_ground_truth_benchmark.py`, `docs/03_evaluations_and_benchmarks/causal_ground_truth_benchmark.md`):**
  - Direkter Abgleich aller 6 Kausalschätzer gegen die exakte Parallelwelten-Ground-Truth (Universum A vs. B/F/G/H).
  - Nachweis, dass KI-gestütztes Targeting bei identischem Förderbudget ($31{,}8\,\%$ Kontingent) **$2{,}39\times$ mehr Studienabbrüche verhindert** als eine Gießkannen-Vergabe ($-4{,}24\,\text{pp}$ vs. $-1{,}78\,\text{pp}$).
- **DGP Version 5.0 Rollout (`src/deepsupport/data_engine/config.py`, `src/deepsupport/simulation/engine.py`):**
  - Entfernung aller 3 historischen V1–V3 Zombie-Parameter (`gewicht_erwerb`, `gewicht_motivation_rauschen`, `gewicht_integration_rauschen`).
  - Explizite Aufnahme aller 5 Szenario-Schalter (`overload_penalty_factor`, `overload_penalty_cap`, `support_kosten_faktor`, `rct_support_uptake`, `disable_apathy_dampening`) sowie Parametrisierung von >25 vormals hartcodierten *Magic Numbers* in `CONFIG`.
  - Empirische Rekalibrierung: $\kappa_{\text{Motivation}} = \kappa_{\text{Integration}} = 9{,}0$ ($\sigma_{\text{total}} = 0{,}1637$, behebt den V4.1-Varianzkollaps), Zero-Inflated Beta Erwerbstätigkeit nach 22. DSW-Sozialerhebung ($37{,}3\,\%$ bei $0\,\text{h}$, Median $15{,}0\,\text{h}$), $28\,\%$ Migrationsquote und fachbereichsspezifische Destatis-Frauenanteile in `STUDIENGAENGE`.
  - Bereitstellung von `get_v41_legacy_config()` für direkte V4.1-vs.-V5.0-Replikationsvergleiche.
- **Automatisierte Unit-Test-Suite (`tests/`, 23 Tests):**
  - 5 Testmodule (`test_survival_losses.py`, `test_causal_estimators.py`, `test_data_splits_and_scaling.py`, `test_metrics_logger.py`, `test_simulation_v5.py`) mit 23 Unit-Tests (`Ran 23 tests in 14.05s — OK`).

### Fixed
- **Breslow Float32 Log-Sum-Exp Underflow & Counting-Process Left-Truncation (`extended_deepsurv.py`, `deep_survival.py`, `survival.py`):**
  - Behebt den Float32-Overflow/Underflow in `tf.cumsum(tf.exp(y_pred))` bei $N \approx 35.000$ Zeilen durch numerisch exaktes Log-Sum-Exp-Centering und exakte Anderson-Gill-Risikomengen ($t_{\text{start},j} < t_i \le t_{\text{stop},j}$). Keras Extended DeepSurv springt von `ROC-AUC = 0.5621` auf **`0.7852`** (`+22.3pp`) bzw. **`0.8692` (Oracle, `+31.8pp`)** und `C-Index = 0.9039`.
- **Competing-Risks Softmax & $F_1(\tau)$ (`dynamic_deephit.py`):**
  - 3-Zustands-Softmax (`Survive`, `Dropout`, `Graduation`) garantiert $\sum_k P_k \equiv 1$ und evaluiert die spezifische Dropout-CIF $F_1(\tau)$ (`ROC-AUC: 0.5924 -> 0.8920`).
- **Multivariate Robinson-Gleichung, Liang-Zeger Cluster-SEs & Multinomialer Bootstrap (`dml_orthogonal.py`, `dml_transformer.py`, `torch_dml.py`):**
  - Gemeinsame Schätzung korrelierter Förderpfade, student-geclusterte Sandwich-Standardfehler und multinomialer Cluster-Multiplizitäts-Bootstrap (`rng.multinomial` statt deduplizierendem `.isin()`).
- **3D-Sequenzskalierung, Mask-Propagation & Hurdle-Regression (`feature_builder.py`, `deep_transformer_regression.py`, `baseline_regressors.py`):**
  - `scale_3d_sequences` für alle Keras-3D-Tensormodelle aktiviert; `SinCosPositionalEncoding` propagiert `_keras_mask`; zweistufige Hurdle-Architektur (`graduates_only=True`) eliminiert die künstliche `3.0`-Imputation bei Studienabbrechern (`MLP-Regressor R²: 0.5223 -> 0.8695`).

## [V4.2.5 Config-Audit, DGP-Leerstellen & V5-Roadmap] - 2026-09-09
### Added
- **Systematischer Config-Audit & V5-Roadmap (`docs/04_causal_and_simulation/config_audit_und_v5_roadmap.md`):**
  - Vollständiger Codeabgleich aller 32 Einträge in `CONFIG` gegen `src/deepsupport/simulation/engine.py`.
  - Aufdeckung historischer "Zombies": `gewicht_erwerb` (durch Zeitkontomodell funktionslos), `gewicht_motivation_rauschen` und `gewicht_integration_rauschen` (durch Beta-Verteilung verwaist).
  - Identifikation von 5 "heimlichen" Defaults in `engine.py` (`overload_penalty_factor`, `overload_penalty_cap`, `support_kosten_faktor`, `rct_support_uptake`, `disable_apathy_dampening`), die in `CONFIG` fehlten.
  - Vollständiger Katalog von über 25 hartcodierten Parametern (*Magic Numbers*) in Merkmalsgenerierung, Zeitkonto, Supportlogik und Dropout-Berechnung.
- **Empirischer Kalibrierungsplan für Version 5:**
  - Motivation: Rekalibrierung von $\kappa=20{,}0$ auf $\kappa=9{,}0$ ($\sigma \approx 0{,}15$), gestützt auf psychometrische Normdaten (Academic Motivation Scale AMS, SELLMO).
  - HZB-Note: KMK-/Destatis-Abiturnotennormierung ($\kappa=6{,}5$, Bundesdurchschnitt $2{,}40$).
  - Erwerbstätigkeit: Zero-inflated Modellierung nach 22. DSW-Sozialerhebung ($37\,\%$ bei $0\,\text{h}$, Erwerbstätige mit Modus $15\,\text{h}$, 20h BAföG-Knick).
  - Migrationshintergrund & Geschlecht: Anpassung auf $28\,\%$ Migrationsquote und studiengangsspezifische Geschlechtermatrizen aus Destatis Fachserie 11.
- **Backlog-Spezifikation (Nice-to-Have):**
  - Dynamische Motivations-Trajektorien als akkumulierter Entfremdungsprozess nach der Erwartungs-Wert-Theorie (Eccles & Wigfield; Heublein et al. 2017/2022).
  - Realism-Mode: Informelle Lerngruppen- und Peer-Netzwerkbildung zur Workload-Pufferung (Tinto-Modell).
- **Aktualisierung von Projekt-Roadmaps:**
  - `ToDo.md` und `LIMITATIONEN_FUTURE_WORK.md` um V5-Phasenplan und Backlog-Einträge erweitert.

## [V4.2.4 Systematische Verteilungsanalyse & Falsifikation der Apathie-Hypothese] - 2026-09-09
### Added
- **Systematische Verteilungs- & Clipping-Analyse (`docs/04_causal_and_simulation/systematische_verteilungsanalyse_v36_vs_v41.md`):**
  - Empirischer $N=50.000$-Vergleich aller Merkmale zwischen V3.6 und V4.1 (Wasserstein-Distanzen, 2-Sample Kolmogorov-Smirnov-Tests, Momente, Quantile, Randaufstauung).
  - Aufdeckung des selektiven Varianz-Kollapses: Während `hzb_note` (Std-Ratio $99{,}98\,\%$) und `alter` ($101{,}82\,\%$) exakt kalibriert wurden, halbierte sich die Standardabweichung von `motivation` ($0{,}2389 \to 0{,}1216$, Ratio $50{,}89\,\%$) und `soziale_integration` ($0{,}2381 \to 0{,}1155$, Ratio $48{,}53\,\%$) durch eine fehlerhafte Rauschannahme ($\sigma=0{,}10$ statt $0{,}25$) in der historischen `calc_kappa.py`.
- **Szenario S16 (`S16_no_apathy_dampening`) & Falsifikation der Apathie-Hypothese:**
  - Implementierung von `disable_apathy_dampening: bool` in `engine.py`, `simulation_v4.py`, `sensitivity_grid.py` und `run_v4_simulation_grid.py`.
  - Empirischer Nachweis auf $N=20.000$ (identischer Seed): Die Apathie-Dämpfung ändert den Selektionsgap in Semester 1 um exakt $0{,}0000$ und im Gesamtverlauf um nur $-0{,}0062$. Die Hypothese, sie sei Hauptursache der V3.6/V4.1-Entzerrung, ist damit eindeutig falsifiziert.
- **Integration des ThinkCentre LXC Full Runs ($N=50.000$, 40 Epochen DML):**
  - Übernahme der LXC-Ergebnisse via Git: H3 (nicht-lineare Cox-Dynamik) hochsignifikant bestätigt ($p < 10^{-15}$, Schutz in vulnerabler Gruppe $HR = 0{,}9927$ vs. Overall $HR = 1{,}0601$).

## [V4.2.3 Hypothesen-Falsifikation & Datenprovenienz-Engine] - 2026-09-08
### Added
- **Projekt-Skill `hypothesis-falsifier` (`.agent/skills/hypothesis-falsifier/SKILL.md`):**
  - Standardisiertes 5-Stufen-Protokoll: Claim Extraction → Formal $H_0/H_1$ Specification → Falsification Criteria & Bounds → Standalone Verification Execution → Decision Matrix & Documentation.
  - Verhindert narrative Spekulation und Post-hoc-Rationalisierungen bei unerwarteten Modell- und Simulationsergebnissen.
- **Automatische Datenprovenienz & Generator-Lineage (`src/export.py`):**
  - Jeder Datenexport via `exportiere_csv()` erzeugt ab sofort automatisch eine `generation_metadata.json` mit Git-Commit, Branch, Dirty-Flag, UTC-Zeitstempel, aufrufendem Skript, Interpreter-Pfad und vollständigem Konfigurations-Dump.
  - Systematisches Zuordnungsdokument in `docs/04_causal_and_simulation/datenprovenienz_und_generator_zuordnung_v36_v41.md`: Lückenlose Zuordnung von V3.1/V3.2 (`output_dl_v2`), V3.6 Clean (`src/output_dl_v36_clean`) und V4.1 Baseline S01 (`data_v4_grid/S01_baseline`).
- **Hypothesen-Untersuchungs-Suite (`src/deepsupport/runners/run_hypothesis_investigation.py` / `run_hypothesis_investigation.py`):**
  - Einheitlicher Runner zur systematischen Überprüfung der drei DGP- und Modell-Hypothesen H1 (Dosis-Skalierung auf V3.6), H2 (Dosis-Halbierung auf V4.1) und H3 (Nicht-lineare Schwellenwert-Dynamik im Extended Cox).
  - Unterstützt `--mode test` (schneller Testlauf auf kleiner Batch mit Subsampling/5 Epochen) und `--mode full` (vollständiger Nachtlauf für ThinkCentre/EliteDesk).
  - Vollständige Verzeichnis-Isolation (`output_hypothesis_test_quick/` bzw. `output_hypothesis_investigation/`), kein Überschreiben bestehender Primärdaten.
- **Empirische Ergebnisse des Testlaufs (Small Batch, Exit-Code 0):**
  - **H3 (Nicht-lineare Cox-Dynamik): CONFIRMED.** Der lineare Extended Cox schätzt aggregiert $HR = 1.0952$ ($p < 0.001$), während die vulnerable Subgruppe (Motivation $< 0.40$) auf $HR = 1.0300$ absinkt ($\Delta = -0.0652$). Belegt, dass lineares Pooling über nicht-lineare Schwellenwerte den Schutzeffekt maskiert.
  - **H2 (Dosis-Halbierung V4.1 S02): CONFIRMED.** Im Szenario S02 (Dosis $+0.05$ statt $+0.10$) schwächt sich der DML-Schutzeffekt von $HR = 0.8500$ ($-15\%$ Risiko) signifikant auf $HR = 0.9385$ ($-6.2\%$ Risiko) ab.
  - **H1 (Dosis-Skalierung V3.6): FALSIFIED für überfachlichen Support ($HR = 1.0001$).** Beweist mikro-empirisch, dass in V3.6 die massive Selektionskluft ($\Delta = -0.2715$) ohne das in V4.1 eingeführte Apathy-Dampening selbst durch eine 5-fache Dosis nicht allein kompensiert werden kann (beim fachlichen Support sank das HR hingegen auf $0.9287$).

### Fixed
- **Auflösung des historischen 1.064-Opfer-Rätsels:**
  - Aufklärung, dass die 1.064 Studierenden, die durch fachlichen Support abgebrochen haben, aus der frühen Version V3.1/V3.2 stammten. In V3.6 Clean war diese Zahl durch den stochastischen Puffer bereits auf 67 geschrumpft, in V4.1 Baseline beträgt sie 207 bei gleichzeitig 1.672 geretteten Studierenden.
- **Modulare Importe & Rückwärtskompatibilität:**
  - Wiederherstellung von `src/models.py` und `src/config.py` (Weiterleitung auf `deepsupport.data_engine.config`), Behebung von Pfad-Präzedenzen zwischen `src/` und `legacy_code/`.

## [V4.2.2 OOP Evaluator Architecture & Knowledge Graph] - 2026-09-06
### Added
- **5 OOP Evaluator-Klassen (`src/deepsupport/evaluation/metrics_logger.py`):**
  - `SurvivalEvaluator`: ROC-AUC, PR-AUC für alle Klassen (Dropout $y=1$ und Non-Dropout $y=0$) mit horizontaler Baseline $\pi_0$, Brier Skill Score ($BSS = 1 - B/B_{\text{ref}}$), Harrell's C-Index, Balanced Accuracy, F1.
  - `RegressionEvaluator`: $R^2$, adj. $R^2$, RMSE, MAE, MedianAE, Explained Variance, Max Error, Parity Plot, Residuen-Histogramm.
  - `MulticlassEvaluator`: Makro-/Weighted-F1, Balanced Accuracy, One-vs-Rest ROC-AUC, OvR PR-AUC je Einzelklasse, zeilennormalisierte Confusion Matrix.
  - `CausalEvaluator`: Dual-CI-Berechnung (sowohl asymptotische Delta-Methode als auch Bootstrap Percentile CIs), Hazard Ratios, prozentuale Risikoreduktion, visualisierter Forest Plot.
  - `DualHeadEvaluator`: Multi-Task Evaluator für autoregressive Architekturen (separater Noten- und Bestehens-Head mit gemeinsamer, prefix-separierter JSON).
  - 100%ige Abwärtskompatibilität aller bestehenden Hilfsfunktionen (`save_metrics`, `plot_roc_curve` etc.).
- **Projekt-Evolution & Kritische Gesamtevaluation (`docs/08_project_evolution/`):**
  - `DeepSupport_Projektentwicklung.md`: Ausführliche Rekonstruktion aller 4 Projektphasen (DE → DA → DL → V4.2) und Identifikation offener Fäden.
  - `DeepSupport_Kritische_Bewertung.md`: Unabhängiges Review mit 8-dimensionaler Bewertungsmatrix (Gesamtnote 8/10) und klaren wissenschaftlichen Empfehlungen.
- **Agenten-Regelsystem (`AGENTS.md`, `GEMINI.md`, `.agent/rules/python_environment.md`):**
  - Verbindliche Richtlinien für Coding-Agenten zur automatischen Nutzung des whitelisted venv `C:\GitHub_public\.venv`.

### Changed
- **Rollout auf alle 14 Modellskripte (`src/deepsupport/models/`):**
  - Sämtliche Modellskripte (Semester GRU/Transformer, Exam GRU/Transformer, Dynamic DeepHit, Extended DeepSurv/Cox, Deep Survival, Baseline Classifiers/Regressors, Autoregressive Dual-Head, DML Orthogonal/Transformer) auf typisierte Evaluatoren umgestellt.
- **Systematisches Markdown-Cross-Linking (59 Dateien, Tier 1–3):**
  - Alle 10 Kern-Dokumente (Tier 1) mit relativen GitHub-Links und `## Verwandte Dokumente`-Tabellen ausgestattet.
  - Alle 7 Sensitivitäts-Synopsen (Tier 2) wechselseitig mit der Master-Synopse und Nachbardimensionen verknüpft.
  - Alle 34 Walkthroughs und 5 Conversation Logs (Tier 3) mit standardisiertem YAML-Frontmatter und Git-Zeitstempeln archiviert.

### Fixed
- **Windows Application Control Bypass:**
  - Identifikation des Fehlers `ImportError: DLL load failed while importing _core` bei nativem SciPy-Aufruf über System-Python; vollständige Behebung durch Umstellung auf das whitelisted Virtual Environment `C:\GitHub_public\.venv`.
- **Terminal Encoding CP1252:**
  - Ersetzung von Sonderzeichen (`π₀`, `τ`, `→`) in den Summary-Printausgaben durch ASCII-Entsprechungen zur Vermeidung von `charmap`-Codec-Abbrüchen auf Windows-Konsolen.
  - Smoke-Test-Suite für alle 5 Evaluatoren erfolgreich verifiziert (**5/5 Tests PASSED**).

## [V4.2.1 Full Grid Completion & Heavy Suite Härtung] - 2026-09-04
### Completed
- **Master Feature Grid Run (S01–S15):**
  - Alle 15 Simulationsszenarien × 15 Modelle = **225 Modelle** zu 100% fehlerfrei trainiert und geloggt (Laufzeit: 23,5 Stunden, Exit-Code 0).
  - Umfassende Metriken (PR-AUC, ROC-AUC, Brier Score, Counterfactual RRs) über alle 15 Welten archiviert in `output_v4_models/`.
- **Heavy Deep Suite & Synopse (S01, S07, S08):**
  - Autonome Ausführung auf Homeserver LXC-Cluster (Lenovo ThinkCentre M70q) für S01, S07, S08 abgeschlossen.
  - Step 2 (`eval_autoregressive_fail.py`) und Step 4 (`landmark_prediction.py`) mit Bugfixes für `StandardScaler` und `HistGradientBoosting` re-evaluiert.
  - Deep Transformer mit Sin/Cos Positional Encoding übertrifft GRU konsistent um $+0.08$ bis $+0.25$ $R^2$ auf Exam-Level.
  - Landmark Representation Learning belegt, dass Transformer-Embeddings nach 2 Fachsemestern bereits $76.5\%$ der Varianz der finalen Abschlussnote erklären.
  - Synoptische Gesamtdokumentation: `docs/03_evaluations_and_benchmarks/synopse_heavy_suite_s01_s07_s08.md`.
- **Cross-Szenario-Synopsen (S01–S15):**
  - Vollständige metrische Synopsen für Zeitkosten, RCT-Selektion, Overload-Penalty, Kombi-Szenario und die Master-Synopse über alle 225 Modelle fertiggestellt.
- **Infrastruktur:**
  - `docs/06_misc/system_and_hardware_stack.md`: Vollständige Hard- und Software-Dokumentation für Workstation (HP EliteDesk) und Cluster-Node (Lenovo ThinkCentre M70q LXC Debian).

## [V4.2 Causal Synthesis & Grid Intermediate Evaluations] - 2026-09-03
### Added
- **Synoptische Zwischenberichte (S01–S08):**
  - `docs/03_evaluations_and_benchmarks/synopse_supportwirkung_s01_s02_s03.md`: Auswertung der Supportwirkung (0.5× bis 2.0×).
  - `docs/03_evaluations_and_benchmarks/synopse_notenboost_s01_s04_s05_s06.md`: Isolierte Analyse des Notenboosts auf den fachlichen Support.
  - `docs/03_evaluations_and_benchmarks/synopse_rauschen_s01_s07_s08.md`: Rausch-Resilienz und Brier-Score-Analyse across all models.
- **Kausale Gesamtsynthese:**
  - `docs/04_causal_and_simulation/04_Kausale_Vergleichsanalyse.md`: Verifizierte empirische Auswertung mit voller Datenprovenienz (JSONs & CSVs).
  - `docs/04_causal_and_simulation/03_Uebersicht_Kausale_Ansaetze.md` & `docs/07_conversation_logs/01_History_Selection_Bias_and_Confounding.md`.

### Progress
- **V4 Grid Runner:** S01 bis S09 vollständig durchgerechnet (120 Modelle, 5 Modi, fehlerfreie Persistierung). S10 in Ausführung.

## [Unreleased / V4 Master Refactoring] - 2026-09-02
### Added
- **`deepsupport/` Package:** Vollständig modulare Architektur. Alle Keras/Scikit-Modelle wurden isoliert (`src/deepsupport/models/`).
- **Annotation Tracking Pattern:** Einführung von `docs/07_conversation_logs/` zur transparenten Aufzeichnung von Design-Entscheidungen und User-Prompts.
- **Git LFS im Archiv:** Das `archive/` Subrepo ist jetzt vollständig auf Git LFS konfiguriert, um große CSV/JSONs effizient zu verwalten.

### Changed
- **Orchestrierung (`grid_runner.py`):** Massiv gehärtet. Der Runner iteriert nun fehlerfrei über die Cross-Szenarien (`S01` bis `S15`) und trennt I/O streng (`data_root` vs `output_root`), sodass Ground-Truth-Daten nie wieder überschrieben werden.
- **Metrics Logger:** Keine `0.0`-Imputation mehr! Fehlende Werte werden strikt als `null` getrackt, um fehlerhafte Aggregationen zu unterbinden.
- **Repository Struktur:** 129 irrelevante Skripte wurden in `legacy_code/` historisiert. Das Projekt-Root-Verzeichnis ist nun 100% sauber.

### Fixed
- Alle 16 Causal Inference-Skripte wurden an die neuen Modulpfade angepasst (Regex-Massenupsdate der `.keras` Imports).
- Beseitigung diverser harter Pfade (z.B. `src/output_dl` Fallbacks im `feature_builder.py`).



## [V4.1.1 Quality Fixes] — 2026-08-30

### 1. Sample Leakage Fix (5 Skripte)
- `autoregressive_next_exam.py`: Split geändert von Zeilenebene auf Studentenebene (group-consistent)
- `autoregressive_deep_transformer.py`: Selber Fix
- `eval_autoregressive_fail.py`: Selber Fix
- `run_transfer_learning.py`: Selber Fix  
- `counterfactual_deepsurv.py`: Selber Fix
- Alle verwenden `random_state=42`, 70/15/15 Split auf `studierenden_id.unique()`

### 2. Future Leakage Fix in Feature Builder
- `feature_builder.py`: `cp_rueckstand` in Exam Tensor und Exam Panel verwendet nun `cp_cum_prev` (shifted) statt `cp_cum` (inklusive aktueller Prüfung)
- `temporal='cum'` Modus nutzt inklusiven cumsum als dokumentierte Design-Entscheidung

### 3. Oracle Feature Erweiterung
- `feature_builder.py`: `hidden_overload` und `hidden_zeit_puffer` als Oracle-Features in ALLEN 5 Build-Funktionen hinzugefügt
- `build_exam_panel_df()`: Oracle-Modus fehlte komplett, jetzt implementiert
- Oracle-Modus hat nun 5 Hidden-Features statt 3:
  - hidden_motivation, hidden_soziale_integration, hidden_erwartete_note (bestehend)
  - hidden_overload (NEU: Workload in Stunden pro Prüfung)
  - hidden_zeit_puffer (NEU: statisch pro Student, individueller Zeitpuffer)

## [V4.1 Sensitivity Grid] — 2026-08-30 — Vollständiger Sensitivitäts-Gitterlauf

### Abgeschlossen
- **Vollständiges Sensitivitäts-Grid:** 15 Szenarien × 8 Universen = 120 Simulationsläufe bei N=50.000
- **Gesamtlaufzeit:** 14,6 Stunden (52.722 Sekunden) mit `ProcessPoolExecutor` und 5 Workern
- **Szenarien** umfassen 6 Parameterdimensionen: Support-Wirkung (S02/S03), Notenboost (S04/S05/S06), Rauschen (S07/S08), Zeitkosten (S09/S10), RCT-Selektion (S11), Overload-Penalty (S12/S13/S14), plus Kombi-Szenario (S15)

### Kernergebnisse
- **Baseline (S01):** Dropout A=29,2%, B=37,1%, ARR=7,9pp, NNT=12,6
- **Perfekte RNG-Synchronisierung:** B=37,1% über alle nicht-globalen Szenarien hinweg
- **Sensitivitäts-Ranking:** Overload-Penalty (8,5pp Spanne) > Support-Wirkung (7,3pp) > Rauschen (6,4pp) > Selektion (3,5pp) > Notenboost (3,0pp) > Zeitkosten (1,2pp)
- **ARR robust** gegenüber Overload-Kalibrierung (7,3–8,4pp)
- **S15 (Kombi):** Kostenverdopplung kostet nur 0,5pp bei gleichzeitiger Wirkungsverdopplung

### Erstellte Analyseskripte
- `analyze_nachtlauf_v41.py` — Synoptische Übersicht und szenarienübergreifender Vergleich
- `paradox_analysis_v41.py` — Paradoxe Statuswechsel-Analyse
- `paradox_s02s03_detail.py` — Detaillierte S02/S03-Divergenzanalyse


## [V4.1] — 2026-08-29/30 — RNG-Synchronisierung & Feature-Restaurierung

### Hintergrund
Beim V4-Refactoring (Performance-Optimierung) gingen mehrere V3-Features
verloren, die für die kausale Interpretierbarkeit der Multiuniversen-Simulation
kritisch waren. V4.1 restauriert alle verlorenen Features.

### Restauriert
- **Per-Student-Seeds (4+1 separate RNG-Streams):** `rng_support`, `rng_social`,
  `rng_dropout`, `rng_anomalie` via `zlib.crc32(studierenden_id) ^ population_seed`.
  Vorher: ein einziger globaler `rng`-Stream → 26% RNG-Artefakt-Statusdifferenzen.
  Nachher: 9,2% Statusdifferenzen (rein kausaler Support-Effekt, 90,8% identisch).
- **Deterministisches Prüfungsrauschen:** `get_exam_noise(base_seed, modul_id, versuch)`
  als reine Funktion statt sequentiellem `rng.normal()`.
- **Pad-Draws für blockierte Angebote:** Iteration über ALLE 12 Angebote in jedem
  Universum; `rng_support.random()` wird immer gezogen, blockierte Angebote nach
  dem Draw ignoriert.
- **Carry-over fachlicher Support-Boost:** 2/3 Nachwirkung aus Vorsemestern.
- **Soziale Integration Drift:** `rng_social.normal(0, 0.05)` statt `rng.beta()`.

### Geändert
- **Overload-Penalty Cap:** Nicht mehr Default, sondern konfigurierbar via
  `overload_penalty_cap` (z.B. `0.15` in Szenario S14). Default: kein Cap.
- **`population_seed` als expliziter Parameter:** `simuliere_verlaeufe()` akzeptiert
  jetzt `population_seed` als Funktionsparameter (wie V3), statt ihn aus der Config
  zu lesen. Runner aktualisiert.
- **Support-Kosten als Faktor:** `support_kosten_faktor` (Default: 1.0) multipliziert
  die individuellen Angebotskosten, statt alle auf denselben Wert zu setzen
  (`support_kosten_override` entfernt).
- **Probabilistischer Modulabwurf:** Statt deterministischem Schwellwert steigt die
  Abwurf-Wahrscheinlichkeit sigmoid mit dem Überschuss über dem individuellen
  Zeitpuffer: `p = ueberschuss / (ueberschuss + 50)`. Eigener RNG-Stream
  `rng_workload` (Stream +5) → keine Beeinflussung anderer Streams.
- **Individueller Zeitpuffer restauriert:** `hidden_zeit_puffer` wieder als
  `Beta(μ=0.33, κ=8) × 180h` pro Student generiert (mean≈60h, std≈26h).
  Ersetzt den fixen +150h-Schwellwert.
- **PruefungsErgebnis Hidden Fields restauriert:** Alle 7 Hidden Fields wieder
  gefüllt (`hidden_overload`, `hidden_zeit_puffer`, `hidden_penalty_capped`,
  `hidden_support_capped`).
- **Butterfly-Effekt Designnotiz:** Kommentar an `get_exam_noise` dokumentiert die
  bewusste Entscheidung zur RNG-Synchronisierung zwischen Universen.

### Hinzugefügt
- **Kalibrierter RCT-Modus:** `rct_support_uptake=True` verwendet jetzt per-Typ
  kalibrierte Raten (fachlich: 0.042, überfachlich: 0.025, psychosozial: 0.023)
  statt pauschal p=0.20, um das Baseline-Teilnahmevolumen beizubehalten.
- **Konfigurierbarer Overload-Penalty-Faktor:** `cfg["overload_penalty_factor"]`
  (Default: 0.1).
- **S14 Overload-Cap-Szenario:** `overload_penalty_cap: 0.15` als Grid-Variante.
- **S15 Kombi-Szenario:** Kosten UND Wirkung verdoppelt (faktor 2 + mult 10.0).

### Nicht geändert
- V4-Features (Precompute, Zeitkonto, Super-Klausur-Boost, dynamische
  erwartete Note, Modulabwurf) bleiben unverändert.
- `simuliere_pruefung`- und `berechne_dropout`-Formeln bleiben identisch.


## [3.6.0] - 2026-08-24

### Hinzugefügt / Erledigt
- **AP0 (3-Way-Backbone & Feature-Factory):** `src/aggregate.py` unterstützt 3 austauschbare Backends (`duckdb`, `numpy`, `pandas`), `cp_attempted`-Spalte und optimierten Multi-Column Student-Join. `src/feature_builder.py` mit Vektorisierung in NumPy (34x Speedup), temporalem Switch (`temporal='prev'|'cum'`), `build_exam_panel_df`, Competing-Risks Dual-Target und flexiblen Landmark-Targets implementiert.
- **AP1 (Feature-Builder-Migration & Skript-Konsolidierung):** Alle 8 Modellklassen auf `src/feature_builder.py` umgestellt. Redundante Skripte (`*_delta.py`, `*_v2.py`) durch transparente, abwärtskompatible Wrapper ersetzt. Automatisierte Smoke-Test-Suite `src/verify_feature_migration.py` erstellt (10/10 Tests PASSED).
- **AP2 & AP4 (Master-Orchestrierung & psutil-Benchmarks):** `src/run_overnight.py` als einheitlicher V3.6 Master-Runner mit `PipelineBenchmarkTracker` (RAM-Delta & CPU-Messung pro Schritt) und automatischer HTML/Markdown-Berichterstellung implementiert.
- **AP3 (Verbose-Modus & Clipping-Diagnostik):** `ClippingTracker` in `src/simulation_v3.py` integriert; protokolliert Capping von Motivation, Integration, Overload-Penalty (Deckelung bei 0.15) und Support-Boost in `output_dl/diagnostics/clipping_report.json`.
- **AP5 (Backbone Sanity Check & Benchmark):** `src/benchmark_backbone_sanity_check.py` ausgeführt. Bit-identische Äquivalenz (0.0 Diff) aller 7 Support-/CP-Merkmale über 812.143 Zeilen bewiesen; DuckDB liefert 1.92x Speedup.
- **AP7 (Autoregressive Next-Exam-Vorhersage):** `src/autoregressive_next_exam.py` mit Dual-Head Multi-Task Architektur (GRU-Encoder + Late-Fusion) implementiert. Erreicht ROC-AUC = 0.9202 für Prüfungsbestehen und $R^2 = 0.4618$ für Noten-Regression auf 114k Test-Prüfungen.
- **AP8 (Strukturelle Mediationsanalyse):** `src/structural_mediation_analysis.py` implementiert (Imai/Pearl Framework). Zerlegt Support-Effekte in direkte (ADE) und vermittelte Leistungs-Pfade (ACME).
- **AP9 (Dokumentation & Changelog):** Vollständige Aktualisierung von `CHANGELOG.md`, `walkthrough.md` und `variablen_kausalitaet_und_temporalitaet.md`.

---

## [3.5.0] - 2026-08-23

### Hinzugefügt
- `src/feature_builder.py`: Zentrale Feature-Factory mit 5 Modi (`standard`, `gradeblind`, `blind`, `oracle`, `realistic`).
- `src/run_feature_grid_experiments.py`: Grid-Runner für 4 Modellklassen über alle 5 Modi.
- Theoretische Vorhersagbarkeits-Schranke: $R^2_{\max} = 0.7816$, Bayes-Risiko $= 0.0348$, $\text{AUC}^* = 0.8974$.
- `Artifacts/projekt_evolution_und_methodenvergleich.md`: Dokumentation der Projektgeschichte über alle Phasen.
- `Artifacts/project_index.md` & `Artifacts/dokumentation_der_dokumentation.md`.

### Behoben
- Oracle-Feature-Bug: Hardcodierte `0.5`-Werte in `aggregate.py`, `feature_builder.py` und `extended_cox_delta.py` korrigiert; dynamische $t-1$ Latenzen wiederhergestellt.
- Mermaid-Diagramm-Rendering in `projekt_evolution_und_methodenvergleich.md` repariert.

---

## [3.3.0] - 2026-08-22

### Hinzugefügt
- Universen F, G, H zur Isolation von Confounder-Strukturen (`simulate_universes_fgh.py`).
- Cross-Modal Causal Transformer-DML Pipeline (`train_transformer_dml.py`).
- Erweiterte Survival-Delta-Modelle (`extended_cox_delta.py`, `extended_deep_survival_delta.py`, `recurrent_exam_survival_delta.py`).
- 27-stufige Orchestrierung in `run_retrain_all.py`.

---

## [3.0.0] - 2026-08-20

### Hinzugefügt
- Simulation V3 mit stochastischem Zeitbudget, Überlastungsmechanismus und 5 Paralleluniversen A–E (`simulation_v3.py`).
- Counterfactual Ground Truth Berechnung (`oracle_lift.py`, `compute_macro_effects.py`).

### 2026-09-02 (Submodules, Synthesis & Grid Run)
- **Portfolio Architecture**: Eingliederung der Legacy-Projekte (\DataAnalysis\, \DataEngineering\, \DeepLearning\) als Read-Only Git Submodules im Hauptprojekt.
- **LFS-Archiv**: Migration historischer Modell- und Simulationsausgaben in ein separates LFS-Archiv (\rchive/\).
- **Methodische Synthese**: Erstellung einer finalen Übersicht über die intellektuelle Projektentwicklung (Dropout-Paradoxon $\rightarrow$ Causal Panels $\rightarrow$ Masked Sequence Models).
- **Grid Run**: Automatisierter Cross-Scenario Grid Run (S02-S15) inkl. automatischer DuckDB-Aggregation angestoßen.
