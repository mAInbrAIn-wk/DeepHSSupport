---
created: 2026-10-17
last_updated: 2026-10-17
status: abgeschlossen
tags: [protokoll, refactoring, v4.2.5, dgp-v5, unit-tests, benchmark, breslow, dml, g-computation]
---

# Protokoll: Post-Refactor Gesamt-Benchmark (V4.2.5), Kausales Policy-Targeting, DGP V5.0 Rollout & Unit-Test-Suite

## 📝 Entwicklungs-Historie & User-Annotationen

**Datum:** 15. Oktober – 17. Oktober 2026  
**Kontext:** Kritischer Methoden- und Code-Audit (5 Arbeitspakete AP 1–AP 5), Behebung gravierender numerischer und biostatistischer Bugs (Breslow Float32 Underflow, Counting-Process Left-Truncation, unskalierte 3D-Tensoren, unclusterte Standardfehler), kompletter Neu-Lauf aller >25 Modelle auf `S01_baseline` (`V4.2.5`), einheitlicher Kausaler Ground-Truth & KI-Policy-Targeting Benchmark sowie Upgrade des Datengenerators auf **Version 5 (DGP V5.0)** inklusive 23 automatisierter Unit-Tests.

### Auslösende User-Annotationen (Zusammenfassung):
> *"Ich habe einen detaillierten Review-Abgleich zu unserem Projekt erhalten... Lass uns die 5 Arbeitspakete (Survival-Losses, Kausal-Inferenz & Cluster-SEs, Feature-Skalierung & Masking, Evaluatoren & Metriken, Kausaler Ground-Truth Benchmark) sauber umsetzen."*
>
> *"Dann schlage ich vor, dass wir die Änderungen jetzt erstmal testen (auf den bestehenden Daten) und eine Gegenüberstellung der verschiedenen Modelle machen (auch um die Verbesserungen zu überprüfen); dazu ein Dokument unter `docs/` das wir im `README.md` verlinken. Und danach gehen wir an Version 5 des Generators... Ich habe auch noch eine Anmerkung ins `README.md` geschrieben, um die Du Dich noch kümmern könntest."*
>
> *"Also, das klingt alles klasse! Das Problem mit der numerischen Instabilität hätte ich nie selbst gefunden, vielen Dank! Kannst Du, wenn Du Dich an v5 machst, ebenfalls um Tests kümmern, auch wenn wir jetzt nicht direkt auf testdriven development umsteigen werden."*
>
> *"Großartig, das klingt sehr gut. Dann müssten wir den Generator mal laufen lassen (bitte in ein neues Verzeichnis), und dann alle Modelle darauf trainieren, oder? Kannst Dich vorher nochmal um ein Update unserer Konversationslogs kümmern und prüfen, ob alles auf GitHub gepusht worden ist? Passen die `requirements.txt` etc. noch? Sind `ToDo` und `Nice2Have` etc. aktuell?"*

---

## 1. Übersicht der Meilensteine

### A. 5-Arbeitspakete Methoden- & Code-Refactoring (`AP 1 – AP 5`)
1. **AP 1 (Survival-Mathematik & Numerische Stabilität):**
   - **Breslow Float32 Log-Sum-Exp Fix ([`extended_deepsurv.py`](../../src/deepsupport/models/extended_deepsurv.py), [`deep_survival.py`](../../src/deepsupport/models/deep_survival.py)):** Das vorherige `tf.cumsum(tf.exp(y_pred))` lief bei $N \approx 35.000$ Zeilen in Float32-Overflow/Underflow und ließ den Keras-MLP-Risk-Score kollabieren (`ROC-AUC = 0.5621`). Durch den numerisch exakten **Log-Sum-Exp-Trick** ($\log \sum_{j \in R(t)} e^{\eta_j} = m + \log \sum e^{\eta_j - m}$) plus vektorisierte Counting-Process-Risikomenge ($t_{\text{start},j} < t_i \le t_{\text{stop},j}$) sprang der identische Keras-DeepSurv auf **`ROC-AUC = 0.7852` ($+22{,}3\,\text{pp}$)** bzw. **`0.8692` (Oracle, $+31{,}8\,\text{pp}$)** und `C-Index = 0.9039`.
   - **Counting-Process Left-Truncation & Active-Interval Masking ([`survival.py`](../../src/deepsupport/models/torch/survival.py)):** Exakte Anderson-Gill-Risikomengen für `PyTorchCoxPH` / `PyTorchCoxTime` sowie aktives Intervall-Masking $[t_{\text{start}}, t_{\text{stop}}]$ für `PyTorchLogisticHazard`.
   - **Competing-Risks Softmax & $F_1(\tau)$ ([`dynamic_deephit.py`](../../src/deepsupport/models/dynamic_deephit.py)):** 3-Zustands-Softmax (`Survive`, `Dropout`, `Graduation`) garantiert $\sum_k P_k \equiv 1$ und evaluiert strikt die Dropout-CIF $F_1(\tau)$ statt der Gesamtereignisrate $1 - S(\tau)$ (`ROC-AUC: 0.5924 -> 0.8920`).
2. **AP 2 (Kausalinferenz, Multivariate Robinson-Gleichung & Cluster-Inferenz):**
   - Gemeinsame multivariate Robinson-Schätzung $\hat{\theta} = (\tilde{A}^\top \tilde{A})^{-1} \tilde{A}^\top \tilde{Y}$ mit **student-geclusterten Liang-Zeger Sandwich-Standardfehlern** und **multinomialem Cluster-Multiplizitäts-Bootstrap** (`rng.multinomial`, behebt den vorherigen `.isin()`-Deduplizierungsfehler, der $\approx 36{,}8\,\%$ der gezogenen Cluster verwarf) in [`dml_orthogonal.py`](../../src/deepsupport/models/dml_orthogonal.py), [`dml_transformer.py`](../../src/deepsupport/models/dml_transformer.py) und [`torch_dml.py`](../../src/deepsupport/causal/torch_dml.py).
   - Mode-konsistente Kovariatenfilterung in [`torch_msm.py`](../../src/deepsupport/causal/torch_msm.py) und [`torch_gcomputation.py`](../../src/deepsupport/causal/torch_gcomputation.py) (verhindert Informations-Leakage im `blind`-Modus).
3. **AP 3 (3D-Sequenzskalierung, Masking & Hurdle-Regression):**
   - `scale_3d_sequences` in [`feature_builder.py`](../../src/deepsupport/data_engine/feature_builder.py) für alle Keras-3D-Tensormodelle aktiviert; `SinCosPositionalEncoding` in [`deep_transformer_regression.py`](../../src/deep_transformer_regression.py) gibt `_keras_mask` sauber an `AttentionPooling` weiter.
   - Zweistufige Hurdle-Architektur für die Abschlussnoten-Regression (`graduates_only=True` in [`baseline_regressors.py`](../../src/deepsupport/models/baseline_regressors.py)): eliminiert die künstliche `3.0`-Imputation bei Studienabbrechern und steigert den MLP-Regressor von `R² = 0.5223` auf **`R² = 0.8695` (`RMSE = 0.1456`)**.
4. **AP 4 (Typisierte Evaluatoren & Skalen-getreue Konfidenzintervalle):**
   - [`CausalEvaluator`](../../src/deepsupport/evaluation/metrics_logger.py) differenziert exakt zwischen relativen Maßen (`HR`/`RR`: Delta-Methode auf Log-Skala $\theta \pm 1{,}96 \cdot \text{SE}_{\log}$) und additiven Maßen (`ATE`/`ARR`: lineare Skala $\hat{\Delta} \pm 1{,}96 \cdot \text{SE}_{\text{ate}}$).
5. **AP 5 (Kausaler Ground-Truth Benchmark & KI-Policy-Targeting):**
   - Neues Benchmark-Modul [`causal_ground_truth_benchmark.py`](../../src/deepsupport/evaluation/causal_ground_truth_benchmark.py) und Bericht [causal_ground_truth_benchmark.md](../03_evaluations_and_benchmarks/causal_ground_truth_benchmark.md): Vergleicht alle 6 Kausalschätzer direkt gegen die exakte Parallelwelten-Ground-Truth (Universum A vs. B/F/G/H) und demonstriert, dass KI-gestütztes Targeting bei gleichem Förderbudget ($31{,}8\,\%$ Kontingent) **$2{,}39\times$ mehr Studienabbrüche verhindert** als eine Gießkannen-Vergabe ($-4{,}24\,\text{pp}$ vs. $-1{,}78\,\text{pp}$).

---

### B. Vollständiger Post-Refactor Benchmark (`V4.2.5`) & Gesamtdokumentation
- Alle 34 Schritte über alle 6 Modellfamilien (>25 Architekturen in Keras 3 & PyTorch 2.x) wurden über [`run_post_refactor_benchmark_s01.py`](../../src/deepsupport/runners/run_post_refactor_benchmark_s01.py) auf `S01_baseline` ausgeführt (`34/34 PASSED`).
- Erstellung des zentralen Referenzdokuments [modell_gegenueberstellung_gesamt_v425.md](../03_evaluations_and_benchmarks/modell_gegenueberstellung_gesamt_v425.md) mit vollständigen Vorher-Nachher-Tabellen (V4.2 vs. V4.2.5) und direktem Framework-Abgleich Keras 3 vs. PyTorch 2.x.
- Einarbeitung der epistemischen Präzisierung zur Selbstselektion in [README.md](../../README.md) (Trennung zwischen realer Unbeobachtbarkeit in Verwaltungsdaten und der auf Entscheidungsplausibilität sowie Beratungserfahrung gestützten Simulationsprämisse).

---

### C. Upgrade auf DGP Version 5.0 (`config.py`, `engine.py`) & 23 Unit-Tests (`tests/`)
- **DGP V5.0 Rollout:**
  - Entfernung aller 3 historischen Zombie-Parameter (`gewicht_erwerb`, `gewicht_motivation_rauschen`, `gewicht_integration_rauschen`).
  - Explizite Aufnahme aller 5 Szenario-Schalter sowie Parametrisierung von >25 vormals hartcodierten *Magic Numbers* in [`config.py`](../../src/deepsupport/data_engine/config.py) und [`engine.py`](../../src/deepsupport/simulation/engine.py).
  - Empirische Rekalibrierung: $\kappa_{\text{Motivation}} = 9{,}0$ ($\sigma_{\text{total}} = 0{,}1637$, $+34{,}6\,\%$ Streuung gegenüber V4.1), Zero-Inflated Beta Erwerbstätigkeit nach 22. DSW-Sozialerhebung ($37{,}3\,\%$ bei $0\,\text{h}$, Median $15{,}0\,\text{h}$, $14{,}0\,\%$ bei $\ge 20\,\text{h}$), $28\,\%$ Migrationsquote und fachbereichsspezifische Destatis-Frauenanteile (`SG01`: $24{,}2\,\%$, `SG02`: $47{,}9\,\%$, `SG03`: $22{,}5\,\%$, `SG04`: $78{,}7\,\%$, `SG05`: $76{,}2\,\%$).
- **Automatisierte Unit-Test-Suite (`23/23 Tests OK`):**
  - [`test_survival_losses.py`](../../tests/test_survival_losses.py) (5 Tests), [`test_causal_estimators.py`](../../tests/test_causal_estimators.py) (4 Tests), [`test_data_splits_and_scaling.py`](../../tests/test_data_splits_and_scaling.py) (5 Tests), [`test_metrics_logger.py`](../../tests/test_metrics_logger.py) (4 Tests) und [`test_simulation_v5.py`](../../tests/test_simulation_v5.py) (5 Tests).

---

### D. Vollständige DGP-V5.0-Generierung ($N = 50{,}000$, 8 Universen A–H) & 34-Schritte-Modell-Benchmark (`outputs/S01_baseline_v5/`)
- **Multi-Universe-Generierung (`data_v5_grid/S01_baseline/`):**
  - Über [`run_v5_generation_and_benchmark.py`](../../src/deepsupport/runners/run_v5_generation_and_benchmark.py) wurden alle 8 kontrafaktischen Universen (`A` bis `H`, je $N = 50{,}000$, `seed=99999`) in `75,84 Min.` generiert und per DuckDB voraggregiert (`0,52 Min.`).
  - Exakte V5.0 Ground Truth: Universum A $= \mathbf{27{,}82\,\%}$ Dropout (`852.646` Prüfungen, `131.616` Support-Teilnahmen) vs. Universum B $= \mathbf{35{,}47\,\%}$ Dropout ($\text{ARR}_{A \text{ vs. } B} = \mathbf{+7{,}65\,\text{pp}}$, $\text{RR}_{A/B} = \mathbf{0{,}7843}$, $\text{NNT} = 13{,}1$).
- **34-Schritte-Modelltraining auf DGP V5.0 (`220,28 Min.`, `34/34 PASSED`):**
  - Sämtliche >25 Modelle über alle 6 Familien wurden auf `data_v5_grid/S01_baseline/universe_A` trainiert (`outputs/S01_baseline_v5/`).
  - Systematischer Diskriminations- und Kalibrierungsgewinn über alle dynamischen Survival- und Sequenzmodelle (`Extended Cox oracle` C-Index: `0,6980` $\to$ **`0,7288`**, `Keras Exam Survival GRU` ROC-AUC: `0,8791` $\to$ **`0,9008`**, `PyTorch Exam Survival GRU`: **`0,9054`**, `Landmark Representation Learning` $R^2$: `0,7865` $\to$ **`0,8404`**) sowie konsistente Kausalidentifikation über alle drei Förderkanäle in `Keras/PyTorch Orthogonal DML` und `PyTorch MSM` ([modell_gegenueberstellung_v41_vs_v50.md](../03_evaluations_and_benchmarks/modell_gegenueberstellung_v41_vs_v50.md)).

---

## Verwandte Dokumente

| Dokument | Bezug |
|:---|:---|
| [Modell-Gegenüberstellung DGP V4.1 vs. DGP V5.0](../03_evaluations_and_benchmarks/modell_gegenueberstellung_v41_vs_v50.md) | Direkter Head-to-Head-Vergleich aller 8 Universen und 6 Modellfamilien zwischen DGP V4.1 und DGP V5.0 |
| [Modell-Gegenüberstellung Gesamt V4.2.5](../03_evaluations_and_benchmarks/modell_gegenueberstellung_gesamt_v425.md) | Vollständiger Vorher-Nachher-Vergleich aller >25 Modelle (V4.2 vs. V4.2.5) |
| [Kausaler Ground-Truth Benchmark](../03_evaluations_and_benchmarks/causal_ground_truth_benchmark.md) | Parallelwelten-Validierung & KI-Policy-Targeting ($2{,}39\times$ Effizienzgewinn) |
| [Config-Audit & V5-Roadmap](../04_causal_and_simulation/config_audit_und_v5_roadmap.md) | DGP V5.0 Spezifikation, $N=50.000$ Verifikation & Spätsemester-Apathie-Analyse |
| [Master-Synopse V4 Gesamt](../03_evaluations_and_benchmarks/master_synopse_v4_gesamt.md) | Cross-Szenario-Referenz über alle 15 Szenarien |
| [ToDo & Aktuelle Baustellen](../../ToDo.md) | Aktueller Projektstand und Backlog |
