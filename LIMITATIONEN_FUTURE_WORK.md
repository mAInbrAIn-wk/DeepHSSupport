---
created: 2026-08-30
last_updated: 2026-10-17
status: abgeschlossen
tags: [limitationen, future-work, dgp-v5, kausalinferenz, methodik]
---

# Limitationen & Future Work

Dieses Dokument fasst die methodischen, datentechnischen und konzeptionellen Einschränkungen des Projekts zusammen und skizziert Ansätze für zukünftige Arbeiten.

---

## 1. Limitationen der Simulations-Engine (`engine.py`)

### A. Studiengangs-Modellierung & Fachbereichsklima
Der synthetische Generator unterscheidet Studiengänge derzeit primär über:
- Die Abfolge, Anzahl und Credit-Points der Pflichtmodule
- Die Modul-Schwierigkeitsgrade und Prüfungshürden
- Die empirische Geschlechterverteilung pro Studiengang (seit **DGP V5.0** fachspezifisch nach Destatis Fachserie 11 kalibriert)

**Einschränkung:** Es gibt derzeit **keine domänenspezifische Modellierung von Fachbereichskulturen**, sozialen Stereotype-Threats (z. B. MINT-spezifisches Hürdenklima für Frauen nach DZHW-Studien) oder psychologischen Gruppenprofilen. Alle Studierenden teilen dieselbe fundamentale Motivations- und Integrationsdynamik.

*Future Work:* Erweiterung des Generators um fachbereichsspezifische Klimafaktoren und empirisch kalibrierte Motivationsverläufe nach DZHW-/CHE-Studierendensurveys.

### B. Parametrische Sensitivität & Sensitivitäts-Grid (V4.1 / V4.2)
Die heuristisch-plausibel gewählten Parameter wurden im **V4.1/V4.2 Sensitivity Grid** (15 Szenarien $\times$ 8 Universen, $N=50.000$) systematisch evaluiert (ARR-Spanne $7{,}3 - 8{,}5\,\text{pp}$). Die Sensitivitätsanalyse zeigte eine hohe strukturelle Stabilität des relativen Risikos, deckte jedoch eine signifikante Nichtlinearität auf: Lineare Cox-Modelle maskieren den Schutzeffekt des Supports ($HR=1{,}06$), während der Schutzeffekt in der tatsächlichen Risikogruppe ($\text{Motivation} < 0{,}40$) greift ($HR=0{,}9927$).

### C. Historischer DGP-Varianzkollaps in V4.1 (in DGP V5.0 behoben)
Bei der Umstellung von Gauß-Clipping auf Beta-Verteilungen in **V4.1** wurde die empirische Varianz von HZB-Note und Alter perfekt repliziert, die Standardabweichung der Motivation und sozialen Integration kollabierte jedoch um $50\,\%$ ($\sigma_{\text{V3}} \approx 0{,}24 \rightarrow \sigma_{\text{V4}} \approx 0{,}12$ bei $\kappa=20{,}0$).
- **Status (in DGP V5.0 behoben):** In **Version 5.0** (`config.py`, `engine.py`) wurde die Konzentration auf $\kappa_{\text{Motivation}} = \kappa_{\text{Integration}} = 9{,}0$ ($\sigma_{\text{total}} = 0{,}1637$) rekalibriert, was den psychometrischen Normdaten (Academic Motivation Scale AMS, SELLMO) exakt entspricht. Zudem wurde die Erwerbstätigkeit auf ein *Zero-Inflated Beta*-Modell ($37{,}3\,\%$ bei $0\,\text{h}$, Median $15{,}0\,\text{h}$) nach der 22. DSW-Sozialerhebung umgestellt. Für historische Replikationsvergleiche steht `get_v41_legacy_config()` bereit; Details siehe [`docs/04_causal_and_simulation/config_audit_und_v5_roadmap.md`](docs/04_causal_and_simulation/config_audit_und_v5_roadmap.md).

### D. Konfigurations-Zombies & Hartcodierte Parameter (in DGP V5.0 behoben)
Der systematische Config-Audit deckte in V4.1 verwaiste Parameter in `CONFIG` (`gewicht_erwerb`, `gewicht_motivation_rauschen`, `gewicht_integration_rauschen`), 5 ungesicherte `.get()`-Defaults in `engine.py` sowie über 25 hartcodierte *Magic Numbers* auf.
- **Status (in DGP V5.0 behoben):** In **Version 5.0** sind alle Zombies entfernt, sämtliche Szenario-Schalter und *Magic Numbers* vollständig in `CONFIG` parametrisiert und durch automatisierte Unit-Tests ([`tests/test_simulation_v5.py`](tests/test_simulation_v5.py)) abgesichert.

### E. Dynamische Trajektorien & Realism-Mode (Backlog / Nice-to-Have)
- **Motivationsverläufe:** Erweiterung der linearen Feedback-Gleichungen um akkumulierte Desillusionsprozesse nach der Erwartungs-Wert-Theorie (Eccles & Wigfield; Heublein et al. 2017/2022).
- **Peer-Gruppendynamik (Realism-Mode):** Modellierung sozialer Netzwerke und informeller Lerngruppen zur Pufferung von Workload und Isolationsrisiken (Tinto-Modell).

---

## 2. Methodische Limitationen der Kausalinferenz

### A. Reaktiver Confounding Bias & Extrapolation in Sequenzmodellen
Im Datengenerator treten Studierende dann dem Support bei, wenn sie sich in einer **akuten Krise** befinden (z. B. verhauene Prüfung $\rightarrow +20\%$ Support-Chance). 

**Einschränkung:** Rekurrierende Modelle (GRU, DeepHit) lernen über ihre temporale Historie, dass das Signal `support_active = 1` ein Indikator für ein tiefes Leistungs- und Motivationstief ist. Werden im kontrafaktischen Testset alle Support-Flags auf `1` gesetzt, extrapolieren Sequenzmodelle außerhalb ihrer Trainingsverteilung, da sie "Dauer-Support in Krisen" selten gesehen haben.

*Future Work / Lösung:* Weiterentwicklung des implementierten **Double Machine Learning (DML)** Ansatzes (`dml_orthogonal_survival.py`) mit zweistufiger Residual-Orthogonalisierung (2SRI).

### B. KPI-Operationalisierung im Dashboard
Die in `kpi.md` definierten Metriken für Wirksamkeit und Zielgruppenerreichung wurden im Analyse-Backend berechnet und geloggt, aus Zeitgründen jedoch nicht vollständig in der Benutzeroberfläche des Dashboards visualisiert.

### C. Feature Engineering Design-Entscheidungen
- **`temporal='cum'`**: Die kummulative Zählweise verwendet einen inklusiven `cumsum` (die aktuelle Prüfung/Semester ist inkludiert). Dies ist eine bewusste Designentscheidung zur Abbildung des momentanen Wissensstands.
- **One-Hot-Encoding (OHE)**: Es besteht eine bewusste Inkonsistenz zwischen den Formaten. Die Tensor-Formate verwenden 5 `stg_OHE`-Spalten, während die Panel-Formate 4 Spalten (Reference Encoding) verwenden. Dies sollte bei der Modellierung beachtet werden.

---

## 3. Dokumentations- & Artefakt-Transparenz

Sämtliche Analyseberichte, Modellvergleiche und Audit-Protokolle im Ordner `Artifacts/` sowie in der System-Historie wurden in transparenter Paarprogrammierung und direkter Generierung durch KI-Agenten (Antigravity IDE, Claude Opus/Sonnet, Gemini Pro/Flash) erstellt.
