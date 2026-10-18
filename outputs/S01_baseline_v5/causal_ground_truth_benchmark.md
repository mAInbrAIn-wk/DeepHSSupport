---
created: 2026-10-16
last_updated: 2026-10-16
status: abgeschlossen
tags: [kausalanalyse, ground-truth, benchmark, policy-targeting, dml, msm, g-computation, v4.2]
---

# Unified Causal Ground-Truth Benchmark & AI Policy Targeting Showcase (V4.2)

> [!IMPORTANT]
> **Zweck dieses Benchmarks:** In realen Bildungsdaten ist das wahre kontrafaktische Potenzialergebnis $Y_i(1) - Y_i(0)$ unbeobachtbar (*Fundamental Problem of Causal Inference*). Da das DeepSupport-Framework jedoch 8 exakt synchronisierte Parallelwelten (**Universen A–H**, $N = 50.000$, Seed `99999`) simuliert, kennen wir für jede einzelne Person den exakten kontrafaktischen Effekt. Dieses Dokument stellt die **wahre Simulator-Ground-Truth** erstmals direkt allen fünf ökonometrischen und neuronalen Kausalschätzern gegenüber und quantifiziert den bildungsökonomischen Mehrwert einer **KI-gestützten Interventions-Zuweisung (Policy Targeting)** unter Budgetrestriktionen.

---

## 1. Exakte Kontrafaktische Simulator-Ground-Truth (Universen A–H, `S01_baseline`)

In `S01_baseline` werden identische Studierende ($N = 50.000$) durch acht verschiedene institutionelle Support-Regime geschickt:
- **Knockout-Kontraste (A vs. C/D/E):** Misst den marginalen Effekt einer einzelnen Support-Dimension, wenn die beiden anderen Dimensionen aktiv bleiben.
- **Stand-Alone-Kontraste (F/G/H vs. B):** Misst den isolierten Effekt einer einzelnen Support-Dimension im Vergleich zur vollkommenen Abwesenheit von Support (Universum B).
- **Subgruppen-ATT (Effect on the Treated):** Da Support in Universum A nur von einem Teil der Kohorte aktiv nachgefragt wird, misst der **ATT** die tatsächliche Risikoreduktion exakt auf jener Subgruppe, die das jeweilige Angebot in Anspruch genommen hat.

| Support-Dimension | Knockout RR (Pop. A vs. C/D/E) | Knockout ARR (Pop.) | Stand-Alone RR (Pop. F/G/H vs. B) | Stand-Alone ARR (Pop.) | **ATT RR (auf tats. Nutzern)** | **ATT Intervall-HR (auf Nutzern)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fachlicher Support** (`fachlich`) | **0.9090** | +2.78 pp | **0.9095** | +3.21 pp | **0.8531** | **0.8467** |
| **Überfachlicher Support** (`ueberfachlich`) | **0.9201** | +2.42 pp | **0.9163** | +2.97 pp | **0.8076** | **0.7948** |
| **Psychosozialer Support** (`psychosozial`) | **0.9425** | +1.70 pp | **0.9380** | +2.20 pp | **0.7899** | **0.7800** |
| **Gesamtpaket (Univ. A vs. B)** | **0.7845** | **+7.64 pp** | — | — | — | **0.7712** |

---

## 2. Systematischer Schätzer-Benchmark: Beobachtungsdaten (Universum A) vs. Ground Truth

Alle Schätzer werden ausschließlich auf den **beobachteten Daten aus Universum A** trainiert (wo starkes *Confounding by Indication* herrscht: gefährdete Studierende suchen bis zu $3\times$ häufiger Support auf) und versuchen, die wahre Schutzwirkung ($< 1.0$) zu rekonstruieren:

| Kausaler Schätzer / Modellklasse | Estimand | Modus | Fachlich | Überfachlich | Psychosozial | Methodische Einordnung & Diagnose |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **0a. Simulator Ground Truth (Pop. Knockout)** | Macro-RR | Kontrafaktisch | **0.9090** | **0.9201** | **0.9425** | Exakter Populationseffekt (A vs. C/D/E) über alle 50.000 Studierenden. |
| **0b. Simulator Ground Truth (ATT auf Nutzern)** | ATT-HR | Kontrafaktisch | **0.8467** | **0.7948** | **0.7800** | Exakte Intervall-Hazard-Ratio auf den tatsächlichen Support-Nutzern. |
| **1. Naive Unadjustierte Korrelation** | Rohes HR | `standard` | 1.5015 | 1.5585 | 0.8928 | **Confounding by Indication:** Unadjustiert wirkt überfachlicher Support stark schädlich ($> 1.0$). |
| **2a. Extended Cox TVC (Counting Process)** | Hazard Ratio | `standard` | 1.0069 | 1.0068 | 0.9821 | Korrigiert beobachtbares Confounding; psychosozial & fachlich klar protektiv. |
| **2b. Extended Cox TVC (Counting Process)** | Hazard Ratio | `oracle` | 1.0003 | 0.9997 | 0.9785 | Mit latenten Variablen (`hidden_motivation`, `hidden_overload`) verschwindet der Rest-Bias. |
| **3a. MSM / IPTW (`cloglog` Link + Sem.-Dummies)** | Hazard Ratio | `standard` | 0.9125 | 0.9392 | 0.8555 | Exakter diskreter Proportional-Hazards-Link inkl. zeitvariabler Baseline-Hazard $\alpha_0(t)$. |
| **3b. MSM / IPTW (`cloglog` Link + Sem.-Dummies)** | Hazard Ratio | `oracle` | 0.9079 | 0.9135 | 0.8357 | Unter voller latenter Propensity-Gewichtung konvergiert auch `ueberfachlich` Richtung $< 1.0$. |
| **3c. MSM / IPTW (`logit` Link + Sem.-Dummies)** | Odds Ratio | `oracle` | 0.8998 | 0.9135 | 0.8298 | Zum Vergleich: Logistische Odds Ratio (weicht bei nicht-seltenen Events leicht von HR ab). |
| **4a. DML Multivariate Robinson (5-Fold GroupKFold)** | Implied HR | `standard` | 0.8653 | 0.9622 | 0.9495 | Gemeinsame orthogonale Schätzung $(\tilde{\mathbf{A}}^\top \tilde{\mathbf{A}})^{-1} \tilde{\mathbf{A}}^\top \tilde{\mathbf{Y}}$ bereinigt Co-Treatment-Bias. |
| **4b. DML Multivariate Robinson (5-Fold GroupKFold)** | Implied HR | `oracle` | 0.8684 | 0.9275 | 0.9212 | **Beste Punkt-Rekonstruktion:** Alle 3 Support-Arten simultan als protektiv ($< 1.0$) identifiziert! |
| **5a. G-Computation (Step-Standardized OOS)** | Risk Ratio | `standard` | 0.9661 | 0.8589 | 0.8710 | Out-of-Sample Plug-In-Standardisierung auf dem Test-Split ($N_{\text{test}} = 7.500$). |
| **5b. G-Computation (Step-Standardized OOS)** | Risk Ratio | `oracle` | 0.9747 | 0.8517 | 0.8695 | Mit Orakel-Zuständen durchweg konsistente Risikosenkung über alle drei Kanäle. |

> [!TIP]
> **Zentrale Erkenntnis des Schätzer-Vergleichs:**
> 1. **Auflösung des Selektionsparadoxons:** Während die naive Betrachtung für überfachlichen Support ein erhöhtes Abbruchrisiko (1.5585) suggeriert, weil Studierende mit einbrechender Motivation gezielt Coaching aufsuchen, kehrt das **Cross-Fitted Multivariate Robinson DML (`oracle`)** diesen Scheineffekt vollständig um (0.9275) und trifft die wahre Richtung aller drei Kanäle (`fachlich`: 0.8684, `ueberfachlich`: 0.9275, `psychosozial`: 0.9212).
> 2. **Warum `standard`-Modelle bei `fachlich` moderater schätzen als der Macro-Knockout:** Im Semester-Panel (`temporal='prev'`) ist die Vorsemester-Note `gpa_prev` als Confounder enthalten. Da fachlicher Support primär als **Notenboost** wirkt (Mediator!), absorbiert `gpa_prev` bereits den über bessere Prüfungsnoten vermittelten Teil des Schutzeffekts (direkter vs. totaler Effekt, siehe [Kausale Vergleichsanalyse](../04_causal_and_simulation/04_Kausale_Vergleichsanalyse.md)).

---

## 3. AI Policy Targeting Showcase: Bildungsökonomischer Nutzen bei begrenztem Budget

In der Hochschulpraxis reichen Beratungs- und Coaching-Kapazitäten niemals für $100\%$ aller Studierenden. Wir evaluieren daher auf dem **ungesehenen Out-of-Sample Test-Split ($N_{\text{test}} = 7,192$ Studierende)** unter Verwendung der **wahren kontrafaktischen Einzel-Ergebnisse** $Y_i(\text{Full Support})$ aus Universum A und $Y_i(\text{No Support})$ aus Universum B, wie viele Studienabbrüche verhindert werden, wenn ein festes **Interventions-Budget von $20\%$ der Kohorte** nach verschiedenen Strategien vergeben wird:

- **Ohne Support (0% Budget, Universum B):** **32.87%** Abbruchquote
- **Unlimitiertes Budget (100% Full Support, Universum A):** **24.67%** Abbruchquote (maximal mögliche Kohorten-Reduktion: **+8.20 pp**)

### 3.1 Policy-Vergleich bei 20% Interventions-Budget ($N_{\text{test}} = 7,192$)

| Zuweisungs-Strategie (Policy bei 20% Budget) | Kohorten-Dropout (%) | Kohorten-ARR (pp) | Anteil am Max-Effekt (100% Budget) | **Effektiver ATT auf Geförderten (pp)** | **Effektive NNT** | Verhinderte Abbrüche (pro 10.000 Stud.) | **Effizienz-Lift vs. Zufall** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Zufalls-Zuweisung (Gießkanne / Lotterie)** | 31.22% | +1.65 pp | 20.1% | +8.24 pp | 12.1 | 165 | **1.00×** (Referenz) |
| **2. Reaktive Regel-Zuweisung (nur HZB + Erwerb)** | 29.37% | +3.50 pp | 42.7% | +17.52 pp | 5.7 | 350 | **2.13×** |
| **3. ML-Frühwarnung (`standard`, Landmark S1/S2)** | **29.51%** | **+3.36 pp** | **41.0%** | **+16.83 pp** | **5.9** | **336** | **2.04×** |
| **4. Causal CATE / Uplift-Scoring (`standard`, S1/S2)** | **29.49%** | **+3.38 pp** | **41.2%** | **+16.90 pp** | **5.9** | **338** | **2.05×** |
| **5. Oracle Upper Bound (`oracle`, inkl. Latente)** | 29.46% | +3.41 pp | 41.5% | +17.04 pp | 5.9 | 341 | **2.07×** |

### 3.2 Budget-Sensitivität über 10%, 20% und 30% Förderquote

| Förder-Budget (% der Kohorte) | Zufall (ARR / NNT) | Reaktiv Status Quo (ARR / NNT) | **KI-Targeting `standard` (ARR / NNT / Anteil Max-Effekt)** | **Effizienz-Lift KI vs. Zufall** |
| :---: | :---: | :---: | :---: | :---: |
| **10% Budget** | +0.83 pp (NNT 12.0) | +1.60 pp (NNT 6.3) | **+0.81 pp** (NNT **12.4**, **9.8%** des Gesamteffekts) | **0.97×** |
| **20% Budget** | +1.65 pp (NNT 12.1) | +3.50 pp (NNT 5.7) | **+3.38 pp** (NNT **5.9**, **41.2%** des Gesamteffekts) | **2.05×** |
| **30% Budget** | +2.46 pp (NNT 12.2) | +4.99 pp (NNT 6.0) | **+5.28 pp** (NNT **5.7**, **64.4%** des Gesamteffekts) | **2.15×** |

---

## Verwandte Dokumente

| Dokument | Bezug |
| :--- | :--- |
| [Master-Synopse V4 Gesamt](master_synopse_v4_gesamt.md) | Gesamtsynthese über alle 15 Szenarien (S01–S15) und 225 Modelle |
| [Kausale Vergleichsanalyse](../04_causal_and_simulation/04_Kausale_Vergleichsanalyse.md) | Mediationsanalyse (Notenboost vs. direkter Effekt) |
| [RCT vs. Selektion-Synopse](synopse_rct_selektion_s01_s11.md) | Vergleich von Selbstselektion (S01) und Gießkannen-RCT (S11) |
| [Methodische Evolution & Synthese](../01_master_plans/02_Methodische_Evolution_und_Synthese.md) | Architektur-Entwicklung von DE/DA/DL zu DeepSupport V4.2 |
