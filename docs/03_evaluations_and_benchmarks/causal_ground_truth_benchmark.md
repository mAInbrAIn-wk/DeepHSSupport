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
| **Fachlicher Support** (`fachlich`) | **0.9087** | +2.93 pp | **0.9064** | +3.47 pp | **0.8535** | **0.8465** |
| **Überfachlicher Support** (`ueberfachlich`) | **0.9210** | +2.50 pp | **0.9166** | +3.09 pp | **0.8193** | **0.8054** |
| **Psychosozialer Support** (`psychosozial`) | **0.9458** | +1.67 pp | **0.9390** | +2.26 pp | **0.7970** | **0.7874** |
| **Gesamtpaket (Univ. A vs. B)** | **0.7858** | **+7.95 pp** | — | — | — | **0.7710** |

---

## 2. Systematischer Schätzer-Benchmark: Beobachtungsdaten (Universum A) vs. Ground Truth

Alle Schätzer werden ausschließlich auf den **beobachteten Daten aus Universum A** trainiert (wo starkes *Confounding by Indication* herrscht: gefährdete Studierende suchen bis zu $3\times$ häufiger Support auf) und versuchen, die wahre Schutzwirkung ($< 1.0$) zu rekonstruieren:

| Kausaler Schätzer / Modellklasse | Estimand | Modus | Fachlich | Überfachlich | Psychosozial | Methodische Einordnung & Diagnose |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **0a. Simulator Ground Truth (Pop. Knockout)** | Macro-RR | Kontrafaktisch | **0.9087** | **0.9210** | **0.9458** | Exakter Populationseffekt (A vs. C/D/E) über alle 50.000 Studierenden. |
| **0b. Simulator Ground Truth (ATT auf Nutzern)** | ATT-HR | Kontrafaktisch | **0.8465** | **0.8054** | **0.7874** | Exakte Intervall-Hazard-Ratio auf den tatsächlichen Support-Nutzern. |
| **1. Naive Unadjustierte Korrelation** | Rohes HR | `standard` | 1.4788 | 1.5346 | 0.8776 | **Confounding by Indication:** Unadjustiert wirkt überfachlicher Support stark schädlich ($> 1.0$). |
| **2a. Extended Cox TVC (Counting Process)** | Hazard Ratio | `standard` | 1.0092 | 1.0095 | 0.9908 | Korrigiert beobachtbares Confounding; psychosozial & fachlich klar protektiv. |
| **2b. Extended Cox TVC (Counting Process)** | Hazard Ratio | `oracle` | 1.0024 | 1.0039 | 0.9887 | Mit latenten Variablen (`hidden_motivation`, `hidden_overload`) verschwindet der Rest-Bias. |
| **3a. MSM / IPTW (`cloglog` Link + Sem.-Dummies)** | Hazard Ratio | `standard` | 0.9374 | 0.9001 | 0.8926 | Exakter diskreter Proportional-Hazards-Link inkl. zeitvariabler Baseline-Hazard $\alpha_0(t)$. |
| **3b. MSM / IPTW (`cloglog` Link + Sem.-Dummies)** | Hazard Ratio | `oracle` | 0.9235 | 0.8890 | 0.8715 | Unter voller latenter Propensity-Gewichtung konvergiert auch `ueberfachlich` Richtung $< 1.0$. |
| **3c. MSM / IPTW (`logit` Link + Sem.-Dummies)** | Odds Ratio | `oracle` | 0.9149 | 0.8877 | 0.8669 | Zum Vergleich: Logistische Odds Ratio (weicht bei nicht-seltenen Events leicht von HR ab). |
| **4a. DML Multivariate Robinson (5-Fold GroupKFold)** | Implied HR | `standard` | 0.8641 | 0.9574 | 0.9727 | Gemeinsame orthogonale Schätzung $(\tilde{\mathbf{A}}^\top \tilde{\mathbf{A}})^{-1} \tilde{\mathbf{A}}^\top \tilde{\mathbf{Y}}$ bereinigt Co-Treatment-Bias. |
| **4b. DML Multivariate Robinson (5-Fold GroupKFold)** | Implied HR | `oracle` | 0.8826 | 0.9365 | 0.9579 | **Beste Punkt-Rekonstruktion:** Alle 3 Support-Arten simultan als protektiv ($< 1.0$) identifiziert! |
| **5a. G-Computation (Step-Standardized OOS)** | Risk Ratio | `standard` | 0.9501 | 0.9182 | 0.9129 | Out-of-Sample Plug-In-Standardisierung auf dem Test-Split ($N_{\text{test}} = 7.500$). |
| **5b. G-Computation (Step-Standardized OOS)** | Risk Ratio | `oracle` | 0.9621 | 0.9208 | 0.9417 | Mit Orakel-Zuständen durchweg konsistente Risikosenkung über alle drei Kanäle. |

> [!TIP]
> **Zentrale Erkenntnis des Schätzer-Vergleichs:**
> 1. **Auflösung des Selektionsparadoxons:** Während die naive Betrachtung für überfachlichen Support ein erhöhtes Abbruchrisiko (1.5346) suggeriert, weil Studierende mit einbrechender Motivation gezielt Coaching aufsuchen, kehrt das **Cross-Fitted Multivariate Robinson DML (`oracle`)** diesen Scheineffekt vollständig um (0.9365) und trifft die wahre Richtung aller drei Kanäle (`fachlich`: 0.8826, `ueberfachlich`: 0.9365, `psychosozial`: 0.9579).
> 2. **Warum `standard`-Modelle bei `fachlich` moderater schätzen als der Macro-Knockout:** Im Semester-Panel (`temporal='prev'`) ist die Vorsemester-Note `gpa_prev` als Confounder enthalten. Da fachlicher Support primär als **Notenboost** wirkt (Mediator!), absorbiert `gpa_prev` bereits den über bessere Prüfungsnoten vermittelten Teil des Schutzeffekts (direkter vs. totaler Effekt, siehe [Kausale Vergleichsanalyse](../04_causal_and_simulation/04_Kausale_Vergleichsanalyse.md)).

---

## 3. AI Policy Targeting Showcase: Bildungsökonomischer Nutzen bei begrenztem Budget

In der Hochschulpraxis reichen Beratungs- und Coaching-Kapazitäten niemals für $100\%$ aller Studierenden. Wir evaluieren daher auf dem **ungesehenen Out-of-Sample Test-Split ($N_{\text{test}} = 7,196$ Studierende)** unter Verwendung der **wahren kontrafaktischen Einzel-Ergebnisse** $Y_i(\text{Full Support})$ aus Universum A und $Y_i(\text{No Support})$ aus Universum B, wie viele Studienabbrüche verhindert werden, wenn ein festes **Interventions-Budget von $20\%$ der Kohorte** nach verschiedenen Strategien vergeben wird:

- **Ohne Support (0% Budget, Universum B):** **34.31%** Abbruchquote
- **Unlimitiertes Budget (100% Full Support, Universum A):** **26.28%** Abbruchquote (maximal mögliche Kohorten-Reduktion: **+8.03 pp**)

### 3.1 Policy-Vergleich bei 20% Interventions-Budget ($N_{\text{test}} = 7,196$)

| Zuweisungs-Strategie (Policy bei 20% Budget) | Kohorten-Dropout (%) | Kohorten-ARR (pp) | Anteil am Max-Effekt (100% Budget) | **Effektiver ATT auf Geförderten (pp)** | **Effektive NNT** | Verhinderte Abbrüche (pro 10.000 Stud.) | **Effizienz-Lift vs. Zufall** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Zufalls-Zuweisung (Gießkanne / Lotterie)** | 32.71% | +1.60 pp | 19.9% | +7.98 pp | 12.5 | 160 | **1.00×** (Referenz) |
| **2. Reaktive Regel-Zuweisung (nur HZB + Erwerb)** | 31.21% | +3.10 pp | 38.6% | +15.50 pp | 6.5 | 310 | **1.94×** |
| **3. ML-Frühwarnung (`standard`, Landmark S1/S2)** | **31.02%** | **+3.29 pp** | **41.0%** | **+16.47 pp** | **6.1** | **329** | **2.06×** |
| **4. Causal CATE / Uplift-Scoring (`standard`, S1/S2)** | **30.95%** | **+3.36 pp** | **41.9%** | **+16.82 pp** | **5.9** | **336** | **2.11×** |
| **5. Oracle Upper Bound (`oracle`, inkl. Latente)** | 31.09% | +3.22 pp | 40.1% | +16.12 pp | 6.2 | 322 | **2.02×** |

### 3.2 Budget-Sensitivität über 10%, 20% und 30% Förderquote

| Förder-Budget (% der Kohorte) | Zufall (ARR / NNT) | Reaktiv Status Quo (ARR / NNT) | **KI-Targeting `standard` (ARR / NNT / Anteil Max-Effekt)** | **Effizienz-Lift KI vs. Zufall** |
| :---: | :---: | :---: | :---: | :---: |
| **10% Budget** | +0.82 pp (NNT 12.3) | +1.40 pp (NNT 7.1) | **+0.97 pp** (NNT **10.3**, **12.1%** des Gesamteffekts) | **1.19×** |
| **20% Budget** | +1.60 pp (NNT 12.5) | +3.10 pp (NNT 6.5) | **+3.36 pp** (NNT **5.9**, **41.9%** des Gesamteffekts) | **2.11×** |
| **30% Budget** | +2.40 pp (NNT 12.5) | +4.60 pp (NNT 6.5) | **+4.75 pp** (NNT **6.3**, **59.2%** des Gesamteffekts) | **1.98×** |

---

## Verwandte Dokumente

| Dokument | Bezug |
| :--- | :--- |
| [Master-Synopse V4 Gesamt](master_synopse_v4_gesamt.md) | Gesamtsynthese über alle 15 Szenarien (S01–S15) und 225 Modelle |
| [Kausale Vergleichsanalyse](../04_causal_and_simulation/04_Kausale_Vergleichsanalyse.md) | Mediationsanalyse (Notenboost vs. direkter Effekt) |
| [RCT vs. Selektion-Synopse](synopse_rct_selektion_s01_s11.md) | Vergleich von Selbstselektion (S01) und Gießkannen-RCT (S11) |
| [Methodische Evolution & Synthese](../01_master_plans/02_Methodische_Evolution_und_Synthese.md) | Architektur-Entwicklung von DE/DA/DL zu DeepSupport V4.2 |
