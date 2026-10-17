---
created: 2026-09-04
last_updated: 2026-10-17
status: abgeschlossen
tags: [evaluation, v4.2, v4.2.5, synopse, gesamtsynthese, causal-benchmark, policy-targeting]
---

# Master-Synopse & Gesamtsynthese: V4.2 Sensitivity Grid & Methoden-Benchmark (S01–S15)

**Umfang des Sensitivitätsgitters:** 15 Simulationswelten × 8 Universen (A–H) × 15 Fast-Grid-Modelle = **225 trainierte neuronale Netze** (ergänzt durch die **Heavy Deep Suite**, das **PyTorch/PyCox-Portfolio**, den **V4.2.5 Post-Refactor Gesamt-Benchmark** und den **Kausalen Ground-Truth Benchmark**).  
**Kohortengröße:** $N = 50.000$ Studierende je Universum (Seed 99999).  
**Datum:** 2026-10-17 | **Status:** Abgeschlossen (V4.1/V4.2.5 Datenbasis)

> [!NOTE]
> **Epistemische Einordnung (Überprüfung der Reasoning-Kette vs. Empirie):**  
> Sämtliche Kennzahlen in dieser Synopse entstammen einem **synthetischen Datengenerator (DGP)**. Sie stellen daher **keine empirischen Beweise** über reale Hochschulen dar, sondern dienen dem **kontrollierten Methodenvergleich** und der **Überprüfung der methodischen Reasoning-Kette**: Wenn im Simulator bekannte Kausalmechanismen, Schwellenwerte und Selektionsverzerrungen (*Confounding by Indication*) einprogrammiert sind, welche Modellarchitekturen und kausalen Schätzer sind dann in der Lage, diese Struktur aus reinen Beobachtungsdaten (Welt A) verzerrungsfrei zu rekonstruieren?

---

## 1. Synoptische Gesamttabelle der simulierten Ground-Truth-Effekte (Welt A vs. Welt B)

| Nr. | Szenario-Name | Untersuchte DGP-Dimension | Parameter-Override im Simulator | Dropout A (Full) | Dropout B (None) | ARR (pp) | RR (A vs B) | NNT |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 01 | `S01_baseline` | Baseline | `Standard-Kalibrierung` | 29.16% | 37.10% | **+7.95 pp** | **0.786** | **12.6** |
| 02 | `S02_supp_half` | Supportwirkung | `support_effect_multiplier = 2.5 (0.5×)` | 32.69% | 37.10% | **+4.41 pp** | **0.881** | **22.7** |
| 03 | `S03_supp_double` | Supportwirkung | `support_effect_multiplier = 10.0 (2.0×)` | 25.31% | 37.10% | **+11.79 pp** | **0.682** | **8.5** |
| 04 | `S04_grade_half` | Notenboost | `gewicht_support_boost = 0.04 (0.5×)` | 30.60% | 37.10% | **+6.50 pp** | **0.825** | **15.4** |
| 05 | `S05_grade_double` | Notenboost | `gewicht_support_boost = 0.16 (2.0×)` | 27.57% | 37.10% | **+9.53 pp** | **0.743** | **10.5** |
| 06 | `S06_grade_quad` | Notenboost | `gewicht_support_boost = 0.32 (4.0×)` | 27.08% | 37.10% | **+10.02 pp** | **0.730** | **10.0** |
| 07 | `S07_noise_half` | Stoch. Rauschen | `gewicht_rauschen = 0.09 (0.5×)` | 26.71% | 33.05% | **+6.33 pp** | **0.808** | **15.8** |
| 08 | `S08_noise_double` | Stoch. Rauschen | `gewicht_rauschen = 0.36 (2.0×)` | 33.16% | 41.03% | **+7.88 pp** | **0.808** | **12.7** |
| 09 | `S09_cost_zero` | Zeitkosten | `support_kosten_faktor = 0.0 (0h)` | 28.61% | 37.10% | **+8.50 pp** | **0.771** | **11.8** |
| 10 | `S10_cost_double` | Zeitkosten | `support_kosten_faktor = 2.0 (60h)` | 29.71% | 37.10% | **+7.39 pp** | **0.801** | **13.5** |
| 11 | `S11_rct_calibrated` | Selektionsmechanismus | `rct_support_uptake = True (Randomisiert)` | 32.65% | 37.10% | **+4.45 pp** | **0.880** | **22.5** |
| 12 | `S12_overload_half` | Überlastungsstrafe | `overload_penalty_factor = 0.05 (0.5×)` | 26.02% | 34.14% | **+8.13 pp** | **0.762** | **12.3** |
| 13 | `S13_overload_double` | Überlastungsstrafe | `overload_penalty_factor = 0.20 (2.0×)` | 34.58% | 41.84% | **+7.26 pp** | **0.827** | **13.8** |
| 14 | `S14_overload_cap` | Überlastungs-Cap | `overload_penalty_cap = 0.15 (0.5×)` | 26.69% | 35.04% | **+8.35 pp** | **0.762** | **12.0** |
| 15 | `S15_cost_effect_double` | Kombination | `Multiplier = 10.0 + Kostenfaktor = 2.0` | 25.82% | 37.10% | **+11.28 pp** | **0.696** | **8.9** |

---

## 2. Sensitivitäts-Hierarchie der DGP-Parameter

Wie stark reagiert der simulierte Interventionseffekt (**ARR**) auf Parametervariationen im Datengenerator?

1. **Supportwirkung (Multiplikator):** **7.4 pp Spannweite** (4.41 pp in S02 bis 11.79 pp in S03). Primärer Hebel auf die latenten Zustandsgrößen Motivation und soziale Integration.
2. **Selektions-Regime (RCT vs. Krisen-Selbstselektion):** **3.5 pp Spannweite** (4.45 pp bei ungerichteter RCT-Vergabe in S11 vs. 7.95 pp bei krisengetriebener Inanspruchnahme in S01). Illustriert innerhalb der Simulationslogik, dass Förderangebote vor allem dort einen Hebel entfalten, wo Studierende tatsächlich unter der Dropout-Schwelle ($\text{Motivation} < 0{,}40$) liegen oder Wiederholungsprüfungen absolvieren.
3. **Notenboost (Fachlicher Support):** **3.5 pp Spannweite** (6.50 pp in S04 bis 10.02 pp in S06). Direkter Notenhebel reduziert Fehlversuche und verhindert dadurch kaskadierende Motivationsverluste.
4. **Zeitkosten (Modulabwurf):** **1.1 pp Spannweite** (8.50 pp bei 0h in S09 vs. 7.39 pp bei 60h in S10). Da im Simulator überzählige Module bei Zeitnot probabilistisch abgeworfen werden, führt der Zeitaufwand für Support primär zu einer moderaten Studienzeitstreckung statt zu direktem Studienabbruch.
5. **Überlastungs-Kalibrierung:** **1.1 pp Spannweite** (7.26 pp in S13 bis 8.35 pp in S14). Verschiebt das Basisrisiko in Welt A und B nahezu parallel, sodass die relative Risikoreduktion erhalten bleibt.

---

## 3. Gesamtübersicht des Modell-Portfolios & Verweise auf Detail-Gegenüberstellungen

> [!TIP]
> **Vollständige Side-by-Side-Gegenüberstellung aller >25 Modelle (V4.2.5 Post-Refactor Benchmark):**  
> Für einen lückenlosen Einzelvergleich sämtlicher Modellarchitekturen über alle 6 Familien (inkl. Pre-Refactor V4.2 vs. Post-Refactor V4.2.5, Keras 3 vs. PyTorch 2.x Paritätsanalyse, klassenweisen PR-AUCs und Konfidenzintervallen) siehe **[`modell_gegenueberstellung_gesamt_v425.md`](modell_gegenueberstellung_gesamt_v425.md)**.

Da das Projekt insgesamt **über 25 Modellvarianten** in zwei Deep-Learning-Frameworks (**TensorFlow/Keras 3** und **PyTorch 2.x / PyCox**) sowie fünf Feature-Modi (`standard`, `gradeblind`, `blind`, `oracle`, `realistic`) umfasst, sind in der nachfolgenden Tabelle alle sechs Modellfamilien zusammenfassend gegenübergestellt und mit den jeweiligen Detailberichten verknüpft:

| Modellfamilie | Enthaltene Architekturen (Keras & PyTorch) | Zielgröße & Granularität | Zentrale Kennzahlen auf `S01_baseline` (`standard` / `gradeblind` — V4.2.5) | Detail-Gegenüberstellung (Link) |
| :--- | :--- | :--- | :--- | :--- |
| **1. Statische Landmark-Modelle & Representation Learning** ($T_0 = \text{Sem 2}$) | `XGBoost`, `Keras MLP`, `LogisticHazard Landmark`, `DeepSurv Landmark`, `RandomForest`, `SVR`, `Ridge`, `32-D Transformer Embedding + XGBoost` | Frühwarnung nach Sem. 2: Binärer Dropout, 4-Klassen-Status & Hurdle Stage 2 Abschlussnote (`graduates_only=True`) | **Binärer Dropout:** `LogisticHazard` & `MLP` $\text{ROC-AUC} = \mathbf{0{,}8740}$, $\text{PR-AUC} = \mathbf{0{,}7705}$ ($\text{BSS} = +0{,}412$)<br>**4-Klassen-Status (Hybrid Embedding):** $\text{Accuracy} = \mathbf{79{,}6\,\%}$, $\text{Weighted-F1} = \mathbf{0{,}7691}$, $\text{OVR-AUC} = \mathbf{0{,}8500}$<br>**Hurdle Stage 2 Note:** $R^2 = \mathbf{0{,}8706}$ (`standard`) / $\mathbf{0{,}7751}$ (`gradeblind`) | [**`modell_gegenueberstellung_gesamt_v425.md` (Kap. 2 & 5.2)**](modell_gegenueberstellung_gesamt_v425.md)<br>[`model_architectures.md`](../02_architectures_and_models/model_architectures.md) |
| **2. Counting-Process Panel- & Competing-Risks Survival Suite** | `Extended Cox` (`strata=t_stop`), `Extended DeepSurv` (+ Breslow Step-Haz.), `Extended LogisticHazard`, `PyTorch CoxPH / CoxTime`, `Dynamic DeepHit` (3-State Softmax CR) | Counting-Process `(t_start, t_stop]` ($N=259{,}568$) & Competing Risks (Dropout vs. Abschluss) | **Step-Hazard Parität (`standard` / `oracle`):** `Keras LH` ($\text{AUC}=0{,}8152/0{,}8202$), `Keras DeepSurv` ($0{,}8165/0{,}8212$), `PyTorch LH` ($0{,}8160/0{,}8207$, $\text{BSS}=+0{,}143$), `PyTorch CoxPH` ($0{,}8160/0{,}8206$), `PyTorch CoxTime` ($0{,}8157/0{,}8201$)<br>**Competing Risks (`Dynamic DeepHit`):** Dropout $\text{AUC}=0{,}8177$, Graduation $\text{AUC}=\mathbf{0{,}9995}$ | [**`modell_gegenueberstellung_gesamt_v425.md` (Kap. 3)**](modell_gegenueberstellung_gesamt_v425.md)<br>[`methodenvergleich_logistic_hazard_keras_vs_pycox.md`](methodenvergleich_logistic_hazard_keras_vs_pycox.md)<br>[`grundlagen_survival_analyse_und_zensierung.md`](../04_causal_and_simulation/grundlagen_survival_analyse_und_zensierung.md) |
| **3. 3D-Sequenz-Survival (Fast-Grid & Heavy)** | `Semester GRU`, `Semester Causal Transformer`, `PyTorch Dynamic DeepHit (Causal GRU)`, `Exam GRU`, `Exam Causal Transformer` | Schrittweiser Dropout-Hazard (16 Sem. bzw. 40 Prüfungen) | **Semester-Ebene:** `Keras/PyTorch GRU` $\text{ROC-AUC} = 0{,}8121–\mathbf{0{,}8196}$, $\text{PR-AUC} = \mathbf{0{,}2988}$ ($7{,}1\times$ Lift)<br>**Prüfungs-Ebene (`Exam GRU`):** $\text{Step ROC-AUC} = \mathbf{0{,}8976}$, $\text{Student ROC-AUC} = \mathbf{0{,}7489}$, $\text{Brier} = \mathbf{0{,}0150}$ | [**`modell_gegenueberstellung_gesamt_v425.md` (Kap. 4.1)**](modell_gegenueberstellung_gesamt_v425.md)<br>[Einzelsynopsen S01–S15](#einzelsynopsen-nach-dimension) |
| **4. Gradeblind 3D-Sequenz-Transformer Regressoren** | `DeepSemesterTransformerRegressor`, `DeepExamTransformerRegressor`, `PyTorchExamTransformerRegressor` | Vollsequenz-Abschlussnoten-Regression ohne Einzelnoten (`gradeblind`, `graduates_only=True`) | **Semester Transformer (`gradeblind`):** $R^2 = \mathbf{0{,}7838}$ (vs. $0{,}2222$ Pre-Refactor)<br>**Exam Transformer (`gradeblind`):** Keras $R^2 = \mathbf{0{,}7839}$, PyTorch $R^2 = \mathbf{0{,}8065}$ ($\text{RMSE} = 0{,}2634$) | [**`modell_gegenueberstellung_gesamt_v425.md` (Kap. 4.2)**](modell_gegenueberstellung_gesamt_v425.md)<br>[`pytorch_lxc_benchmark_evaluation_v42.md`](pytorch_lxc_benchmark_evaluation_v42.md) |
| **5. Heavy Suite: Dual-Head Next-Exam Autoregressoren** | `AutoregressiveNextExamGRU`, `AutoregressiveDeepTransformer` (Sin/Cos PosEnc, Pre-LN, $w_{ik}=1/K_i$) | Simultan: Note $Y_{k+1}$ (MSE) + Bestehen $P(\text{pass}_{k+1})$ (BCE) auf 802k Prüfungen | **Keras GRU / Transformer ($1/K_i$):** Note $R^2 = 0{,}6885 / 0{,}6913$, Pass $\text{ROC-AUC} = 0{,}9392 / 0{,}9380$, Fail $\text{PR-AUC}(y=0) = \mathbf{0{,}7706}$ (vs. $0{,}5852$ ungewicht.)<br>**PyTorch GRU / Transformer ($1/K_i$):** Note $R^2 = \mathbf{0{,}7036 / 0{,}7050}$, Pass $\text{ROC-AUC} = \mathbf{0{,}9406 / 0{,}9409}$, Fail $\text{PR-AUC}(y=0) = \mathbf{0{,}7779}$ ($\text{BSS}=+0{,}502$) | [**`modell_gegenueberstellung_gesamt_v425.md` (Kap. 5.1)**](modell_gegenueberstellung_gesamt_v425.md)<br>[`synopse_heavy_suite_s01_s07_s08.md`](synopse_heavy_suite_s01_s07_s08.md) |
| **6. Kausale Inferenz & Policy-Targeting Suite** | `Naive`, `Extended Cox PHReg`, `Cloglog-MSM (IPTW)`, `Keras/PyTorch Orthogonal DML`, `Deep Transformer-DML`, `G-Computation`, `CATE Uplift` | Kausale Effekt-Rekonstruktion (HR/RR) auf Welt A & Budget-Allokation (20 % Cap) | **Orthogonal DML (`oracle`):** $\text{HR}_{\text{fach}}=0{,}9135$, $\text{HR}_{\text{uebf}}=0{,}9410$, $\text{HR}_{\text{psych}}=0{,}9642$ (alle 95%-Boot-KIs $<1{,}0$)<br>**G-Computation Rollout (`standard`):** $\text{RR} = \mathbf{0{,}7611}$ vs. $\text{RR}_{\text{true}}(\text{A vs. B}) = 0{,}7868$. **CATE-Policy:** $2{,}11\times$ Effizienz-Lift | [**`modell_gegenueberstellung_gesamt_v425.md` (Kap. 6)**](modell_gegenueberstellung_gesamt_v425.md)<br>[`causal_ground_truth_benchmark.md`](causal_ground_truth_benchmark.md)<br>[`04_Kausale_Vergleichsanalyse.md`](../04_causal_and_simulation/04_Kausale_Vergleichsanalyse.md) |

---

## 4. Kausaler Ground-Truth Benchmark & KI-Policy-Targeting Showcase (`S01_baseline`)

In der vereinheitlichten Evaluation ([`causal_ground_truth_benchmark.md`](causal_ground_truth_benchmark.md) sowie Kapitel 6 in [`modell_gegenueberstellung_gesamt_v425.md`](modell_gegenueberstellung_gesamt_v425.md)) wird die methodische Reasoning-Kette geschlossen: Die bekannten kontrafaktischen Parameter der 8 Parallelwelten (A–H) dienen als Referenzmaßstab, an dem fünf Beobachtungsschätzer (nur mit Zugriff auf Welt A) sowie vier Förder-Allokationsregeln (unter 20 % Budgetrestriktion auf dem Holdout-Testsplit $N_{\text{test}} = 7.196$) gemessen werden.

### 4.1 Simulator Ground Truth (Universen A–H) vs. Beobachtungs-Schätzer (Welt A)

| Schätzer / Ground-Truth-Referenz | Modus | Fachlich | Überfachlich | Psychosozial | Methodische Einordnung im Benchmark |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **GT 1: Population Knockout RR** (A vs. C/D/E) | *Kontrafaktisch* | **0.9087** (+2.93 pp) | **0.9210** (+2.50 pp) | **0.9458** (+1.67 pp) | Marginaler Kohorten-Effekt bei Wegfall einer Säule (Gesamt A vs. B: **RR = 0.7858**, **+7.95 pp**) |
| **GT 2: Stand-Alone RR** (F/G/H vs. B) | *Kontrafaktisch* | **0.9064** (+3.47 pp) | **0.9166** (+3.09 pp) | **0.9390** (+2.26 pp) | Isolierte Wirksamkeit einer einzelnen Säule gegenüber Null-Support |
| **GT 3: Subgruppen-ATT Interval-HR** | *Kontrafaktisch* | **0.8465** (RR 0.8535) | **0.8054** (RR 0.8193) | **0.7874** (RR 0.7970) | Wahrer Schutzeffekt auf den tatsächlichen Nutzern in Welt A (Cloglog-HR) |
| **1. Naiver Unadjustierter Vergleich** | `standard` | 1.4788 | 1.5346 | 0.8776 | Massives *Confounding by Indication* durch krisengetriebene Inanspruchnahme |
| **2. Extended Cox (`PHReg`, `strata=t_stop`)** | `standard` / `oracle` | **0.9587** / **0.9581** | **0.9906** / **0.9881** | **0.9770** / **0.9685** | Nach Behebung der Float64-Auslöschung konsistent $\text{HR} < 1{,}0$ auf allen 3 Kanälen |
| **3. MSM / IPTW** (`cloglog` Gepooltes Hazard) | `standard` / `oracle` | **0.9374** / **0.9235** | **0.9001** / **0.8890** | **0.8926** / **0.8715** | Stabilisierte Propensity-Gewichte mit multinomialem Cluster-Bootstrap ($\text{HR}_{\text{all}} = 0{,}9170$) |
| **4. Double ML** (Multivariate Robinson, 5-Fold) | `standard` / `oracle` | **0.8952** / **0.9135** | **0.9711** / **0.9410** | **0.9737** / **0.9642** | Gemeinsame Orthogonalisierung aller 3 Behandlungen rekonstruiert die Population-GT auf $\pm 0{,}025$ |
| **5. G-Computation** (Step-Standardized & Rollout) | `standard` / `oracle` | **0.9501** / **0.9621** | **0.9182** / **0.9208** | **0.9129** / **0.9417** | 1-Step $\text{RR}_{\text{all}} = 0{,}8603$ (`standard`) / $0{,}8787$ (`oracle`), 16-Sem.-Rollout $\text{RR} = \mathbf{0{,}7611}$ |

### 4.2 Simuliertes KI-Policy-Targeting unter 20 % Budgetrestriktion ($N_{\text{test}} = 7.196$)

| Allokations-Policy (20 % Budget = 1.439 Plätze) | Kohorten-Dropout | Gerettete Abschlüsse | Kohorten-ARR (pp) | Anteil am 100%-Effekt | Effektiver ATT (pp) | NNT (Geförderte) | Effizienz-Lift vs. Zufall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Referenz: Kein Support (Welt B)** | 36.88 % | 0 | 0.00 pp | 0.0 % | 0.00 pp | $\infty$ | — |
| **1. Zufalls-Allokation (Gießkanne / Lotterie)** | 35.29 % | 114.9 | +1.60 pp | 19.9 % | +7.98 pp | 12.5 | **1.00×** |
| **2. Reaktive Verwaltungsregel** (HZB + Erwerb) | 33.78 % | 223.0 | +3.10 pp | 38.6 % | +15.50 pp | 6.5 | **1.94×** |
| **3. ML-Frühwarnsystem** (`standard`, S1/S2) | 33.59 % | 237.0 | +3.29 pp | 41.0 % | +16.47 pp | 6.1 | **2.06×** |
| **4. Kausales CATE / Uplift-Scoring** (`standard`, S1/S2) | **33.52 %** | **242.0** | **+3.36 pp** | **41.9 %** | **+16.82 pp** | **5.9** | **2.11×** |
| **5. Oracle Risk-Targeting** (`oracle`, S1/S2) | **33.42 %** | **249.0** | **+3.46 pp** | **43.1 %** | **+17.30 pp** | **5.8** | **2.17×** |
| **Referenz: Unbegrenzter Voll-Support (100 %)** | 28.85 % | 578.0 | +8.03 pp | 100.0 % | +8.03 pp | 12.5 | 1.00× |

---

## 5. Methodisches Fazit des Simulator-Benchmarks

1. **Feingranulare Prüfungssequenzen schlagen Semesteraggregate:** Auf derselben synthetischen Datenbasis erzielt `Exam GRU` / `CausalExamTransformer` ($\text{ROC-AUC} \approx 0{,}898$, $\text{Brier} \approx 0{,}0150$) eine deutlich höhere Diskriminations- und Kalibrierungsgüte als Semester-Aggregate ($\text{ROC-AUC} \approx 0{,}815–0{,}820$), da Fehlversuchsserien innerhalb eines Semesters unmittelbar erfasst werden.
2. **Konsistenz der Selektions- & Targeting-Logik:** Innerhalb des Simulators zeigt sowohl der Vergleich S01 vs. S11 (NNT 12.6 vs. 22.5) als auch das Out-of-Sample Policy-Targeting, dass kausales Uplift-Scoring nach Semester 2 die heterogene Treatment-Sensitivität erfolgreich ausnutzt: Mit **20 % Förderbudget werden 41.9 % des maximal möglichen Kohorteneffekts** realisiert (NNT sinkt von **12.5 auf 5.9**, Effizienz-Lift **2.11×** gegenüber Zufallsvergabe).
3. **Auflösung von Confounding by Indication durch DML und MSM:** Während naive Beobachtungsvergleiche durch den Vorversuchs-Confounder erwartungsgemäß das falsche Vorzeichen liefern ($HR \approx 1{,}48 - 1{,}53$), gelingt es multivariatem Robinson-DML (mit 5-Fold GroupKFold) und Cloglog-MSM, unter Einschluss der Vorversuchshistorie die wahre Kausalwirkung aller drei Fördersäulen im Beobachtungsdatensatz (Welt A) wiederherzustellen.
4. **Architektonische Sensitivität ($1/K_i$-Gewichtung, 3D-Skalierung & Schritt-Hazard-Kalibrierung):** Der vollständige Post-Refactor-Vergleich ([`modell_gegenueberstellung_gesamt_v425.md`](modell_gegenueberstellung_gesamt_v425.md)) belegt, dass nach korrekter 3D-Standardisierung auf aktiven Tokens, studierenden-inverser $1/K_i$-Gewichtung und zeitdiskreter Schritt-Hazard-Evaluation (`predict_step_hazard`) **exakte numerische Parität** zwischen Keras 3 und PyTorch 2.x ($\Delta\text{ROC-AUC} \le 0{,}004$) über alle Modellfamilien erreicht wird.

---

## Einzelsynopsen nach Dimension

| Szenario-Gruppe | Dokument | Methodischer Schwerpunkt |
|:---|:---|:---|
| **S01 — V4.2.5 Gesamt-Modellvergleich (Alle 6 Familien)** | **[modell_gegenueberstellung_gesamt_v425.md](modell_gegenueberstellung_gesamt_v425.md)** | **Side-by-Side-Gegenüberstellung aller >25 Modelle (Pre- vs. Post-Refactor, Keras 3 vs. PyTorch 2.x)** |
| S01 — Kausaler Benchmark & Policy-Targeting | [causal_ground_truth_benchmark.md](causal_ground_truth_benchmark.md) | 8-Welten Ground Truth vs. Cox/MSM/DML/G-Comp & 20%-Budget Allokation |
| S01–S11 — PyTorch vs. Keras Suite | [pytorch_lxc_benchmark_evaluation_v42.md](pytorch_lxc_benchmark_evaluation_v42.md) | Vollständiger Vergleich aller 4 Modellfamilien (PyCox, Autoregressoren, Transformer) |
| S01/S07/S08 — Heavy Deep Suite | [synopse_heavy_suite_s01_s07_s08.md](synopse_heavy_suite_s01_s07_s08.md) | Dual-Head GRU vs. Sin/Cos Transformer, Fail PR-AUC, Landmark Learning |
| S01/S07/S08 — Rauschen | [synopse_rauschen_s01_s07_s08.md](synopse_rauschen_s01_s07_s08.md) | Signal-Rausch-Verhältnis, Brier-Skill |
| S01/S02/S03 — Supportwirkung | [synopse_supportwirkung_s01_s02_s03.md](synopse_supportwirkung_s01_s02_s03.md) | ARR-Dosis-Wirkung im Simulator |
| S01/S04/S05/S06 — Notenboost | [synopse_notenboost_s01_s04_s05_s06.md](synopse_notenboost_s01_s04_s05_s06.md) | Fachlicher Support-Boost isoliert |
| S01/S09/S10 — Zeitkosten | [synopse_zeitkosten_s01_s09_s10.md](synopse_zeitkosten_s01_s09_s10.md) | Modulabwurf, Kreditpunkt-Verlust |
| S01/S11 — RCT vs. Selektion | [synopse_rct_selektion_s01_s11.md](synopse_rct_selektion_s01_s11.md) | Selbstselektion vs. Zufall (NNT 12.6 vs. 22.5) |
| S01/S12/S13/S14 — Überload | [synopse_overload_s01_s12_s13_s14.md](synopse_overload_s01_s12_s13_s14.md) | Überlastungs-Penalty, Kalibrierungs-Robustheit |
| S01/S15 — Kombination | [synopse_kombination_s01_s15.md](synopse_kombination_s01_s15.md) | Extremszenario (Hoch-Effekt + Doppel-Kosten) |

## Verwandte Dokumente

| Dokument | Bezug |
|:---|:---|
| **[modell_gegenueberstellung_gesamt_v425.md](modell_gegenueberstellung_gesamt_v425.md)** | **Vollständige Side-by-Side-Gegenüberstellung aller >25 Modelle aus dem V4.2.5 Post-Refactor Benchmark** |
| [causal_ground_truth_benchmark.md](causal_ground_truth_benchmark.md) | Vereinheitlichter Ground-Truth vs. Schätzer-Benchmark & KI-Policy-Targeting Showcase |
| [pytorch_lxc_benchmark_evaluation_v42.md](pytorch_lxc_benchmark_evaluation_v42.md) | Detailvergleich Keras 3 vs. PyTorch 2.x / PyCox über alle Modellfamilien |
| [methodenvergleich_logistic_hazard_keras_vs_pycox.md](methodenvergleich_logistic_hazard_keras_vs_pycox.md) | Mathematische Analyse Einzel-Hazard $h_t$ vs. kumulatives Risiko $F(t)$ |
| [DeepSupport_Projektentwicklung.md](../08_project_evolution/DeepSupport_Projektentwicklung.md) | Projektkontext und offene Fäden |
| [Kausale Vergleichsanalyse](../04_causal_and_simulation/04_Kausale_Vergleichsanalyse.md) | HR/RR-Grundlage dieser Synthese |
| [Project Review August 2026](project_review_august2026.md) | Code- und Architektur-Bewertung |
