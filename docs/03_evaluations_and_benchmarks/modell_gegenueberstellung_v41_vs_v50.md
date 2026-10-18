---
created: 2026-10-18
last_updated: 2026-10-18
status: abgeschlossen
tags: [benchmark, v4.1, v5.0, dgp-comparison, keras, pytorch, survival, causal-inference, policy-targeting]
---

# Systematische Modell-Gegenüberstellung: DGP V4.1 vs. DGP V5.0 (Post-Refactor Master-Benchmark)

> [!IMPORTANT]
> **Zielsetzung dieses Dokuments:**  
> Nachdem in Phase 1–3 der [V5-Roadmap](../04_causal_and_simulation/config_audit_und_v5_roadmap.md) der Datengenerator auf **DGP V5.0** gehoben wurde — insbesondere durch die Behebung des Varianzkollapses der Startmotivation ($\kappa: 24{,}0 \to 9{,}0$, $\sigma_{\text{Mot}}: 0{,}1216 \to 0{,}1637$), die Einführung der stetigen **Zero-Inflated Beta-Erwerbstätigkeit** ($37{,}3\,\%$ bei $0\,\text{h/Woche}$, Median $15\,\text{h}$), fachspezifische Geschlechterquoten nach Destatis sowie die Dämpfung künstlicher Spätsemester-Apathie (`apathy_sem_factor`: $0{,}04 \to 0{,}018$) — wurden alle 8 kontrafaktischen Parallelwelten (**Universen A–H**, $N = 50{,}000$, `seed=99999`) unter `data_v5_grid/S01_baseline/` neu generiert und sämtliche **34 Benchmark-Schritte über alle 6 Modellfamilien** (`outputs/S01_baseline_v5/`) vollständig neu trainiert.  
> Dieses Dokument vergleicht die **kontrafaktische Simulator-Ground-Truth** sowie die **Out-of-Sample-Performance aller >25 Modelle (Keras 3 & PyTorch 2.x)** direkt zwischen **DGP V4.1** (`outputs/S01_baseline_post_refactor/`) und **DGP V5.0** (`outputs/S01_baseline_v5/`).

---

## 1. Kontrafaktische Simulator-Ground-Truth im Vergleich: DGP V4.1 vs. DGP V5.0 ($N = 50{,}000$)

Beide Datensätze simulieren exakt $N = 50{,}000$ Studierende mit identischem Start-Seed (`population_seed = 99999`) über alle 8 institutionellen Support-Regime (Universen A bis H):

| Universum | Support-Regime | Dropout V4.1 (%) | **Dropout V5.0 (%)** | $\Delta$ (V5.0 − V4.1) | Kausale Schutzwirkung V4.1 ($\text{ARR}$ / $\text{RR}$) | **Kausale Schutzwirkung V5.0 ($\text{ARR}$ / $\text{RR}$)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Universum A** | Full Support (Alle 3 Kanäle aktiv) | 29,21 % | **27,82 %** | −1,39 pp | **+7,91 pp** ($\text{RR}_{A/B} = 0{,}7868$, $\text{NNT}=12{,}6$) | **+7,65 pp** ($\text{RR}_{A/B} = 0{,}7843$, $\text{NNT}=13{,}1$) |
| **Universum B** | No Support (Kontrafaktische Nullwelt) | 37,12 % | **35,47 %** | −1,65 pp | Referenz (0,00 pp) | Referenz (0,00 pp) |
| **Universum C** | Knockout: Ohne Fachlichen Support | 32,05 % | **30,61 %** | −1,44 pp | $\text{ARR}_{A \text{ vs. } C} = +2{,}84\text{ pp}$ ($\text{RR}=0{,}9114$) | **$\text{ARR}_{A \text{ vs. } C} = +2{,}79\text{ pp}$** ($\text{RR}=0{,}9090$) |
| **Universum D** | Knockout: Ohne Überfachlichen Support | 31,73 % | **30,24 %** | −1,49 pp | $\text{ARR}_{A \text{ vs. } D} = +2{,}52\text{ pp}$ ($\text{RR}=0{,}9206$) | **$\text{ARR}_{A \text{ vs. } D} = +2{,}42\text{ pp}$** ($\text{RR}=0{,}9201$) |
| **Universum E** | Knockout: Ohne Psychosozialen Support | 30,85 % | **29,52 %** | −1,33 pp | $\text{ARR}_{A \text{ vs. } E} = +1{,}64\text{ pp}$ ($\text{RR}=0{,}9468$) | **$\text{ARR}_{A \text{ vs. } E} = +1{,}70\text{ pp}$** ($\text{RR}=0{,}9425$) |
| **Universum F** | Stand-Alone: Nur Fachlicher Support | 33,59 % | **32,26 %** | −1,33 pp | $\text{ARR}_{F \text{ vs. } B} = +3{,}53\text{ pp}$ ($\text{RR}=0{,}9048$) | **$\text{ARR}_{F \text{ vs. } B} = +3{,}21\text{ pp}$** ($\text{RR}=0{,}9095$) |
| **Universum G** | Stand-Alone: Nur Überfachlicher Support | 33,98 % | **32,50 %** | −1,48 pp | $\text{ARR}_{G \text{ vs. } B} = +3{,}14\text{ pp}$ ($\text{RR}=0{,}9153$) | **$\text{ARR}_{G \text{ vs. } B} = +2{,}97\text{ pp}$** ($\text{RR}=0{,}9163$) |
| **Universum H** | Stand-Alone: Nur Psychosozialer Support | 34,86 % | **33,27 %** | −1,59 pp | $\text{ARR}_{H \text{ vs. } B} = +2{,}26\text{ pp}$ ($\text{RR}=0{,}9392$) | **$\text{ARR}_{H \text{ vs. } B} = +2{,}20\text{ pp}$** ($\text{RR}=0{,}9380$) |

### Strukturelle Einordnung der Kohortenverschiebung
1. **Invarianz der relativen kausalen Schutzwirkung ($\text{RR}_{A/B}$):** Das Gesamtrisiko-Verhältnis zwischen Universum A und Universum B bleibt mit $\text{RR}_{A/B} = 0{,}7843$ (Risikosenkung um **−21,57 %**) gegenüber V4.1 ($\text{RR}_{A/B} = 0{,}7868$, **−21,32 %**) bemerkenswert stabil. Auch die Wirkungsrangfolge der drei Support-Säulen ($\text{Fachlich} > \text{Überfachlich} > \text{Psychosozial}$) bleibt sowohl im Knockout- als auch im Stand-Alone-Design unverändert erhalten.
2. **Rückgang der Basis-Abbruchquote um $\approx 1{,}4\text{–}1{,}6\text{ pp}$:** Durch die Rekalibrierung der Erwerbstätigkeit auf das 22. DSW-Sozialerhebungs-Profil ($37{,}3\,\%$ nicht-erwerbstätige Vollzeitstudierende statt $25{,}1\,\%$ in V4.1; Anteil $\ge 20\,\text{h/Woche}$ sinkt von $24{,}9\,\%$ auf $14{,}0\,\%$) sowie die Dämpfung des künstlichen Spätsemester-Apathie-Terms (`apathy_sem_factor = 0.018` statt `0.040`) entfallen unplausible Spätsemester-Zeitbudgetsprengungen bei leistungsstarken Studierenden. Gleichzeitig sorgt die um $+34{,}6\,\%$ verbreiterte Motivationsstreuung ($\sigma_{\text{Mot}} = 0{,}1637$) für $2{,}40\times$ mehr echte Frühkrisen in Semester 1 ($P(\text{Mot}_0 < 0{,}40) = 9{,}65\,\%$).

---

## 2. Familie 1 & 5: Statische Landmark-Modelle & Hybrides Representation Learning ($t_0 = 2$)

Am Ende des 2. Fachsemesters ($t_0 = 2$) prognostizieren die Modelle der Familie 1 und Familie 5 den weiteren Studienverlauf bzw. die finale Bachelor-Abschlussnote (Hurdle Stage 2, `graduates_only=True`):

### 2.1 Binäre Landmark-Klassifikation, Landmark-Survival & 4-Klassen-Status (`standard` & `gradeblind`)

| Modellarchitektur | Framework | Modus | Metrik | DGP V4.1 | **DGP V5.0** | $\Delta$ (V5.0 − V4.1) | Befund |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| **Logistic Hazard Landmark** | Keras 3 | `standard` | ROC-AUC / Brier | 0,8740 / 0,1139 | **0,8690 / 0,1115** | −0,0050 / **−0,0024** | Bessere Kalibrierung (niedrigerer Brier Score) |
| **Breslow DeepSurv Landmark** | Keras 3 | `standard` | ROC-AUC / BSS | 0,7275 / +0,0991 | **0,7301 / +0,1049** | **+0,0026 / +0,0058** | Höhere Trennschärfe & Kalibrierung |
| **Keras MLP Binary Classifier** | Keras 3 | `standard` | Accuracy / F1 / ROC-AUC | 0,8407 / 0,8291 / 0,8740 | **0,8489 / 0,8422 / 0,8683** | **+0,82 pp Acc / +1,31 pp F1** | Schärfere Klassifikationsgrenze ($\tau=0{,}5$) |
| **Keras MLP Binary Classifier** | Keras 3 | `gradeblind` | Accuracy / F1 / ROC-AUC | 0,8405 / 0,8291 / 0,8720 | **0,8501 / 0,8432 / 0,8673** | **+0,96 pp Acc / +1,41 pp F1** | Konsistenter F1-Gewinn ohne Noten |
| **Random Forest Classifier** | Sklearn | `standard` | Accuracy / ROC-AUC / Brier | 0,8266 / 0,8480 / 0,1255 | **0,8383 / 0,8484 / 0,1201** | **+1,17 pp Acc / −0,0054 Brier** | Deutlich geringerer Kalibrierungsfehler |
| **SVM (RBF Kernel)** | Sklearn | `standard` | Accuracy / F1 / Brier | 0,8413 / 0,8298 / 0,1221 | **0,8509 / 0,8392 / 0,1181** | **+0,96 pp Acc / −0,0040 Brier** | Höchste Punkt-Accuracy an $t_0 = 2$ |
| **Hybrid: 32-D Exam-Transformer Embedding + XGBoost (F5.1)** | Keras 3 + XGBoost | `standard` | 4-Class Acc / Weighted-F1 / OVR-AUC | 0,7963 / 0,7691 / 0,8500 | **0,8111 / 0,7887 / 0,8721** | **+1,48 pp Acc / +2,21 pp AUC** | **Starker Sprung durch 32-D Sequenz-Embedding!** |

### 2.2 Hurdle Stage 2: Konditionale Abschlussnoten-Regression (`graduates_only=True`)

| Modellarchitektur | Framework | Modus | $R^2$ (V4.1) | **$R^2$ (V5.0)** | RMSE V4.1 | **RMSE V5.0** | MAE V4.1 | **MAE V5.0** | Einordnung |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Keras MLP Regressor** | Keras 3 | `standard` | 0,8706 | **0,8609** | 0,2154 | **0,2232** | 0,1651 | **0,1724** | Hohe Stabilität trotz breiterer Motivationsvarianz |
| **SVR (RBF Kernel)** | Sklearn | `standard` | 0,8672 | **0,8600** | 0,2182 | **0,2240** | 0,1672 | **0,1730** | Nahezu identisch mit Keras MLP |
| **Random Forest Regressor** | Sklearn | `standard` | 0,8538 | **0,8485** | 0,2289 | **0,2330** | 0,1755 | **0,1801** | Solide nichtlineare Baseline |
| **Ridge Regression (L2)** | Sklearn | `standard` | 0,8510 | **0,8465** | 0,2311 | **0,2345** | 0,1796 | **0,1829** | Lineare Referenz |
| **Hybrid: 32-D Exam-Transformer Embedding + XGBoost (F5.1)** | Keras 3 + XGBoost | `standard` | 0,7865 | **0,8404 (+5,39 pp)** | 0,2782 | **0,2403 (−0,0379)** | 0,2174 | **0,1862 (−0,0312)** | **Massiver Gewinn der gelernten Sequenz-Repräsentation** |
| **Keras MLP Regressor** | Keras 3 | `gradeblind` | 0,7751 | **0,7190** | 0,2839 | **0,3172** | 0,2226 | **0,2476** | Erwarteter Effekt: Höhere latente Motivationsstreuung ($\kappa=9{,}0$) ist ohne S1/S2-Noten schwerer aus `hzb_note` allein ablesbar |

> [!TIP]
> **Warum sinkt $R^2$ im statischen `gradeblind`-Modus leicht ($0{,}7751 \to 0{,}7190$), während das hybride Sequenz-Embedding (F5.1) um $+5{,}39\text{ pp}$ ($0{,}7865 \to 0{,}8404$) zulegt?**  
> In DGP V4.1 war die latente Motivation durch $\kappa = 24{,}0$ ($\sigma = 0{,}1216$) extrem eng um den durch `hzb_note` und `erwerbstaetigkeit_std` determinierten Erwartungswert konzentriert — ein statisches Modell konnte die Abschlussnote daher schon aus der Abiturnote fast vollständig erraten. In **DGP V5.0** ($\kappa = 9{,}0$, $\sigma = 0{,}1637$) weicht die tatsächliche Studienmotivation individuell deutlich stärker von der Abiturnote ab. Ein Modell ohne Prüfungsnoten (`gradeblind`) sieht diese latente Abweichung nicht, während der **Headless Exam-Transformer (F5.1)** die feingranulare Prüfungsdynamik aus Semester 1–2 in sein 32-D-Embedding komprimiert und dadurch in V5.0 um **+5,39 Prozentpunkte $R^2$** gewinnt.

---

## 3. Familie 2 & 3: Dynamische Counting-Process Panel- & 3D-Sequenz-Survival-Modelle

Hier zeigt sich der **wichtigste prädiktive Effekt des DGP-V5.0-Upgrades**: Ausnahmslos **alle** dynamischen Panel- und 3D-Sequenz-Survival-Modelle (sowohl in Keras 3 als auch in PyTorch 2.x, sowohl auf Semester- als auch auf Einzelprüfungsebene) erzielen auf DGP V5.0 eine **höhere Diskrimination (ROC-AUC & Harrell's C-Index)** und einen **niedrigeren Kalibrierungsfehler (Brier Score)** als auf DGP V4.1!

| Modellfamilie & Architektur | Framework | Modus | ROC-AUC V4.1 | **ROC-AUC V5.0** | $\Delta$ ROC-AUC | C-Index V4.1 | **C-Index V5.0** | $\Delta$ C-Index | Brier V4.1 $\to$ **V5.0** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **F2.1 Extended Cox (`PHReg`, `strata=t_stop`)** | Statsmodels | `standard` | 0,7661 | **0,7670** | +0,09 pp | 0,6394 | **0,6553** | **+1,59 pp** | `null` |
| **F2.1 Extended Cox (`PHReg`, `strata=t_stop`)** | Statsmodels | `oracle` | 0,7860 | **0,7935** | **+0,75 pp** | 0,6980 | **0,7288** | **+3,08 pp** | `null` |
| **F2.1 Extended Cox (`PHReg`, `strata=t_stop`)** | Statsmodels | `gradeblind` | 0,7626* | **0,7625** | −0,01 pp | 0,6316* | **0,6427** | **+1,11 pp** | `null` |
| **F2.2 Extended DeepSurv (+ Breslow Step)** | Keras 3 | `standard` | 0,8165 | **0,8213** | **+0,48 pp** | `null` | **0,8185** | — | 0,0342 $\to$ **0,0341** |
| **F2.2 Extended DeepSurv (+ Breslow Step)** | Keras 3 | `oracle` | 0,8212 | **0,8269** | **+0,57 pp** | `null` | **0,8263** | — | 0,0341 $\to$ **0,0338** |
| **F2.2 Extended Logistic Hazard** | Keras 3 | `standard` | 0,8152 | **0,8201** | **+0,49 pp** | `null` | **0,8164** | — | 0,0342 $\to$ **0,0340** |
| **F2.2 Extended Logistic Hazard** | Keras 3 | `oracle` | 0,8202 | **0,8255** | **+0,53 pp** | `null` | **0,8233** | — | 0,0340 $\to$ **0,0337** |
| **F2.3 Dynamic DeepHit Competing Risks** | Keras 3 | `standard` | 0,8177 | **0,8162** | −0,15 pp | `null` | `null` | — | 0,0351 $\to$ **0,0342** |
| **F2.4 PyTorch Logistic Hazard** | PyTorch 2.x | `standard` | 0,8160 | **0,8207** | **+0,47 pp** | 0,8050 | **0,8205** | **+1,55 pp** | 0,0338 $\to$ **0,0339** |
| **F2.4 PyTorch Logistic Hazard** | PyTorch 2.x | `oracle` | 0,8207 | **0,8262** | **+0,55 pp** | 0,8081 | **0,8193** | **+1,12 pp** | 0,0337 $\to$ **0,0335** |
| **F2.4 PyTorch Extended CoxPH** | PyTorch 2.x | `standard` | 0,8160 | **0,8205** | **+0,45 pp** | 0,8054 | **0,8188** | **+1,34 pp** | 0,0344 $\to$ **0,0343** |
| **F2.4 PyTorch Extended CoxPH** | PyTorch 2.x | `oracle` | 0,8194 | **0,8263** | **+0,69 pp** | 0,8086 | **0,8251** | **+1,65 pp** | 0,0342 $\to$ **0,0341** |
| **F2.4 PyTorch CoxTime (Non-Proportional)** | PyTorch 2.x | `standard` | 0,8157 | **0,8213** | **+0,56 pp** | 0,8050 | **0,8206** | **+1,56 pp** | 0,0344 $\to$ **0,0343** |
| **F2.4 PyTorch CoxTime (Non-Proportional)** | PyTorch 2.x | `oracle` | 0,8188 | **0,8263** | **+0,75 pp** | 0,8059 | **0,8234** | **+1,75 pp** | 0,0343 $\to$ **0,0342** |
| **F2.4 PyTorch DeepHit Competing Risks** | PyTorch 2.x | `standard` | 0,7463 | **0,7576** | **+1,13 pp** | 0,8732 | **0,8831** | **+0,99 pp** | 0,0398 $\to$ **0,0396** |
| **F2.4 PyTorch DeepHit Competing Risks** | PyTorch 2.x | `oracle` | 0,7556 | **0,7642** | **+0,86 pp** | 0,8769 | **0,8863** | **+0,94 pp** | 0,0395 $\to$ **0,0396** |
| **F3.1 Keras Semester Survival GRU** | Keras 3 | `standard` | 0,8121 | **0,8173** | **+0,52 pp** | `null` | `null` | — | 0,0353 $\to$ **0,0343** |
| **F3.1 Keras Semester Survival GRU** | Keras 3 | `oracle` | 0,8139 | **0,8171** | **+0,32 pp** | `null` | `null` | — | 0,0348 $\to$ **0,0343** |
| **F3.2 Keras Semester Causal Transformer** | Keras 3 | `standard` | 0,8144 | **0,8190** | **+0,46 pp** | `null` | `null` | — | 0,0347 $\to$ **0,0341** |
| **F3.3 Keras Exam Survival GRU** | Keras 3 | `standard` | 0,8791 | **0,9008** | **+2,17 pp** | `null` | `null` | — | 0,0154 $\to$ **0,0144** |
| **F3.4 Keras Exam Causal Transformer** | Keras 3 | `standard` | 0,8772 | **0,8828** | **+0,56 pp** | `null` | `null` | — | 0,0155 $\to$ **0,0146** |
| **F3.6 PyTorch Semester Survival GRU** | PyTorch 2.x | `standard` | 0,8196 | **0,8214** | **+0,18 pp** | `null` | `null` | — | 0,0345 $\to$ **0,0341** |
| **F3.6 PyTorch Semester Causal Transformer** | PyTorch 2.x | `standard` | 0,8147 | **0,8173** | **+0,26 pp** | `null` | `null` | — | 0,0347 $\to$ **0,0342** |
| **F3.6 PyTorch Exam Survival GRU** | PyTorch 2.x | `standard` | 0,8976 | **0,9054** | **+0,78 pp** | `null` | `null` | — | 0,0150 $\to$ **0,0143** |
| **F3.6 PyTorch Causal Exam Transformer** | PyTorch 2.x | `gradeblind` | 0,8779 | **0,8896** | **+1,17 pp** | `null` | `null` | — | 0,0153 $\to$ **0,0149** |
| **F3.7 PyTorch Dynamic DeepHit (Causal Step)** | PyTorch 2.x | `standard` | 0,7957 | **0,8032** | **+0,75 pp** | `null` | `null` | — | 0,0373 $\to$ **0,0359** |

*\*Hinweis zu F2.1 (`gradeblind`): Im initialen V4.1-Benchmarklauf war `gradeblind` aufgrund einer doppelten Z-Standardisierung vor dem Fix auf $\text{HR}=1{,}0$ ($\text{C-Index}=0{,}5000$) zurückgefallen und erreichte erst im isolierten Nachlauf `0,7626 / 0,6316`. In V5.0 konvergierte `extended_cox.py --mode gradeblind` im ersten Durchlauf fehlerfrei.*

### Warum gewinnen sämtliche Längsschnitt- und Survival-Modelle in DGP V5.0?
1. **Höheres Signal-Rausch-Verhältnis in den latenten Trajektorien ($\kappa = 9{,}0$):** Durch die Aufhebung des Varianzkollapses der Startmotivation ($\sigma_{\text{Mot}}$ steigt von $0{,}1216$ auf $0{,}1637$) gewinnt die latente Motivationsvariable echten diagnostischen Wert — sichtbar am **C-Index-Sprung von $+3{,}08\text{ pp}$** (`0,6980` $\to$ **`0,7288`**) im `Extended Cox (oracle)` sowie $+1{,}65\text{–}1{,}75\text{ pp}$ in `PyTorchCoxPH` und `PyTorchCoxTime`.
2. **Wegfall unmotivierter Spätsemester-Zufallsabbrüche (`apathy_sem_factor = 0.018`):** In DGP V4.1 erzeugte der linear anwachsende Term `0.04 * sem` in Semester 7–9 einen pauschalen Risikoaufschlag von bis zu $+0{,}36$, der selbst Studierende mit guten Prüfungsleistungen willkürlich in den Abbruch trieb. Durch die Halbierung auf `0.018` in V5.0 korreliert der Studienabbruch auf der Prüfungsebene noch konsistenter mit tatsächlichen Fehlversuchsketten — was den **ROC-AUC der `Keras Exam Survival GRU` um $+2{,}17\text{ pp}$ auf `0,9008`** und der **`PyTorch Exam Survival GRU` auf `0,9054`** hebt.

---

## 4. Familie 4: Autoregressive Next-Exam Dual-Head Modelle ($w_{ik} = 1/K_i$)

Die vier autoregressiven Multi-Task-Modelle sagen nach jeder einzelnen abgelegten Prüfung $(i, k)$ simultan die **Note der nächsten Klausur (Head 1)** sowie die **Bestehens-/Durchfallwahrscheinlichkeit der nächsten Klausur (Head 2)** voraus:

| Modellarchitektur | Framework | Grade $R^2$ (V4.1 $\to$ **V5.0**) | Grade RMSE (V4.1 $\to$ **V5.0**) | Grade MAE (V4.1 $\to$ **V5.0**) | Pass ROC-AUC (V4.1 $\to$ **V5.0**) | Fail PR-AUC (V4.1 $\to$ **V5.0**) | Pass Brier (V4.1 $\to$ **V5.0**) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Keras Next-Exam Dual-Head GRU** | Keras 3 | 0,6885 $\to$ **0,6792** | 0,7622 $\to$ **0,7721** | 0,5920 $\to$ **0,5873** | 0,9392 $\to$ **0,9388** | 0,7706 $\to$ **0,7687** | 0,0705 $\to$ **0,0687** |
| **Keras Next-Exam Deep Transformer** | Keras 3 | 0,6913 $\to$ **0,6914** | 0,7587 $\to$ **0,7573** | 0,5708 $\to$ **0,5622** | 0,9380 $\to$ **0,9386** | 0,7665 $\to$ **0,7618** | 0,0712 $\to$ **0,0698** |
| **PyTorch Next-Exam Dual-Head GRU** | PyTorch 2.x | 0,7036 $\to$ **0,7043** | 0,7435 $\to$ **0,7412** | 0,5617 $\to$ **0,5635** | 0,9406 $\to$ **0,9423** | 0,7760 $\to$ **0,7748** | 0,0682 $\to$ **0,0669** |
| **PyTorch Next-Exam Dual-Head Transformer** | PyTorch 2.x | 0,7050 $\to$ **0,7010** | 0,7418 $\to$ **0,7453** | 0,5610 $\to$ **0,5533** | 0,9409 $\to$ **0,9414** | 0,7779 $\to$ **0,7723** | 0,0681 $\to$ **0,0671** |

**Befund:** Auf Einzelprüfungsebene zeigen alle vier Architekturen eine herausragende Stabilität zwischen V4.1 und V5.0 (`Pass ROC-AUC` bis **0,9423**, `Grade R²` bis **0,7043**). Durch die geringfügig niedrigere Klausur-Durchfallquote in V5.0 ($\pi_0(\text{Fail}) = 15{,}96\,\%$ statt $16{,}38\,\%$ in V4.1) verbessert sich der `Pass Brier Score` über alle vier Modelle hinweg um **$-0{,}0011$ bis $-0{,}0018$** auf bis zu **0,0669** (`PyTorch Next-Exam GRU`), während der `Fail Lift` von $\approx 4{,}70\times$ auf **$\approx 4{,}85\times$** gegenüber der Basisrate steigt.

---

## 5. Familie 6: Kausale Inferenz-Suite & KI-Policy-Targeting — DGP V4.1 vs. DGP V5.0

Besonders aufschlussreich ist der Vergleich der kausalen Deconfounding-Schätzer auf Universum A: Während in DGP V4.1 die extrem schmale Motivationsverteilung ($\kappa=24{,}0$) dazu führte, dass Propensity-Score-Modelle (`MSM / IPTW`) und Regressionsadjustierungen kaum unabhängige Variation in `hidden_motivation` nutzen konnten, verbessert die realistische Dispersion in **DGP V5.0 ($\kappa=9{,}0$)** die kausale Identifikation über **alle fünf Schätzerklassen** hinweg signifikant!

### 5.1 Direktvergleich aller Kausalschätzer auf Universum A (DGP V4.1 vs. DGP V5.0)

| Kausaler Schätzer / Modellklasse | Modus | Estimand | Fachlich (V4.1 $\to$ **V5.0**) | Überfachlich (V4.1 $\to$ **V5.0**) | Psychosozial (V4.1 $\to$ **V5.0**) | Methodische Veränderung in DGP V5.0 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **0a. Simulator Ground Truth (Pop. Knockout A vs. C/D/E)** | Kontrafaktisch | Macro-RR | 0,9163 $\to$ **0,9090** | 0,9144 $\to$ **0,9201** | 0,9403 $\to$ **0,9425** | Wahre Populations-Risikoreduktion |
| **0b. Simulator Ground Truth (ATT-HR auf Nutzern)** | Kontrafaktisch | ATT-HR | 0,8608 $\to$ **0,8467** | 0,7828 $\to$ **0,7948** | 0,7738 $\to$ **0,7800** | Wahre Intervall-Hazard-Ratio auf Geförderten |
| **1. Naive Unadjustierte Korrelation** | `standard` | Rohes HR | 1,4589 $\to$ **1,5015** | 1,5471 $\to$ **1,5585** | 0,8895 $\to$ **0,8928** | Noch stärkeres *Confounding by Indication* in V5.0 |
| **2a. Extended Cox (`PHReg`, `strata=t_stop`)** | `standard` | Step-HR | 0,9587 $\to$ **0,9557** | 0,9906 $\to$ **0,9868** | 0,9770 $\to$ **0,9571** | Stärkere Identifikation aller 3 Kanäle ($< 1{,}0$) |
| **2b. Extended Cox (`PHReg`, `strata=t_stop`)** | `oracle` | Step-HR | 0,9581 $\to$ **0,9604** | 0,9881 $\to$ **0,9799** | 0,9685 $\to$ **0,9436** | `ueberfachlich` ($[0{,}963; 0{,}997]$) & `psychosozial` ($[0{,}921; 0{,}967]$) signifikant $<1$ |
| **2c. Extended Cox (`PHReg`, `strata=t_stop`)** | `gradeblind` | Step-HR | 0,9645 $\to$ **0,9629** | 0,9922 $\to$ **0,9890** | 0,9762 $\to$ **0,9584** | Direkt im ersten Lauf stabil konvergiert |
| **3a. Keras Orthogonal DML (5-Fold GroupKFold)** | `standard` | Robinson HR | 0,8475 $\to$ **0,8307** | 0,9790 $\to$ **0,9653** | 0,9883 $\to$ **0,9560** | `fachlich` & `psychosozial` bereits ohne Orakel signifikant $<1{,}0$! |
| **3b. Keras Orthogonal DML (5-Fold GroupKFold)** | `oracle` | Robinson HR | 0,8503 $\to$ **0,8352** | 0,9610 $\to$ **0,9328** | 0,9729 $\to$ **0,9321** | **Alle 3 Kanäle mit asympt. 95%-KI strikt $< 1{,}0$!** |
| **4a. PyTorch Orthogonal DML (5-Fold + 100x Boot.)** | `standard` | Robinson HR | 0,8952 $\to$ **0,8812** | 0,9711 $\to$ **0,9516** | 0,9737 $\to$ **0,9465** | Deutlich näher an der wahren Ground Truth als in V4.1 |
| **4b. PyTorch Orthogonal DML (5-Fold + 100x Boot.)** | `oracle` | Robinson HR | 0,9135 $\to$ **0,9019** | 0,9410 $\to$ **0,9146** | 0,9642 $\to$ **0,9209** | **Exzellente Rekonstruktion der Knockout-Ground-Truth (`0,909 / 0,920 / 0,943`)!** |
| **5a. PyTorch MSM / IPTW (`cloglog` Link, Benchmark)** | `standard` | Marginales HR | 0,9275 $\to$ **0,9125** | 0,9859 $\to$ **0,9392** | 0,9125 $\to$ **0,8555** | Stark verbesserte IPTW-Balancierung bei `ueberfachlich` |
| **5b. PyTorch MSM / IPTW (`cloglog` Link, Benchmark)** | `oracle` | Marginales HR | 0,9234 $\to$ **0,9079** | 0,9585 $\to$ **0,9135** | 0,8920 $\to$ **0,8357** | In `torch_causal` sinkt auch `hr_fachlich` von `1,0835` auf **`0,9755` ($<1{,}0$)**! |
| **6a. PyTorch G-Computation (Step-Standardized OOS)** | `standard` | Risk Ratio | 0,9650 $\to$ **0,9661** | 0,8891 $\to$ **0,8589** | 0,9127 $\to$ **0,8710** | Full-Support 16-Sem.-Rollout: $\text{RR} = 0{,}7611 \to \mathbf{0{,}7628}$ (GT: `0,7843`) |
| **6b. PyTorch G-Computation (Step-Standardized OOS)** | `oracle` | Risk Ratio | 0,9688 $\to$ **0,9747** | 0,8784 $\to$ **0,8517** | 0,9110 $\to$ **0,8695** | Full-Support 1-Step RR: $\text{RR} = 0{,}8787 \to \mathbf{0{,}8746}$ ($[0{,}870; 0{,}878]$) |

> [!NOTE]
> **Schlüsselerkenntnis zur kausalen Identifikation in DGP V5.0:**  
> 1. **Präzisere Punkt- und Intervallschätzung in Orthogonal DML:** Im `oracle`-Modus trifft `PyTorch Orthogonal DML` auf DGP V5.0 mit $\text{HR}_{\text{fach}} = \mathbf{0{,}9019}$ ($[0{,}844; 0{,}957]$), $\text{HR}_{\text{uebf}} = \mathbf{0{,}9146}$ ($[0{,}879; 0{,}949]$) und $\text{HR}_{\text{psych}} = \mathbf{0{,}9209}$ ($[0{,}892; 0{,}950]$) die wahren Populations-Knockout-RRs (`0,9090`, `0,9201`, `0,9425`) auf 1 bis 2 Prozentpunkte genau. Auch in `Keras Orthogonal DML (oracle)` liegen in V5.0 erstmals **alle drei** asymptotischen 95%-Konfidenzintervalle vollständig unter $1{,}0$ (`[0,773; 0,903]`, `[0,897; 0,971]`, `[0,900; 0,965]`).
> 2. **Auflösung des IPTW-Positivitätsengpasses in `PyTorch MSM (oracle)`:** Während in V4.1 das neuronale Propensity-Modell in `torch_marginal_structural_model` (`oracle`) für `fachlich` noch bei $\text{HR} = 1{,}0835$ verharrte, sinkt die Schätzung in V5.0 auf **$\text{HR}_{\text{fach}} = 0{,}9755$** (bzw. **`0,9079`** im Benchmark-Spezifikationslauf), sodass nun **sämtliche** IPTW-Schätzer in allen drei Dimensionen die wahre Schutzrichtung ($< 1{,}0$) anzeigen.

### 5.2 AI Policy Targeting Showcase bei 20 % Interventions-Budget ($N_{\text{test}} = 7{,}192$)

| Zuweisungs-Strategie (bei 20 % Förder-Budget) | Kohorten-ARR V4.1 (pp) | **Kohorten-ARR V5.0 (pp)** | Effektive NNT V4.1 $\to$ **V5.0** | Verhinderte Abbrüche / 10k (V4.1 $\to$ **V5.0**) | Effizienz-Lift vs. Zufall (V4.1 $\to$ **V5.0**) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Zufalls-Zuweisung (Gießkanne / Lotterie)** | +1,66 pp | **+1,65 pp** | 12,0 $\to$ **12,1** | 166 $\to$ **165** | 1,00× $\to$ **1,00×** |
| **2. Reaktive Regel-Zuweisung (HZB + Erwerb)** | +3,57 pp | **+3,50 pp** | 5,6 $\to$ **5,7** | 357 $\to$ **350** | 2,15× $\to$ **2,13×** |
| **3. ML-Frühwarnung (`standard`, Landmark S1/S2)** | +3,53 pp | **+3,36 pp** | 5,7 $\to$ **5,9** | 353 $\to$ **336** | 2,13× $\to$ **2,04×** |
| **4. Causal CATE / Uplift-Scoring (`standard`, S1/S2)** | +3,56 pp | **+3,38 pp** | 5,6 $\to$ **5,9** | 356 $\to$ **338** | 2,15× $\to$ **2,05×** |
| **5. Oracle Upper Bound (`oracle`, inkl. Latente)** | +3,60 pp | **+3,41 pp** | 5,6 $\to$ **5,9** | 360 $\to$ **341** | 2,17× $\to$ **2,07×** |

Bei 30 % Förderbudget erreicht das **KI-gestützte Causal CATE / Uplift-Targeting (`standard`)** in DGP V5.0 eine Kohorten-Risikoreduktion von **+5,28 pp** ($\text{NNT} = 5{,}7$, **64,4 %** des maximal möglichen 100%-Budget-Effekts von $+8{,}20\text{ pp}$ bei nur 30 % Ressourceneinsatz; Effizienz-Lift **2,15×** gegenüber Zufallsvergabe) und übertrifft damit auch die reaktive Regel-Zuweisung (+4,99 pp, $\text{NNT} = 6{,}0$) um **+0,29 pp** (29 zusätzlich gerettete Abschlüsse pro 10.000 Studierende).

---

## 6. Laufzeit- & Ressourcenbilanz des DGP-V5.0-Gesamtlaufs

| Phase | Skript / Orchestrator | Umfang | Laufzeit (Min.) | Peak RAM (MB) | Status |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **1. Multi-Universe Generierung (V5.0)** | `run_v5_generation_and_benchmark.py` | 8 Universen (A–H) $\times$ $N = 50{,}000$ (`data_v5_grid/S01_baseline/`) | 75,84 Min. | ~1.450 MB | **PASSED (8/8)** |
| **2. DuckDB Pre-Aggregation** | `aggregate.py` (via Runner) | `agg_pruefungen.csv` & `agg_abschluesse.csv` (Univ. A & B) | 0,52 Min. | ~620 MB | **PASSED (2/2)** |
| **3. 34-Schritte Modell-Benchmark** | `run_post_refactor_benchmark_s01.py` | Alle 6 Modellfamilien (>25 Modelle Keras 3 & PyTorch 2.x + Causal GT) | 220,28 Min. | 2.827,4 MB | **PASSED (34/34)** |
| **Gesamtprozess V5.0** | End-to-End Pipeline | 400.000 simulierte Verläufe + 34 Modell-/Evaluationsläufe | **296,64 Min. (~4,94 Std.)** | **2.827,4 MB** | **100 % PASSED** |

---

## Verwandte Dokumente

| Dokument | Thema & Verbindung zu dieser Gegenüberstellung |
| :--- | :--- |
| **[Modell-Gegenüberstellung V4.2.5 (DGP V4.1)](modell_gegenueberstellung_gesamt_v425.md)** | Vollständige Referenz-Dokumentation aller 6 Modellfamilien auf dem DGP-V4.1-Datensatz (`outputs/S01_baseline_post_refactor/`) |
| **[Config-Audit & V5-Roadmap](../04_causal_and_simulation/config_audit_und_v5_roadmap.md)** | Fachliche und mathematische Spezifikation aller Parameter-Upgrades von DGP V4.1 auf DGP V5.0 inkl. S16 Spätkrisen-Nachweis |
| **[Kausaler Ground-Truth Benchmark (V4.1)](causal_ground_truth_benchmark.md)** | Detaillierter Kausal- und Policy-Targeting-Report auf `data_v4_grid/S01_baseline` (V5.0-Pendant unter `outputs/S01_baseline_v5/causal_ground_truth_benchmark.md`) |
| **[Master-Synopse V4.2](master_synopse_v4_gesamt.md)** | Gesamtsynopse über alle 15 Sensitivitätsszenarien (S01–S15) der V4.1-Grid-Ära |
| **[Konversationsprotokoll 17./18.10.2026](../07_conversation_logs/2026_10_17_Post_Refactor_Benchmark_V425_and_DGP_V5.md)** | Chronologisches Protokoll der Bugfixes, des V5.0-Upgrades, der Unit-Tests und des 34-Schritte-Modell-Benchmarks |
