---
created: 2026-10-17
last_updated: 2026-10-17
status: abgeschlossen
tags: [benchmark, v4.2.5, post-refactor, keras, pytorch, model-comparison, survival, causal-inference]
---

# Umfassende Modell-Gegenüberstellung (V4.2.5 Post-Refactor Benchmark)

> [!IMPORTANT]
> **Epistemischer Rahmen & Synthetische Benchmark-Validierung**  
> Sämtliche in diesem Dokument berichteten Metriken entstammen dem vollständigen **Post-Refactor Master-Benchmark (V4.2.5)** auf dem synthetischen Referenz-Szenario `data_v4_grid/S01_baseline/universe_A` ($N = 50.000$ Studierende, `seed=99999`, Split-Seed `42`). Da der datengenerierende Prozess (DGP V4.1) vollständig spezifiziert ist, dienen diese Ergebnisse **nicht** dem empirischen Nachweis realweltlicher hochschuldidaktischer Hypothesen, sondern der **methodischen Verifikation der Modellierungs-, Kalibrierungs- und Deconfounding-Kette**:
> 1. **Prädiktive Diskrimination & Wahrscheinlichkeitskalibrierung:** Vergleich klassischer tabellarischer Verfahren, zeitdiskreter Counting-Process-Survival-Modelle, kausaler 3D-Sequenzarchitekturen (GRU / Transformer) und autoregressiver Multi-Task-Modelle.
> 2. **Framework-Parität (Keras 3 vs. PyTorch 2.x):** Überprüfung, ob mathematisch äquivalente Architekturen in beiden Deep-Learning-Backends konsistente Schätzwerte liefern.
> 3. **Kausale Identifikation unter Selektionsbias:** Abgleich geschätzter Behandlungseffekte ($\text{HR}, \text{RR}, \text{ARR}$) gegen die exakt bekannte kontrafaktische Simulator-Ground-Truth ($\text{RR}_{\text{true}}(\text{A vs. B}) = 0{,}7868$, $\text{ARR}_{\text{true}} = +7{,}91\text{ pp}$).

---

## 1. Überblick: Post-Refactor Master-Benchmark & Behobene Altlasten

Im Rahmen des **OOP-Evaluator- und Pipeline-Refactorings (V4.2.5)** wurden alle >25 Modellarchitekturen in sechs Modellfamilien auf die zentralisierten Feature-Builder-Tensoren (`src/deepsupport/data_engine/feature_builder.py`) und die fünf standardisierten Evaluator-Klassen in [`src/deepsupport/evaluation/metrics_logger.py`](../../src/deepsupport/evaluation/metrics_logger.py) umgestellt:

| Evaluator-Klasse | Einsatzbereich | Standardisierte Kernmetriken (Zero-Imputation Policy: fehlende Werte als `null`) |
| :--- | :--- | :--- |
| **`SurvivalEvaluator`** | Binäre Landmark-Klassifikation, Counting-Process Panel-Survival, 3D-Sequenz-Survival | `roc_auc`, `pr_auc_dropout` ($y=1$), `pr_auc_nondropout` ($y=0$), Prävalenz $\pi_0$, Lift, `brier_score`, `brier_skill_score` (BSS), `c_index` |
| **`RegressionEvaluator`** | Hurdle Stage 2 Abschlussnoten-Regression (`graduates_only=True`), 3D-Sequenz-Regressoren | `r2_score`, `rmse`, `mae`, `median_ae`, `explained_variance`, `max_error` |
| **`MulticlassEvaluator`** | 4-Klassen-Landmark-Statusprognose ($t_0 = 2$) | `accuracy`, `f1_macro`, `f1_weighted`, `balanced_accuracy`, `roc_auc_ovr_macro`, klassenweise `pr_auc_0..3` |
| **`DualHeadEvaluator`** | Autoregressive Next-Exam Multi-Task-Modelle (Note + Bestehen) | Simultan: Regressionsmetriken (Head 1) + Kalibrierungs-/PR-AUC-Metriken für `Pass` ($y=1$) und `Fail` ($y=0$) (Head 2) |
| **`CausalEvaluator`** | Extended Cox, Orthogonal DML, Deep Transformer-DML, MSM (IPTW), G-Computation | `hr_fachlich`, `hr_ueberfachlich`, `hr_psychosozial`, `gcomputation_rr_full_support`, 95%-Konfidenzintervalle (asymptotisch & 100-Fold Student-Cluster-Bootstrap) |

### 1.1 Fünf durch den Post-Refactor-Benchmark aufgedeckte und behobene Methodik-/Numerik-Bugs

Der systematische Gesamtlauf über `src/deepsupport/runners/run_post_refactor_benchmark_s01.py` (34 Benchmark-Schritte, 100 % `PASSED`) deckte fünf subtile methodische bzw. numerische Inkonsistenzen aus der Vor-Refactoring-Ära (V4.2) auf, die vor der finalen Auswertung behoben wurden:

1. **Numerische Auslöschung (`Float64 Catastrophic Cancellation`) & Newton-Overshoot in `statsmodels` `PHReg` (`extended_cox.py`):**
   - *Befund:* In `statsmodels.duration.hazard_regression.PHReg.breslow_loglike` wird die Risikomenge rückwärts über $t = 15 \dots 1$ mittels `xp0 += ...; xp0 -= ...` bei globaler Zentrierung `linpred -= linpred.max()` akkumuliert. Bei $N = 259.568$ Personen-Semestern führte dies zu IEEE-754 Float64-Auslöschung (`1.0 + 1e-18 - 1.0 == 0.0`) sowie einer singulären Hesse-Matrix durch die innerhalb jeder Risikomenge $R(t)$ konstante Indikatorvariable `is_first_semester` ($\mathbb{I}(t_{\text{stop}} == 1)$).
   - *Lösung:* Ausschluss von `is_first_semester` aus den zeitvariablen In-Risk-Set-Kovariaten, Übergabe von `strata=t_stop` (da die Risikomengen $R(t) = \{i : t_{\text{stop},i} = t\}$ im diskreten Personen-Semester-Panel paarweise disjunkt sind, ist die stratifizierte Partial-Likelihood mathematisch zu 100 % identisch zur unstratifizierten Likelihood mit $\max |\Delta \nabla \ell| = 1{,}16 \times 10^{-10}$, normiert `linpred` jedoch lokal pro Semester $t$) sowie ein gedämpfter Newton-Raphson-Solver mit Backtracking-Line-Search.
2. **Future Sequence-Length Leakage in `PyTorchDynamicDeepHit` (`torch/sequence.py`):**
   - *Befund:* Die ursprüngliche PyTorch-Implementierung nutzte ein bidirektionales `AttentionPooling` über die gesamte Sequenz $1 \dots T_i$ zur Vorhersage jedes Schrittes $t$, wodurch die finale Sequenzlänge $T_i$ (Abbruchzeitpunkt) in frühe Semester leckte (`ROC-AUC = 0.9980`).
   - *Lösung:* Vollständige Neuimplementierung als strikt kausales unidirektionales 2-Layer-GRU-Schritt-Hazard-Modell mit kombinierter Verlustfunktion $(1 - \alpha_{\text{rank}}) \mathcal{L}_{\text{NLL}} + \alpha_{\text{rank}} \mathcal{L}_{\text{rank}}$ ($\alpha_{\text{rank}} = 0{,}1$) zum paarweisen Vergleich von $F_i(T_i)$ vs. $F_j(T_i)$ am tatsächlichen Ereigniszeitpunkt $T_i$ nach Lee et al. (2018). Ergebnis: leak-freier `Step ROC-AUC = 0.7957` (`Brier = 0.0373, BSS = +0.079`).
3. **Intervall-Hazard $h_t(X_{it})$ vs. Kumulative Inzidenz $F_i(t)$ auf Personen-Semester-Panels (`extended_deepsurv.py`, `torch/survival.py`, `run_torch_lxc.py`):**
   - *Befund:* Im Counting-Process-Panel repräsentiert das Ziel-Label `e_test` den bedingten Austritt im aktuellen Semesterintervall $(t-1, t]$. Mehrere Panel-Modelle (`Extended DeepSurv`, `PyTorchLogisticHazard`, `PyTorchCoxPH`, `PyTorchCoxTime`) evaluierten zuvor unkalibrierte Log-Risiken $\hat{\eta}_{it}$ oder kumulative Risiken $P(T \le t \mid X_{it})$ gegen das Intervall-Label `e_test`, was zu künstlich niedrigen Panel-AUCs (`0.5570–0.7631`) und stark negativen Brier Skill Scores führte.
   - *Lösung:* Implementierung von `predict_step_hazard(x, timestep)` (bedingte Schritt-Hazard $h_t(X_{it}) = P(T=t \mid T \ge t, X_{it})$) sowie Breslow-Schritt-Kalibrierung $1 - \exp(-\Delta \hat{H}_0(t) e^{\hat{\eta}_{it}})$. Dadurch steigt `Extended DeepSurv` von `ROC-AUC = 0.5570` auf `0.8165` (`standard`) bzw. `0.8212` (`oracle`) und erreicht exakte Parität mit `PyTorchCoxPH` (`0.8160` / `0.8206`) und `PyTorchLogisticHazard` (`0.8160` / `0.8200`).
4. **3D-Sequenz-Standardisierung, Exakte Maskierung & Hurdle Stage 2 (`deep_transformer_regression.py`, `landmark_prediction.py`):**
   - *Befund:* Die tiefen 3D-Sequenz-Transformer (`Deep Semester/Exam Transformer Regressor & Survival`) wurden in V4.2 auf unskalierten Roh-Tensoren (in denen `cp_cum` bis 180 und `overload` bis 400 die Sinus/Cosinus-Positionskodierung im ersten Layer übersteuerten) und ohne Hurdle-Stage-2-Filter (`graduates_only=True`) trainiert (`R² = 0.2222` bzw. `0.3347`). Zudem berechnete `landmark_prediction.py` die Sequenzmaske *nach* dem `StandardScaler` mittels `(X != 0).any(axis=-1)`, wodurch 0-Padding-Tokens nach Mittelwertzentrierung als aktive Tokens behandelt wurden.
   - *Lösung:* Exakte Längenberechnung vor der Skalierung, 3D-`StandardScaler`-Fit ausschließlich auf aktiven Nicht-Padding-Tokens des Trainingssplits (Nullsetzen der Padding-Positionen nach Transformation) und strikte Hurdle-Stage-2-Filterung (`graduates_only=True`). Ergebnis: `Deep Semester Transformer Regressor` steigt im `gradeblind`-Modus von $R^2 = 0{,}2222$ auf **$R^2 = 0{,}7838$** (+56,16 pp), `PyTorch Exam Transformer Regressor` erreicht **$R^2 = 0{,}8065$**.
5. **Studierenden-Inverse Stichprobengewichtung ($w_{ik} = 1/K_i$) & Multi-Output-Mapping (`autoregressive_gru.py`, `autoregressive_transformer.py`):**
   - *Befund:* In den autoregressiven Next-Exam-Modellen gingen Langzeitstudierende mit bis zu $K_i = 40$ Prüfungen viermal stärker in den Gradienten ein als frühe Abbrecher.
   - *Lösung:* Durchgängige Gewichtung jedes Prüfungsübergangs $(i, k)$ mit $w_{ik} = 1/K_i$ (normiert auf $\bar{w} = 1{,}0$) in Keras 3 und PyTorch. Ergebnis: Der `Keras Autoregressive Next-Exam GRU` verbessert sich von Grade $R^2 = 0{,}5706$ auf **$R^2 = 0{,}6885$** (+11,79 pp) und beim `Fail PR-AUC` ($y=0$) von $0{,}5852$ auf **$0{,}7706$** (+18,54 pp).

---

## 2. Familie 1: Statische Landmark-Modelle am Ende von Semester 2 ($t_0 = 2$)

Alle Modelle der Familie 1 operieren auf dem streng kausal am Ende des 2. Fachsemesters ($t_0 = 2$) eingefrorenen Querschnitts-Datensatz (`build_landmark_dataset()`, $N_{\text{test}} = 7.196$ Studierende für Klassifikation/Survival bzw. $N_{\text{test,grad}} = 5.314$ Absolventen in Hurdle Stage 2). Studierende, die bereits in Semester 1–2 ausgeschieden sind, werden bei den nachgelagerten Multi-State-Evaluierungen separat ausgewiesen (`exclude_early_dropouts=True` in Familie 5 mit $N_{\text{test}} = 6.941$).

### 2.1 4-Klassen-Multiclass-Statusprognose (`MulticlassEvaluator`)

Zielvariable ist der finale 4-Klassen-Studienstatus (`0 = Abgeschlossen`, `1 = Abgebrochen`, `2 = Exmatrikuliert`, `3 = Zeitüberschreitung`). Verglichen werden die rein tabellarischen Landmark-Klassifikatoren (`baseline_classifiers.py`) mit dem hybriden **Transformer-Embedding + XGBoost**-Ansatz aus Familie 5 (`landmark_prediction.py`, der ein 32-dimensionales Pre-LayerNorm-Prüfungssequenz-Embedding aus S1–S2 mit den tabellarischen Landmark-Features fusioniert):

| Modellarchitektur | Framework | Feature-Modus | $N_{\text{test}}$ | Accuracy | Macro-F1 | Weighted-F1 | Balanced Acc. | OVR Macro ROC-AUC | PR-AUC Kl. 0 (Abschl.) | PR-AUC Kl. 1 (Abbr.) | PR-AUC Kl. 2 (Exmat.) | PR-AUC Kl. 3 (Zeitüb.) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Multiclass** | XGBoost | `standard` | 7.196 | 0,7385 | 0,4090 | 0,7037 | 0,4039 | 0,8351 | 0,9382 | 0,6812 | 0,3140 | 0,0588 |
| **XGBoost Multiclass** | XGBoost | `gradeblind` | 7.196 | 0,7232 | 0,3822 | 0,6845 | 0,3811 | 0,8114 | 0,9295 | 0,6540 | 0,2715 | 0,0492 |
| **Keras MLP Multiclass** | Keras 3 | `standard` | 7.196 | 0,7390 | 0,3986 | 0,6982 | 0,3954 | 0,8384 | 0,9395 | 0,6890 | 0,3085 | 0,0541 |
| **Keras MLP Multiclass** | Keras 3 | `gradeblind` | 7.196 | 0,7246 | 0,3834 | 0,6851 | 0,3825 | 0,8142 | 0,9308 | 0,6610 | 0,2690 | 0,0485 |
| **Hybrid: 32-D Exam-Transformer Embedding + XGBoost (F5.1)** | Keras 3 + XGBoost | `standard` | 6.941 | **0,7963** | **0,4382** | **0,7691** | **0,4148** | **0,8500** | **0,9396** | 0,3896* | **0,4468** | **0,0624** |

*\*Hinweis zu F5.1 (`landmark_prediction.py`): Durch `exclude_early_dropouts=True` ($t > 2$) sinkt die Basisrate für freiwilligen Studienabbruch nach Semester 2 deutlich, während die Erkennung späterer Zwangsexmatrikulationen (Klasse 2: PR-AUC steigt von `0,3140` auf **`0,4468`**) und des Gesamterfolgs (`Weighted-F1 = 0,7691`, `OVR ROC-AUC = 0,8500`) durch das gelernte 32-D-Sequenz-Embedding signifikant gewinnt.*

---

### 2.2 Binäre Landmark-Abbruchprognose & Landmark-Survival (`SurvivalEvaluator`, $t_0 = 2$)

Zielvariable ist das binäre Ereignis `Dropout vs. Abschluss` ($N_{\text{test}} = 7.196$, Ereignisprävalenz $\pi_0(y=1) = 0{,}2615$, Komplementärprävalenz $\pi_0(y=0) = 0{,}7385$). Gemäß der **Evaluation PR-AUC Policy** werden die Precision-Recall-AUCs für beide Klassen samt Lift gegenüber $\pi_0$ ausgewiesen:

| Modellarchitektur | Framework | Modus | ROC-AUC | PR-AUC Dropout ($y=1$) | Lift ($y=1$) | PR-AUC Non-Drop ($y=0$) | Brier Score | Brier Skill Score (BSS) | F1 ($\tau=0{,}5$) | Balanced Acc. | Harrell C-Index |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Hazard Landmark** | Keras 3 | `standard` | **0,8740** | **0,7705** | **2,95x** | **0,9387** | 0,1139 | +0,4105 | **0,6347** | **0,7402** | **0,7907** |
| **Keras MLP Binary Classifier** | Keras 3 | `standard` | **0,8740** | 0,7701 | **2,94x** | 0,9382 | **0,1135** | **+0,4121** | 0,6336 | 0,7393 | `null` |
| **Keras MLP Binary Classifier** | Keras 3 | `gradeblind` | 0,8720 | 0,7675 | 2,93x | 0,9370 | 0,1142 | +0,4089 | 0,6346 | 0,7401 | `null` |
| **XGBoost Binary Classifier** | XGBoost | `standard` | 0,8698 | 0,7642 | 2,92x | 0,9358 | 0,1154 | +0,4024 | 0,6310 | 0,7375 | `null` |
| **XGBoost Binary Classifier** | XGBoost | `gradeblind` | 0,8351 | 0,7118 | 2,72x | 0,9185 | 0,1284 | +0,3350 | 0,5980 | 0,7142 | `null` |
| **Gaussian Naive Bayes** | Sklearn | `standard` | 0,8576 | 0,7476 | 2,86x | 0,9284 | 0,1487 | +0,2299 | 0,6215 | 0,7480 | `null` |
| **Gaussian Naive Bayes** | Sklearn | `gradeblind` | 0,8614 | 0,7522 | 2,88x | 0,9301 | 0,1395 | +0,2775 | 0,6240 | 0,7492 | `null` |
| **Random Forest Classifier** | Sklearn | `standard` | 0,8480 | 0,7286 | 2,79x | 0,9210 | 0,1255 | +0,3499 | 0,5890 | 0,7085 | `null` |
| **Random Forest Classifier** | Sklearn | `gradeblind` | 0,8339 | 0,7089 | 2,71x | 0,9142 | 0,1333 | +0,3098 | 0,5745 | 0,6990 | `null` |
| **SVM (RBF Kernel)** | Sklearn | `standard` | 0,8254 | 0,7414 | 2,83x | 0,9085 | 0,1221 | +0,3678 | 0,6012 | 0,7160 | `null` |
| **SVM (RBF Kernel)** | Sklearn | `gradeblind` | 0,8210 | 0,7398 | 2,83x | 0,9062 | 0,1225 | +0,3655 | 0,5985 | 0,7140 | `null` |
| **Breslow DeepSurv Landmark** | Keras 3 | `standard` | 0,7275* | 0,4933* | 1,89x | 0,8381 | 0,1740 | +0,0991 | 0,3765 | 0,6065 | 0,7898 |

*\*Methodischer Befund zu `Breslow DeepSurv Landmark`: In der Ereigniszeit-Konkordanz (`Harrell C-Index = 0,7898` vs. `0,7907` bei `Logistic Hazard`) sind Cox-Proportional-Hazards-Netze und diskrete Hazard-Netze nahezu identisch. Für die kumulative binäre Abschluss-/Abbruchwahrscheinlichkeit über ein 16-Semester-Fenster liefert das zeitdiskrete `Logistic Hazard Landmark`-Modell jedoch eine deutlich schärfere Wahrscheinlichkeitskalibrierung (`BSS = +0,4105` vs. `+0,0991`), da es keine Proportional-Hazards-Restriktion über frühe und späte Semester hinweg erzwingt.*

---

### 2.3 Hurdle Stage 2: Konditionale Abschlussnoten-Regression (`RegressionEvaluator`, `graduates_only=True`)

Da Studienabbrecher keine Abschlussnote besitzen (strukturelles Fehlen durch Abbruch; siehe Hurdle-Modellierung in der [Master-Synopse](master_synopse_v4_gesamt.md)), wird die Regressionsstufe strikt auf den tatsächlichen Absolventen des Testsplits ($N_{\text{test,grad}} = 5.314$) evaluiert:

| Modellarchitektur | Framework | Feature-Modus | $N_{\text{test,grad}}$ | $R^2$ | RMSE (Notenpunkte) | MAE | Median AE | Explained Variance | Max Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Keras MLP Regressor** | Keras 3 | `standard` | 5.314 | **0,8706** | **0,2154** | **0,1651** | **0,1297** | **0,8706** | **0,9693** |
| **XGBoost Regressor** | XGBoost | `standard` | 5.314 | 0,8673 | 0,2181 | 0,1670 | 0,1315 | 0,8673 | 0,9850 |
| **SVR (RBF Kernel)** | Sklearn | `standard` | 5.314 | 0,8672 | 0,2182 | 0,1672 | 0,1318 | 0,8672 | 0,9912 |
| **Random Forest Regressor** | Sklearn | `standard` | 5.314 | 0,8538 | 0,2289 | 0,1755 | 0,1390 | 0,8538 | 1,0420 |
| **Ridge Regression (L2)** | Sklearn | `standard` | 5.314 | 0,8510 | 0,2311 | 0,1796 | 0,1435 | 0,8510 | 1,0615 |
| **Hybrid: 32-D Exam-Transformer Embedding + XGBoost (F5.1)** | Keras 3 + XGBoost | `standard` (OOF-Emb.) | 5.327 | 0,7865 | 0,2782 | 0,2174 | 0,1735 | 0,7866 | 1,1791 |
| **Keras MLP Regressor** | Keras 3 | `gradeblind` | 5.314 | **0,7751** | **0,2839** | **0,2226** | **0,1799** | **0,7753** | **1,2137** |
| **XGBoost Regressor** | XGBoost | `gradeblind` | 5.314 | 0,7686 | 0,2880 | 0,2261 | 0,1830 | 0,7686 | 1,2410 |
| **Ridge Regression (L2)** | Sklearn | `gradeblind` | 5.314 | 0,7654 | 0,2900 | 0,2285 | 0,1852 | 0,7654 | 1,2580 |
| **Random Forest Regressor** | Sklearn | `gradeblind` | 5.314 | 0,7525 | 0,2978 | 0,2342 | 0,1905 | 0,7525 | 1,2940 |

**Befund zum Informationsgehalt früher Prüfungsnoten vs. Eingangsvoraussetzungen:**
Selbst im strikten `gradeblind`-Modus (ohne jede Hochschulprüfungsnote aus Semester 1–2, nur mit `hzb_note`, Studiengang, Erwerbstätigkeit, CP-Fortschritt und Fehlversuchszählern) erklären die Regressoren **$77{,}5\,\%$ der Varianz** der späteren Bachelor-Abschlussnote ($\text{RMSE} = 0{,}2839$). Die Hinzunahme der tatsächlichen Prüfungsnoten aus Semester 1–2 (`standard`) steigert das Bestimmtheitsmaß um $+9{,}55\text{ pp}$ auf **$R^2 = 0{,}8706$** ($\text{RMSE} = 0{,}2154$).

---

## 3. Familie 2: Counting-Process Panel-Survival & Competing Risks (Semester-Ebene)

Familie 2 modelliert den Studienverlauf als zeitdiskretes Counting-Process-Panel $(t_{\text{start}}, t_{\text{stop}}]$ über alle aktiven Personen-Semester (`build_semester_panel_df()`, $N_{\text{total}} = 259.568$ Personen-Semester, Testsplit $N_{\text{test}} = 51.921$ Personen-Semester von $7.500$ Test-Studierenden, Intervall-Abbruchprävalenz $\pi_0(y=1) = 0{,}0410$).

### 3.1 Gesamtvergleich aller Panel-Survival- & Competing-Risks-Modelle (Pre-Refactor V4.2 vs. Post-Refactor V4.2.5)

| Modellarchitektur | Framework | Modus | Pre-Refactor V4.2 Panel ROC-AUC | Post-Refactor V4.2.5 Step ROC-AUC | PR-AUC Dropout ($y=1$, $\pi_0=0{,}041$) | Lift ($y=1$) | PR-AUC Non-Drop ($y=0$) | Brier Score | Brier Skill Score (BSS) | Harrell C-Index | Kumulativer CIF ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Extended Cox (`PHReg`, `strata=t_stop`)** | Statsmodels | `standard` | 0,7410* | 0,7795 | 0,2180 | 5,32x | 0,9865 | `null` | `null` | 0,6892 | `null` |
| **Extended Cox (`PHReg`, `strata=t_stop`)** | Statsmodels | `oracle` | — | 0,7860 | 0,2295 | 5,60x | 0,9871 | `null` | `null` | 0,6980 | `null` |
| **Extended Logistic Hazard** | Keras 3 | `standard` | 0,8002 | **0,8152** | 0,2722 | 6,65x | 0,9893 | 0,0342 | +0,1302 | 0,6725 | — |
| **Extended Logistic Hazard** | Keras 3 | `oracle` | — | **0,8202** | 0,2823 | 6,89x | 0,9896 | 0,0340 | +0,1342 | 0,6750 | — |
| **Extended DeepSurv (+ Breslow Step-Haz.)** | Keras 3 | `standard` | 0,5570 | **0,8165** | **0,2831** | **6,91x** | 0,9893 | 0,0342 | +0,1296 | 0,6808 | 0,6596 |
| **Extended DeepSurv (+ Breslow Step-Haz.)** | Keras 3 | `oracle` | — | **0,8212** | **0,2885** | **7,04x** | **0,9896** | 0,0341 | +0,1323 | 0,7068 | 0,6908 |
| **Dynamic DeepHit (3-State Softmax CR)** | Keras 3 | `standard` | 0,8164 | **0,8177** | 0,2630 | 6,33x | 0,9894 | 0,0351 | +0,1196 | `null` | — |
| **PyTorch Logistic Hazard (`predict_step_hazard`)** | PyTorch 2.x | `standard` | 0,5586 | **0,8160** | 0,2821 | 6,89x | 0,9891 | **0,0338** | **+0,1391** | 0,8050 | 0,7088 |
| **PyTorch Logistic Hazard (`predict_step_hazard`)** | PyTorch 2.x | `oracle` | 0,7384 | **0,8207** | 0,2880 | 7,03x | 0,9894 | **0,0337** | **+0,1430** | 0,8081 | 0,7384 |
| **PyTorch CoxPH Extended (+ Breslow Step-Haz.)** | PyTorch 2.x | `standard` | 0,7014 | **0,8160** | 0,2795 | 6,82x | 0,9890 | 0,0344 | +0,1253 | **0,8054** | 0,7013 |
| **PyTorch CoxPH Extended (+ Breslow Step-Haz.)** | PyTorch 2.x | `oracle` | 0,7075 | **0,8194** | 0,2838 | 6,93x | 0,9892 | 0,0342 | +0,1290 | **0,8086** | 0,7075 |
| **PyTorch CoxTime (Non-Proportional Step-Haz.)** | PyTorch 2.x | `standard` | 0,7631 | **0,8157** | 0,2788 | 6,81x | 0,9890 | 0,0344 | +0,1248 | 0,8050 | 0,7606 |
| **PyTorch CoxTime (Non-Proportional Step-Haz.)** | PyTorch 2.x | `oracle` | 0,7712 | **0,8188** | 0,2849 | 6,95x | 0,9892 | 0,0343 | +0,1280 | 0,8059 | 0,7712 |
| **PyTorch DeepHit (Single-Risk Ranking + NLL)** | PyTorch 2.x | `standard` | 0,7400 | 0,7058 | 0,1051 | 2,57x | 0,9815 | 0,0445 | -0,1322 | **0,8828** | 0,7422 |
| **PyTorch DeepHit Competing Risks (Multi-Event)** | PyTorch 2.x | `standard` | 0,7712 | 0,7463 | 0,1410 | 3,44x | 0,9851 | 0,0398 | -0,0130 | **0,8732** | 0,7675 |

*\*Hinweis zu Competing Risks (`Dynamic DeepHit 3-State Softmax`): Neben dem Abbruchrisiko (`ROC-AUC = 0,8177`) schätzt das 3-Zustands-Modell simultan die Wahrscheinlichkeit eines regulären Studienabschlusses im jeweiligen Semester mit **`ROC-AUC_Graduation = 0,9995`** und **`PR-AUC_Graduation = 0,9959`** (da der Abschluss im DGP deterministisch ab $\text{CP}_{\text{cum}} \ge 180$ eintritt).*

**Zentrale methodische Erkenntnis aus Familie 2:**
1. **Exakte Konvergenz über Modellklassen nach Schritt-Hazard-Kalibrierung:** Sobald alle Panel-Modelle konsistent über die bedingte Intervall-Hazard $h_t(X_{it}) = P(T=t \mid T \ge t, X_{it})$ ausgewertet werden, konvergieren `Keras Extended Logistic Hazard` (`0,8152`), `Keras Extended DeepSurv` (`0,8165`), `PyTorch Logistic Hazard` (`0,8160`), `PyTorch CoxPH Extended` (`0,8160`) und `PyTorch CoxTime` (`0,8157`) im `standard`-Modus innerhalb eines Korridors von lediglich **$\pm 0{,}0008$ ROC-AUC-Punkten**!
2. **Trade-off zwischen Konkordanz-Ranking (`DeepHit`) und Brier-Kalibrierung (`Logistic Hazard`):** Die Paar-Ranking-Strafe $\mathcal{L}_{\text{rank}}$ in `PyTorchDeepHit` maximiert den globalen **Harrell C-Index auf `0,8828`** (Single-Risk) bzw. **`0,8732`** (Competing Risks), verzerrt jedoch die absoluten Schritt-Wahrscheinlichkeiten (`BSS = -0,1322`). Für kalibrierte Frühwarnsysteme und kausale G-Computation/DML-Nuisance-Modelle sind daher strikt log-likelihood-basierte Schritt-Hazard-Modelle (`Logistic Hazard`, `BSS = +0,1391`) zwingend vorzuziehen.

---

## 4. Familie 3: 3D-Sequenz-Survival & Gradeblind Sequenz-Regressoren

Familie 3 verarbeitet die vollständige Studienhistorie als maskierten 3D-Tensor:
- **Semester-Sequenz-Tensor (`build_semester_sequence_tensor()`):** $(N, 16, F)$ mit $F = 18$ (`standard`), $F = 17$ (`gradeblind`) bzw. $F = 23$ (`oracle`) Features; Testsplit $N_{\text{test}} = 51.668\text{--}51.698$ aktive Semester-Schritte ($\pi_0 = 0{,}0417\text{--}0{,}0423$).
- **Prüfungs-Sequenz-Tensor (`build_exam_sequence_tensor()`):** $(N, 40, F)$ mit $F = 24$ (`standard`), $F = 23$ (`gradeblind`) bzw. $F = 29$ (`oracle`) Features; Testsplit $N_{\text{test}} = 127.514\text{--}128.198$ aktive Prüfungsschritte ($\pi_0 = 0{,}0171$).

### 4.1 3D-Sequenz-Survival (Semester- & Prüfungsebene, `SurvivalEvaluator`)

| Modellarchitektur | Granularität | Framework | Modus | Pre-Refactor V4.2 Step ROC-AUC | Post-Refactor V4.2.5 Step ROC-AUC | Step PR-AUC ($y=1$) | Lift ($y=1$) | Step PR-AUC ($y=0$) | Brier Score | Brier Skill Score (BSS) | Student-Level ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Semester Survival GRU (`semester_gru.py`)** | Semester ($T \le 16$) | Keras 3 | `standard` | 0,8113 | **0,8121** | 0,2676 | 6,41x | 0,9884 | 0,0353 | +0,1176 | 0,7018* |
| **Semester Survival GRU (`semester_gru.py`)** | Semester ($T \le 16$) | Keras 3 | `oracle` | — | **0,8139** | 0,2768 | 6,63x | 0,9886 | 0,0348 | +0,1313 | 0,7125* |
| **PyTorch Semester Survival GRU** | Semester ($T \le 16$) | PyTorch 2.x | `standard` | — | **0,8196** | **0,2988** | **7,06x** | **0,9891** | **0,0345** | **+0,1481** | **0,7018** |
| **Semester Causal Transformer (`semester_transformer.py`)** | Semester ($T \le 16$) | Keras 3 | `standard` | 0,8132 | **0,8144** | 0,2763 | 6,62x | 0,9886 | 0,0347 | +0,1326 | 0,6817* |
| **PyTorch Semester Causal Transformer** | Semester ($T \le 16$) | PyTorch 2.x | `standard` | — | **0,8147** | 0,2930 | 6,93x | 0,9884 | 0,0347 | +0,1438 | 0,6817 |
| **PyTorch Dynamic DeepHit (Causal Step-GRU)** | Semester ($T \le 16$) | PyTorch 2.x | `standard` | 0,9980 (Leak) / 0,5707 | **0,7957** | 0,2404 | 5,68x | 0,9871 | 0,0373 | +0,0790 | 0,5568 |
| **Exam Survival GRU (`exam_gru.py`)** | Prüfung ($K \le 40$) | Keras 3 | `standard` | 0,8933 | **0,8791** | 0,1678 | 9,80x | 0,9975 | 0,0154 | +0,0843 | 0,7379 |
| **PyTorch Exam Survival GRU** | Prüfung ($K \le 40$) | PyTorch 2.x | `standard` | — | **0,8976** | **0,2038** | **11,91x** | **0,9980** | **0,0150** | **+0,1059** | **0,7489** |
| **Exam Causal Transformer (`exam_transformer.py`)** | Prüfung ($K \le 40$) | Keras 3 | `standard` | 0,8859 | **0,8772** | 0,1601 | 9,35x | 0,9974 | 0,0155 | +0,0787 | 0,6885 |
| **Deep Exam Transformer Survival (BCE)** | Prüfung ($K \le 40$) | Keras 3 | `gradeblind` | 0,7751 | **0,8687** (+9,36 pp) | 0,1554 | 9,08x | 0,9971 | 0,0156 | +0,0739 | — |
| **PyTorch Causal Exam Transformer Survival** | Prüfung ($K \le 40$) | PyTorch 2.x | `gradeblind` | — | **0,8779** | 0,1596 | 9,32x | 0,9973 | 0,0153 | +0,0825 | — |

*\*Technischer Hinweis zur Student-Level-Aggregation auf Semesterebene: In den PyTorch-Sequenzmodellen wird das kumulative Studierenden-Abbruchrisiko über das zeitliche Maximum $\max_t \hat{h}_{it}$ aggregiert (`Student ROC-AUC = 0,7018` auf Semester- bzw. `0,7489` auf Prüfungsebene), während der Legacy-Helper in `semester_gru.py` das survival-gewichtete Produkt $1 - \prod_t (1 - \hat{h}_{it})$ über unterschiedlich lange Trajektorien bildete (wodurch Langzeitstudierende mit 14 Semestern mechanisch höhere kumulative Summen erhielten als frühe Abbrecher in Semester 1).*

---

### 4.2 Gradeblind 3D-Sequenz-Regressoren (Hurdle Stage 2, `RegressionEvaluator`, `graduates_only=True`)

Diese drei tiefen Transformer-Regressoren (`deep_transformer_regression.py` und `run_torch_lxc.py`) prüfen eine zentrale strukturelle Frage des Simulators: **Wie genau lässt sich die finale Bachelor-Abschlussnote aus der kompletten Studienverlaufshistorie vorhersagen, wenn sämtliche Einzelnoten (`note`, `gpa_prev`, `hzb_note` erhalten, aber keine Modulnoten) maskiert sind (`mode="gradeblind"`)?**

| Modellarchitektur | Granularität | Framework | Modus | Pre-Refactor V4.2 $R^2$ (ohne Hurdle/Scaler) | Post-Refactor V4.2.5 $R^2$ (`graduates_only=True`) | RMSE | MAE | Median AE | Explained Variance | Max Error | Trainingszeit |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Deep Semester Transformer Regressor** | Semester ($T \le 16$) | Keras 3 | `gradeblind` | 0,2222 | **0,7838** (+56,16 pp) | 0,2784 | 0,2188 | **0,1793** | 0,7859 | 1,2353 | 195,4 s |
| **Deep Exam Transformer Regressor** | Prüfung ($K \le 40$) | Keras 3 | `gradeblind` | 0,3347 | **0,7839** (+44,92 pp) | 0,2783 | 0,2202 | 0,1801 | 0,7858 | 1,2972 | 384,0 s |
| **PyTorch Exam Transformer Regressor** | Prüfung ($K \le 40$) | PyTorch 2.x | `gradeblind` | — | **0,8065** | **0,2634** | **0,2070** | — | — | — | 142,1 s |
| *Referenz: Statischer Landmark MLP Regressor ($t_0=2$)* | Querschnitt ($t_0=2$) | Keras 3 | `gradeblind` | 0,7751 | 0,7751 | 0,2839 | 0,2226 | 0,1799 | 0,7753 | 1,2137 | 18,2 s |

**Methodischer Befund:**
Durch den 3D-`StandardScaler` (nur auf aktiven Tokens) und die Hurdle-Stage-2-Konditionierung auf Absolventen ($N_{\text{test,grad}} = 5.314$) übertreffen die `gradeblind` Sequenz-Transformer ($R^2 = 0{,}7838\text{--}0{,}8065$, $\text{RMSE} = 0{,}2634\text{--}0{,}2784$) den statischen Landmark-Regressor aus Semester 2 ($R^2 = 0{,}7751$, $\text{RMSE} = 0{,}2839$) um bis zu **$+3{,}14\text{ pp}$ $R^2$**, da die kompletten Wiederholungsversuchs-, Modulschwierigkeits- und Support-Trajektorien über alle Studiensemester hinweg zusätzliche Information über das latente Leistungsniveau enthalten.

---

## 5. Familie 4 & 5: Autoregressive Next-Exam Dual-Head Suite & Representation Learning

### 5.1 Familie 4: Autoregressive Next-Exam Multi-Task-Modelle (`DualHeadEvaluator`, $1/K_i$-Gewichtung)

Die Modelle der Familie 4 sagen vor jedem einzelnen Prüfungsantritt $(i, k+1)$ anhand der bisherigen Prüfungssequenz $1 \dots k$ und des Zielmodul-Kontextes (`schwierigkeit`, `cp`, `versuch`, `semester`, Support-Indikatoren) simultan die **exakte Prüfungsnote** (Head 1: Regression) und die **Bestehenswahrscheinlichkeit** (Head 2: Klassifikation, `Pass` $y=1$ mit $\pi_0 = 0{,}8362$ vs. `Fail` $y=0$ mit $\pi_0 = 0{,}1638$) voraus ($N_{\text{test}} = 120.655$ Prüfungsübergänge):

| Modellarchitektur | Framework | Gewichtung | Head 1: Grade $R^2$ | Head 1: RMSE | Head 1: MAE | Head 1: MedAE | Head 2: Pass ROC-AUC | PR-AUC Pass ($y=1$, $\pi_0=0{,}836$) | PR-AUC Fail ($y=0$, $\pi_0=0{,}164$) | Lift Fail ($y=0$) | Brier Score | Brier Skill Score (BSS) | F1 Pass ($\tau=0{,}5$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *Pre-Refactor V4.2: Keras Next-Exam GRU* | Keras 3 | Ungewichtet | 0,5706 | 0,8948 | 0,6667 | — | 0,9367 | 0,9868 | 0,5852 | 3,57x | 0,0766 | +0,4405 | — |
| **Keras Next-Exam Dual-Head GRU** | Keras 3 | $w_{ik} = 1/K_i$ | 0,6885 (+11,8 pp) | 0,7622 | 0,5920 | 0,4788 | 0,9392 | 0,9873 | 0,7706 (+18,5 pp) | 4,70x | 0,0705 | +0,4850 | 0,9408 |
| **Keras Next-Exam Deep Transformer (Sin/Cos PE)** | Keras 3 | $w_{ik} = 1/K_i$ | 0,6913 | 0,7587 | 0,5708 | 0,4448 | 0,9380 | 0,9870 | 0,7665 | 4,68x | 0,0712 | +0,4805 | 0,9425 |
| **PyTorch Next-Exam Dual-Head GRU** | PyTorch 2.x | $w_{ik} = 1/K_i$ | 0,7036 | 0,7435 | 0,5617 | **0,4312** | 0,9406 | 0,9876 | 0,7760 | 4,74x | 0,0682 | +0,5018 | **0,9437** |
| **PyTorch Next-Exam Dual-Head Transformer** | PyTorch 2.x | $w_{ik} = 1/K_i$ | **0,7050** | **0,7418** | **0,5610** | 0,4314 | **0,9409** | **0,9877** | **0,7779** | **4,75x** | **0,0681** | **+0,5025** | 0,9432 |

**Zentrale Erkenntnisse aus Familie 4:**
1. **Durchbruch bei der Klausur-Durchfallprognose (`Fail PR-AUC`):** Während Nichtbestehen nur $\pi_0(y=0) = 16{,}38\,\%$ aller Prüfungsantritte betrifft, erreichen alle vier refaktorierten Dual-Head-Architekturen mit $1/K_i$-Gewichtung einen **`Fail PR-AUC` von `0,7665` bis `0,7779`** (ein **4,75-facher Präzisions-Lift** gegenüber dem Zufallsbaseline-Wert und $+18{,}5\text{ pp}$ gegenüber dem ungewichteten V4.2-Lauf) sowie einen **Brier Skill Score von bis zu $+0{,}5025$**.
2. **Nahezu perfekte Framework- und Architektur-Konsistenz:** Zwischen GRU und Transformer sowie zwischen Keras 3 und PyTorch 2.x liegen die Unterschiede beim `Pass ROC-AUC` (`0,9380` bis `0,9409`) bei weniger als $0{,}3\text{ pp}$ und beim `Grade R²` (`0,6885` bis `0,7050`) bei $1{,}65\text{ pp}$.

### 5.2 Familie 5: Landmark Representation Learning (`landmark_prediction.py`)

Familie 5 untersucht, ob ein auf den Einzelprüfungen der ersten beiden Semester ($t \le 2$) vortrainierter **Pre-LayerNorm Transformer-Encoder**, dessen 32-dimensionaler `GlobalAveragePooling1D`-Bottleneck-Vektor mittels 5-Fold Out-of-Fold-Cross-Fitting extrahiert und mit den 16 tabellarischen Landmark-Features konkateniert wird, rein tabellarische GBDTs (`XGBoost`) schlägt:
- **4-Klassen-Statusprognose (`landmark_representation_status`):** `Accuracy = 0,7963`, `Macro-F1 = 0,4382` (vs. `0,4090` rein tabellarisch), `Weighted-F1 = 0,7691` (vs. `0,7037`), `OVR Macro ROC-AUC = 0,8500` (vs. `0,8351`).
- **Hurdle Stage 2 Abschlussnoten-Regression (`landmark_representation_grade`):** $R^2 = 0{,}7865$, $\text{RMSE} = 0{,}2782$, $\text{MAE} = 0{,}2174$.

---

## 6. Familie 6: Kausale Inferenz-Suite vs. Kontrafaktische Simulator-Ground-Truth (`CausalEvaluator`)

Familie 6 bildet das methodische Herzstück zur Auflösung des **Selbstselektions-Paradoxons**: Da im Simulator vor allem Studierende in akuten Leistungskrisen (`note > 3.0`, `versuch >= 2`, niedrige `motivation`) Support in Anspruch nehmen, weisen Support-Nutzende in Universum A naiv betrachtet eine um **+72,5 % höhere Abbruchrate** auf ($\text{RR}_{\text{naiv}} = 1{,}725$), obwohl Support in Wahrheit kausal schützt ($\text{RR}_{\text{true}}(\text{A vs. B}) = 0{,}7868$, Risikoreduktion $-21{,}32\,\%$, $\text{ARR}_{\text{true}} = +7{,}91\text{ pp}$).

### 6.1 Synoptischer Vergleich aller Kausal-Schätzer gegen die kontrafaktische Ground Truth

*(Detaillierte Herleitung der drei Estimand-Ebenen — Robinson Step-Hazard-HR, 1-Step Step-Standardized RR und 16-Semester Sequential Rollout RR — siehe [Kausaler Ground-Truth Benchmark](causal_ground_truth_benchmark.md) und [Kausale Vergleichsanalyse](../04_causal_and_simulation/04_Kausale_Vergleichsanalyse.md)).*

| Schätzer / Modellfamilie | Framework | Modus | Estimand-Ebene | $\text{HR}_{\text{fach}}$ / $\text{RR}_{\text{fach}}$ [95%-KI] | $\text{HR}_{\text{uebf}}$ / $\text{RR}_{\text{uebf}}$ [95%-KI] | $\text{HR}_{\text{psych}}$ / $\text{RR}_{\text{psych}}$ [95%-KI] | Kumulativer Gesamt-Effekt (All Support) | Nuisance Panel ROC-AUC / Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0. Kontrafaktische Simulator-Ground-Truth (`Universe A vs. B/F/G/H`)** | DGP V4.1 ($N=50\text{k}$) | Exakt (Seed-Sync) | 16-Sem. Kumulatives Risiko | **$\text{RR}(\text{F vs. B}) = 0{,}9048$**<br>($-9{,}52\,\%$, $\text{ARR}=+3{,}53\text{ pp}$) | **$\text{RR}(\text{G vs. B}) = 0{,}9153$**<br>($-8{,}47\,\%$, $\text{ARR}=+3{,}14\text{ pp}$) | **$\text{RR}(\text{H vs. B}) = 0{,}9392$**<br>($-6{,}08\,\%$, $\text{ARR}=+2{,}26\text{ pp}$) | **$\text{RR}(\text{A vs. B}) = 0{,}7868$**<br>($-21{,}32\,\%$, $\text{ARR}=+7{,}91\text{ pp}$) | — |
| **1. Naiver Gruppenvergleich in Universum A** | Empirisch unbereinigt | — | Kumulatives Risiko | — | — | — | $\text{RR}_{\text{naiv}} = 1{,}725$<br>(**+72,5 % Schein-Risiko**) | — |
| **2a. Extended Cox (`PHReg`, Pre-Refactor V4.2)** | Statsmodels | `gradeblind` | Bedingtes Step-HR | 1,1196 (+12,0 % Bias) | 1,0344 (+3,4 % Bias) | 1,0288 (+2,9 % Bias) | Nicht deconfounded | — |
| **2b. Extended Cox (`PHReg`, Post-Refactor V4.2.5)** | Statsmodels | `standard` | Bedingtes Step-HR (`strata=t_stop`) | **0,9587** $[0{,}917;\, 1{,}002]$ | **0,9906** $[0{,}974;\, 1{,}007]$ | **0,9770** $[0{,}953;\, 1{,}001]$ | Vorzeichenwechsel ($\text{HR} < 1$) | 0,7795 / `null` |
| **2c. Extended Cox (`PHReg`, Post-Refactor V4.2.5)** | Statsmodels | `oracle` | Bedingtes Step-HR (`strata=t_stop`) | **0,9581** $[0{,}916;\, 1{,}002]$ | **0,9881** $[0{,}972;\, 1{,}005]$ | **0,9685** $[0{,}945;\, 0{,}993]$ | Alle 3 Kanäle $\text{HR} < 1$ | 0,7860 / `null` |
| **3a. Keras Orthogonal DML (`dml_orthogonal.py`)** | Keras 3 (5-Fold GroupKFold) | `standard` | Multivariate Robinson Step-HR | **0,8475** $[0{,}786;\, 0{,}913]$<br>(Partial $\text{RR}=0{,}9034$) | **0,9790** $[0{,}943;\, 1{,}016]$<br>(Partial $\text{RR}=0{,}9432$) | **0,9883** $[0{,}958;\, 1{,}020]$<br>(Partial $\text{RR}=0{,}9545$) | Alle 3 Kanäle $\text{HR} < 1$ | 0,8145 / 0,0341 |
| **3b. Keras Orthogonal DML (`dml_orthogonal.py`)** | Keras 3 (5-Fold GroupKFold) | `oracle` | Multivariate Robinson Step-HR | **0,8503** $[0{,}789;\, 0{,}916]$<br>(Partial $\text{RR}=0{,}9033$) | **0,9610** $[0{,}925;\, 0{,}998]$<br>(Partial $\text{RR}=0{,}9401$) | **0,9729** $[0{,}942;\, 1{,}005]$<br>(Partial $\text{RR}=0{,}9521$) | Alle 3 Kanäle signifikant/konsistent $<1$ | 0,8188 / 0,0339 |
| **4a. PyTorch Orthogonal DML (`causal.py`)** | PyTorch 2.x (5-Fold + 100x Boot.) | `standard` | Multivariate Robinson Step-HR | **0,8952** $[0{,}843;\, 0{,}940]_{\text{boot}}$ | **0,9711** $[0{,}938;\, 1{,}003]_{\text{boot}}$ | **0,9737** $[0{,}943;\, 1{,}003]_{\text{boot}}$ | Alle 3 Kanäle $\text{HR} < 1$ | 0,8145 / 0,0340 |
| **4b. PyTorch Orthogonal DML (`causal.py`)** | PyTorch 2.x (5-Fold + 100x Boot.) | `oracle` | Multivariate Robinson Step-HR | **0,9135** $[0{,}861;\, 0{,}958]_{\text{boot}}$ | **0,9410** $[0{,}907;\, 0{,}980]_{\text{boot}}$ | **0,9642** $[0{,}931;\, 0{,}992]_{\text{boot}}$ | **Alle 3 Kanäle 95%-Boot-KI $< 1{,}0$!** | 0,8195 / 0,0338 |
| **5a. Keras Deep Transformer-DML (`dml_transformer.py`)** | Keras 3 (Sequenz-Kontext) | `standard` | Robinson Step-HR & Partial RR | **0,9585** $[0{,}839;\, 1{,}095]$ | 1,0354 $[0{,}982;\, 1{,}092]$ | **0,9505** $[0{,}897;\, 1{,}008]$ | Sequenzielles Deconfounding | 0,8184 / 0,0335 |
| **5b. Keras Deep Transformer-DML (`dml_transformer.py`)** | Keras 3 (Sequenz-Kontext) | `oracle` | Robinson Step-HR & Partial RR | **0,4625** $[0{,}349;\, 0{,}613]$<br>(Partial $\text{RR}=0{,}9145$) | 1,0265 $[0{,}978;\, 1{,}077]$ | 1,0059 $[0{,}956;\, 1{,}058]$ | Starke Fach-Identifikation | **0,8200 / 0,0333** |
| **6a. PyTorch Marginal Structural Model (IPTW `cloglog`)** | PyTorch 2.x | `standard` | Marginales Step-HR | 1,0116 $[0{,}959;\, 1{,}067]$ | **0,9076** $[0{,}882;\, 0{,}934]$ | **0,9103** $[0{,}879;\, 0{,}943]$ | **$\text{HR}_{\text{all}} = 0{,}9170$** $[0{,}880;\, 0{,}955]$ | — |
| **6b. PyTorch Marginal Structural Model (IPTW `cloglog`)** | PyTorch 2.x | `oracle` | Marginales Step-HR | 1,0835 $[1{,}027;\, 1{,}143]$* | **0,9170** $[0{,}891;\, 0{,}943]$ | **0,9110** $[0{,}880;\, 0{,}944]$ | **$\text{HR}_{\text{all}} = 0{,}9170$** $[0{,}880;\, 0{,}955]$ | — |
| **7a. PyTorch Parametric G-Computation** | PyTorch 2.x (100x Boot.) | `standard` | 1-Step Standardized vs. 16-Sem. Rollout | — | — | — | **1-Step $\text{RR} = 0{,}8603$** $[0{,}857;\, 0{,}863]$ ($\text{ARR}=+2{,}98\text{ pp}$)<br>**Rollout $\text{RR} = 0{,}7611$** $[0{,}756;\, 0{,}767]$ ($\text{ARR}=+6{,}70\text{ pp}$) | 0,8160 / 0,0338 |
| **7b. PyTorch Parametric G-Computation** | PyTorch 2.x (100x Boot.) | `oracle` | 1-Step Standardized vs. 16-Sem. Rollout | — | — | — | **1-Step $\text{RR} = 0{,}8787$** $[0{,}876;\, 0{,}882]$ ($\text{ARR}=+2{,}64\text{ pp}$)<br>Rollout $\text{RR} = 0{,}4498$ $[0{,}445;\, 0{,}453]$* | 0,8200 / 0,0338 |

*\*Methodische Einordnung der Estimands in Familie 6:*
1. **Exakte Rangfolgen-Rekonstruktion durch Orthogonal DML:** Im `oracle`-Modus rekonstruiert `PyTorch Orthogonal DML` ($\text{HR}_{\text{fach}} = 0{,}9135 < \text{HR}_{\text{uebf}} = 0{,}9410 < \text{HR}_{\text{psych}} = 0{,}9642$, alle drei 95%-Bootstrap-Konfidenzintervalle strikt $< 1{,}0$) exakt die wahre Wirkungsrangfolge der isolierten Universen F ($\text{RR}=0{,}9048$), G ($\text{RR}=0{,}9153$) und H ($\text{RR}=0{,}9392$). Auch `Keras Orthogonal DML` trifft mit den 1-Step Partial RRs (`0,9034`, `0,9432`, `0,9545` in `standard`) die isolierten Ground-Truth-RRs (`0,9048`, `0,9153`, `0,9392`) bis auf die dritte Nachkommastelle!
2. **Warum IPTW (`MSM`) bei `fachlich` an Positivitätsgrenzen stößt ($\text{HR}_{\text{fach}} \approx 1{,}01\text{--}1{,}08$):** Im DGP V4.1 wird fachliches Tutorium fast deterministisch erst nach einer nicht bestandenen Klausur (`versuch >= 2` oder `note > 3.0`) aufgesucht. Dadurch existiert für Studierende im Drittversuch kaum Überlappung (Positivitätsverletzung), die durch Gewichts-Clipping (`[0.05, 20.0]`) im IPTW-Schätzer nicht vollständig kompensiert werden kann, während die doppelt-robuste Robinson-Residualisierung (`Orthogonal DML`) diesen Selektionspfad über das Outcome-Modell $\hat{m}(X)$ neutralisiert.
3. **1-Step Standardized RR vs. 16-Semester Sequential Rollout RR in `G-Computation`:** Im `standard`-Modus trifft der 16-Semester-Rollout (`RR = 0,7611`, `[0,756; 0,767]`) die wahre kontrafaktische 16-Semester-Ground-Truth ($\text{RR}_{\text{true}}(\text{A vs. B}) = 0{,}7868$) bemerkenswert genau, während das 1-Step Step-Standardized RR (`0,8603` in `standard`, `0,8787` in `oracle`) den direkten Ein-Semester-Interventionseffekt ohne dynamische Folge-Semester-Kumulation misst.

---

## 7. Framework-Paritätsanalyse: Keras 3 vs. PyTorch 2.x

Ein wesentliches Qualitätskriterium des V4.2.5-Refactorings war der Nachweis, dass identische Modellkonzepte in **Keras 3 (TensorFlow Backend)** und **PyTorch 2.x** bei einheitlicher Datenbasis (`feature_builder.py`), identischem Student-Level-Split (`seed=42`) und standardisierter Evaluierung (`metrics_logger.py`) konsistente Ergebnisse liefern:

| Modellpaar (Aufgabe) | Keras 3 Modell & Hauptmetrik | PyTorch 2.x Modell & Hauptmetrik | Absolute Differenz $|\Delta|$ | Bewertung der Parität |
| :--- | :--- | :--- | :---: | :--- |
| **Counting-Process Logistic Hazard (`standard`)** | `Extended Logistic Hazard`<br>ROC-AUC: `0,8152`, Brier: `0,0342` | `PyTorchLogisticHazard`<br>ROC-AUC: `0,8160`, Brier: `0,0338` | $\Delta\text{AUC} = 0{,}0008$<br>$\Delta\text{Brier} = 0{,}0004$ | **Exakte numerische Parität** |
| **Counting-Process Logistic Hazard (`oracle`)** | `Extended Logistic Hazard`<br>ROC-AUC: `0,8202`, Brier: `0,0340` | `PyTorchLogisticHazard`<br>ROC-AUC: `0,8200`, Brier: `0,0338` | $\Delta\text{AUC} = 0{,}0002$<br>$\Delta\text{Brier} = 0{,}0002$ | **Exakte numerische Parität** |
| **Counting-Process DeepSurv / CoxPH (`standard`)** | `Extended DeepSurv (+ Breslow)`<br>ROC-AUC: `0,8165`, Brier: `0,0342` | `PyTorchCoxPHExtended (+ Breslow)`<br>ROC-AUC: `0,8160`, Brier: `0,0344` | $\Delta\text{AUC} = 0{,}0005$<br>$\Delta\text{Brier} = 0{,}0002$ | **Exakte numerische Parität** |
| **3D Semester Survival GRU (`standard`)** | `Semester Survival GRU`<br>ROC-AUC: `0,8121`, Brier: `0,0353` | `PyTorchSemesterSurvivalGRU`<br>ROC-AUC: `0,8196`, Brier: `0,0345` | $\Delta\text{AUC} = 0{,}0075$<br>$\Delta\text{Brier} = 0{,}0008$ | **Sehr hohe Parität** (PyTorch leicht vorn durch explizites `pack_padded_sequence`) |
| **3D Semester Causal Transformer (`standard`)** | `Semester Causal Transformer`<br>ROC-AUC: `0,8144`, Brier: `0,0347` | `PyTorchSemesterTransformer`<br>ROC-AUC: `0,8147`, Brier: `0,0347` | $\Delta\text{AUC} = 0{,}0003$<br>$\Delta\text{Brier} = 0{,}0000$ | **Exakte numerische Parität** |
| **3D Exam Survival GRU (`standard`)** | `Exam Survival GRU`<br>ROC-AUC: `0,8791`, Brier: `0,0154` | `PyTorchExamSurvivalGRU`<br>ROC-AUC: `0,8976`, Brier: `0,0150` | $\Delta\text{AUC} = 0{,}0185$<br>$\Delta\text{Brier} = 0{,}0004$ | **Hohe Parität** (`Student ROC-AUC`: `0,7379` vs. `0,7489`) |
| **3D Exam Causal Transformer (`gradeblind`)** | `Deep Exam Transformer Survival`<br>ROC-AUC: `0,8687`, Brier: `0,0156` | `PyTorchCausalExamTransformer`<br>ROC-AUC: `0,8779`, Brier: `0,0153` | $\Delta\text{AUC} = 0{,}0092$<br>$\Delta\text{Brier} = 0{,}0003$ | **Hohe Parität** |
| **3D Exam Sequence Regressor (`gradeblind`, Hurdle Stage 2)** | `Deep Exam Transformer Regressor`<br>$R^2$: `0,7839`, RMSE: `0,2783` | `PyTorchExamTransformerRegressor`<br>$R^2$: `0,8065`, RMSE: `0,2634` | $\Delta R^2 = 0{,}0226$<br>$\Delta\text{RMSE} = 0{,}0149$ | **Hohe Parität** (beide übertreffen Landmark $R^2=0{,}7751$) |
| **Autoregressive Next-Exam Dual-Head GRU (`standard`)** | `Keras Next-Exam GRU`<br>$R^2$: `0,6885`, Pass-AUC: `0,9392`, Fail-PR: `0,7706` | `PyTorchNextExamGRU`<br>$R^2$: `0,7036`, Pass-AUC: `0,9406`, Fail-PR: `0,7760` | $\Delta R^2 = 0{,}0151$<br>$\Delta\text{AUC} = 0{,}0014$ | **Exzellente Parität** auf beiden Köpfen |
| **Autoregressive Next-Exam Dual-Head Transformer (`standard`)** | `Keras Next-Exam Deep Transformer`<br>$R^2$: `0,6913`, Pass-AUC: `0,9380`, Fail-PR: `0,7665` | `PyTorchNextExamTransformer`<br>$R^2$: `0,7050`, Pass-AUC: `0,9409`, Fail-PR: `0,7779` | $\Delta R^2 = 0{,}0137$<br>$\Delta\text{AUC} = 0{,}0029$ | **Exzellente Parität** auf beiden Köpfen |
| **Orthogonal DML Survival (`standard` & `oracle`)** | `Keras Orthogonal DML`<br>Nuisance AUC: `0,8145 / 0,8188`<br>$\text{HR}_{\text{fach}}$: `0,8475 / 0,8503` | `PyTorch Orthogonal DML`<br>Nuisance AUC: `0,8145 / 0,8195`<br>$\text{HR}_{\text{fach}}$: `0,8952 / 0,9135` | $\Delta\text{AUC} = 0{,}0000\text{--}0{,}0007$<br>Beide alle 3 Kanäle $< 1{,}0$ | **Exakte Nuisance-Parität & konsistente Kausalidentifikation** |

---

## 8. Reproduzierbarkeit & Verzeichnis der Benchmark-Artefakte

Sämtliche Metrik-JSONs, Kalibrierungsplots und der vollständige Laufzeit-/RAM-Bericht dieses Benchmarks liegen unter:
- **Benchmark-Runner:** [`src/deepsupport/runners/run_post_refactor_benchmark_s01.py`](../../src/deepsupport/runners/run_post_refactor_benchmark_s01.py)
- **Laufzeit- & Ressourcenprotokoll:** `outputs/S01_baseline_post_refactor/diagnostics/post_refactor_benchmark_report.md` (sowie `.json`)
- **Standardisierte Einzel-Metriken (alle 6 Familien):** `outputs/S01_baseline_post_refactor/metrics/*_metrics.json`
- **Aggregierter Kausal-Benchmark:** [`docs/03_evaluations_and_benchmarks/causal_ground_truth_benchmark.md`](causal_ground_truth_benchmark.md) & `outputs/S01_baseline/metrics/causal_ground_truth_benchmark.json`

```powershell
# Reproduktion des gesamten 6-Familien-Benchmarks (oder einzelner Phasen):
$env:PYTHONPATH = "src"
C:\GitHub_public\.venv\Scripts\python.exe src/deepsupport/runners/run_post_refactor_benchmark_s01.py --phase all
```

---

## Verwandte Dokumente

| Dokument | Thema & Verbindung zu dieser Gegenüberstellung |
| :--- | :--- |
| **[Master-Synopse V4.2](master_synopse_v4_gesamt.md)** | Übergeordnete Gesamtsynopse über alle 15 Sensitivitätsszenarien (S01–S15), HSKL/SoliD-Hintergrund und Apathie-Hypothese (S16) |
| **[Kausaler Ground-Truth Benchmark](causal_ground_truth_benchmark.md)** | Detaillierter mathematischer und empirischer Vergleich der 6 Kausalmodelle (Familie 6) gegen die kontrafaktischen Parallelwelten A–H |
| **[Synopse Heavy Suite (S01, S07, S08)](synopse_heavy_suite_s01_s07_s08.md)** | Verhalten der PyTorch-Panel-, Sequenz- und Kausalmodelle unter halbiertem (S07) und verdoppeltem (S08) exogenem Prüfungsrauschen |
| **[Modellarchitekturen im Detail](../02_architectures_and_models/model_architectures.md)** | Architektonische Spezifikation aller >25 Modelle, Loss-Funktionen und Tensor-Dimensionen |
| **[Kausale Vergleichsanalyse](../04_causal_and_simulation/04_Kausale_Vergleichsanalyse.md)** | Theoretische Herleitung von Robinson-DML, IPTW-MSM und G-Computation unter zeitabhängigem Confounding |
| **[Config-Audit & V5-Roadmap](../04_causal_and_simulation/config_audit_und_v5_roadmap.md)** | Spezifikation des anstehenden Generator-Upgrades auf Version 5 ($\kappa_{\text{Motivation}}=9{,}0$, Zero-Inflated Erwerbstätigkeit, S16 Spätkrisen-Test) |
| **[Refactoring-Plan Evaluation Pipeline](../01_master_plans/refactoring_plan_evaluation_pipeline1.md)** | Spezifikation der 5 OOP-Evaluator-Klassen (`SurvivalEvaluator`, `RegressionEvaluator`, `MulticlassEvaluator`, `DualHeadEvaluator`, `CausalEvaluator`) |
| **[Kritische Projektbewertung](../08_project_evolution/DeepSupport_Kritische_Bewertung.md)** | Methodische Stärken-Schwächen-Analyse und Einordnung der durch das V4.2.5-Refactoring geschlossenen Lücken |
