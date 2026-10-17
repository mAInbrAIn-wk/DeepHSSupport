# Post-Refactor Master Benchmark Report (V4.2.5)

**Aktualisiert am:** 2026-10-17 09:44:26  
**Kumulierte Laufzeit:** 244.40 Minuten

| Schritt | Status | Dauer (s) | RAM Start (MB) | RAM Ende (MB) | RAM Delta (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| F1.1: Landmark Baseline Classifiers (4-Class & Binary | standard) | PASSED | 205.99 | 571.9 | 718.1 | +146.3 |
| F1.2: Landmark Baseline Classifiers (4-Class & Binary | gradeblind) | PASSED | 213.09 | 718.1 | 797.1 | +78.9 |
| F1.3: Landmark Baseline Regressors (Hurdle Stage 2 | standard) | PASSED | 69.99 | 797.1 | 876.0 | +78.9 |
| F1.4: Landmark Baseline Regressors (Hurdle Stage 2 | gradeblind) | PASSED | 94.17 | 876.0 | 788.1 | -87.8 |
| F1.5: Deep Survival Landmark (Logistic Hazard & Breslow DeepSurv | standard) | PASSED | 109.01 | 788.1 | 995.4 | +207.3 |
| F2.1: Extended Cox Counting-Process [standard] | PASSED | 10.82 | 570.7 | 815.0 | +244.3 |
| F2.1: Extended Cox Counting-Process [oracle] | PASSED | 11.93 | 617.9 | 877.1 | +259.1 |
| F2.1: Extended Cox Counting-Process [gradeblind] | PASSED | 9.85 | 659.7 | 859.9 | +200.2 |
| F2.2: Keras Extended DeepSurv & Logistic Hazard [standard] | PASSED | 489.3 | 601.8 | 814.6 | +212.8 |
| F2.2: Keras Extended DeepSurv & Logistic Hazard [oracle] | PASSED | 440.32 | 814.6 | 896.4 | +81.7 |
| F2.3: Keras Dynamic DeepHit 3-State Softmax Competing Risks [standard] | PASSED | 136.96 | 896.4 | 1064.9 | +168.6 |
| F3.1: Keras Semester Survival GRU [standard] | PASSED | 119.29 | 568.7 | 812.1 | +243.4 |
| F3.1: Keras Semester Survival GRU [oracle] | PASSED | 123.49 | 812.1 | 871.4 | +59.3 |
| F3.2: Keras Semester Causal Transformer Survival [standard] | PASSED | 84.84 | 871.4 | 986.8 | +115.4 |
| F3.3: Keras Exam Survival GRU [standard] | PASSED | 205.14 | 986.8 | 1038.7 | +51.9 |
| F3.4: Keras Exam Causal Transformer Survival [standard] | PASSED | 697.57 | 1038.7 | 1296.4 | +257.6 |
| F3.5: Keras Modern Deep Transformer Suite (Sem-Reg, Exam-Reg, Exam-Surv | gradeblind) | PASSED | 1194.05 | 1296.4 | 1635.9 | +339.5 |
| F3.6: PyTorch Sequence Survival & Transformer Suite (Sem-GRU, Sem-Trans, Exam-GRU, Dyn-DeepHit, Exam-Reg, Exam-Surv) | PASSED | 891.7 | 1635.9 | 1750.1 | +114.2 |
| F3.7: PyTorch Dynamic DeepHit (Causal Step-Hazard Fix | standard) | PASSED | 39.39 | 571.0 | 794.6 | +223.6 |
| F4.1: Keras Autoregressive Next-Exam Dual-Head GRU (1/K_i weighted | standard) | PASSED | 373.66 | 762.7 | 1173.4 | +410.7 |
| F4.2: Evaluation Autoregressive Next-Exam Fail PR-AUC | PASSED | 48.44 | 1173.4 | 1239.0 | +65.7 |
| F4.3: Keras Autoregressive Next-Exam Deep Transformer (Sin/Cos PE, 1/K_i weighted | standard) | PASSED | 1880.88 | 1239.0 | 1468.4 | +229.4 |
| F5.1: Landmark Representation Learning (Headless Transformer -> XGBoost 4-Class & Hurdle Reg) | PASSED | 49.72 | 2734.9 | 2804.3 | +69.3 |
| F4.4: PyTorch Autoregressive Next-Exam Dual-Head GRU (1/K_i weighted | standard) | PASSED | 529.42 | 2804.3 | 1078.0 | -1726.2 |
| F4.5: PyTorch Autoregressive Next-Exam Dual-Head Transformer (1/K_i weighted | standard) | PASSED | 2262.78 | 1057.0 | 1459.9 | +402.8 |
| F6.1: Keras Orthogonal DML Survival (5-Fold GroupKFold + Multivariate Robinson) [standard] | PASSED | 45.4 | 569.8 | 715.8 | +145.9 |
| F6.2: Keras Deep Transformer-DML (64-D Causal Embedding + 5-Fold GroupKFold) [standard] | PASSED | 288.45 | 569.2 | 906.7 | +337.5 |
| F6.3: PyTorch Causal Suite (DML, MSM cloglog/logit, G-Computation) [standard] | PASSED | 990.01 | 873.5 | 1006.9 | +133.4 |
| F6.1: Keras Orthogonal DML Survival (5-Fold GroupKFold + Multivariate Robinson) [oracle] | PASSED | 38.0 | 1007.0 | 1039.6 | +32.6 |
| F6.2: Keras Deep Transformer-DML (64-D Causal Embedding + 5-Fold GroupKFold) [oracle] | PASSED | 191.82 | 906.9 | 1051.4 | +144.5 |
| F6.3: PyTorch Causal Suite (DML, MSM cloglog/logit, G-Computation) [oracle] | PASSED | 1289.48 | 1157.9 | 1129.1 | -28.8 |
| F6.4: Unified Ground-Truth Benchmark & KI-Policy-Targeting Uplift Showcase | PASSED | 114.32 | 1051.4 | 1041.6 | -9.7 |
| F2.4: PyTorch Panel Survival Suite (LH, DeepHit, CoxPH, CoxTime, DeepHit-CR) [standard] | PASSED | 731.73 | 569.1 | 1013.2 | +444.1 |
| F2.4: PyTorch Panel Survival Suite (LH, DeepHit, CoxPH, CoxTime, DeepHit-CR) [oracle] | PASSED | 682.75 | 1013.4 | 1009.3 | -4.0 |
