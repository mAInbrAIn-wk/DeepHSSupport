# Post-Refactor Master Benchmark Report (V4.2.5)

**Aktualisiert am:** 2026-10-18 03:04:07  
**Kumulierte Laufzeit:** 220.28 Minuten

| Schritt | Status | Dauer (s) | RAM Start (MB) | RAM Ende (MB) | RAM Delta (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| F1.1: Landmark Baseline Classifiers (4-Class & Binary | standard) | PASSED | 334.18 | 568.5 | 717.5 | +148.9 |
| F1.2: Landmark Baseline Classifiers (4-Class & Binary | gradeblind) | PASSED | 348.32 | 717.5 | 797.6 | +80.1 |
| F1.3: Landmark Baseline Regressors (Hurdle Stage 2 | standard) | PASSED | 101.34 | 797.6 | 864.4 | +66.8 |
| F1.4: Landmark Baseline Regressors (Hurdle Stage 2 | gradeblind) | PASSED | 116.44 | 864.4 | 799.2 | -65.3 |
| F1.5: Deep Survival Landmark (Logistic Hazard & Breslow DeepSurv | standard) | PASSED | 190.66 | 799.2 | 992.6 | +193.4 |
| F2.1: Extended Cox Counting-Process [standard] | PASSED | 12.63 | 992.6 | 1219.5 | +227.0 |
| F2.1: Extended Cox Counting-Process [oracle] | PASSED | 15.0 | 1033.5 | 1280.0 | +246.5 |
| F2.1: Extended Cox Counting-Process [gradeblind] | PASSED | 12.61 | 1075.4 | 1260.9 | +185.6 |
| F2.2: Keras Extended DeepSurv & Logistic Hazard [standard] | PASSED | 786.94 | 1079.5 | 995.9 | -83.6 |
| F2.2: Keras Extended DeepSurv & Logistic Hazard [oracle] | PASSED | 430.51 | 995.9 | 1099.2 | +103.3 |
| F2.3: Keras Dynamic DeepHit 3-State Softmax Competing Risks [standard] | PASSED | 96.65 | 1099.2 | 1062.9 | -36.2 |
| F2.4: PyTorch Panel Survival Suite (LH, DeepHit, CoxPH, CoxTime, DeepHit-CR) [standard] | PASSED | 514.83 | 1062.9 | 1301.5 | +238.5 |
| F2.4: PyTorch Panel Survival Suite (LH, DeepHit, CoxPH, CoxTime, DeepHit-CR) [oracle] | PASSED | 518.28 | 1301.6 | 1326.4 | +24.8 |
| F3.1: Keras Semester Survival GRU [standard] | PASSED | 92.26 | 1326.4 | 1424.6 | +98.2 |
| F3.1: Keras Semester Survival GRU [oracle] | PASSED | 90.21 | 1424.6 | 1476.5 | +51.8 |
| F3.2: Keras Semester Causal Transformer Survival [standard] | PASSED | 61.46 | 1476.5 | 1536.5 | +60.0 |
| F3.3: Keras Exam Survival GRU [standard] | PASSED | 156.74 | 1536.5 | 1715.4 | +178.9 |
| F3.4: Keras Exam Causal Transformer Survival [standard] | PASSED | 417.76 | 1715.4 | 1863.8 | +148.4 |
| F3.5: Keras Modern Deep Transformer Suite (Sem-Reg, Exam-Reg, Exam-Surv | gradeblind) | PASSED | 892.93 | 1863.8 | 2257.2 | +393.4 |
| F3.6: PyTorch Sequence Survival & Transformer Suite (Sem-GRU, Sem-Trans, Exam-GRU, Dyn-DeepHit, Exam-Reg, Exam-Surv) | PASSED | 751.77 | 2257.2 | 2200.3 | -56.9 |
| F3.7: PyTorch Dynamic DeepHit (Causal Step-Hazard Fix | standard) | PASSED | 37.29 | 2200.3 | 2229.4 | +29.1 |
| F4.1: Keras Autoregressive Next-Exam Dual-Head GRU (1/K_i weighted | standard) | PASSED | 453.38 | 2213.3 | 2407.3 | +193.9 |
| F4.2: Evaluation Autoregressive Next-Exam Fail PR-AUC | PASSED | 52.48 | 2407.3 | 2463.4 | +56.1 |
| F4.3: Keras Autoregressive Next-Exam Deep Transformer (Sin/Cos PE, 1/K_i weighted | standard) | PASSED | 2063.42 | 2463.4 | 2480.9 | +17.5 |
| F5.1: Landmark Representation Learning (Headless Transformer -> XGBoost 4-Class & Hurdle Reg) | PASSED | 40.45 | 2480.9 | 2524.1 | +43.2 |
| F4.4: PyTorch Autoregressive Next-Exam Dual-Head GRU (1/K_i weighted | standard) | PASSED | 370.49 | 2524.1 | 2556.7 | +32.6 |
| F4.5: PyTorch Autoregressive Next-Exam Dual-Head Transformer (1/K_i weighted | standard) | PASSED | 1837.71 | 2548.7 | 2764.1 | +215.4 |
| F6.1: Keras Orthogonal DML Survival (5-Fold GroupKFold + Multivariate Robinson) [standard] | PASSED | 39.54 | 2749.1 | 2718.8 | -30.3 |
| F6.2: Keras Deep Transformer-DML (64-D Causal Embedding + 5-Fold GroupKFold) [standard] | PASSED | 196.94 | 2718.8 | 2773.0 | +54.2 |
| F6.3: PyTorch Causal Suite (DML, MSM cloglog/logit, G-Computation) [standard] | PASSED | 934.92 | 2773.0 | 2745.4 | -27.6 |
| F6.1: Keras Orthogonal DML Survival (5-Fold GroupKFold + Multivariate Robinson) [oracle] | PASSED | 39.53 | 2745.4 | 2770.7 | +25.3 |
| F6.2: Keras Deep Transformer-DML (64-D Causal Embedding + 5-Fold GroupKFold) [oracle] | PASSED | 193.01 | 2770.7 | 2827.4 | +56.7 |
| F6.3: PyTorch Causal Suite (DML, MSM cloglog/logit, G-Computation) [oracle] | PASSED | 944.57 | 2827.4 | 2764.0 | -63.4 |
| F6.4: Unified Ground-Truth Benchmark & KI-Policy-Targeting Uplift Showcase | PASSED | 71.5 | 2764.0 | 2777.2 | +13.2 |
