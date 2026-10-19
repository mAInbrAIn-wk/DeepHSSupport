# Fast Core Suite Benchmark Report

**Generiert am:** 2026-09-06 23:22:02  
**Gesamtlaufzeit:** 94.34 Minuten

| Schritt | Status | Dauer (s) | RAM Start (MB) | RAM Ende (MB) | RAM Delta (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Extended Cox [standard] | PASSED | 15.14 | 644.7 | 1066.9 | +422.2 |
| Extended DeepSurv [standard] | FAILED | 196.32 | 887.8 | 1361.8 | +473.9 |
| Recurrent Survival GRU [standard] | PASSED | 300.84 | 1196.5 | 1172.9 | -23.6 |
| Dynamic DeepHit Competing Risks [standard] | FAILED | 198.32 | 1172.9 | 1380.7 | +207.8 |
| Transformer Survival [standard] | PASSED | 104.81 | 1251.9 | 1233.0 | -18.9 |
| Recurrent Exam Survival GRU [standard] | PASSED | 447.68 | 1233.0 | 1446.4 | +213.4 |
| Transformer Exam Survival [standard] | PASSED | 778.09 | 1446.4 | 1894.0 | +447.6 |
| Landmark Baseline Classifiers [standard] | FAILED | 0.0 | 1894.0 | 1894.0 | +0.0 |
| Landmark Regression [standard] | FAILED | 0.0 | 1894.0 | 1894.0 | +0.0 |
| DML Orthogonal Survival [standard] | PASSED | 74.63 | 1894.0 | 2129.0 | +235.0 |
| Transformer DML [standard] | PASSED | 494.16 | 2129.0 | 2108.2 | -20.8 |
| Extended Cox [gradeblind] | PASSED | 15.16 | 2108.2 | 2336.2 | +228.0 |
| Extended DeepSurv [gradeblind] | FAILED | 210.3 | 2200.9 | 2208.7 | +7.8 |
| Recurrent Survival GRU [gradeblind] | PASSED | 309.48 | 2076.6 | 2011.4 | -65.2 |
| Dynamic DeepHit Competing Risks [gradeblind] | FAILED | 238.73 | 2011.4 | 2067.9 | +56.5 |
| Transformer Survival [gradeblind] | PASSED | 123.43 | 2016.0 | 2073.9 | +58.0 |
| Recurrent Exam Survival GRU [gradeblind] | PASSED | 447.99 | 2073.9 | 2232.4 | +158.5 |
| Transformer Exam Survival [gradeblind] | PASSED | 776.36 | 2232.4 | 2404.5 | +172.1 |
| Landmark Baseline Classifiers [gradeblind] | FAILED | 0.0 | 2404.5 | 2404.5 | +0.0 |
| Landmark Regression [gradeblind] | FAILED | 0.0 | 2404.5 | 2404.5 | +0.0 |
| DML Orthogonal Survival [gradeblind] | PASSED | 83.36 | 2404.5 | 2639.6 | +235.1 |
| Transformer DML [gradeblind] | PASSED | 494.51 | 2639.6 | 2518.8 | -120.8 |
| Oracle Models (Lift Analysis) | PASSED | 98.24 | 2518.8 | 2531.9 | +13.1 |
| DSGVO Realistic Models | PASSED | 56.29 | 2531.9 | 2546.1 | +14.2 |
| Deep Survival Landmark (LH & DS) | FAILED | 40.25 | 2546.1 | 2408.1 | -137.9 |
| Strukturelle Mediationsanalyse | PASSED | 8.19 | 2408.1 | 2588.3 | +180.1 |
| Plot Calibration Curves | PASSED | 29.84 | 2587.3 | 2515.6 | -71.7 |
| Counterfactual HR Analyzer (Extended Cox/Panel) | PASSED | 56.79 | 2515.6 | 2900.0 | +384.4 |
| Counterfactual DeepHit Competing Risks | PASSED | 14.83 | 2900.0 | 2618.1 | -281.8 |
| Counterfactual Grade Transformer | PASSED | 0.0 | 2618.1 | 2618.1 | +0.0 |
| Counterfactual Oracle Logistic Hazard | PASSED | 46.87 | 2618.1 | 2739.4 | +121.2 |
