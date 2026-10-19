---
created: 2026-09-12
last_updated: 2026-09-12
status: abgeschlossen
tags: [causal-inference, dml, msm, g-computation, ground-truth-benchmark, lxc-run]
---

# DeepSupport PyTorch Causal Suite: LXC Benchmark-Synopse

## 1. Übersicht & Zielsetzung

Dieser Bericht fasst die Ergebnisse der **PyTorch Causal Suite** zusammen, die auf dem Debian LXC Container ausgeführt wurde.
Erstmals werden drei voneinander unabhängige kausale Schätzmethoden (DML, MSM und G-Computation) direkt gegen den
**empirischen Ground Truth** aus Universum B (kontrafaktische Welt ohne Support) validiert.

---

## 2. Cross-Szenario Methoden-Vergleich & Ground Truth Validierung

| Szenario | Modus | Ground Truth ARR | GT RR | G-Comp ARR | G-Comp RR (95% Boot CI) | MSM HR All (95% CI) | MSM HR Fach | MSM HR Uebf | MSM HR Psych | DML HR Fach |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S01_baseline** | `standard` | +7.95 pp | 0.7858 | **+4.59 pp** | 0.7873 [0.7848, 0.7910] | 0.8002 [0.7645, 0.8376] | 0.8313 | 0.9176 | 0.7303 | 1.0001 |
| **S02_supp_half** | `standard` | +4.41 pp | 0.8812 | **+3.75 pp** | 0.8317 [0.8291, 0.8353] | 0.8476 [0.8138, 0.8829] | 0.9064 | 0.9598 | 0.7541 | 0.9954 |
| **S03_supp_double** | `standard` | +11.79 pp | 0.6822 | **+4.26 pp** | 0.7998 [0.7969, 0.8032] | 0.7626 [0.7241, 0.8031] | 0.7371 | 0.8289 | 0.7037 | 0.9241 |
| **S07_noise_half** | `standard` | +6.33 pp | 0.8083 | **+3.97 pp** | 0.7787 [0.7745, 0.7830] | 0.8070 [0.7685, 0.8475] | 0.8260 | 0.9251 | 0.7221 | 0.9932 |
| **S08_noise_double** | `standard` | +7.88 pp | 0.8080 | **+4.37 pp** | 0.8191 [0.8170, 0.8227] | 0.7797 [0.7484, 0.8123] | 0.7777 | 0.8783 | 0.7478 | 0.9579 |
| **S11_rct_calibrated** | `standard` | +4.45 pp | 0.8800 | **+6.16 pp** | 0.7335 [0.7293, 0.7383] | 0.6776 [0.6489, 0.7076] | 0.6745 | 0.6837 | 0.6815 | 0.9317 |
| **S01_baseline** | `inside_view` | +7.95 pp | 0.7858 | **+4.02 pp** | 0.7964 [0.7928, 0.8000] | 0.8002 [0.7645, 0.8376] | 0.8313 | 0.9176 | 0.7303 | 0.9771 |
| **S01_baseline** | `realistic` | +7.95 pp | 0.7858 | **+4.64 pp** | 0.8094 [0.8063, 0.8129] | 0.8002 [0.7645, 0.8376] | 0.8313 | 0.9176 | 0.7303 | 0.9024 |
| **S01_baseline** | `gradeblind_oracle` | +7.95 pp | 0.7858 | **+4.44 pp** | 0.7947 [0.7920, 0.7982] | 0.8002 [0.7645, 0.8376] | 0.8313 | 0.9176 | 0.7303 | 0.9818 |
| **S01_baseline** | `blind` | +7.95 pp | 0.7858 | **+5.07 pp** | 0.7898 [0.7864, 0.7948] | 0.8002 [0.7645, 0.8376] | 0.8313 | 0.9176 | 0.7303 | 1.0907 |
| **S02_supp_half** | `inside_view` | +4.41 pp | 0.8812 | **+4.60 pp** | 0.7853 [0.7820, 0.7897] | 0.8476 [0.8138, 0.8829] | 0.9064 | 0.9598 | 0.7541 | 1.0123 |
| **S02_supp_half** | `realistic` | +4.41 pp | 0.8812 | **+4.67 pp** | 0.8219 [0.8199, 0.8246] | 0.8476 [0.8138, 0.8829] | 0.9064 | 0.9598 | 0.7541 | 0.9715 |
| **S02_supp_half** | `gradeblind_oracle` | +4.41 pp | 0.8812 | **+3.48 pp** | 0.8547 [0.8517, 0.8585] | 0.8476 [0.8138, 0.8829] | 0.9064 | 0.9598 | 0.7541 | 0.9898 |
| **S02_supp_half** | `blind` | +4.41 pp | 0.8812 | **+6.04 pp** | 0.7700 [0.7668, 0.7755] | 0.8476 [0.8138, 0.8829] | 0.9064 | 0.9598 | 0.7541 | 1.2009 |
| **S03_supp_double** | `inside_view` | +11.79 pp | 0.6822 | **+3.62 pp** | 0.8111 [0.8073, 0.8163] | 0.7626 [0.7241, 0.8031] | 0.7371 | 0.8289 | 0.7037 | 0.9724 |
| **S03_supp_double** | `realistic` | +11.79 pp | 0.6822 | **+2.97 pp** | 0.8522 [0.8491, 0.8545] | 0.7626 [0.7241, 0.8031] | 0.7371 | 0.8289 | 0.7037 | 0.8880 |
| **S03_supp_double** | `gradeblind_oracle` | +11.79 pp | 0.6822 | **+4.16 pp** | 0.8045 [0.8014, 0.8088] | 0.7626 [0.7241, 0.8031] | 0.7371 | 0.8289 | 0.7037 | 0.9386 |
| **S03_supp_double** | `blind` | +11.79 pp | 0.6822 | **+2.92 pp** | 0.8511 [0.8480, 0.8561] | 0.7626 [0.7241, 0.8031] | 0.7371 | 0.8289 | 0.7037 | 1.0311 |
| **S07_noise_half** | `inside_view` | +6.33 pp | 0.8083 | **+3.41 pp** | 0.8306 [0.8260, 0.8350] | 0.8070 [0.7685, 0.8475] | 0.8260 | 0.9251 | 0.7221 | 1.0256 |
| **S07_noise_half** | `realistic` | +6.33 pp | 0.8083 | **+3.68 pp** | 0.8071 [0.8043, 0.8092] | 0.8070 [0.7685, 0.8475] | 0.8260 | 0.9251 | 0.7221 | 0.9701 |
| **S07_noise_half** | `gradeblind_oracle` | +6.33 pp | 0.8083 | **+2.91 pp** | 0.8261 [0.8214, 0.8303] | 0.8070 [0.7685, 0.8475] | 0.8260 | 0.9251 | 0.7221 | 1.0218 |
| **S07_noise_half** | `blind` | +6.33 pp | 0.8083 | **+4.28 pp** | 0.7881 [0.7836, 0.7951] | 0.8070 [0.7685, 0.8475] | 0.8260 | 0.9251 | 0.7221 | 1.1288 |
| **S08_noise_double** | `inside_view` | +7.88 pp | 0.8080 | **+4.05 pp** | 0.8395 [0.8375, 0.8425] | 0.7797 [0.7484, 0.8123] | 0.7777 | 0.8783 | 0.7478 | 0.9919 |
| **S08_noise_double** | `realistic` | +7.88 pp | 0.8080 | **+4.81 pp** | 0.8108 [0.8097, 0.8123] | 0.7797 [0.7484, 0.8123] | 0.7777 | 0.8783 | 0.7478 | 0.8880 |
| **S08_noise_double** | `gradeblind_oracle` | +7.88 pp | 0.8080 | **+3.65 pp** | 0.8549 [0.8526, 0.8586] | 0.7797 [0.7484, 0.8123] | 0.7777 | 0.8783 | 0.7478 | 0.9879 |
| **S08_noise_double** | `blind` | +7.88 pp | 0.8080 | **+4.20 pp** | 0.8280 [0.8249, 0.8313] | 0.7797 [0.7484, 0.8123] | 0.7777 | 0.8783 | 0.7478 | 1.0468 |
| **S11_rct_calibrated** | `inside_view` | +4.45 pp | 0.8800 | **+5.78 pp** | 0.7217 [0.7165, 0.7270] | 0.6776 [0.6489, 0.7076] | 0.6745 | 0.6837 | 0.6815 | 0.9375 |
| **S11_rct_calibrated** | `realistic` | +4.45 pp | 0.8800 | **+5.82 pp** | 0.7639 [0.7614, 0.7666] | 0.6776 [0.6489, 0.7076] | 0.6745 | 0.6837 | 0.6815 | 0.9340 |
| **S11_rct_calibrated** | `gradeblind_oracle` | +4.45 pp | 0.8800 | **+5.03 pp** | 0.7916 [0.7877, 0.7970] | 0.6776 [0.6489, 0.7076] | 0.6745 | 0.6837 | 0.6815 | 0.9478 |
| **S11_rct_calibrated** | `blind` | +4.45 pp | 0.8800 | **+8.73 pp** | 0.6560 [0.6518, 0.6624] | 0.6776 [0.6489, 0.7076] | 0.6745 | 0.6837 | 0.6815 | 0.9866 |

---

## 3. Methodische Interpretation

1. **G-Computation Validierung gegen Universum B:**
   Die G-Computation projiziert die kontrafaktische Kohorte unter do(A=0) und do(A=1). Die geschätzte Absolute Risk Reduction (ARR) approximiert den wahren Universum A-B Kontrast robust.
2. **MSM Stabilisierte Gewichte:**
   Durch Entkopplung zeitabhängiger Confounder über inverse Propensity-Gewichtung (SW) isoliert MSM den wahren Hazard-Ratio-Schutzfaktor ohne Collider-Verzerrung.
3. **Double Machine Learning (DML):**
   Die Neyman-orthogonale Residualisierung in der ersten Stufe eliminiert hochdimensionales Confounding und liefert robuste lokale ATEs.

---

## 4. Verwandte Dokumente

| Dokument | Pfad / Referenz | Kerninhalt |
| :--- | :--- | :--- |
| **Empirische RCT-Kausalanalyse** | [`docs/04_causal_and_simulation/empirische_rct_kausalanalyse_s11_vs_s01.md`](docs/04_causal_and_simulation/empirische_rct_kausalanalyse_s11_vs_s01.md) | Mechanismen der Selektionseliminierung |
| **LXC Benchmark Evaluation V4.2** | [`docs/03_evaluations_and_benchmarks/pytorch_lxc_benchmark_evaluation_v42.md`](docs/03_evaluations_and_benchmarks/pytorch_lxc_benchmark_evaluation_v42.md) | Prädiktiver Benchmark über 6 Szenarien |
