---
created: 2026-09-12
last_updated: 2026-09-12
status: abgeschlossen
tags: [ablation-study, keras-vs-pytorch, architecture-hypotheses, state-decay, pre-ln, logits-loss, adamw]
---

# Synopse: Faktorielle $2^4$-Ablationsstudie (Keras vs. PyTorch)

## 1. Übersicht & Versuchsmatrix

| Run-ID | Framework | State Agg | Normalisierung | Loss | Optimizer | Noten $R^2$ | Noten RMSE | Bestehen ROC-AUC | Nichtbest. PR-AUC ($y=0$) | Brier Score | Zeit (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **R0** | `keras` | `diffusion` | `post_ln` | `sigmoid` | `adam_fix` | **0.6838** | 0.7680 | 0.9421 | 0.7816 | 0.0677 | 471.7 |
| **R1** | `keras` | `gather` | `post_ln` | `sigmoid` | `adam_fix` | **0.6771** | 0.7760 | 0.9420 | 0.7806 | 0.0684 | 550.5 |
| **R2** | `keras` | `gather` | `pre_ln` | `sigmoid` | `adam_fix` | **0.6765** | 0.7767 | 0.9409 | 0.7765 | 0.0735 | 552.8 |
| **R3** | `keras` | `gather` | `pre_ln` | `logits` | `adam_fix` | **0.6554** | 0.8016 | 0.9401 | 0.7761 | 0.0733 | 586.7 |
| **R4** | `keras` | `gather` | `pre_ln` | `logits` | `adamw_cosine` | **0.6705** | 0.7839 | 0.9404 | 0.7754 | 0.0693 | 482.8 |
| **R5** | `pytorch` | `gather` | `pre_ln` | `logits` | `adamw_cosine` | **0.6664** | 0.7887 | 0.9425 | 0.7822 | 0.0677 | 541.1 |
| **R6** | `pytorch` | `diffusion` | `pre_ln` | `logits` | `adamw_cosine` | **0.6436** | 0.8153 | 0.9417 | 0.7799 | 0.0675 | 535.9 |
| **R7** | `pytorch` | `gather` | `post_ln` | `logits` | `adamw_cosine` | **0.7117** | 0.7332 | 0.9433 | 0.7870 | 0.0666 | 554.1 |
| **R8** | `pytorch` | `gather` | `pre_ln` | `logits` | `adam_fix` | **0.6526** | 0.8049 | 0.9420 | 0.7814 | 0.0680 | 535.3 |

---

## 2. Quantitative Hypothesen-Prüfung

### Hypothese 1 (State Gathering vs. Padding Diffusion):
- **Keras Gewinn (R1 - R0):** $\Delta R^2 = -0.0066$
- **PyTorch Verlust durch Downgrade (R5 - R6):** $\Delta R^2 = +0.0228$
- **Befund:** Falsifiziert (Schwellenwert $\Delta R^2 \ge 0.05$).

### Hypothese 2 (Pre-LayerNorm vs. Post-LayerNorm):
- **Keras Pre-LN Effekt (R2 - R1):** $\Delta R^2 = -0.0006$
- **PyTorch Pre-LN Effekt (R5 - R7):** $\Delta R^2 = -0.0453$
- **Befund:** Bestätigt.

### Hypothese 3 (Logits Loss Numerik):
- **Brier Score Reduktion (R2 -> R3):** +0.36 %
- **Befund:** Falsifiziert.

### Hypothese 4 (AdamW + Cosine Decay vs. Standard Adam):
- **Keras Optimierer-Effekt (R4 - R3):** $\Delta R^2 = +0.0151$
- **PyTorch Optimierer-Effekt (R5 - R8):** $\Delta R^2 = +0.0138$

---

## 3. Verwandte Dokumente

| Dokument | Pfad / Referenz | Kerninhalt |
| :--- | :--- | :--- |
| **Ablations-Masterplan** | [`../01_master_plans/ablationsplan_keras_vs_pytorch_architekturhypothesen.md`](../01_master_plans/ablationsplan_keras_vs_pytorch_architekturhypothesen.md) | Theoretische Herleitung & Hypothesen |
| **LXC Benchmark Evaluation V4.2** | [`../03_evaluations_and_benchmarks/pytorch_lxc_benchmark_evaluation_v42.md`](../03_evaluations_and_benchmarks/pytorch_lxc_benchmark_evaluation_v42.md) | Ausgangsbefund des Keras-vs-PyTorch-Vergleichs |
