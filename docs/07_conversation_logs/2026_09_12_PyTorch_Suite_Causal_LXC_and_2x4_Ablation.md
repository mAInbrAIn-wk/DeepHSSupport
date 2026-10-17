---
created: 2026-09-12
last_updated: 2026-10-17
status: abgeschlossen
tags: [protokoll, pytorch, pycox, causal-ml, lxc, ablation, erd, v4.2]
---

# Protokoll: PyTorch/PyCox Modeling Suite, LXC Multi-Mode Causal Benchmark, $2^4$-Architektur-Ablation & DGP-Audit

## 📝 Entwicklungs-Historie & User-Annotationen

**Zeitraum:** 08. September – 12. September 2026  
**Kontext:** Systematische Hypothesen-Falsifikation (V3.6 vs. V4.1 Varianzkollaps), 11-Tabellen-ERD & visuelle EDA, Modernisierung der Deep-Transformer-Suite, vollständiger Aufbau der parallelen PyTorch 2.x / PyCox Suite inklusive Kausalmodellen (DML, MSM, G-Computation), Ausführung auf dem Debian-LXC-Cluster sowie $2^4$-faktorielle Architektur-Ablation zwischen Keras 3 und PyTorch 2.x.

### Auslösende Leitfragen & User-Impulse (Zusammenfassung):
> *"Warum unterscheiden sich die Deconfounding-Raten zwischen V3.6 und V4.1 so stark? Liegt es an der Apathie-Dämpfung oder an den Startverteilungen?"*
>
> *"Kannst Du einen vollständigen Audit von `CONFIG` und `engine.py` machen – welche Parameter sind Zombies, welche sind heimliche `.get()`-Defaults, und welche Werte sind als Magic Numbers hartcodiert?"*
>
> *"Lass uns eine parallele PyTorch / PyCox Suite aufbauen, die wir als Headless-Turnkey-Runner auf dem Debian LXC-Container (ThinkCentre) über die Kernszenarien und Kausal-Modi laufen lassen können – inklusive einer sauberen $2^4$-Ablationsstudie, warum PyTorch und Keras vorher bei Sequenzmodellen auseinanderliefen."*

---

## 1. Übersicht der Meilensteine

### A. Hypothesen-Falsifikation & Aufdeckung des V4.1-Varianzkollapses (`2026-09-08 / 09`)
- **Skill `hypothesis-falsifier` (`.agent/skills/hypothesis-falsifier/SKILL.md`):** Einführung eines strikten 5-Stufen-Protokolls gegen narrative Post-hoc-Spekulationen.
- **Szenario S16 (`disable_apathy_dampening`) & Verteilungsanalyse (`N=50.000`):**
  - Empirische Falsifikation der Apathie-Hypothese für Semester 1 ($\Delta = 0{,}0000$).
  - Aufdeckung der wahren Ursache in [systematische_verteilungsanalyse_v36_vs_v41.md](../04_causal_and_simulation/systematische_verteilungsanalyse_v36_vs_v41.md): Während `hzb_note` und `alter` bei der Umstellung auf Beta-Verteilungen in V4.1 ihre Varianz zu $100\,\%$ behielten, halbierte $\kappa = 20{,}0$ die Standardabweichung von `motivation_initial` ($\sigma: 0{,}239 \to 0{,}122$) und `soziale_integration_initial` ($\sigma: 0{,}238 \to 0{,}116$).
- **Technischer Config-Audit:** Vollständige Dokumentation aller 3 Zombie-Parameter, 5 heimlichen `.get()`-Defaults und >25 Magic Numbers sowie Ausarbeitung der empirischen V5-Kalibrierung in [config_audit_und_v5_roadmap.md](../04_causal_and_simulation/config_audit_und_v5_roadmap.md).

### B. 11-Tabellen-ERD, Sunburst-Visualisierungen & Survival-Mathematik (`2026-09-09`)
- **Datenarchitektur:** Vollständiges 11-Tabellen Mermaid-ERD und Dekonstruktion des Overload-Zeitmangels in [datenarchitektur_und_eda_v4.md](../04_causal_and_simulation/datenarchitektur_und_eda_v4.md).
- **Visuelle Exploration:** Wiederherstellung der Sunburst-I/II-Diagramme und 6 Publikationsgrafiken in [visuelle_datenexploration_v4.md](../04_causal_and_simulation/visuelle_datenexploration_v4.md).
- **Biostatistische Fundierung:** Formale Herleitung der Greenwood-Varianz, Erklärung des `SurvfuncRight`-Randabsturzes bei $n_K = d_K$ und Competing-Risks-Beweis in [grundlagen_survival_analyse_und_zensierung.md](../04_causal_and_simulation/grundlagen_survival_analyse_und_zensierung.md).

### C. Deep Transformer Modernisierung & Kausale Mediation (`2026-09-10`)
- **`src/deep_transformer_regression.py`:** Schlanker Backbone ($d=64$, 4 Heads, 2 Blocks, `SinCosPositionalEncoding`, L2-Regularisierung). 9-Run-Regularisierungsbenchmark belegt $3\times$ schnellere Konvergenz ohne Overfitting.
- **Kausale Mediation (Imai/Pearl):** 4-Stufen-Prüfplan in [kausale_mediationsanalyse_v42.md](../04_causal_and_simulation/kausale_mediationsanalyse_v42.md) (Selektions-Audit $d = -0{,}945$, Oracle-Entzauberung $OR \le 0{,}999$).

### D. PyTorch 2.x & PyCox Modeling Suite + LXC Multi-Scenario Benchmarks (`2026-09-11`)
- **Portierung von 4 prädiktiven & 3 kausalen Modellfamilien (`src/deepsupport/models/torch/`, `src/deepsupport/causal/`):**
  - Panel-Survival (`LogisticHazard`, `DeepHit`, `CoxPH`, `CoxTime`, `DeepHitCompetingRisks`), Exam-Transformer (`ExamTransformerRegressor`, `CausalExamTransformerSurvival`), Dual-Head Autoregressoren (`NextExamGRU`, `NextExamTransformer`), Sequentielle Verlaufsmodelle (`SemesterGRU`, `SemesterTransformer`).
  - Kausale PyTorch-Suite: `PyTorchDMLSurvival`, `PyTorchMSM`, `PyTorchGComputation`.
- **LXC Cluster-Läufe & Berichte:**
  - Prädiktiver 6-Szenarien-Benchmark (`S01`, `S02`, `S03`, `S07`, `S08`, `S11`) in [pytorch_lxc_benchmark_evaluation_v42.md](../03_evaluations_and_benchmarks/pytorch_lxc_benchmark_evaluation_v42.md).
  - Multi-Mode Kausal-Benchmark (`inside_view`, `realistic`, `gradeblind_oracle`, `blind`) in [pytorch_lxc_causal_multi_mode_evaluation_v42.md](../03_evaluations_and_benchmarks/pytorch_lxc_causal_multi_mode_evaluation_v42.md).
- **Docker & Proxmox Kapselung:** Bereitstellung von `Dockerfile`, `docker-compose.yml` und [docker_proxmox_deployment_guide.md](../06_misc/docker_proxmox_deployment_guide.md).

### E. $2^4$-Faktorielle Architektur-Ablation Keras 3 vs. PyTorch 2.x (`2026-09-12`)
- Systematische Prüfung der 4 Architekturfaktoren (Recurrent Dropout/State Decay, Pre-LayerNorm vs. Post-LayerNorm, Logits-BCE vs. Sigmoid-BCE, AdamW Weight Decay) über 16 Modellkonfigurationen in [ablationsstudie_2x4_keras_vs_pytorch_ergebnisse.md](../03_evaluations_and_benchmarks/ablationsstudie_2x4_keras_vs_pytorch_ergebnisse.md).

---

## Verwandte Dokumente

| Dokument | Bezug |
|:---|:---|
| [Config-Audit & V5-Roadmap](../04_causal_and_simulation/config_audit_und_v5_roadmap.md) | Vollständiger Parameter-Audit und V5-Rekalibrierungsplan |
| [Systematische Verteilungsanalyse V3.6 vs. V4.1](../04_causal_and_simulation/systematische_verteilungsanalyse_v36_vs_v41.md) | Nachweis des Motivations-Varianzkollapses ($\kappa=20.0$) |
| [PyTorch LXC Benchmark Evaluation](../03_evaluations_and_benchmarks/pytorch_lxc_benchmark_evaluation_v42.md) | Ergebnisse der PyTorch/PyCox Suite über 6 Szenarien |
| [PyTorch LXC Multi-Mode Causal Evaluation](../03_evaluations_and_benchmarks/pytorch_lxc_causal_multi_mode_evaluation_v42.md) | Kausale Deconfounding-Analyse über 4 Informationsmodi |
| [$2^4$-Ablationsstudie Keras vs. PyTorch](../03_evaluations_and_benchmarks/ablationsstudie_2x4_keras_vs_pytorch_ergebnisse.md) | Empirische Dekonstruktion der Architekturunterschiede |
| [Protokoll Post-Refactor V4.2.5 & DGP V5.0](2026_10_17_Post_Refactor_Benchmark_V425_and_DGP_V5.md) | Direkte Fortsetzung: Numerische Bugfixes, V4.2.5 Gesamtvergleich & V5.0 Rollout |
