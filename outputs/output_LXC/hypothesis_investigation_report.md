# Untersuchungsbericht: Hypothesenüberprüfung (H1, H2, H3)

---
created: 2026-09-09
mode: full
status: abgeschlossen
tags: [hypothesis-investigation, causal-inference, cox-stratification, dml, dgp-scaling]
---

## 1. Übersicht & Zielsetzung
Überprüfung der empirischen Mechanismen hinter der kausalen Entzerrung im DeepSupport-Projekt gemäß Protocol `.agent/skills/hypothesis-falsifier/SKILL.md`.

## 2. Zusammenfassung der Ergebnisse

| Hypothese | Gegenstand | Prüfmetrik | Referenz / Erwartung | Gemessener Wert | Status |
|:---|:---|:---|:---|:---|:---:|
| **H3** | Nicht-lineare Cox-Dynamik | $HR_{\text{vuln}} \text{ vs. } HR_{\text{all}}$ | $HR_{\text{vuln}} < HR_{\text{all}}$ | $HR_{\text{vuln}} = 0.9927$ (All: 1.0601) | **CONFIRMED** |
| **H2** | Dosis-Halbierung V4.1 (S02) | $HR_{\text{S02}} \text{ vs. } HR_{\text{S01}}$ | $HR_{\text{S02}} > 0.8500$ | $HR_{\text{S02}} = 0.8122$ | **FALSIFIED** |
| **H1** | Dosis-Skalierung V3.6 (5.0x) | $HR_{\text{V3.6(5x)}}$ | $HR < 0.9200$ | $HR = 0.9607$ (Ref: 0.9809) | **AMBIGUOUS** |

## 3. Detailergebnisse
Vollständige Maschinendaten hinterlegt in `hypothesis_investigation_results.json`.
