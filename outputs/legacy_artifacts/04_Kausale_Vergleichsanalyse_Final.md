# Quantitative Kausalanalyse & Bias-Auswertung (V3.6 Baseline)

> **Methodischer Kontext:** Dieses Dokument stellt empirische Belege aus den archivierten V3.6 Läufen bereit. Sämtliche Werte stammen aus verifizierten JSON-Metrikdateien und CSV-Beständen im Verzeichnis `archive/legacy_outputs/`.

## 1. Kausale Schätzer auf Dropout (Relative Risiken / Hazard Ratios)

Vergleich der relativen Risiken bzw. Hazard Ratios (Werte < 1.0 signalisieren Risikosenkung = Schutzwirkung). Alle Werte sind aus realen Modell-Output-Dateien ausgelesen:

| Modell / Methode | Exakter Dateipfad der Metrik | Fachlich | Überfachlich | Psychosozial |
| :--- | :--- | :---: | :---: | :---: |
| **Ground Truth (Parallelwelten A vs C/D/E)** | `output_dl/metrics/true_macro_effects_v3.json` | **0.9326** | **0.9194** | **0.9448** |
| **Extended Cox Panel (TVC)** | `output_dl/metrics/extended_cox_panel_metrics.json` | 0.9234 | 0.9648 | 0.9005 |
| **DML Orthogonal Survival (Panel)** | `output_dl/metrics/dml_orthogonal_survival_metrics.json` | 0.9863 | 0.9977 | 0.9941 |
| **Deep Transformer DML (Sequenz)** | `output_dl/analysis/deep_transformer_dml_results.json` | 1.0172 | 0.9957 | 0.9569 |
| **Oracle DeepSurv (Latente Variablen)** | `src/output_dl/metrics/counterfactual_oracle_deepsurv_metrics_metrics.json` | 0.9933 | 0.9897 | 0.9892 |
| **Oracle Logistic Hazard (Latente Variablen)** | `src/output_dl/metrics/counterfactual_oracle_logistic_hazard_metrics_metrics.json` | 0.9897 | 0.9900 | 0.9870 |

## 2. Der kausale "Notenboost" (Abschlussnoten der Absolventen)

Da der primäre Wirkungskanal des fachlichen Supports über die Prüfungsleistung läuft, muss die Notenwirkung isoliert betrachtet werden. Die Daten stammen direkt aus den CSV-Faktentabellen der jeweiligen Universen (nur erfolgreiche Absolventen mit `abschlussnote`):

| Universum / Kohorte | Exakter Datenpfad | Ø Abschlussnote (GPA) | Median GPA | Erfolgsquote | Kausaler Effekt vs. Univ A |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Univ A (Full Support)** | `archive/legacy_outputs/output_dl/abschluesse.csv` | **2.0277** | 2.03 | 69.2% | Referenz (0.0000) |
| **Univ B (No Support)** | `archive/legacy_outputs/output_dl/universe_B/abschluesse.csv` | **2.1629** | 2.19 | 61.0% | +0.1353 Pkt. |
| **Univ C (No Fachlich)** | `archive/legacy_outputs/output_dl/universe_C/abschluesse.csv` | **2.1177** | 2.12 | 67.0% | +0.0901 Pkt. |
| **Univ D (No Ueberfachlich)** | `archive/legacy_outputs/output_dl/universe_D/abschluesse.csv` | **2.0492** | 2.06 | 66.5% | +0.0215 Pkt. |
| **Univ E (No Psychosozial)** | `archive/legacy_outputs/output_dl/universe_E/abschluesse.csv` | **2.0685** | 2.09 | 67.4% | +0.0409 Pkt. |

### 2.1 Confounding by Indication bei den Noten (innerhalb Universum A)
- **Support-Nutzer (fachlich):** Ø Abschlussnote = **2.0965** (N = 23648)
- **Nicht-Nutzer (fachlich):** Ø Abschlussnote = **1.8788** (N = 10944)
- **Naiver Unterschied (Nutzer - Nicht-Nutzer):** **+0.2177 Notenpunkte** *(Nutzer schneiden scheinbar schlechter ab!)*
- **Tatsächlicher kausaler Ground-Truth-Effekt (Univ A vs. C):** **-0.0901 Notenpunkte** *(In Wahrheit verbessert fachlicher Support die Gesamtnote um ~0.09 Punkte).*

> [!IMPORTANT]
> **Doppeltes Paradoxon:** Auf Beobachtungsebene in Univ A erzielen Fachsupport-Nutzer schlechtere Noten (+0.2177 schlechter), weil leistungsschwache Studierende selektiv Support nachfragen (*Confounding by Indication*). Vergleicht man jedoch Univ A mit Univ C (wo fachlicher Support komplett gesperrt ist), zeigt sich der kausale Effekt: Ohne den Support verschlechtert sich der Gesamtschnitt der Absolventen von 2.0277 auf 2.1177 (+0.0901 Notenpunkte Verlust).

## 3. Strukturelle Mediation (Imai / Pearl)

Aus der Datei `src/output_dl_seed99999/metrics/oracle_mediation_analysis_metrics.json` lässt sich die Trennung von direktem und mediiertem Effekt (Odds Ratios) ablesen:

| Supportart | Konfiguration | Total OR | Direct OR | Mediated OR | Interpretation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Fachlich** | Realistic | 0.9710 | 0.9773 | 0.9936 | Partiell protektiv; Mediation über Prüfungsergebnisse. |
| **Fachlich** | Oracle Both | 0.9338 | 0.9465 | 0.9866 | Unter voller Kontrolle sinkt die Total OR auf 0.9338. |
| **Überfachlich** | Realistic | 1.1247 | 1.0631 | 1.0580 | **Scheinbar schädlich (OR > 1)** durch unkontrolliertes Motivations-Confounding. |
| **Überfachlich** | Oracle Confounder | 1.0095 | 1.0029 | 1.0065 | Mit latenter Motivation bricht der Scheineffekt in sich zusammen (OR ~ 1.00). |
| **Psychosozial** | Realistic | 0.9849 | 0.9637 | 1.0220 | Direkter Schutzeffekt bereits in der realistischen Variante sichtbar. |
| **Psychosozial** | Oracle Confounder | 0.9346 | 0.9376 | 0.9968 | Mit voller Kontrolle verstärkt sich die Schutzwirkung auf Total OR = 0.9346. |

## 4. Fazit und Synthese der Feedback-Schleifen
1. **Fachlicher Support:** Wirkt vor allem auf die Prüfungsebene (Notenboost von -0.09 GPA). Sequenzmodelle wie der Transformer DML (RR = 1.0172) tendieren zur Überdämpfung, weil sie die Notenverbesserung als Prädiktor aufsaugen und dem Support-Flag wenig Restvarianz belassen.
2. **Überfachlicher Support:** Schafft im realistischen Beobachtungsraum ein massives Confounding (Total OR = 1.1247), da Studierende mit abstürzender Motivation den Support aufsuchen. Erst wenn die latente Variable `hidden_motivation` dem Modell beigegeben wird, lösen sich die Scheinkorrelationen auf (Oracle DeepSurv HR = 0.9897, Oracle Logistic Hazard RR = 0.9899).
3. **Psychosozialer Support:** Ist der stabilste Schützer quer über alle Modelle (Ground Truth RR = 0.9448, Cox HR = 0.9005, Deep Transformer RR = 0.9569, Oracle Logistic Hazard RR = 0.9870), da Kriseninterventionen in der Datengenerierung exogener auftreten.