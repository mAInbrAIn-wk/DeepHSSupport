# Abstract

## Wirksamkeitsanalyse von Hochschulsupport und Prognose des Studienverlaufs

### Vorgeschichte

Da ich in meiner letzten Arbeitsstelle selbst einen hybriden Mathesupport entwickelt habe, treibt mich seitdem die Frage einer datengebtriebenen Evaluierung desselben um. In DE habe ich mir viele Gedanken um ER- bzw. Datenstruktur gemacht (und dies mehr recht als schlecht mit SQL simuliert), in DA konnte ich aufsetzend auf eine dynamisch-stochastische Python-Simulation diese Fragestellung mit einer Survival Analyse und Cos-Regression untersuchen und in Dashboards visualisieren. Die jetzige Arbeit greift dieses Thema erneut auf, und erweitert sie (neben Verbesserungen der Simulation) um Techniken des Maschinellen und Deep Learnings.

### Zielsetzung

Neben dem Anwenden konkreter ML-/DL-Techniken geht es mir gerade um den Metaaspekt der Evaluation dieser Techniken in Bezug auf die Fragestellung Wirksamkeit des Supports sowie der Früherkennung von Studienabbruch, um die Zielgruppenerreichung zu optimieren.

### Verwendete Modelle und Ansätze

Für die Wirksamkeitsanalyse der Supportmaßnahmen kommen zeitvariante Cox-Regressionsmodelle zum Einsatz, die es ermöglichen, dem Problem zeitvarianter Confounder — etwa dem Zusammenhang zwischen verschlechterten Noten, der Inanspruchnahme von Support und dem gleichzeitig steigenden Abbruchrisiko — methodisch zu begegnen; dabei werden aber auch DeepSurv Ansätze ausprobiert. 

Darüber hinaus wird die Prognose des individuellen Studienverlaufs mittels verschiedener Methoden des maschinellen Lernens und Deep Learnings adressiert: Statische Baseline-Modelle auf Basis von Querschnittsdaten nach zwei Semestern (Landmark-Ansatz) werden mit sequenziellen Modellen verglichen, die den Studienverlauf als Zeitreihe auf Semester- und Prüfungsebene abbilden (LSTM, GRU, Causal Transformer). Zur differenzierten Modellierung von Studienabbruch und Studienabschluss als konkurrierende Ereignisse wird zudem ein Dynamic-DeepHit-Modell eingesetzt.

### (Vorläufige) Ergebnisse

Die erweiterte Survivalanalyse löst das Problem der zeitveränderlichen Störfaktoren, an dem die Analyse aus dem DA Kurs gescheitert wäre: Der Wechsel von HR > 1 (statisch) zu HR ≈ 0.37 (zeitveränderlich) ist deutlich und unterstreicht die Wichtigkeit der korrekten Methoden-/Toolauswahl. 
Bereits auf Basis weniger Früherkennungsmerkmale nach zwei Semestern lässt sich eine Klassifikationsgenauigkeit von ca. 79 % für den Studienstatus erreichen, während die sequenziellen Modelle die Abschlussnote mit $R^2 > 0,90$ vorhersagen können.

**Anmerkung:** Zur Codeerstellung und -dokumentation wurde stark auf KI Unterstützung (Gemini 3.1 Pro/3.6 Flash und Claude Opus 4.6) zurückgegriffen, da mir so ein für 2 Tage (trotz der Vorarbeiten) kaum zu erreichender Arbeitsumfang gelungen ist. Für Fehler übernehme ich natürlich die volle Verantwortung!