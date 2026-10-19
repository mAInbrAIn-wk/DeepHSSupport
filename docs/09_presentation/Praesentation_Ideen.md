# Was muss ich alles erklären?

## Sammlung von Punkten und Ideen

1. Setting: Synthetischer Datensatz, dynamisches Modell zur Simulation von Prüfungen und Supportteilnehmen bzw. deren Wirkung
2. Verschiedene Fragestellungen:
    - Vorhersage der Dropout Wahrscheinlichkeit oder der (avg) Note pro Semester/Prüfung
    - Identifikation der wichtigsten Prädiktoren, insbesondere: Effektivität des Supports untersuchen
        - Das hat wiederum verschiedene Dimensionen (kpi): Wirkung, Erreichung der Zielgruppe, Timing etc.
3. Welche Probleme treten auf?
    - Problem des Selektion Bias bei der Analyse des Supports (Supportnutzung ist selektiv, was die Wirkungsanalyse verzerren kann - ist umgekehrt aber für die Zielgruppenerreichung sogar wünschenswert)
    - Problem der hidden ground truth variablen als time varying confounder (insb. Motivation, soz. Integration)
    - nicht-lineare Wechselwirkungen der Hintergrrundvariablen ergeben eine komplexe Dynamik, welche nicht trivial zu analysieren ist
4. Welche Analyseansätze gibt es und welche haben sich als geeignet erwiesen?
    - traditionelle multivariaten Ansätze wie Logist. Regression etc.
    - Survival-Analyse nach Kaplan-Meier und Cox-Regressionsmodell -- proportional hazards sind aber ziemlich sicher nicht gegeben
    - Extended Cox mit einer Zeitreihen Analyse
    - Discrete Time Logistic Hazards
    - DeepSurv, Dynamic DeepHit, Cox-PH Neural Net, Recurrent Cox-PH, DeepSurv++
5. Metaanalysen:
    - Welche Arten von Tests sind besonders geeignet?
    - Welche Methoden des Deep Learning haben sich als besonders geeignet erwiesen?

Kriege ich das in 10 Minuten hin? Fokus sollte auf Verständlichkeit liegen, nicht auf den Details der Modelle

## Nun zu den Folien

### Folie 1: Titel

- Titel: "Wirksamkeitsanalyse von Hochschulsupport und Prognose des Studienverlaufs"
- Wilfried Keller
- Deep Learning bei Dr. Bernd Ebenhoch

### Folie 2: Motivation

- Vorgeschichte aus abstact.md

### Folie 3: Zielsetzung

- Zielsetzung aus abstract.md

### Folie 4: Setup (vereinfachte Darstellung)

- Simulation: Dynamische Struktur (z.B. in einer Graphdarstellung)
- Änderungen: Support nicht mehr (fast) zufällig, Zeitkontenmodell

### Folien 5-6: Was ist ein confounder und wie kann man ihn kontrollieren?

- alt vs. neu: confounder statisch und dynamisch
- alt: Supportbesuche wurden nur von der HZB-Note beeinflusst (nicht einmal von dem erwarteten Nutzen)
    - Cox-Regression konnte (mit Kontrollvariable HZB Note!) klare Supportwirkung zeigen
- neu: Supportbesuche wurden zusätzlich von der Fehlversuchen beeinflusst, die aber auch die Motivation senken (dynamischer Confounder)
    - Cox-Regressor zeigt negative Supportwirkung (HR > 1!) -- Das sogenannte Dropout-Paradoxon!
    - Erst die Längsschnitt-Panel Modelle (Extended Cox) und Double Machine Learning (DML) mit Treatment-Orthogonalisierung können diesen Bias erfolgreich auflösen und einen realistischen, positiven Effekt ausweisen (RR < 1).

Dazu vielleicht etwas anschauliches? Eine Grafik?

### Folie 7: Ergebnisse der Survivalanalyse

- Hier zusammentragen und vergleichen
- Besonderheit der Loss-Funktion und der Discrete Time Modelle (BCE-Loss)
- Vergleich mit DeepSurv++ und rekurrente/Transformer Modelle

### Folie 8: Prognose

- Klassifikation vs Regression -> was will ich? Wie kann ich das Problem in beides übersetzen?
- Landmark approach vs. sequenzielle Modelle

### Folie 9: Ergebnisse der Prognose

- Hier zusammentragen und vergleichen

### Folie 10: Fazit

### Folie 11: Ausblick und Limitations