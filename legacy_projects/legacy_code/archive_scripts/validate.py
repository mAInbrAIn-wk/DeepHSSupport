"""
Validierung & Dokumentation für HSDS Datensatz (DL-Edition)
===========================================================
Führt automatisierte Konsistenz-Checks durch und generiert eine
ausführliche DATENSATZ_DOKU.md im Output-Verzeichnis.
"""

from pathlib import Path
import pandas as pd
import json

def _df_to_markdown(df: pd.DataFrame) -> str:
    """Konvertiert ein Pandas DataFrame manuell in Markdown-Tabellenformat ohne externe Abhängigkeit (wie tabulate)."""
    cols = [str(col) for col in df.columns]
    index_name = str(df.index.name) if df.index.name else "Studiengang"
    
    header = "| " + index_name + " | " + " | ".join(cols) + " |"
    sep = "| " + "--- | " + " | ".join(["---"] * len(cols)) + " |"
    
    rows = [header, sep]
    for idx, row in df.iterrows():
        r_vals = [str(idx)] + [str(val) for val in row.values]
        rows.append("| " + " | ".join(r_vals) + " |")
        
    return "\n".join(rows)

def validiere_und_dokumentiere(output_dir: Path):
    print("Starte Validierung und Dokumentations-Generierung ...")
    
    # 1. CSV-Dateien laden
    studierende_df = pd.read_csv(output_dir / 'studierende.csv')
    studiengaenge_df = pd.read_csv(output_dir / 'studiengaenge.csv')
    module_df = pd.read_csv(output_dir / 'module.csv')
    pruefungen_df = pd.read_csv(output_dir / 'pruefungen.csv')
    semester_df = pd.read_csv(output_dir / 'semester.csv')
    einschreibungen_df = pd.read_csv(output_dir / 'einschreibungen.csv')
    support_angebote_df = pd.read_csv(output_dir / 'support_angebote.csv')
    support_teilnahmen_df = pd.read_csv(output_dir / 'support_teilnahmen.csv')
    abschluesse_df = pd.read_csv(output_dir / 'abschluesse.csv')
    
    # Anreichern von Studiengang-Namen
    if 'stg_name' not in abschluesse_df.columns:
        abschluesse_df = abschluesse_df.merge(
            studierende_df[['studierenden_id', 'studiengang_id']], on='studierenden_id', how='left'
        ).merge(
            studiengaenge_df[['studiengang_id', 'name']].rename(columns={'name': 'stg_name'}), on='studiengang_id', how='left'
        )

    config_pfad = output_dir / 'config_used.json'
    config_data = {}
    if config_pfad.exists():
        with open(config_pfad, 'r') as f:
            config_data = json.load(f)
            
    # 2. Statistische Kennzahlen berechnen
    n_studis = len(studierende_df)
    n_abschluss = (abschluesse_df['status'] == 'abgeschlossen').sum()
    n_abbruch = (abschluesse_df['status'] == 'abgebrochen').sum()
    n_exmat = (abschluesse_df['status'] == 'exmatrikuliert').sum()
    n_zeit = (abschluesse_df['status'] == 'zeitueberschreitung').sum()
    
    abschluss_quote = n_abschluss / n_studis if n_studis > 0 else 0.0
    abbruch_quote = (n_abbruch + n_exmat + n_zeit) / n_studis if n_studis > 0 else 0.0
    
    bestandene_abschluesse = abschluesse_df[abschluesse_df['status'] == 'abgeschlossen']
    avg_note = bestandene_abschluesse['abschlussnote'].mean() if len(bestandene_abschluesse) > 0 else 0.0
    avg_dauer = bestandene_abschluesse['studiendauer_semester'].mean() if len(bestandene_abschluesse) > 0 else 0.0
    avg_ba_note = bestandene_abschluesse['bachelorarbeitsnote'].mean() if 'bachelorarbeitsnote' in bestandene_abschluesse and len(bestandene_abschluesse) > 0 else 0.0
    
    anomalie_rate = (abschluesse_df['anomalie_typ'].notna()).sum() / n_studis if n_studis > 0 else 0.0
    
    # Statistische Aufschlüsselung nach Abgangsarten & Studiengängen
    abgang_summary = abschluesse_df.groupby(['stg_name', 'status']).size().unstack(fill_value=0)
    for col in ['abgeschlossen', 'abgebrochen', 'exmatrikuliert', 'zeitueberschreitung']:
        if col not in abgang_summary.columns:
            abgang_summary[col] = 0
            
    abgang_summary['Gesamt'] = abgang_summary.sum(axis=1)
    abgang_summary['Abschlussquote'] = (abgang_summary['abgeschlossen'] / abgang_summary['Gesamt'] * 100).round(1).astype(str) + '%'

    # 3. Konsistenz-Checks
    checks = []
    
    # Check 1: Foreign Keys Prüfungen -> Studierende
    studi_ids = set(studierende_df['studierenden_id'])
    pr_valid = set(pruefungen_df['studierenden_id']).issubset(studi_ids)
    checks.append(("Alle Prüfungen referenzieren existierende Studierende", pr_valid))
    
    # Check 2: Foreign Keys Einschreibungen -> Studierende
    ein_valid = set(einschreibungen_df['studierenden_id']).issubset(studi_ids)
    checks.append(("Alle Einschreibungen referenzieren existierende Studierende", ein_valid))
    
    # Check 3: Foreign Keys Abschlüsse -> Studierende
    abs_valid = set(abschluesse_df['studierenden_id']).issubset(studi_ids)
    checks.append(("Alle Abschluss-Datensätze referenzieren existierende Studierende", abs_valid))
    
    # Check 4: Genau 1 Abschluss pro Studi
    genau_ein_abs = len(abschluesse_df) == n_studis and abschluesse_df['studierenden_id'].nunique() == n_studis
    checks.append(("Jede/r Studierende hat genau einen Abschluss-Datensatz", genau_ein_abs))
    
    # Check 5: Noten im Bereich 1.0 - 5.0
    noten_valid = (pruefungen_df['note'].between(1.0, 5.0)).all()
    checks.append(("Alle Noten in gültigem Bereich (1.0–5.0)", noten_valid))
    
    # Check 6: Max 3 Versuche
    versuche_valid = (pruefungen_df['versuch'] <= 3).all()
    checks.append(("Max. 3 Versuche pro Modul", versuche_valid))
    
    # Check 7: Bestanden <-> Note <= 4.0
    bestanden_valid = ((pruefungen_df['bestanden'] & (pruefungen_df['note'] <= 4.0)) | (~pruefungen_df['bestanden'] & (pruefungen_df['note'] > 4.0))).all()
    checks.append(("Bestanden ⇔ Note ≤ 4.0", bestanden_valid))
    
    # Check 8: Nur erfolgreiche Abschlüsse haben eine Abschlussnote
    nicht_bestanden_df = abschluesse_df[abschluesse_df['status'] != 'abgeschlossen']
    keine_note_bei_nicht_abschluss = nicht_bestanden_df['abschlussnote'].isna().all()
    checks.append(("Nur erfolgreiche Abschlüsse ('abgeschlossen') enthalten eine Abschlussnote", keine_note_bei_nicht_abschluss))
    
    # Check 9: Bachelorarbeitsnote vorhanden bei allen erfolgreichen Abschlüssen
    if 'bachelorarbeitsnote' in abschluesse_df.columns:
        ba_valid = bestandene_abschluesse['bachelorarbeitsnote'].notna().all()
        checks.append(("Erfolgreiche Abschlüsse besitzen eine valide Bachelorarbeitsnote", ba_valid))

    # Check 10: Plausible Gesamtabschlussquote > 50%
    quote_valid = abschluss_quote > 0.50
    checks.append((f"Gesamtabschlussquote ist größer als 50% (ist: {abschluss_quote:.1%})", quote_valid))

    # Check 11: Abschlussquote pro Studiengang > 50%
    sg_quoten = (abgang_summary['abgeschlossen'] / abgang_summary['Gesamt']) > 0.50
    sg_quote_valid = sg_quoten.all()
    checks.append((f"Abschlussquote in allen Studiengängen einzeln > 50% (Min: {(abgang_summary['abgeschlossen'] / abgang_summary['Gesamt']).min():.1%})", sg_quote_valid))

    # Check 12: Ground Truth Counterfactual Notes vorhanden
    cf_valid = 'note_counterfactual' in pruefungen_df.columns and (pruefungen_df['note_counterfactual'].between(1.0, 5.0)).all()
    checks.append(("Ground Truth Counterfactual Notes korrekt generiert", cf_valid))

    # 4. Dokumentation schreiben
    table_markdown = _df_to_markdown(abgang_summary[['abgeschlossen', 'abgebrochen', 'exmatrikuliert', 'zeitueberschreitung', 'Gesamt', 'Abschlussquote']])

    doku_content = f"""# Synthetische Studienverlaufsdaten – Design & Validierung (DL-Edition)

## Zweck
Synthetischer Datensatz zur Analyse der Wirkung von Studierenden-Support-Angeboten auf Studienerfolg, Abschlussnote und Abbruchverhalten. Speziell optimiert für **Deep Learning / Zeitreihenmodelle**, Kausalanalyse (Counterfactual Ground Truth) und Survival-Analysen.

## Umfang
- **Studierende**: {n_studis:,}
- **Prüfungen gesamt**: {len(pruefungen_df):,}
- **Support-Teilnahmen gesamt**: {len(support_teilnahmen_df):,}
- Kohorten: {config_data.get('start_jahr', 2015)}–{config_data.get('end_jahr', 2024)} (WS-Start)
- {len(studiengaenge_df)} Studiengänge, {len(module_df)} Module, {len(support_angebote_df)} Support-Angebote

## Datenmodell (Tabellen)
| Tabelle | Inhalt |
|---|---|
| `studierende` | Stammdaten + latente Eigenschaften (Motivation, soziale Integration, Erwerb) |
| `studiengaenge` | 5 Studiengänge mit Regelstudienzeit und CP-Summe |
| `module` | Module mit CP, Schwierigkeit, Turnus (WS/SS/beides), Workload (h) |
| `modul_studiengang` | n:m-Zuordnung Modul ↔ Studiengang mit empfohlenem Fachsemester |
| `semester` | Chronologische Semester-Liste (SS/WS) |
| `einschreibungen` | Pro Studi × Semester (aktive Einschreibung & Fachsemester) |
| `pruefungen` | Prüfungsversuche mit Note, Bestehen und `note_counterfactual` (Ground Truth) |
| `support_angebote` | Tutorien, Beratung, Sprachkurse etc. (3 Kategorien) mit Zeitkosten |
| `support_modul_zuordnung` | Welches Angebot unterstützt welches Modul, mit Wirkungsstärke |
| `support_teilnahmen` | Studi × Angebot × Semester |
| `abschluesse` | Endstatus pro Studi (`abgeschlossen`, `abgebrochen`, `exmatrikuliert`, `zeitueberschreitung`) mit `bachelorarbeitsnote` |
| `agg_pruefungen` | **(Aggregiert)** Längsschnitt-Tabelle aller Prüfungen inkl. Support-Exposition (vorher/gleichzeitig) |
| `agg_abschluesse` | **(Aggregiert)** Querschnitt-Tabelle auf Studierenden-Ebene mit allen Kontrollvariablen für EDA/Dashboards |

## Statistische Übersicht der Abgangsarten

### Gesamtübersicht
- **Abschlussquote gesamt**: {abschluss_quote:.1%}
- **Insg. nicht erfolgreich** (Abbruch/Exmatrikulation/Zeitüberschreitung): {abbruch_quote:.1%}
- **Mittlere Abschlussnote**: {avg_note:.2f}
- **Mittlere Bachelorarbeitsnote**: {avg_ba_note:.2f}
- **Mittlere Studiendauer (Erfolgreiche)**: {avg_dauer:.2f} Semester

### Aufschlüsselung nach Studiengängen
{table_markdown}

## Modellarchitektur & Erweiterungen (DL-Edition)

### 1. Zeitkontenmodell (Time Budgeting)
- Studierende verfügen pro Semester über ein festes **Zeitbudget** (Standard: 900 Stunden für Vollzeit).
- Erwerbstätigkeit (z.B. 15h/Woche) zieht direkt Stunden vom Zeitkonto ab.
- Das Belegen von Modulen (Workload = CP × 30h) und die Teilnahme an Support-Angeboten zehren ebenfalls am Zeitkonto.
- Wenn der Gesamtaufwand das verfügbare Budget übersteigt, tritt eine **Overload-Penalty** ein (lineare Senkung der latenten Prüfungsleistung).

### 2. Turnus & Kalendarische Semester
- Module finden spezifisch im **Wintersemester (WS)**, **Sommersemester (SS)** oder **beiden** statt.
- Die Simulation unterscheidet strikt zwischen chronologischem Kalendersemester und absolviertem Fachsemester. 

### 3. Reaktive Support-Nutzung
- **Fachbezug**: Fachlicher Support (z.B. Mathe-Tutorium) wird nur gewählt, wenn in dem Semester auch ein passendes Modul belegt wird.
- **Reaktivität auf Fehlversuche**: Nach einem Nichtbestehen im Vorsemester steigt die Inanspruchnahme von Support-Angeboten für dieses Modul im Wiederholungsversuch um bis zu +20 Prozentpunkte.

### 4. Counterfactual Ground Truth
- In `pruefungen.csv` wird die **`note_counterfactual`** geloggt (die hypothetische Note ohne den Support-Boost).
- Dies ermöglicht die spätere kausale Evaluierung von Deep-Learning-Modellen gegen die wahre Ground Truth.

## Validierungsergebnisse (Konsistenz-Checks)

"""
    for desc, is_ok in checks:
        icon = "✅" if is_ok else "❌"
        doku_content += f"- {icon} {desc}\n"

    doku_content += """
## Reproduzierbarkeit
Seed: `42` (in CONFIG). Bei gleichem Seed identische Daten.
"""

    (output_dir / 'DATENSATZ_DOKU.md').write_text(doku_content, encoding='utf-8')
    print("  [OK] DATENSATZ_DOKU.md erfolgreich geschrieben.")

if __name__ == '__main__':
    validiere_und_dokumentiere(Path('../output_dl'))
