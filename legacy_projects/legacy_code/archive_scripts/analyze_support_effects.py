"""
Umfassende Analyse der kontrafaktischen Support-Effekte.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
Dieses Skript beantwortet vier zentrale Fragen:
1. Warum wirkt der fachliche Support kaum auf die Dropout-Rate?
2. Individuelle Studierenden-Migration zwischen den Universen (Wer schafft es in A, nicht in C/D/E?)
3. Noteneffekte des fachlichen Supports (erwartete_note → tatsächliche Noten)
4. Synergie-Effekte (Summe der Einzeleffekte vs. Gesamteffekt)

Liest aus den 5 Universumsordnern in output_dl/.
"""

import pandas as pd
import numpy as np
import json
import os
import sys

# --- Pfade ---
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "output_dl")
UNIVERSE_DIRS = {
    "A": BASE_DIR,                              # Baseline
    "B": os.path.join(BASE_DIR, "universe_B"),  # Kein Support
    "C": os.path.join(BASE_DIR, "universe_C"),  # Kein Fachlich
    "D": os.path.join(BASE_DIR, "universe_D"),  # Kein Überfachlich
    "E": os.path.join(BASE_DIR, "universe_E"),  # Kein Psychosozial
}
UNIVERSE_LABELS = {
    "A": "Baseline (alle Support-Typen)",
    "B": "Kein Support (komplett blockiert)",
    "C": "Kein fachlicher Support",
    "D": "Kein überfachlicher Support",
    "E": "Kein psychosozialer Support",
}
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_universe(key):
    """Lädt studierende.csv und pruefungen.csv für ein Universum."""
    d = UNIVERSE_DIRS[key]
    stud = pd.read_csv(os.path.join(d, "studierende.csv"))
    pruef = pd.read_csv(os.path.join(d, "pruefungen.csv"))
    return stud, pruef

def is_dropout(stud_df):
    """Bestimmt Dropout-Status. Sucht nach Spalte 'dropout' oder leitet ab."""
    # Prüfe ob es eine explizite dropout-Spalte gibt
    if "dropout" in stud_df.columns:
        return stud_df["dropout"].astype(bool)
    # Sonst: abschluesse.csv laden und mergen
    # Fallback: Studierende ohne Abschluss in den Abschlüssen
    return None


def main():
    print("=" * 80)
    print("ANALYSE DER KONTRAFAKTISCHEN SUPPORT-EFFEKTE")
    print("=" * 80)

    # ================================================================
    # DATEN LADEN
    # ================================================================
    print("\n[1/4] Lade Daten aus allen 5 Universen...")
    universes_stud = {}
    universes_pruef = {}
    for key in ["A", "B", "C", "D", "E"]:
        stud, pruef = load_universe(key)
        universes_stud[key] = stud
        universes_pruef[key] = pruef
        print(f"  Universum {key}: {len(stud)} Studierende, {len(pruef)} Prüfungen")

    # Prüfe Spalten der Studierenden
    print(f"\n  Spalten studierende.csv: {list(universes_stud['A'].columns)}")
    print(f"  Spalten pruefungen.csv: {list(universes_pruef['A'].columns)}")

    # ================================================================
    # ABSCHLUSS-STATUS BESTIMMEN
    # ================================================================
    # Lade abschluesse.csv für jedes Universum
    universes_abschl = {}
    for key in ["A", "B", "C", "D", "E"]:
        d = UNIVERSE_DIRS[key]
        abschl_path = os.path.join(d, "abschluesse.csv")
        if os.path.exists(abschl_path):
            abschl = pd.read_csv(abschl_path)
            universes_abschl[key] = abschl
            print(f"  Abschlüsse {key}: {len(abschl)} Zeilen, Spalten: {list(abschl.columns)}")
        else:
            print(f"  WARNUNG: {abschl_path} nicht gefunden!")

    # ================================================================
    # FRAGE 1: DROPOUT-RATEN UND EFFEKTSTÄRKEN
    # ================================================================
    print("\n" + "=" * 80)
    print("FRAGE 1: Dropout-Raten und wahre Support-Effekte")
    print("=" * 80)

    # Ermittle Dropout-Status pro Universum
    dropout_counts = {}
    for key in ["A", "B", "C", "D", "E"]:
        stud = universes_stud[key]
        abschl = universes_abschl.get(key)

        # Bestimme Status aus abschluesse.csv
        if abschl is not None and "status" in abschl.columns:
            # Zaehle unique Studierende mit status == 'abbruch' oder aehnlich
            status_counts = abschl["status"].value_counts()
            print(f"\n  Universum {key} - Status-Verteilung:")
            for s, c in status_counts.items():
                print(f"    {s}: {c}")
            # Dropout = abgebrochen + exmatrikuliert + zeitueberschreitung
            n_dropout = len(abschl[abschl["status"].isin(["abgebrochen", "exmatrikuliert", "zeitueberschreitung"])])
            n_total = len(stud)
        elif abschl is not None and "studierenden_id" in abschl.columns:
            abschluss_ids = set(abschl["studierenden_id"].unique())
            stud_ids = set(stud["studierenden_id"].unique())
            n_dropout = len(stud_ids - abschluss_ids)
            n_total = len(stud)
        else:
            n_dropout = 0
            n_total = len(stud)

        dropout_rate = n_dropout / n_total if n_total > 0 else 0
        dropout_counts[key] = {"n_total": n_total, "n_dropout": n_dropout, "rate": dropout_rate}

    print("\n  --- Vergleich der Dropout-Raten ---")
    print(f"  {'Universum':<12} {'Label':<40} {'Dropout':>7} {'Rate':>8} {'Diff vsA':>8} {'RR':>8}")
    print(f"  {'-'*12} {'-'*40} {'-'*7} {'-'*8} {'-'*8} {'-'*8}")
    rate_a = dropout_counts["A"]["rate"]
    for key in ["A", "B", "C", "D", "E"]:
        dc = dropout_counts[key]
        diff = dc["rate"] - rate_a
        rr = rate_a / dc["rate"] if dc["rate"] > 0 else float('inf')
        print(f"  {key:<12} {UNIVERSE_LABELS[key]:<40} {dc['n_dropout']:>7} {dc['rate']:>7.2%} {diff:>+7.3f} {rr:>7.4f}")

    # Lade die offizielle true_macro_effects_v2.json zum Vergleich
    macro_path = os.path.join(BASE_DIR, "metrics", "true_macro_effects_v2.json")
    if os.path.exists(macro_path):
        with open(macro_path) as f:
            macro_effects = json.load(f)
        print("\n  --- Offizielle true_macro_effects_v2.json (zum Vergleich) ---")
        for key, data in macro_effects.items():
            if isinstance(data, dict) and "dropout_rate" in data:
                print(f"  {key}: dropout_rate = {data['dropout_rate']:.5f}")

    # ================================================================
    # FRAGE 2: INDIVIDUELLE MIGRATION (Wer droppt in B/C/D/E, aber nicht in A?)
    # ================================================================
    print("\n" + "=" * 80)
    print("FRAGE 2: Individuelle Studierenden-Migration zwischen Universen")
    print("=" * 80)

    # Wir identifizieren Dropout-Status pro Student pro Universum
    # aus den abschluesse.csv (Status-Spalte)
    student_outcomes = pd.DataFrame({"studierenden_id": universes_stud["A"]["studierenden_id"]})

    for key in ["A", "B", "C", "D", "E"]:
        abschl = universes_abschl.get(key)
        if abschl is not None:
            # Finde die relevante Status-Spalte
            status_col = None
            for col in abschl.columns:
                if "status" in col.lower() or "ergebnis" in col.lower():
                    status_col = col
                    break

            if status_col:
                # Bestimme Dropout pro Student
                # Nehme den letzten Status pro Student
                sort_col = "abschluss_semester_id" if "abschluss_semester_id" in abschl.columns else ("semester_id" if "semester_id" in abschl.columns else abschl.columns[1])
                last_status = abschl.sort_values(sort_col)\
                    .groupby("studierenden_id")[status_col].last()
                student_outcomes[f"status_{key}"] = student_outcomes["studierenden_id"].map(last_status)
                student_outcomes[f"dropout_{key}"] = student_outcomes[f"status_{key}"].isin(["abgebrochen", "exmatrikuliert", "zeitueberschreitung"])
            else:
                # Fallback: Wer hat Prüfungen aber keinen Abschluss?
                pruef_ids = set(universes_pruef[key]["studierenden_id"].unique())
                abschl_ids = set(abschl["studierenden_id"].unique()) if "studierenden_id" in abschl.columns else set()
                student_outcomes[f"dropout_{key}"] = student_outcomes["studierenden_id"].isin(pruef_ids - abschl_ids)

    # Migrationsanalyse
    dropout_cols = [c for c in student_outcomes.columns if c.startswith("dropout_")]
    if len(dropout_cols) >= 2 and "dropout_A" in dropout_cols:
        print("\n  --- Migrationsmatrix: Wer schafft es in A, aber nicht in X? ---")
        for key in ["B", "C", "D", "E"]:
            col_a = "dropout_A"
            col_x = f"dropout_{key}"
            if col_x not in student_outcomes.columns:
                continue

            # Kreuztabelle
            both_ok = ((~student_outcomes[col_a]) & (~student_outcomes[col_x])).sum()
            a_ok_x_drop = ((~student_outcomes[col_a]) & (student_outcomes[col_x])).sum()
            a_drop_x_ok = ((student_outcomes[col_a]) & (~student_outcomes[col_x])).sum()
            both_drop = ((student_outcomes[col_a]) & (student_outcomes[col_x])).sum()

            print(f"\n  A vs. {key} ({UNIVERSE_LABELS[key]}):")
            print(f"  {'':>25} {'Nicht-Dropout in '+key:>22} {'Dropout in '+key:>18}")
            print(f"  {'Nicht-Dropout in A':>25} {both_ok:>22} {a_ok_x_drop:>18}")
            print(f"  {'Dropout in A':>25} {a_drop_x_ok:>22} {both_drop:>18}")
            print(f"  → {a_ok_x_drop} Studierende werden durch Wegfall von {UNIVERSE_LABELS[key].replace('Kein ', '')} zum Dropout getrieben")
            print(f"  → {a_drop_x_ok} Studierende schaffen es OHNE {UNIVERSE_LABELS[key].replace('Kein ', '')} sogar besser (Zeitgewinn?)")
    else:
        print("  WARNUNG: Dropout-Status konnte nicht zuverlässig bestimmt werden!")
        print(f"  Verfügbare Spalten: {list(student_outcomes.columns)}")

    # ================================================================
    # FRAGE 3: NOTENEFFEKTE DES FACHLICHEN SUPPORTS
    # ================================================================
    print("\n" + "=" * 80)
    print("FRAGE 3: Noteneffekte des fachlichen Supports")
    print("=" * 80)

    pruef_a = universes_pruef["A"]

    # 3a: Vergleich note vs. note_counterfactual in Universum A
    if "note_counterfactual" in pruef_a.columns and "support_genutzt" in pruef_a.columns:
        support_exams = pruef_a[pruef_a["support_genutzt"] == True].copy()
        no_support_exams = pruef_a[pruef_a["support_genutzt"] == False].copy()

        print(f"\n  Prüfungen mit Support: {len(support_exams)}")
        print(f"  Prüfungen ohne Support: {len(no_support_exams)}")

        if len(support_exams) > 0:
            support_exams["note_diff"] = support_exams["note_counterfactual"] - support_exams["note"]
            mean_diff = support_exams["note_diff"].mean()
            median_diff = support_exams["note_diff"].median()
            n_improved = (support_exams["note_diff"] > 0).sum()
            n_unchanged = (support_exams["note_diff"] == 0).sum()

            # Wie viele wurden durch Support vor Durchfallen gerettet?
            rescued = ((support_exams["note"] <= 4.0) & (support_exams["note_counterfactual"] > 4.0)).sum()
            # Bestehensquote mit/ohne Support
            pass_with = (support_exams["note"] <= 4.0).mean()
            pass_without = (support_exams["note_counterfactual"] <= 4.0).mean()

            print(f"\n  --- Effekt des fachlichen Supports auf Prüfungsnoten ---")
            print(f"  Mittlere Notenverbesserung (ATT):  {mean_diff:+.4f} Notenpunkte")
            print(f"  Median Notenverbesserung:          {median_diff:+.4f} Notenpunkte")
            print(f"  Prüfungen mit Verbesserung:        {n_improved} ({n_improved/len(support_exams)*100:.1f}%)")
            print(f"  Prüfungen ohne Veränderung:        {n_unchanged} ({n_unchanged/len(support_exams)*100:.1f}%)")
            print(f"  Vor Durchfallen gerettet:          {rescued} ({rescued/len(support_exams)*100:.2f}%)")
            print(f"  Bestehensquote MIT Support:        {pass_with*100:.2f}%")
            print(f"  Bestehensquote OHNE Support:       {pass_without*100:.2f}%")
            print(f"  Bestehensquoten-Differenz:         {(pass_with - pass_without)*100:+.2f} Prozentpunkte")

            # Verteilung der Notenverbesserungen
            print(f"\n  --- Verteilung der Notenverbesserungen ---")
            bins = [-0.01, 0.01, 0.3, 0.5, 1.0, 2.0, 5.0]
            labels = ["0 (kein Effekt)", "0.01-0.3 (klein)", "0.3-0.5 (mittel)",
                      "0.5-1.0 (groß)", "1.0-2.0 (sehr groß)", "2.0+ (extrem)"]
            support_exams["diff_bin"] = pd.cut(support_exams["note_diff"], bins=bins, labels=labels, right=False)
            for label in labels:
                count = (support_exams["diff_bin"] == label).sum()
                pct = count / len(support_exams) * 100
                print(f"    {label:<30} {count:>8} ({pct:>5.1f}%)")

    # 3b: Vergleich Noten zwischen Universum A und C (ohne fachlichen Support)
    pruef_c = universes_pruef["C"]
    print(f"\n  --- Direkter Notenvergleich A vs. C (gleiche Studis, gleiche Module) ---")

    # Merge auf Studierenden-ID + Modul-ID + Semester
    merge_cols = ["studierenden_id", "semester_id", "modul_id"]
    if all(c in pruef_a.columns for c in merge_cols) and all(c in pruef_c.columns for c in merge_cols):
        merged = pruef_a[merge_cols + ["note", "bestanden", "support_genutzt"]].merge(
            pruef_c[merge_cols + ["note", "bestanden"]],
            on=merge_cols, suffixes=("_A", "_C"), how="inner"
        )
        print(f"  Gematchte Prüfungen: {len(merged)}")

        # Nur Prüfungen, wo in A Support genutzt wurde
        with_support = merged[merged["support_genutzt"] == True]
        print(f"  Davon mit Support in A: {len(with_support)}")

        if len(with_support) > 0:
            note_diff = with_support["note_C"] - with_support["note_A"]
            print(f"  Mittlerer Notenunterschied (C - A): {note_diff.mean():+.4f} (positiv = A ist besser)")
            print(f"  Median Notenunterschied:            {note_diff.median():+.4f}")

            # Bestehensvergleich
            pass_a = (with_support["bestanden_A"] == True).sum() if "bestanden_A" in with_support else (with_support["note_A"] <= 4.0).sum()
            pass_c = (with_support["bestanden_C"] == True).sum() if "bestanden_C" in with_support else (with_support["note_C"] <= 4.0).sum()
            rescued_ac = ((with_support["note_A"] <= 4.0) & (with_support["note_C"] > 4.0)).sum()
            print(f"  Bestanden in A: {pass_a}, Bestanden in C: {pass_c}")
            print(f"  Nur durch fachlichen Support bestanden: {rescued_ac}")

    # 3c: Endnoten-Vergleich auf Studierendenebene
    stud_a = universes_stud["A"]
    stud_c = universes_stud["C"]
    if "hidden_erwartete_note_final" in stud_a.columns:
        merged_stud = stud_a[["studierenden_id", "hidden_erwartete_note_final"]].merge(
            stud_c[["studierenden_id", "hidden_erwartete_note_final"]],
            on="studierenden_id", suffixes=("_A", "_C")
        )
        note_diff = merged_stud["hidden_erwartete_note_final_C"] - merged_stud["hidden_erwartete_note_final_A"]
        print(f"\n  --- Vergleich der finalen erwarteten Note (hidden_erwartete_note_final) ---")
        print(f"  Mittlerer Unterschied (C - A):  {note_diff.mean():+.4f} (positiv = A besser)")
        print(f"  Median:                         {note_diff.median():+.4f}")
        print(f"  Std:                            {note_diff.std():.4f}")
        n_a_better = (note_diff > 0.01).sum()
        n_c_better = (note_diff < -0.01).sum()
        n_equal = len(note_diff) - n_a_better - n_c_better
        print(f"  A besser (Diff > 0.01):         {n_a_better} ({n_a_better/len(note_diff)*100:.1f}%)")
        print(f"  C besser (Diff < -0.01):        {n_c_better} ({n_c_better/len(note_diff)*100:.1f}%)")
        print(f"  Gleich (|Diff| ≤ 0.01):         {n_equal} ({n_equal/len(note_diff)*100:.1f}%)")

    # ================================================================
    # FRAGE 4: SYNERGIE-EFFEKTE
    # ================================================================
    print("\n" + "=" * 80)
    print("FRAGE 4: Synergie-Effekte (Summe der Einzeleffekte vs. Gesamteffekt)")
    print("=" * 80)

    # Lade offizielle Makro-Effekte
    if os.path.exists(macro_path):
        with open(macro_path) as f:
            macro = json.load(f)

        rate_a = macro["universe_A_baseline"]["dropout_rate"]
        rate_b = macro["universe_B"]["dropout_rate"]
        rate_c = macro["universe_C"]["dropout_rate"]
        rate_d = macro["universe_D"]["dropout_rate"]
        rate_e = macro["universe_E"]["dropout_rate"]

        total_effect = rate_b - rate_a  # Gesamter Support-Effekt (positiv = Support schützt)
        fach_effect = rate_c - rate_a   # Effekt von fachlichem Support allein
        uebf_effect = rate_d - rate_a   # Effekt von überfachlichem Support allein
        psych_effect = rate_e - rate_a  # Effekt von psychosozialem Support allein

        sum_individual = fach_effect + uebf_effect + psych_effect
        synergy = total_effect - sum_individual

        print(f"\n  Dropout-Raten:")
        print(f"    A (Baseline):           {rate_a*100:.3f}%")
        print(f"    B (Kein Support):       {rate_b*100:.3f}%")
        print(f"    C (Kein Fachlich):      {rate_c*100:.3f}%")
        print(f"    D (Kein Überfachlich):  {rate_d*100:.3f}%")
        print(f"    E (Kein Psychosozial):  {rate_e*100:.3f}%")

        print(f"\n  Support-Effekte (Dropout-Reduktion durch Support, %-Punkte):")
        print(f"    Fachlicher Support:       {fach_effect*100:+.3f} pp")
        print(f"    Überfachlicher Support:   {uebf_effect*100:+.3f} pp")
        print(f"    Psychosozialer Support:   {psych_effect*100:+.3f} pp")
        print(f"    ---")
        print(f"    Summe Einzeleffekte:      {sum_individual*100:+.3f} pp")
        print(f"    Tatsächl. Gesamteffekt:   {total_effect*100:+.3f} pp")
        print(f"    Synergie (Rest):          {synergy*100:+.3f} pp")
        print(f"    Synergie-Anteil:          {abs(synergy/total_effect)*100:.1f}% des Gesamteffekts")

        if abs(synergy) > 0.001:
            print(f"\n  → Es gibt eine {'positive' if synergy > 0 else 'negative'} Synergie von {synergy*100:+.3f} pp.")
            print(f"    Das bedeutet: Die Support-Typen {'verstärken' if synergy > 0 else 'schwächen'} sich gegenseitig.")
        else:
            print(f"\n  → Die Synergie ist vernachlässigbar klein. Die Effekte sind nahezu additiv.")

    # ================================================================
    # KAUSAL-DIAGNOSE: WARUM FACHLICHER SUPPORT FAST UNWIRKSAM IST
    # ================================================================
    print("\n" + "=" * 80)
    print("KAUSAL-DIAGNOSE: Warum fachlicher Support kaum auf Dropout wirkt")
    print("=" * 80)

    # Analysiere den Mechanismus im Detail
    print("""
  BEFUND: Fachlicher Support reduziert die Dropout-Rate nur um 0.07 pp (RR = 0.997).
          Im Vergleich: Überfachlich -1.88 pp, Psychosozial -1.66 pp, Gesamt -3.80 pp.

  URSACHE 1: Kein direkter Pfad zu Dropout-Variablen
  ─────────────────────────────────────────────────────
  berechne_dropout() hängt ab von:
    • motivation        (Gewicht 0.30, Schwelle 0.4)
    • soz_integration   (Gewicht 0.20, Schwelle 0.4)  
    • cp_rueckstand     (Gewicht 0.15)
    • durchgefallen     (Gewicht 0.04 pro Fail)
    • overload_penalty  (Gewicht 0.10)

  Fachlicher Support → verbessert NUR Prüfungsnoten (fachlicher_boost)
  → KEIN direkter Effekt auf motivation oder soz_integration!
  → Überfachlich/Psychosozial boosten diese DIREKT.

  URSACHE 2: Schwacher indirekter Pfad
  ─────────────────────────────────────
  Fachlicher Support → bessere Note → weniger Durchfallen → weniger Demotivation
  ABER: durchgefallen_aktuell hat nur Gewicht 0.04 pro Fail.
  → Ein vermiedenes Durchfallen senkt die Dropout-Prob. um nur ~0.02 pp pro Semester.

  URSACHE 3: Zeitkosten konterkarieren den Noteneffekt
  ────────────────────────────────────────────────────
  Fachliche Support-Angebote kosten 30h/Semester.
  Überfachliche: 10h, Psychosoziale: 5-15h.
  → Die 30h erhöhen den overload_penalty (+0.03 Dropout-Risiko pro Angebot).
  → Der kleine Noteneffekt wird durch die Zeitkosten teilweise aufgefressen.
  
  URSACHE 4: Notenverbesserung ≠ Dropout-Vermeidung
  ──────────────────────────────────────────────────
  Eine Note von 3.3 → 2.3 ist eine Verbesserung, hat aber NULL Effekt auf Dropout,
  weil berechne_dropout() nur binäre Failures zählt (Note ≤ 4.0 vs. 5.0).
  Support hilft nur bei "Grenzfällen" (Note ~4.0-5.0), die relativ selten sind.
""")

    # Quantifiziere den Grenzfall-Anteil
    if "note_counterfactual" in pruef_a.columns and "support_genutzt" in pruef_a.columns:
        support_exams = pruef_a[pruef_a["support_genutzt"] == True]
        if len(support_exams) > 0:
            near_fail_cf = ((support_exams["note_counterfactual"] >= 4.0) & (support_exams["note_counterfactual"] <= 5.0)).sum()
            rescued = ((support_exams["note"] <= 4.0) & (support_exams["note_counterfactual"] > 4.0)).sum()
            print(f"  QUANTIFIZIERUNG:")
            print(f"  Prüfungen mit Support:            {len(support_exams)}")
            print(f"  Davon Grenzfälle (Note ≥ 4.0 cf): {near_fail_cf} ({near_fail_cf/len(support_exams)*100:.1f}%)")
            print(f"  Tatsächlich gerettet:              {rescued} ({rescued/len(support_exams)*100:.2f}%)")
            print(f"  → Nur {rescued/len(support_exams)*100:.2f}% der Support-Prüfungen haben überhaupt")
            print(f"    einen Dropout-relevanten Effekt!")

    # ================================================================
    # VERGLEICH: DML-SCHÄTZUNGEN vs. GROUND TRUTH
    # ================================================================
    print("\n" + "=" * 80)
    print("VERGLEICH: DML-Modell-Schätzungen vs. kontrafaktische Ground Truth")
    print("=" * 80)

    dml_path = os.path.join(BASE_DIR, "metrics", "dml_orthogonal_survival_metrics.json")
    if os.path.exists(dml_path) and os.path.exists(macro_path):
        with open(dml_path) as f:
            dml = json.load(f)
        with open(macro_path) as f:
            macro = json.load(f)

        print(f"\n  {'Support-Typ':<25} {'Ground Truth RR':>15} {'DML Mean RR':>12} {'Differenz':>10} {'Bewertung':>12}")
        print(f"  {'-'*25} {'-'*15} {'-'*12} {'-'*10} {'-'*12}")

        comparisons = [
            ("Fachlich", macro["universe_C"]["vs_A_relative_risk"], dml.get("Mean_RR_fach_DML", None)),
            ("Überfachlich", macro["universe_D"]["vs_A_relative_risk"], dml.get("Mean_RR_uebf_DML", None)),
            ("Psychosozial", macro["universe_E"]["vs_A_relative_risk"], dml.get("Mean_RR_psych_DML", None)),
        ]

        for name, gt_rr, dml_rr in comparisons:
            if dml_rr is not None:
                diff = dml_rr - gt_rr
                if abs(diff) < 0.05:
                    bewertung = "✓ Gut"
                elif abs(diff) < 0.10:
                    bewertung = "~ Akzeptabel"
                else:
                    bewertung = "✗ Abweichung!"
                print(f"  {name:<25} {gt_rr:>15.4f} {dml_rr:>12.4f} {diff:>+10.4f} {bewertung:>12}")

        # Gesamteffekt-Vergleich
        gt_total_rr = macro["universe_B"]["vs_A_relative_risk"]
        print(f"\n  Gesamteffekt (alle Support-Typen zusammen):")
        print(f"    Ground Truth RR (B vs A):  {gt_total_rr:.4f}")
        # DML hat keinen kombinierten RR - das wäre ein separates Experiment

    # ================================================================
    # EXPORT: Zusammenfassung als CSV
    # ================================================================
    summary_data = []
    for key in ["A", "B", "C", "D", "E"]:
        row = {
            "universum": key,
            "label": UNIVERSE_LABELS[key],
            "n_studierende": dropout_counts[key]["n_total"],
            "n_dropout": dropout_counts[key]["n_dropout"],
            "dropout_rate": dropout_counts[key]["rate"],
        }
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(OUTPUT_DIR, "universe_comparison_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Zusammenfassung exportiert: {summary_path}")

    # Export der Migrationsmatrix
    if "dropout_A" in student_outcomes.columns:
        migration_path = os.path.join(OUTPUT_DIR, "student_migration_matrix.csv")
        migration_cols = ["studierenden_id"] + [c for c in student_outcomes.columns if c.startswith("dropout_")]
        student_outcomes[migration_cols].to_csv(migration_path, index=False)
        print(f"  Migrationsmatrix exportiert: {migration_path}")

    print("\n" + "=" * 80)
    print("ANALYSE ABGESCHLOSSEN")
    print("=" * 80)


if __name__ == "__main__":
    main()
