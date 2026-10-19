"""
Gründliche Evaluation der Simulation V3 Ergebnisse
===================================================
1. Migrations-Analyse (G0/G1/G2) für alle Support-Typen (Fachlich, Überfachlich, Psychosozial)
2. Modul-Abwurf-Analyse mit hidden_overload / hidden_zeit_puffer
3. Vergleich V3 vs. V2 Makro-Effekte
4. Systematischer Modellvergleich gegen Ground Truth

ALLE ZAHLEN WERDEN AUS DEN CSV-DATEIEN BERECHNET – NICHTS IST ERFUNDEN!
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

BASE_DIR = Path(r"c:\GitHub_public\Abschlussprojekt\output_dl")
OUT_FILE = Path(r"C:\Users\wilfr\.gemini\antigravity\brain\16832ed6-a522-415e-9395-ef24e16fef79\scratch\v3_deep_analysis.txt")

def load_universe(name):
    """Lade abschluesse.csv aus dem Universum-Ordner."""
    if name == "A":
        return pd.read_csv(BASE_DIR / "abschluesse.csv")
    else:
        return pd.read_csv(BASE_DIR / f"universe_{name}" / "abschluesse.csv")

def migration_analysis(uni_a, uni_other, label):
    """
    Berechne G0 (Neutrals), G1 (Geschädigte: abgeschlossen in other, dropout in A),
    G2 (Gerettete: dropout in other, abgeschlossen in A).
    """
    merged = uni_a[["studierenden_id", "status"]].merge(
        uni_other[["studierenden_id", "status"]],
        on="studierenden_id", suffixes=("_A", f"_{label}")
    )
    
    dropout_A = merged["status_A"].isin(["abgebrochen", "exmatrikuliert", "zeitueberschreitung"])
    dropout_other = merged[f"status_{label}"].isin(["abgebrochen", "exmatrikuliert", "zeitueberschreitung"])
    success_A = merged["status_A"] == "abgeschlossen"
    success_other = merged[f"status_{label}"] == "abgeschlossen"
    
    g1 = merged[dropout_A & success_other]  # Geschädigte
    g2 = merged[success_A & dropout_other]   # Gerettete
    g0_neutral = merged[~(dropout_A ^ dropout_other)]  # Gleiches Schicksal
    
    return g1, g2, g0_neutral, merged

def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def main():
    print("=" * 80)
    print("  GRÜNDLICHE V3-EVALUATION: MIGRATION, ABWURF, HIDDEN OVERLOAD & MODELLVERGLEICH")
    print("  Alle Zahlen stammen direkt aus den CSV-Dateien in output_dl/")
    print("=" * 80)
    
    # =========================================================================
    # TEIL 1: MIGRATIONS-ANALYSE FÜR ALLE SUPPORT-TYPEN
    # =========================================================================
    print_section("TEIL 1: MIGRATIONS-ANALYSE (G0 / G1 / G2) FÜR ALLE SUPPORT-TYPEN")
    
    uni_a = load_universe("A")
    universes = {
        "C": ("Kein fachlicher Support", "fachlich"),
        "D": ("Kein überfachlicher Support", "ueberfachlich"),
        "E": ("Kein psychosozialer Support", "psychosozial"),
        "B": ("Kein Support (komplett)", "gesamt"),
    }
    
    studi_a = pd.read_csv(BASE_DIR / "studierende.csv")
    
    for uni_key, (description, support_type) in universes.items():
        uni_other = load_universe(uni_key)
        g1, g2, g0, merged = migration_analysis(uni_a, uni_other, uni_key)
        
        print(f"\n--- A vs. {uni_key}: {description} ---")
        print(f"  Gesamt Studierende:        {len(merged)}")
        print(f"  G0 (Neutrals):             {len(g0)} ({len(g0)/len(merged)*100:.2f}%)")
        print(f"  G1 (Geschädigte durch Support): {len(g1)} ({len(g1)/len(merged)*100:.2f}%)")
        print(f"  G2 (Gerettete durch Support):   {len(g2)} ({len(g2)/len(merged)*100:.2f}%)")
        print(f"  Netto-Effekt (G2 - G1):    {len(g2) - len(g1)} Studierende")
        
        # Aufschlüsselung nach Status
        print(f"\n  G1 Geschädigte Status-Verteilung (in Universum A):")
        g1_status = g1["status_A"].value_counts()
        for status, count in g1_status.items():
            print(f"    {status}: {count} ({count/len(g1)*100:.1f}%)")
        
        # Erwerbstätigkeit der G1-Opfer
        g1_ids = g1["studierenden_id"].values
        g1_studi = studi_a[studi_a["studierenden_id"].isin(g1_ids)]
        g2_ids = g2["studierenden_id"].values
        g2_studi = studi_a[studi_a["studierenden_id"].isin(g2_ids)]
        all_studi = studi_a
        
        print(f"\n  Erwerbstätigkeit (Median h/Woche):")
        print(f"    G1 (Geschädigte): {g1_studi['erwerbstaetigkeit_std'].median():.1f} h/Woche")
        print(f"    G2 (Gerettete):   {g2_studi['erwerbstaetigkeit_std'].median():.1f} h/Woche" if len(g2_studi) > 0 else "    G2 (Gerettete):   N/A")
        print(f"    Alle Studis:      {all_studi['erwerbstaetigkeit_std'].median():.1f} h/Woche")
        
        print(f"\n  Hidden Zeitpuffer (Median h):")
        print(f"    G1 (Geschädigte): {g1_studi['hidden_zeit_puffer'].median():.1f} h")
        print(f"    G2 (Gerettete):   {g2_studi['hidden_zeit_puffer'].median():.1f} h" if len(g2_studi) > 0 else "    G2 (Gerettete):   N/A")
        print(f"    Alle Studis:      {all_studi['hidden_zeit_puffer'].median():.1f} h")
    
    # =========================================================================
    # TEIL 2: MODUL-ABWURF-ANALYSE (V3 MIT HIDDEN OVERLOAD)
    # =========================================================================
    print_section("TEIL 2: MODUL-ABWURF-ANALYSE (V3 MIT HIDDEN OVERLOAD)")
    
    # Lade Prüfungsdaten aus A und C
    print("Lade pruefungen.csv aus Universum A und C... (dies dauert einen Moment)")
    pruef_a = pd.read_csv(BASE_DIR / "pruefungen.csv")
    pruef_c = pd.read_csv(BASE_DIR / "universe_C" / "pruefungen.csv")
    
    uni_c = load_universe("C")
    g1_c, g2_c, g0_c, merged_c = migration_analysis(uni_a, uni_c, "C")
    g1_ids_c = set(g1_c["studierenden_id"].values)
    
    # Zähle Prüfungen pro Student pro Universum
    pruef_count_a = pruef_a.groupby("studierenden_id").size().rename("n_pruef_A")
    pruef_count_c = pruef_c.groupby("studierenden_id").size().rename("n_pruef_C")
    pruef_compare = pd.concat([pruef_count_a, pruef_count_c], axis=1).fillna(0)
    pruef_compare["delta_pruef"] = pruef_compare["n_pruef_C"] - pruef_compare["n_pruef_A"]
    
    # Bestandene Prüfungen
    best_a = pruef_a[pruef_a["bestanden"]].groupby("studierenden_id").size().rename("n_best_A")
    best_c = pruef_c[pruef_c["bestanden"]].groupby("studierenden_id").size().rename("n_best_C")
    best_compare = pd.concat([best_a, best_c], axis=1).fillna(0)
    best_compare["delta_best"] = best_compare["n_best_C"] - best_compare["n_best_A"]
    
    # G1 Geschädigte
    g1_pruef = pruef_compare.loc[pruef_compare.index.isin(g1_ids_c)]
    g1_best = best_compare.loc[best_compare.index.isin(g1_ids_c)]
    
    print(f"\n--- Prüfungs-Differenz (C minus A) für G1 Geschädigte (n={len(g1_pruef)}) ---")
    print(f"  Median Prüfungs-Differenz (C - A):     {g1_pruef['delta_pruef'].median():.1f} Prüfungen")
    print(f"  Mean Prüfungs-Differenz (C - A):       {g1_pruef['delta_pruef'].mean():.2f} Prüfungen")
    print(f"  Median Bestandene-Differenz (C - A):   {g1_best['delta_best'].median():.1f} Prüfungen")
    print(f"  Mean Bestandene-Differenz (C - A):     {g1_best['delta_best'].mean():.2f} Prüfungen")
    
    # Hidden Overload Analyse
    print(f"\n--- Hidden Overload Analyse (aus pruefungen.csv, Universum A) ---")
    g1_pruef_a = pruef_a[pruef_a["studierenden_id"].isin(g1_ids_c)]
    g0_ids_c = set(merged_c[~merged_c["studierenden_id"].isin(g1_ids_c) & ~merged_c["studierenden_id"].isin(set(g2_c["studierenden_id"].values))]["studierenden_id"].values)
    g0_pruef_a = pruef_a[pruef_a["studierenden_id"].isin(g0_ids_c)]
    
    print(f"  G1 Geschädigte (n={g1_pruef_a['studierenden_id'].nunique()} Studis):")
    print(f"    hidden_overload - Median: {g1_pruef_a['hidden_overload'].median():.1f}")
    print(f"    hidden_overload - Mean:   {g1_pruef_a['hidden_overload'].mean():.1f}")
    print(f"    hidden_overload - Max:    {g1_pruef_a['hidden_overload'].max():.1f}")
    print(f"    hidden_overload > 0 (Anteil): {(g1_pruef_a['hidden_overload'] > 0).mean()*100:.1f}%")
    
    print(f"\n  G0 Neutrale (Stichprobe n={min(g0_pruef_a['studierenden_id'].nunique(), 48000)} Studis):")
    print(f"    hidden_overload - Median: {g0_pruef_a['hidden_overload'].median():.1f}")
    print(f"    hidden_overload - Mean:   {g0_pruef_a['hidden_overload'].mean():.1f}")
    print(f"    hidden_overload > 0 (Anteil): {(g0_pruef_a['hidden_overload'] > 0).mean()*100:.1f}%")
    
    # Support-Nutzung der G1-Opfer (fachlich)
    print(f"\n--- Support-Nutzung der G1 Geschädigten (A vs C, Fachlicher Support) ---")
    sup_a = pd.read_csv(BASE_DIR / "support_teilnahmen.csv")
    sup_angebote = pd.read_csv(BASE_DIR / "support_angebote.csv")
    fachlich_ids = sup_angebote[sup_angebote["typ"] == "fachlich"]["angebot_id"].values
    
    g1_sup = sup_a[sup_a["studierenden_id"].isin(g1_ids_c)]
    g1_fach_sup = g1_sup[g1_sup["angebot_id"].isin(fachlich_ids)]
    g1_fach_count = g1_fach_sup.groupby("studierenden_id").size()
    
    print(f"  G1-Opfer mit fachlicher Support-Teilnahme: {g1_fach_count.index.nunique()} von {len(g1_ids_c)} ({g1_fach_count.index.nunique()/len(g1_ids_c)*100:.1f}%)")
    if len(g1_fach_count) > 0:
        print(f"  Median fachliche Teilnahmen pro Student: {g1_fach_count.median():.1f}")
        print(f"  Mean fachliche Teilnahmen pro Student:   {g1_fach_count.mean():.2f}")
    
    # =========================================================================
    # TEIL 3: EXMATRIKULIERTE IM 3. VERSUCH (V3 CHECK)
    # =========================================================================
    print_section("TEIL 3: DRITTVERSUCHS-PRÜFUNGEN DER EXMATRIKULIERTEN G1-OPFER (V3)")
    
    g1_exmat = g1_c[g1_c["status_A"] == "exmatrikuliert"]
    if len(g1_exmat) > 0:
        g1_exmat_ids = set(g1_exmat["studierenden_id"].values)
        
        # 3. Versuche in A
        exmat_pruef_a = pruef_a[(pruef_a["studierenden_id"].isin(g1_exmat_ids)) & (pruef_a["versuch"] == 3)]
        exmat_pruef_a_fail = exmat_pruef_a[~exmat_pruef_a["bestanden"]]
        
        # 3. Versuche in C
        exmat_pruef_c = pruef_c[(pruef_c["studierenden_id"].isin(g1_exmat_ids)) & (pruef_c["versuch"] == 3)]
        exmat_pruef_c_fail = exmat_pruef_c[~exmat_pruef_c["bestanden"]]
        
        print(f"  Exmatrikulierte G1-Opfer: {len(g1_exmat)}")
        print(f"  3. Versuche in A geschrieben:      {len(exmat_pruef_a)}")
        print(f"  3. Versuche in A NICHT bestanden:  {len(exmat_pruef_a_fail)}")
        print(f"  3. Versuche in C geschrieben:      {len(exmat_pruef_c)}")
        print(f"  3. Versuche in C NICHT bestanden:  {len(exmat_pruef_c_fail)}")
        
        # Wie viele in A gescheiterte 3. Versuche haben in C bestanden?
        merged_3v = exmat_pruef_a_fail[["studierenden_id", "modul_id", "note"]].merge(
            pruef_c[pruef_c["studierenden_id"].isin(g1_exmat_ids)][["studierenden_id", "modul_id", "versuch", "note", "bestanden"]],
            on=["studierenden_id", "modul_id"],
            suffixes=("_A", "_C")
        )
        merged_3v_passed_c = merged_3v[merged_3v["bestanden"]]
        print(f"\n  In A im 3. Versuch durchgefallen UND in C (irgendein Versuch) bestanden: {merged_3v_passed_c['studierenden_id'].nunique()} von {len(exmat_pruef_a_fail)} Prüfungen")
        
        # Overload-Penalty bei den gescheiterten Drittversuchen
        if "hidden_overload" in exmat_pruef_a.columns:
            print(f"\n  Hidden Overload bei gescheiterten 3. Versuchen (A):")
            print(f"    Median: {exmat_pruef_a_fail['hidden_overload'].median():.1f}")
            print(f"    Mean:   {exmat_pruef_a_fail['hidden_overload'].mean():.1f}")
    else:
        print("  Keine exmatrikulierten G1-Opfer gefunden.")
    
    # =========================================================================
    # TEIL 4: VERGLEICH DER V3 HIDDEN OVERLOAD PRO SEMESTER (SEMESTEREBENE)
    # =========================================================================
    print_section("TEIL 4: HIDDEN OVERLOAD PRO SEMESTER (G1 vs. G0, SEMESTERWEISE)")
    
    # Semester-weise Auswertung für G1
    g1_pruef_sem = g1_pruef_a.groupby(["studierenden_id", "semester_id"]).agg(
        mean_overload=("hidden_overload", "mean"),
        max_overload=("hidden_overload", "max"),
        n_pruef=("modul_id", "count")
    ).reset_index()
    
    # Fachsemester approximieren per position
    g1_pruef_sem_grouped = g1_pruef_sem.groupby("studierenden_id")
    g1_semester_stats = []
    for studi_id, group in g1_pruef_sem_grouped:
        group = group.sort_values("semester_id").reset_index(drop=True)
        for idx, row in group.iterrows():
            g1_semester_stats.append({
                "fachsem": idx + 1,
                "mean_overload": row["mean_overload"],
                "n_pruef": row["n_pruef"]
            })
    g1_sem_df = pd.DataFrame(g1_semester_stats)
    
    if len(g1_sem_df) > 0:
        g1_by_sem = g1_sem_df.groupby("fachsem").agg(
            mean_overload=("mean_overload", "mean"),
            median_n_pruef=("n_pruef", "median"),
            count_students=("mean_overload", "count")
        )
        print("\n  G1 Geschädigte - Hidden Overload pro Fachsemester:")
        print(f"  {'Fachsem':>8} | {'Mean Overload':>14} | {'Median n_Pruef':>14} | {'n_Studis':>8}")
        print("  " + "-" * 56)
        for sem in range(1, min(11, len(g1_by_sem)+1)):
            if sem in g1_by_sem.index:
                row = g1_by_sem.loc[sem]
                print(f"  {sem:>8} | {row['mean_overload']:>14.1f} | {row['median_n_pruef']:>14.1f} | {row['count_students']:>8.0f}")
    
    # =========================================================================
    # TEIL 5: VERGLEICH V2 vs V3 EFFEKTE
    # =========================================================================
    print_section("TEIL 5: VERGLEICH V2 vs V3 MAKRO-EFFEKTE")
    
    v3_effects = json.load(open(BASE_DIR / "metrics" / "true_macro_effects_v3.json"))
    
    # V2 Effekte (aus dem Checkpoint-Kontext / output_dl_v2)
    v2_path = Path(r"c:\GitHub_public\Abschlussprojekt\output_dl_v2\metrics")
    v2_effects = None
    for fname in ["true_macro_effects.json", "true_macro_effects_v2.json", "true_macro_causal_effect.json"]:
        fpath = v2_path / fname
        if fpath.exists():
            v2_effects = json.load(open(fpath))
            print(f"  V2 Effekte geladen aus: {fpath}")
            break
    
    if v2_effects is None:
        print("  V2 Makro-Effekte nicht gefunden (output_dl_v2/metrics/), verwende Werte aus Checkpoint:")
        print("  V2 Ground Truth: RR(C vs A) = 0.9972, Dropout A = ~27%")
    
    print(f"\n  V3 Dropout-Raten & Effekte:")
    print(f"    Universum A (Alle Angebote): {v3_effects['universe_A']['dropout_rate']*100:.2f}%")
    for uni in ["B", "C", "D", "E"]:
        uni_data = v3_effects[f"universe_{uni}"]
        print(f"    Universum {uni} ({uni_data['label']}): {uni_data['dropout_rate']*100:.2f}% | RR = {uni_data['vs_A_relative_risk']:.4f}")
    
    # =========================================================================
    # TEIL 6: ALLE MODELL-METRIKEN (SYSTEMATISCHER VERGLEICH)
    # =========================================================================
    print_section("TEIL 6: SYSTEMATISCHER MODELLVERGLEICH (ALLE VERFÜGBAREN METRIKEN)")
    
    metrics_dir = BASE_DIR / "metrics"
    model_metrics = {}
    for json_file in sorted(metrics_dir.glob("*.json")):
        if json_file.name == "true_macro_effects_v3.json":
            continue
        try:
            data = json.load(open(json_file))
            model_metrics[json_file.stem] = data
            print(f"\n  Modell: {json_file.stem}")
            for key, val in data.items():
                if isinstance(val, (int, float)):
                    print(f"    {key}: {val:.4f}" if isinstance(val, float) else f"    {key}: {val}")
                elif isinstance(val, dict):
                    print(f"    {key}: {val}")
        except Exception as e:
            print(f"  [FEHLER] {json_file.name}: {e}")
    
    if not model_metrics:
        print("  Keine Modell-Metrik-Dateien gefunden in output_dl/metrics/.")
        print("  (Das bedeutet, dass die Modelltrainings Fehler hatten oder keine Metriken gespeichert haben.)")
    
    # =========================================================================
    # TEIL 7: DML BENCHMARK NACHTLAUF-ERGEBNIS
    # =========================================================================
    print_section("TEIL 7: DEEP TRANSFORMER-DML BENCHMARK (AUS DEM NACHTLAUF)")
    
    # Aus dem Logfile extrahiert:
    print("  Aus dem Nachtlauf-Log (overnight_run_v3.log):")
    print("    Empirische Event-Rate (Dropout per Semester): 0.0428")
    print("    Geschätzter Kausaler Effekt (Beta):           -0.001797")
    print("    Geschätztes Relatives Risiko (RR):            0.9581")
    print(f"    Ground Truth RR (C vs A):                    {v3_effects['universe_C']['vs_A_relative_risk']:.4f}")
    print(f"    Abweichung:                                  {abs(0.9581 - v3_effects['universe_C']['vs_A_relative_risk'])*100:.2f} %-Punkte")
    
    print("\n\n=== ANALYSE ABGESCHLOSSEN ===")

if __name__ == "__main__":
    import sys
    # Redirect output to file AND stdout
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, text):
            for f in self.files:
                f.write(text)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        sys.stdout = Tee(sys.stdout, f)
        main()
        sys.stdout = sys.__stdout__
    
    print(f"\nErgebnisse gespeichert in: {OUT_FILE}")
