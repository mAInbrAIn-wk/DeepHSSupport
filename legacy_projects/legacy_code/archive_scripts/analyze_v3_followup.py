"""
Vertiefte V3-Analyse: Antworten auf alle Nutzer-Rückfragen
==========================================================
1. B-zu-C Migration (warum weniger G1 in B als in C?)
2. Non-fachlicher Support G1: Warum gibt es überhaupt Geschädigte?
3. Per-Semester normierte Prüfungsdifferenz (nicht nur kumuliert)
4. Overload-Cap Analyse: Wie oft wird der Deckel von 0.15 erreicht?
5. Modulabwurf-Nachweis auf Prüfungsebene (verzögerte Module)
6. CP-Rückstand Profil der G1 vs G2 vs G0 Subgruppen
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"c:\GitHub_public\Abschlussprojekt\output_dl")

def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def load_universe(name):
    if name == "A":
        return pd.read_csv(BASE_DIR / "abschluesse.csv")
    else:
        return pd.read_csv(BASE_DIR / f"universe_{name}" / "abschluesse.csv")

def migration(uni_a, uni_other, label):
    merged = uni_a[["studierenden_id", "status"]].merge(
        uni_other[["studierenden_id", "status"]],
        on="studierenden_id", suffixes=("_A", f"_{label}"))
    dropout_A = merged["status_A"].isin(["abgebrochen", "exmatrikuliert", "zeitueberschreitung"])
    dropout_other = merged[f"status_{label}"].isin(["abgebrochen", "exmatrikuliert", "zeitueberschreitung"])
    success_A = merged["status_A"] == "abgeschlossen"
    success_other = merged[f"status_{label}"] == "abgeschlossen"
    g1 = merged[dropout_A & success_other]
    g2 = merged[success_A & dropout_other]
    return g1, g2, merged

def main():
    uni_a = load_universe("A")
    uni_b = load_universe("B")
    uni_c = load_universe("C")
    uni_d = load_universe("D")
    uni_e = load_universe("E")
    studi = pd.read_csv(BASE_DIR / "studierende.csv")
    
    # =========================================================================
    # FRAGE 1: B-zu-C Migration -- Warum weniger G1 in B (666) als in C (759)?
    # =========================================================================
    print_section("FRAGE 1: B-zu-C MIGRATION (Warum B weniger G1 als C?)")
    
    # B vs C direkt: Wer ist in B dropout, aber in C nicht?
    merged_bc = uni_b[["studierenden_id", "status"]].merge(
        uni_c[["studierenden_id", "status"]],
        on="studierenden_id", suffixes=("_B", "_C"))
    
    dropout_B = merged_bc["status_B"].isin(["abgebrochen", "exmatrikuliert", "zeitueberschreitung"])
    dropout_C = merged_bc["status_C"].isin(["abgebrochen", "exmatrikuliert", "zeitueberschreitung"])
    success_B = merged_bc["status_B"] == "abgeschlossen"
    success_C = merged_bc["status_C"] == "abgeschlossen"
    
    bc_geschaedigt = merged_bc[dropout_B & success_C]  # In B dropout, in C success
    bc_gerettet = merged_bc[success_B & dropout_C]     # In B success, in C dropout
    
    print(f"  B vs C Migration:")
    print(f"    In B dropout & in C erfolgreich: {len(bc_geschaedigt)} (non-fachlicher Support schadet in B)")
    print(f"    In B erfolgreich & in C dropout: {len(bc_gerettet)} (non-fachlicher Support rettet in B)")
    
    # Die G1(A vs B) und G1(A vs C) überlappen sich teilweise
    g1_ac, g2_ac, _ = migration(uni_a, uni_c, "C")
    g1_ab, g2_ab, _ = migration(uni_a, uni_b, "B")
    
    g1_ac_ids = set(g1_ac["studierenden_id"])
    g1_ab_ids = set(g1_ab["studierenden_id"])
    
    print(f"\n  Überlappung:")
    print(f"    G1(A vs C): {len(g1_ac_ids)} Geschädigte")
    print(f"    G1(A vs B): {len(g1_ab_ids)} Geschädigte")
    print(f"    Schnittmenge (in beiden G1): {len(g1_ac_ids & g1_ab_ids)}")
    print(f"    Nur in G1(A vs C): {len(g1_ac_ids - g1_ab_ids)}")
    print(f"    Nur in G1(A vs B): {len(g1_ab_ids - g1_ac_ids)}")
    
    # Warum sind manche in G1(A vs C) aber NICHT in G1(A vs B)?
    # D.h. sie sind in C erfolgreich und in A dropout, aber in B AUCH dropout
    # -> Der non-fachliche Support rettet sie nicht, nur der fachliche hätte geholfen
    only_ac = g1_ac_ids - g1_ab_ids
    if only_ac:
        # Was ist ihr Status in B?
        only_ac_in_b = uni_b[uni_b["studierenden_id"].isin(only_ac)]["status"].value_counts()
        print(f"\n  G1(A vs C) die NICHT G1(A vs B) sind - ihr Status in B:")
        for status, count in only_ac_in_b.items():
            print(f"    {status}: {count}")
        print(f"  -> Diese {len(only_ac)} sind in B AUCH Dropout, d.h. non-fachlicher Support hat sie nicht gerettet,")
        print(f"    aber in C (wo ALLE non-fachlichen Supports aktiv sind) wären sie erfolgreich.")
    
    # Und andersherum: G1(A vs B) die nicht in G1(A vs C) sind
    only_ab = g1_ab_ids - g1_ac_ids
    if only_ab:
        only_ab_in_c = uni_c[uni_c["studierenden_id"].isin(only_ab)]["status"].value_counts()
        print(f"\n  G1(A vs B) die NICHT G1(A vs C) sind - ihr Status in C:")
        for status, count in only_ab_in_c.items():
            print(f"    {status}: {count}")

    # =========================================================================
    # FRAGE 2: Non-fachlicher G1 -- Warum gibt es Geschädigte bei D und E?
    # =========================================================================
    print_section("FRAGE 2: NON-FACHLICHER SUPPORT G1 -- Woher kommen die?")
    
    g1_ad, g2_ad, _ = migration(uni_a, uni_d, "D")
    g1_ae, g2_ae, _ = migration(uni_a, uni_e, "E")
    
    # Haben die G1(A vs D) überfachlichen Support genutzt?
    sup = pd.read_csv(BASE_DIR / "support_teilnahmen.csv")
    sup_angebote = pd.read_csv(BASE_DIR / "support_angebote.csv")
    
    for label, g1_df, typ in [("D (überfachlich)", g1_ad, "ueberfachlich"),
                                ("E (psychosozial)", g1_ae, "psychosozial")]:
        g1_ids = set(g1_df["studierenden_id"])
        angebote = sup_angebote[sup_angebote["typ"] == typ]["angebot_id"].values
        g1_sup = sup[(sup["studierenden_id"].isin(g1_ids)) & (sup["angebot_id"].isin(angebote))]
        n_mit_sup = g1_sup["studierenden_id"].nunique()
        
        print(f"\n  G1(A vs {label}): {len(g1_ids)} Geschädigte")
        print(f"    Mit {typ} Support-Teilnahme: {n_mit_sup} ({n_mit_sup/len(g1_ids)*100:.1f}%)")
        
        # Haben sie auch fachlichen Support genutzt?
        fach_angebote = sup_angebote[sup_angebote["typ"] == "fachlich"]["angebot_id"].values
        g1_fach = sup[(sup["studierenden_id"].isin(g1_ids)) & (sup["angebot_id"].isin(fach_angebote))]
        n_mit_fach = g1_fach["studierenden_id"].nunique()
        print(f"    AUCH mit fachlichem Support: {n_mit_fach} ({n_mit_fach/len(g1_ids)*100:.1f}%)")
        
        # Erwerbstätigkeit
        g1_studi = studi[studi["studierenden_id"].isin(g1_ids)]
        print(f"    Erwerbstätigkeit Median: {g1_studi['erwerbstaetigkeit_std'].median():.1f} h/Woche")
        print(f"    hidden_zeit_puffer Median: {g1_studi['hidden_zeit_puffer'].median():.1f} h")
    
    # =========================================================================
    # FRAGE 3: Per-Semester normierte Prüfungsdifferenz
    # =========================================================================
    print_section("FRAGE 3: PER-SEMESTER NORMIERTE PRÜFUNGSDIFFERENZ (G1, A vs C)")
    
    print("  Lade pruefungen.csv A und C...")
    pruef_a = pd.read_csv(BASE_DIR / "pruefungen.csv")
    pruef_c = pd.read_csv(BASE_DIR / "universe_C" / "pruefungen.csv")
    
    g1_ac_ids_set = set(g1_ac["studierenden_id"])
    
    # Pro Student pro Semester: Anzahl Prüfungen und bestandene
    def per_sem_stats(df, ids):
        sub = df[df["studierenden_id"].isin(ids)]
        # Fachsemester approximieren
        sem_order = sorted(sub["semester_id"].unique())
        sem_map = {s: i+1 for i, s in enumerate(sem_order)}
        
        # Per student, per semester
        grouped = sub.groupby(["studierenden_id", "semester_id"]).agg(
            n_pruef=("modul_id", "count"),
            n_best=("bestanden", "sum")
        ).reset_index()
        
        # Map to fachsemester per student
        student_sem = grouped.groupby("studierenden_id")["semester_id"].rank(method="dense").astype(int)
        grouped["fachsem"] = student_sem
        
        return grouped[["studierenden_id", "fachsem", "n_pruef", "n_best"]]
    
    g1_a_sem = per_sem_stats(pruef_a, g1_ac_ids_set)
    g1_c_sem = per_sem_stats(pruef_c, g1_ac_ids_set)
    
    # Merge by student + fachsemester
    merged_sem = g1_a_sem.merge(g1_c_sem, on=["studierenden_id", "fachsem"],
                                 suffixes=("_A", "_C"), how="outer").fillna(0)
    merged_sem["delta_pruef"] = merged_sem["n_pruef_C"] - merged_sem["n_pruef_A"]
    merged_sem["delta_best"] = merged_sem["n_best_C"] - merged_sem["n_best_A"]
    
    sem_summary = merged_sem.groupby("fachsem").agg(
        mean_delta_pruef=("delta_pruef", "mean"),
        mean_delta_best=("delta_best", "mean"),
        n_students=("studierenden_id", "nunique")
    )
    
    print(f"\n  G1-Geschädigte: Delta Prüfungen pro Semester (C - A, >0 = mehr in C)")
    print(f"  {'Fachsem':>8} | {'D_ Prüf (mean)':>14} | {'D_ Bestanden':>12} | {'n_Studis':>8}")
    print("  " + "-" * 52)
    for sem in range(1, min(13, len(sem_summary)+1)):
        if sem in sem_summary.index:
            r = sem_summary.loc[sem]
            print(f"  {sem:>8} | {r['mean_delta_pruef']:>+14.2f} | {r['mean_delta_best']:>+12.2f} | {r['n_students']:>8.0f}")
    
    # =========================================================================
    # FRAGE 4: Overload-Cap Analyse
    # =========================================================================
    print_section("FRAGE 4: OVERLOAD-CAP ANALYSE (Wie oft wird der Deckel erreicht?)")
    
    # Der Deckel in der Simulation ist 0.15 auf die overload_penalty.
    # hidden_overload ist die Stundenzahl ÜBER dem verfügbaren Zeitkonto.
    # Die Penalty wird berechnet als: min(overload / available_time, 0.15)
    # Wenn overload = 0.15 * available_time, ist der Deckel erreicht.
    # Wir können das nicht direkt berechnen, da available_time nicht geloggt wird.
    # Aber wir können die Verteilung von hidden_overload analysieren.
    
    g1_pruef = pruef_a[pruef_a["studierenden_id"].isin(g1_ac_ids_set)]
    g1_overload_pos = g1_pruef[g1_pruef["hidden_overload"] > 0]
    
    print(f"  G1-Geschädigte Prüfungen mit overload > 0: {len(g1_overload_pos)} von {len(g1_pruef)} ({len(g1_overload_pos)/len(g1_pruef)*100:.1f}%)")
    print(f"\n  Verteilung hidden_overload (nur >0):")
    for pct in [25, 50, 75, 90, 95, 99]:
        val = g1_overload_pos["hidden_overload"].quantile(pct/100)
        print(f"    P{pct}: {val:.1f} h")
    
    # Abschätzung: Bei ~500h verfügbarer Zeit wäre der Deckel bei 0.15*500 = 75h
    # Bei ~600h wäre es bei 90h, bei ~400h bei 60h
    # Also: overload > 60-90h -> Deckel wahrscheinlich erreicht
    thresholds = [30, 45, 60, 75, 90]
    for t in thresholds:
        n = (g1_overload_pos["hidden_overload"] >= t).sum()
        print(f"    overload >= {t}h: {n} ({n/len(g1_overload_pos)*100:.1f}% der Overload-Prüfungen)")
    
    print(f"\n  HINWEIS: Der exakte Penalty-Deckel kann nur aus dem Code abgeleitet werden.")
    print(f"  Wir empfehlen, in V3.1 auch 'penalty_reached_cap' als Boolean zu loggen.")
    
    # =========================================================================
    # FRAGE 5: MODULABWURF-NACHWEIS auf Prüfungsebene
    # =========================================================================
    print_section("FRAGE 5: MODULABWURF-NACHWEIS AUF PRÜFUNGSEBENE")
    
    print("  Methode: Für jeden G1-Geschädigten prüfen, ob es Module gibt, die in C")
    print("  in einem früheren Fachsemester geschrieben werden als in A.")
    print("  (Verschobene Module = Hinweis auf Modulabwurf)")
    print("  ACHTUNG: Rechts-zensierte Module (in A nie geschrieben wegen Dropout) werden separat gezählt.")
    
    # Finde für jeden G1-Studierenden die Module und ihre ersten Fachsemester
    def first_attempt_per_module(df, ids):
        sub = df[df["studierenden_id"].isin(ids)]
        # Fachsemester per student
        sub = sub.copy()
        sub["fachsem"] = sub.groupby("studierenden_id")["semester_id"].rank(method="dense").astype(int)
        first = sub.groupby(["studierenden_id", "modul_id"])["fachsem"].min().reset_index()
        first.columns = ["studierenden_id", "modul_id", "first_fachsem"]
        return first
    
    first_a = first_attempt_per_module(pruef_a, g1_ac_ids_set)
    first_c = first_attempt_per_module(pruef_c, g1_ac_ids_set)
    
    # Merge: Module die in beiden Welten vorkommen
    merged_mod = first_a.merge(first_c, on=["studierenden_id", "modul_id"],
                                suffixes=("_A", "_C"), how="outer")
    
    # Kategorie 1: Modul in C früher als in A -> Verzögerung in A
    both = merged_mod.dropna(subset=["first_fachsem_A", "first_fachsem_C"])
    delayed_in_a = both[both["first_fachsem_A"] > both["first_fachsem_C"]]
    earlier_in_a = both[both["first_fachsem_A"] < both["first_fachsem_C"]]
    same = both[both["first_fachsem_A"] == both["first_fachsem_C"]]
    
    # Kategorie 2: Modul nur in C (nie in A versucht -> rechts-zensiert)
    only_c = merged_mod[merged_mod["first_fachsem_A"].isna() & merged_mod["first_fachsem_C"].notna()]
    # Kategorie 3: Modul nur in A (nie in C versucht)
    only_a = merged_mod[merged_mod["first_fachsem_C"].isna() & merged_mod["first_fachsem_A"].notna()]
    
    n_g1 = len(g1_ac_ids_set)
    print(f"\n  Ergebnisse (n={n_g1} G1-Geschädigte):")
    print(f"    Module in beiden Welten versucht:          {len(both)} ({both['studierenden_id'].nunique()} Studis)")
    print(f"      davon in A VERZÖGERT (später als in C):  {len(delayed_in_a)} ({delayed_in_a['studierenden_id'].nunique()} Studis)")
    print(f"      davon in A FRÜHER (früher als in C):     {len(earlier_in_a)} ({earlier_in_a['studierenden_id'].nunique()} Studis)")
    print(f"      davon GLEICHZEITIG:                      {len(same)}")
    print(f"    Module NUR in C versucht (rechts-zensiert): {len(only_c)} ({only_c['studierenden_id'].nunique()} Studis)")
    print(f"    Module NUR in A versucht:                   {len(only_a)} ({only_a['studierenden_id'].nunique()} Studis)")
    
    # Delay-Statistik
    delayed_in_a_delay = delayed_in_a["first_fachsem_A"] - delayed_in_a["first_fachsem_C"]
    print(f"\n  Verzögerung (Fachsemester, A minus C) bei verzögerten Modulen:")
    print(f"    Median: {delayed_in_a_delay.median():.1f} Semester")
    print(f"    Mean:   {delayed_in_a_delay.mean():.1f} Semester")
    print(f"    Max:    {delayed_in_a_delay.max():.0f} Semester")
    
    # Pro Student: Wie viele Module sind verzögert?
    delay_per_studi = delayed_in_a.groupby("studierenden_id").size()
    cens_per_studi = only_c.groupby("studierenden_id").size()
    
    print(f"\n  Pro Student:")
    print(f"    Verzögerte Module pro Student (Median): {delay_per_studi.median():.1f}")
    print(f"    Rechts-zensierte Module pro Student (Median): {cens_per_studi.median():.1f}")
    print(f"    Studis mit mindestens 1 verzögertem Modul: {len(delay_per_studi)} von {n_g1} ({len(delay_per_studi)/n_g1*100:.1f}%)")
    print(f"    Studis mit mindestens 1 rechts-zensiertem Modul: {len(cens_per_studi)} von {n_g1} ({len(cens_per_studi)/n_g1*100:.1f}%)")
    
    # =========================================================================
    # FRAGE 6: CP-RÜCKSTAND PROFIL DER SUBGRUPPEN
    # =========================================================================
    print_section("FRAGE 6: CP-RÜCKSTAND PROFIL (G0, G1, G2)")
    
    # Berechne CP-Rückstand pro Student pro Semester
    # CP = bestandene_module * 5 ECTS; Soll = fachsem * 30 ECTS
    def cp_profile(df, ids, label):
        sub = df[df["studierenden_id"].isin(ids)].copy()
        sub["fachsem"] = sub.groupby("studierenden_id")["semester_id"].rank(method="dense").astype(int)
        
        # Kumulative bestandene Module pro Student
        sub_best = sub[sub["bestanden"]].groupby(["studierenden_id", "fachsem"]).size().reset_index(name="n_bestanden_sem")
        
        # Alle Student-Semester Kombinationen
        all_sems = sub.groupby(["studierenden_id", "fachsem"]).size().reset_index(name="n_pruef_sem")
        all_sems = all_sems.merge(sub_best, on=["studierenden_id", "fachsem"], how="left").fillna(0)
        
        # Kumulative bestandene
        all_sems = all_sems.sort_values(["studierenden_id", "fachsem"])
        all_sems["kum_bestanden"] = all_sems.groupby("studierenden_id")["n_bestanden_sem"].cumsum()
        all_sems["cp_ist"] = all_sems["kum_bestanden"] * 5  # 5 ECTS pro Modul
        all_sems["cp_soll"] = all_sems["fachsem"] * 30       # 30 ECTS pro Semester
        all_sems["cp_rueckstand"] = all_sems["cp_soll"] - all_sems["cp_ist"]
        
        # Delta-Rückstand (Veränderung zum Vorsemester)
        all_sems["cp_rueckstand_prev"] = all_sems.groupby("studierenden_id")["cp_rueckstand"].shift(1)
        all_sems["delta_rueckstand"] = all_sems["cp_rueckstand"] - all_sems["cp_rueckstand_prev"]
        
        return all_sems
    
    g2_ac_ids = set(g2_ac["studierenden_id"])
    g0_ids = set(uni_a["studierenden_id"]) - g1_ac_ids_set - g2_ac_ids
    # Sample G0 for efficiency
    g0_sample = set(list(g0_ids)[:5000])
    
    cp_g1 = cp_profile(pruef_a, g1_ac_ids_set, "G1")
    cp_g2 = cp_profile(pruef_a, g2_ac_ids, "G2")
    cp_g0 = cp_profile(pruef_a, g0_sample, "G0")
    
    print(f"\n  CP-Rückstand pro Fachsemester (Universum A):")
    print(f"  {'Fachsem':>8} | {'G0 Mean':>10} | {'G1 Mean':>10} | {'G2 Mean':>10} | {'G0 D_':>8} | {'G1 D_':>8} | {'G2 D_':>8}")
    print("  " + "-" * 76)
    
    for sem in range(1, 11):
        vals = []
        for df, label in [(cp_g0, "G0"), (cp_g1, "G1"), (cp_g2, "G2")]:
            sem_data = df[df["fachsem"] == sem]
            if len(sem_data) > 0:
                vals.append((sem_data["cp_rueckstand"].mean(), sem_data["delta_rueckstand"].mean()))
            else:
                vals.append((np.nan, np.nan))
        
        g0_r, g0_d = vals[0]
        g1_r, g1_d = vals[1]
        g2_r, g2_d = vals[2]
        
        print(f"  {sem:>8} | {g0_r:>10.1f} | {g1_r:>10.1f} | {g2_r:>10.1f} | {g0_d:>+8.1f} | {g1_d:>+8.1f} | {g2_d:>+8.1f}")
    
    # Letztes aktives Semester: Endstand CP-Rückstand
    last_sem_g1 = cp_g1.sort_values("fachsem").groupby("studierenden_id").last()
    last_sem_g2 = cp_g2.sort_values("fachsem").groupby("studierenden_id").last()
    last_sem_g0 = cp_g0.sort_values("fachsem").groupby("studierenden_id").last()
    
    print(f"\n  Endstand CP-Rückstand (letztes aktives Semester):")
    print(f"    G0 (Neutrale): Median {last_sem_g0['cp_rueckstand'].median():.0f} CP, Mean {last_sem_g0['cp_rueckstand'].mean():.1f} CP")
    print(f"    G1 (Geschädigte): Median {last_sem_g1['cp_rueckstand'].median():.0f} CP, Mean {last_sem_g1['cp_rueckstand'].mean():.1f} CP")
    print(f"    G2 (Gerettete): Median {last_sem_g2['cp_rueckstand'].median():.0f} CP, Mean {last_sem_g2['cp_rueckstand'].mean():.1f} CP")
    
    print("\n\n=== VERTIEFTE ANALYSE ABGESCHLOSSEN ===")

if __name__ == "__main__":
    main()
