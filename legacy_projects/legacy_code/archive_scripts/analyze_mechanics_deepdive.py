import os
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    base_dir = Path(r"c:\GitHub_public\Abschlussprojekt\output_dl")
    dir_A = base_dir
    dir_C = base_dir / "universe_C"

    print("================================================================================")
    print("DEEP DIVE SIMULATIONS-MECHANIK & G1-OPFER ANALYSE")
    print("================================================================================")

    # 1. Kohorten definieren
    stud_A = pd.read_csv(dir_A / "studierende.csv")
    abschl_A = pd.read_csv(dir_A / "abschluesse.csv")
    abschl_C = pd.read_csv(dir_C / "abschluesse.csv")
    pruef_A = pd.read_csv(dir_A / "pruefungen.csv")
    pruef_C = pd.read_csv(dir_C / "pruefungen.csv")

    dropout_statuses = ["abgebrochen", "exmatrikuliert", "zeitueberschreitung"]
    
    last_A = abschl_A.sort_values("abschluss_semester_id").groupby("studierenden_id").last().reset_index()
    last_C = abschl_C.sort_values("abschluss_semester_id").groupby("studierenden_id").last().reset_index()

    stud_A["status_A"] = stud_A["studierenden_id"].map(last_A.set_index("studierenden_id")["status"])
    stud_A["status_C"] = stud_A["studierenden_id"].map(last_C.set_index("studierenden_id")["status"])

    stud_A["dropout_A"] = stud_A["status_A"].isin(dropout_statuses)
    stud_A["dropout_C"] = stud_A["status_C"].isin(dropout_statuses)

    g1_mask = (stud_A["dropout_A"] == True) & (stud_A["dropout_C"] == False)
    g1_studis = stud_A[g1_mask].copy()

    # --- Q1: Genauer Status der G1 Opfer ---
    print("\n1. G1-Opfer Status-Verteilung in Universum A:")
    print(g1_studis["status_A"].value_counts().to_string())

    # --- Q2: Die 1.4% (15 Studierende) ohne Modul-Abwurf ---
    count_A = pruef_A.groupby(["studierenden_id", "semester_id"]).size().reset_index(name="n_pruef_A")
    count_C = pruef_C.groupby(["studierenden_id", "semester_id"]).size().reset_index(name="n_pruef_C")
    
    merged_sem = count_A.merge(count_C, on=["studierenden_id", "semester_id"], how="outer").fillna(0)
    merged_sem["modul_abgeworfen"] = merged_sem["n_pruef_A"] < merged_sem["n_pruef_C"]
    
    lifetime_abwurf = merged_sem.groupby("studierenden_id")["modul_abgeworfen"].any().reset_index(name="has_abwurf")
    g1_studis = g1_studis.merge(lifetime_abwurf, on="studierenden_id")
    
    g1_no_abwurf = g1_studis[g1_studis["has_abwurf"] == False]
    print(f"\n2. G1-Opfer OHNE Modul-Abwurf (N={len(g1_no_abwurf)}):")
    print(g1_no_abwurf[["studierenden_id", "status_A", "erwerbstaetigkeit_std", "hzb_note"]].head(10).to_string())

    # --- Q3: Kaskaden-Effekt (Wie schaukeln sich 8 Module Unterschied auf?) ---
    # Untersuche die zeitliche Abfolge der Modulprüfungen pro Semester in A vs C für G1
    print("\n3. Kaskaden-Analyse der Abwürfe pro Semester bei G1-Opfern:")
    g1_sem_diff = merged_sem[merged_sem["studierenden_id"].isin(g1_studis["studierenden_id"])].copy()
    g1_sem_diff["diff_C_minus_A"] = g1_sem_diff["n_pruef_C"] - g1_sem_diff["n_pruef_A"]
    
    sem_stats = g1_sem_diff.groupby("semester_id")["diff_C_minus_A"].agg(["count", "mean", "sum"]).rename(columns={"mean": "avg_pruef_diff_C_minus_A"})
    print(sem_stats.head(10).to_string())

if __name__ == "__main__":
    main()
