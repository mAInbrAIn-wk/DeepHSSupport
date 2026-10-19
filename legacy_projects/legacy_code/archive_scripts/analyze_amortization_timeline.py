import os
import pandas as pd
import numpy as np
import json
from pathlib import Path

def main():
    base_dir = Path(r"c:\GitHub_public\Abschlussprojekt\output_dl")
    dir_A = base_dir
    dir_C = base_dir / "universe_C"
    output_dir = base_dir / "analysis"
    output_dir.mkdir(exist_ok=True, parents=True)

    print("================================================================================")
    print("1. TIMELINE & OVERLOAD-FALLE ANALLYSEN (G1 OPFER)")
    print("================================================================================")

    # Daten laden
    stud_A = pd.read_csv(dir_A / "studierende.csv")
    abschl_A = pd.read_csv(dir_A / "abschluesse.csv")
    abschl_C = pd.read_csv(dir_C / "abschluesse.csv")
    pruef_A = pd.read_csv(dir_A / "pruefungen.csv")

    dropout_statuses = ["abgebrochen", "exmatrikuliert", "zeitueberschreitung"]
    
    last_A = abschl_A.sort_values("abschluss_semester_id").groupby("studierenden_id").last().reset_index()
    last_C = abschl_C.sort_values("abschluss_semester_id").groupby("studierenden_id").last().reset_index()

    stud_A["dropout_A"] = stud_A["studierenden_id"].map(last_A.set_index("studierenden_id")["status"]).isin(dropout_statuses)
    stud_A["dropout_C"] = stud_A["studierenden_id"].map(last_C.set_index("studierenden_id")["status"]).isin(dropout_statuses)
    
    stud_A["sem_A"] = stud_A["studierenden_id"].map(last_A.set_index("studierenden_id")["abschluss_semester_id"])
    stud_A["sem_C"] = stud_A["studierenden_id"].map(last_C.set_index("studierenden_id")["abschluss_semester_id"])

    # G1: Dropout in A (mit Support), Überlebt in C (ohne fachl. Support)
    # G2: Überlebt in A (mit Support), Dropout in C (ohne fachl. Support)
    g1_mask = (stud_A["dropout_A"] == True) & (stud_A["dropout_C"] == False)
    g2_mask = (stud_A["dropout_A"] == False) & (stud_A["dropout_C"] == True)

    stud_A["gruppe"] = "Andere"
    stud_A.loc[g1_mask, "gruppe"] = "G1_Geschaedigte"
    stud_A.loc[g2_mask, "gruppe"] = "G2_Gerettete"

    g1_ids = set(stud_A[stud_A["gruppe"] == "G1_Geschaedigte"]["studierenden_id"])
    g2_ids = set(stud_A[stud_A["gruppe"] == "G2_Gerettete"]["studierenden_id"])

    # Fachliche Support-Teilnahmen in Universum A
    pruef_A_fach = pruef_A[pruef_A["support_genutzt"].astype(str) == "True"].copy()
    
    # Semester der Support-Nutzung vs. Semester des Abruchs für G1
    g1_supp = pruef_A_fach[pruef_A_fach["studierenden_id"].isin(g1_ids)].copy()
    g1_supp = g1_supp.merge(stud_A[["studierenden_id", "sem_A"]], on="studierenden_id")
    
    # Extract integer semester numbers
    g1_supp["sem_A_num"] = g1_supp["sem_A"].astype(str).str.extract(r'(\d+)')[0].astype(int)
    g1_supp["sem_id_num"] = g1_supp["semester_id"].astype(str).str.extract(r'(\d+)')[0].astype(int)
    g1_supp["sem_diff"] = g1_supp["sem_A_num"] - g1_supp["sem_id_num"]
    
    print("\n--- G1 (Geschädigte, N=1.064): Zeitpunkt der Support-Nutzung relativ zum Abbruch-Semester ---")
    print("sem_diff = 0 bedeutet: Abbruch erfolgte im EXAKTEN Semester der Support-Nutzung!")
    print(g1_supp["sem_diff"].value_counts().sort_index().to_string())

    # Zeitkonto & Overload-Check
    # Verfügbare Zeit pro Semester = max(0, 900 - erwerbstaetigkeit_std * 16)
    # Workload pro Modul ~ 150h, Support kosten = 30h
    print("\n--- Overload-Check beim Support-Kauf ---")
    
    # Berechne den geplanten Workload pro Studi und Semester in A
    sem_workload = pruef_A.groupby(["studierenden_id", "semester_id"]).size().reset_index(name="n_pruefungen")
    sem_workload["workload_h"] = sem_workload["n_pruefungen"] * 150
    
    # Fachlicher Support Stunden pro Semester
    supp_h = pruef_A_fach.groupby(["studierenden_id", "semester_id"]).size().reset_index(name="n_support")
    supp_h["support_h"] = supp_h["n_support"] * 30
    
    merged_sem = sem_workload.merge(supp_h, on=["studierenden_id", "semester_id"], how="left").fillna(0)
    merged_sem = merged_sem.merge(stud_A[["studierenden_id", "erwerbstaetigkeit_std", "gruppe"]], on="studierenden_id")
    
    # 900h Semester-Budget
    merged_sem["verfuegbare_zeit"] = np.maximum(0, 900 - merged_sem["erwerbstaetigkeit_std"] * 16)
    merged_sem["total_workload"] = merged_sem["workload_h"] + merged_sem["support_h"]
    merged_sem["is_overload"] = merged_sem["total_workload"] > merged_sem["verfuegbare_zeit"]
    
    # Betrachte nur Semester mit Support-Nutzung
    supp_sem_only = merged_sem[merged_sem["n_support"] > 0]
    
    overload_by_group = supp_sem_only.groupby("gruppe")["is_overload"].agg(["count", "sum", "mean"]).rename(columns={"count": "n_support_semesters", "sum": "n_overload_semesters", "mean": "pct_in_overload"})
    print(overload_by_group.to_string())

if __name__ == "__main__":
    main()
