import os
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    base_dir = Path(r"c:\GitHub_public\Abschlussprojekt\output_dl")
    dir_A = base_dir
    dir_C = base_dir / "universe_C"
    output_dir = base_dir / "analysis"
    output_dir.mkdir(exist_ok=True, parents=True)

    print("================================================================================")
    print("1. LIFETIME MODUL-ABWURF & CP-DYNAMIK ANALLYSEN (1:1 WELTEN-VERGLEICH A vs C)")
    print("================================================================================")

    # 1. Kohorten definieren
    stud_A = pd.read_csv(dir_A / "studierende.csv")
    abschl_A = pd.read_csv(dir_A / "abschluesse.csv")
    abschl_C = pd.read_csv(dir_C / "abschluesse.csv")

    dropout_statuses = ["abgebrochen", "exmatrikuliert", "zeitueberschreitung"]
    
    last_A = abschl_A.sort_values("abschluss_semester_id").groupby("studierenden_id")["status"].last()
    last_C = abschl_C.sort_values("abschluss_semester_id").groupby("studierenden_id")["status"].last()

    stud_A["dropout_A"] = stud_A["studierenden_id"].map(last_A).isin(dropout_statuses)
    stud_A["dropout_C"] = stud_A["studierenden_id"].map(last_C).isin(dropout_statuses)

    g1_mask = (stud_A["dropout_A"] == True) & (stud_A["dropout_C"] == False)
    g2_mask = (stud_A["dropout_A"] == False) & (stud_A["dropout_C"] == True)

    stud_A["gruppe"] = "G0_Andere"
    stud_A.loc[g1_mask, "gruppe"] = "G1_Geschaedigte"
    stud_A.loc[g2_mask, "gruppe"] = "G2_Gerettete"

    print("\nKohorten-Verteilung:")
    print(stud_A["gruppe"].value_counts().to_string())

    # 2. Prüfungs-Aktivitäten laden und nach (studierenden_id, semester_id) aggregieren
    pruef_A = pd.read_csv(dir_A / "pruefungen.csv")
    pruef_C = pd.read_csv(dir_C / "pruefungen.csv")

    # Anzahl geschriebener Prüfungen pro Studi & Semester
    count_A = pruef_A.groupby(["studierenden_id", "semester_id"]).size().reset_index(name="n_pruef_A")
    count_C = pruef_C.groupby(["studierenden_id", "semester_id"]).size().reset_index(name="n_pruef_C")

    # Bestandene CP pro Studi & Semester
    # Annahme: Jedes Modul bringt ~5 CP (oder genauer: count der bestandenen Prüfungen)
    cp_A = pruef_A[pruef_A["bestanden"] == True].groupby(["studierenden_id", "semester_id"]).size().reset_index(name="n_passed_A")
    cp_C = pruef_C[pruef_C["bestanden"] == True].groupby(["studierenden_id", "semester_id"]).size().reset_index(name="n_passed_C")

    # Merge über Semester & Studierende
    merged_sem = count_A.merge(count_C, on=["studierenden_id", "semester_id"], how="outer").fillna(0)
    merged_sem = merged_sem.merge(cp_A, on=["studierenden_id", "semester_id"], how="outer").fillna(0)
    merged_sem = merged_sem.merge(cp_C, on=["studierenden_id", "semester_id"], how="outer").fillna(0)

    # Identifiziere Modul-Abwurf: In A wurden WENIGER Prüfungen geschrieben als in C
    merged_sem["modul_abgeworfen"] = merged_sem["n_pruef_A"] < merged_sem["n_pruef_C"]
    merged_sem["n_abgeworfen"] = np.maximum(0, merged_sem["n_pruef_C"] - merged_sem["n_pruef_A"])

    # Zuordnung der Kohorte
    merged_sem = merged_sem.merge(stud_A[["studierenden_id", "gruppe"]], on="studierenden_id")

    # Aggregation pro Student über die gesamte Lebensdauer
    studi_lifetime = merged_sem.groupby("studierenden_id").agg(
        gruppe=("gruppe", "first"),
        total_semesters=("semester_id", "nunique"),
        has_modul_abwurf=("modul_abgeworfen", "any"),
        total_abgeworfene_module=("n_abgeworfen", "sum"),
        total_passed_A=("n_passed_A", "sum"),
        total_passed_C=("n_passed_C", "sum")
    ).reset_index()

    studi_lifetime["passed_diff_C_minus_A"] = studi_lifetime["total_passed_C"] - studi_lifetime["total_passed_A"]

    print("\n--------------------------------------------------------------------------------")
    print("LIFETIME MODUL-ABWURF EVALUATION NACH GRUPPEN")
    print("--------------------------------------------------------------------------------")
    
    summary = studi_lifetime.groupby("gruppe").agg(
        n_studis=("studierenden_id", "count"),
        n_studis_with_abwurf=("has_modul_abwurf", "sum"),
        pct_studis_with_abwurf=("has_modul_abwurf", "mean"),
        avg_abgeworfene_module=("total_abgeworfene_module", "mean"),
        avg_passed_diff_C_minus_A=("passed_diff_C_minus_A", "mean")
    )
    
    summary["pct_studis_with_abwurf"] = summary["pct_studis_with_abwurf"] * 100
    print(summary.to_string())

    # 3. Spezieller Check: Fachliche Support-Nutzung in den Abwurf-Semestern
    pruef_A_fach = pruef_A[pruef_A["support_genutzt"].astype(str) == "True"]
    supp_sem_set = set(zip(pruef_A_fach["studierenden_id"], pruef_A_fach["semester_id"]))

    merged_sem["hatte_fach_support"] = merged_sem.apply(
        lambda row: (row["studierenden_id"], row["semester_id"]) in supp_sem_set, axis=1
    )

    print("\n--------------------------------------------------------------------------------")
    print("MODUL-ABWURF IN SEMESTERN MIT FACHLICHEM SUPPORT")
    print("--------------------------------------------------------------------------------")
    
    supp_sems_only = merged_sem[merged_sem["hatte_fach_support"] == True]
    supp_summary = supp_sems_only.groupby("gruppe").agg(
        n_support_sems=("semester_id", "count"),
        n_abwurf_sems=("modul_abgeworfen", "sum"),
        pct_abwurf_in_support_sem=("modul_abgeworfen", "mean")
    )
    supp_summary["pct_abwurf_in_support_sem"] = supp_summary["pct_abwurf_in_support_sem"] * 100
    print(supp_summary.to_string())

if __name__ == "__main__":
    main()
