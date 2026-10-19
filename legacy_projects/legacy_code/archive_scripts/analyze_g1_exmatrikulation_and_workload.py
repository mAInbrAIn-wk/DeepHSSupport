import os
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    base_dir = Path(r"c:\GitHub_public\Abschlussprojekt\output_dl")
    dir_A = base_dir
    dir_C = base_dir / "universe_C"

    print("================================================================================")
    print("EMPIRISCHER VERGLEICH: ERWERBSTÄTIGKEIT & EXMATRIKULATIONEN (G1 vs G2 vs G0)")
    print("================================================================================")

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
    g2_mask = (stud_A["dropout_A"] == False) & (stud_A["dropout_C"] == True)

    stud_A["gruppe"] = "G0_Andere"
    stud_A.loc[g1_mask, "gruppe"] = "G1_Geschaedigte"
    stud_A.loc[g2_mask, "gruppe"] = "G2_Gerettete"

    # --- 1. Erwerbstätigkeit-Vergleich ---
    print("\n1. Durchschnittliche Erwerbstätigkeit (Wochenstunden) nach Gruppe:")
    erwerb_stats = stud_A.groupby("gruppe")["erwerbstaetigkeit_std"].agg(["count", "mean", "median", "std"])
    print(erwerb_stats.to_string())

    # --- 2. Exmatrikulations-Analyse (73 G1-Studierende) ---
    exmat_g1_ids = set(stud_A[(stud_A["gruppe"] == "G1_Geschaedigte") & (stud_A["status_A"] == "exmatrikuliert")]["studierenden_id"])
    print(f"\n2. Exmatrikulations-Mechanismus bei den {len(exmat_g1_ids)} exmatrikulierten G1-Opfern:")

    # Pruefungen dieser 73 Studierenden in A und C
    p_exmat_A = pruef_A[pruef_A["studierenden_id"].isin(exmat_g1_ids)]
    p_exmat_C = pruef_C[pruef_C["studierenden_id"].isin(exmat_g1_ids)]

    # Finde 3. Versuche
    v3_A = p_exmat_A[p_exmat_A["versuch"] == 3]
    v3_C = p_exmat_C[p_exmat_C["versuch"] == 3]

    print(f"3. Versuche geschrieben in A (mit Support): {len(v3_A)} | Durchgefallen (5.0): {(v3_A['bestanden'] == False).sum()}")
    print(f"3. Versuche geschrieben in C (ohne Support): {len(v3_C)} | Durchgefallen (5.0): {(v3_C['bestanden'] == False).sum()}")

    # Abgleich der 3. Versuche im selben Modul für exmatrikulierte G1 Studierende
    merged_v3 = v3_A.merge(v3_C, on=["studierenden_id", "modul_id"], suffixes=("_A", "_C"))
    merged_v3["exmat_durch_overload"] = (merged_v3["bestanden_A"] == False) & (merged_v3["bestanden_C"] == True)
    
    print(f"Modulprüfungen im 3. Versuch, die in A DURCHGEFALLEN (5.0) sind, aber in C BESTANDEN wurden: {merged_v3['exmat_durch_overload'].sum()}")

if __name__ == "__main__":
    main()
