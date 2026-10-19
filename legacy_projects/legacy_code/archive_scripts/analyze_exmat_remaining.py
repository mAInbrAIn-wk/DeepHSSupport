import os
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    base_dir = Path(r"c:\GitHub_public\Abschlussprojekt\output_dl")
    dir_A = base_dir
    dir_C = base_dir / "universe_C"

    stud_A = pd.read_csv(dir_A / "studierende.csv")
    abschl_A = pd.read_csv(dir_A / "abschluesse.csv")
    abschl_C = pd.read_csv(dir_C / "abschluesse.csv")
    pruef_A = pd.read_csv(dir_A / "pruefungen.csv")
    pruef_C = pd.read_csv(dir_C / "pruefungen.csv")

    dropout_statuses = ["abgebrochen", "exmatrikuliert", "zeitueberschreitung"]
    last_A = abschl_A.sort_values("abschluss_semester_id").groupby("studierenden_id")["status"].last()
    last_C = abschl_C.sort_values("abschluss_semester_id").groupby("studierenden_id")["status"].last()

    stud_A["status_A"] = stud_A["studierenden_id"].map(last_A)
    stud_A["status_C"] = stud_A["studierenden_id"].map(last_C)
    stud_A["dropout_A"] = stud_A["status_A"].isin(dropout_statuses)
    stud_A["dropout_C"] = stud_A["status_C"].isin(dropout_statuses)

    g1_mask = (stud_A["dropout_A"] == True) & (stud_A["dropout_C"] == False)
    g1_exmat_ids = set(stud_A[g1_mask & (stud_A["status_A"] == "exmatrikuliert")]["studierenden_id"])

    print(f"Gesamt exmatrikulierte G1 Studierende: {len(g1_exmat_ids)}")

    # 3. Versuche in A, die durchgefallen sind (5.0)
    v3_A_fails = pruef_A[(pruef_A["studierenden_id"].isin(g1_exmat_ids)) & (pruef_A["versuch"] == 3) & (pruef_A["bestanden"] == False)]
    exmat_studis_with_v3_fail_in_A = set(v3_A_fails["studierenden_id"])
    
    print(f"G1 Studierende mit 3. Versuch Fehlschlag in A: {len(exmat_studis_with_v3_fail_in_A)}")

    # Finde die verbleibenden Studierenden
    other_g1_exmat = g1_exmat_ids - exmat_studis_with_v3_fail_in_A
    print(f"Verbleibende G1 exmatrikulierte Studierende: {len(other_g1_exmat)}")

    # Analyse was mit den 73 exmatrikulierten Studierenden in C passiert ist:
    # Wie haben die 73 G1-Exmatrikulierten in A ihre exmatrikulierten Module in C abgeschlossen?
    # Finde für jeden 3. Versuch Fail in A die Versuche in C für dasselbe Modul
    rows = []
    for _, row_a in v3_A_fails.iterrows():
        s_id = row_a["studierenden_id"]
        m_id = row_a["modul_id"]
        
        # Alle Versuche dieses Studis für dieses Modul in C
        p_c = pruef_C[(pruef_C["studierenden_id"] == s_id) & (pruef_C["modul_id"] == m_id)]
        if p_c.empty:
            outcome_c = "Modul in C nie belegt"
        else:
            best_c = p_c[p_c["bestanden"] == True]
            if not best_c.empty:
                versuch_c = best_c.iloc[0]["versuch"]
                note_c = best_c.iloc[0]["note"]
                outcome_c = f"Bestanden in Versuch {versuch_c} (Note {note_c})"
            else:
                outcome_c = "In C durchgefallen"
        
        rows.append({"studierenden_id": s_id, "modul_id": m_id, "outcome_in_C": outcome_c})

    df_outcomes = pd.DataFrame(rows)
    print("\n--- Wie haben die in A durchgefallenen 3. Versuche in Welt C geendet? ---")
    print(df_outcomes["outcome_in_C"].value_counts().to_string())

if __name__ == "__main__":
    main()
