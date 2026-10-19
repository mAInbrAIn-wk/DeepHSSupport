import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, export_text

def main():
    base_dir = Path(r"c:\GitHub_public\Abschlussprojekt\output_dl")
    dir_A = base_dir
    dir_C = base_dir / "universe_C"
    output_dir = base_dir / "analysis"
    output_dir.mkdir(exist_ok=True, parents=True)

    print("================================================================================")
    print("ANALYSE DER OVERLOAD-VICTIMS (GRUPPE 1: DROPOUT IN A, GERETTET IN C)")
    print("================================================================================")

    # 1. Daten laden
    stud_A = pd.read_csv(dir_A / "studierende.csv")
    abschl_A = pd.read_csv(dir_A / "abschluesse.csv")
    pruef_A = pd.read_csv(dir_A / "pruefungen.csv")

    stud_C = pd.read_csv(dir_C / "studierende.csv")
    abschl_C = pd.read_csv(dir_C / "abschluesse.csv")
    pruef_C = pd.read_csv(dir_C / "pruefungen.csv")

    # Status bestimmen (breite Definition: abgebrochen, exmatrikuliert, zeitueberschreitung)
    dropout_statuses = ["abgebrochen", "exmatrikuliert", "zeitueberschreitung"]
    
    last_status_A = abschl_A.sort_values("abschluss_semester_id").groupby("studierenden_id")["status"].last()
    last_status_C = abschl_C.sort_values("abschluss_semester_id").groupby("studierenden_id")["status"].last()

    stud_A["dropout_A"] = stud_A["studierenden_id"].map(last_status_A).isin(dropout_statuses)
    stud_C["dropout_C"] = stud_C["studierenden_id"].map(last_status_C).isin(dropout_statuses)

    merged = stud_A.merge(stud_C[["studierenden_id", "dropout_C", "hidden_erwartete_note_final", "motivation_final", "soziale_integration_final"]], 
                          on="studierenden_id", suffixes=("_A", "_C"))

    # Gruppen definieren
    g1_mask = (merged["dropout_A"] == True) & (merged["dropout_C"] == False)  # Geschädigte (1.064)
    g2_mask = (merged["dropout_A"] == False) & (merged["dropout_C"] == True)  # Gerettete (1.340)
    g3_mask = (merged["dropout_A"] == False) & (merged["dropout_C"] == False) # Immer Erfolgreich (33.690)
    g4_mask = (merged["dropout_A"] == True) & (merged["dropout_C"] == True)   # Immer Dropout (13.906)

    merged["gruppe"] = "Andere"
    merged.loc[g1_mask, "gruppe"] = "G1_Geschaedigte"
    merged.loc[g2_mask, "gruppe"] = "G2_Gerettete"
    merged.loc[g3_mask, "gruppe"] = "G3_Immer_Erfolg"
    merged.loc[g4_mask, "gruppe"] = "G4_Immer_Dropout"

    counts = merged["gruppe"].value_counts()
    print("\n--- Gruppengrößen ---")
    for grp, cnt in counts.items():
        print(f"  {grp}: {cnt} Studierende ({cnt / len(merged):.2%})")

    # 2. Demografischer Profil-Vergleich
    print("\n================================================================================")
    print("DEMOGRAFISCHES UND INITIALES PROFIL PRO GRUPPE")
    print("================================================================================")
    
    cols_to_compare = [
        "erwerbstaetigkeit_std", 
        "hzb_note", 
        "motivation_initial", 
        "soziale_integration_initial",
        "hidden_erwartete_note_initial"
    ]
    
    profile_df = merged.groupby("gruppe")[cols_to_compare].agg(["mean", "median", "std"])
    print(profile_df.to_string())

    # Ergänzende Quantile für Erwerbstätigkeit
    print("\n--- Erwerbstätigkeit Quantile (Std/Woche) ---")
    for grp in ["G1_Geschaedigte", "G2_Gerettete", "G3_Immer_Erfolg", "G4_Immer_Dropout"]:
        sub = merged[merged["gruppe"] == grp]["erwerbstaetigkeit_std"]
        print(f"  {grp}: Mean={sub.mean():.2f}, Median={sub.median():.2f}, 75%={sub.quantile(0.75):.2f}, 90%={sub.quantile(0.90):.2f}")

    # 3. Noten, CP-Rückstand & Prüfungsverhalten
    print("\n================================================================================")
    print("NOTEN, SUPPORT-NUTZUNG UND PRÜFUNGSLEISTUNG IN UNIVERSUM A vs C")
    print("================================================================================")

    # Prüfungsdaten aggregieren pro Student
    pruef_A_fach_supp = pruef_A[pruef_A["support_genutzt"].astype(str) == "True"]
    supp_count_A = pruef_A_fach_supp.groupby("studierenden_id")["modul_id"].count()
    
    pruef_summary_A = pruef_A.groupby("studierenden_id").agg(
        n_pruefungen_A=("modul_id", "count"),
        n_bestanden_A=("bestanden", "sum"),
        avg_note_A=("note", "mean")
    )
    pruef_summary_C = pruef_C.groupby("studierenden_id").agg(
        n_pruefungen_C=("modul_id", "count"),
        n_bestanden_C=("bestanden", "sum"),
        avg_note_C=("note", "mean")
    )

    merged = merged.merge(pruef_summary_A, on="studierenden_id", how="left")
    merged = merged.merge(pruef_summary_C, on="studierenden_id", how="left")
    merged["n_fach_support_A"] = merged["studierenden_id"].map(supp_count_A).fillna(0)

    print("\n--- Support-Nutzung (Anzahl fachliche Support-Angebote in A) ---")
    print(merged.groupby("gruppe")["n_fach_support_A"].agg(["mean", "median", "max"]).to_string())

    print("\n--- Prüfungs-Anzahl & Bestehensraten in A vs C ---")
    merged["failed_A"] = merged["n_pruefungen_A"] - merged["n_bestanden_A"]
    merged["failed_C"] = merged["n_pruefungen_C"] - merged["n_bestanden_C"]
    
    perf_df = merged.groupby("gruppe")[["n_pruefungen_A", "n_pruefungen_C", "failed_A", "failed_C", "avg_note_A", "avg_note_C"]].mean()
    print(perf_df.to_string())

    # 4. Dynamische Hidden-Variables & Notenänderungen
    print("\n================================================================================")
    print("DYNAMISCHE HIDDEN-VARIABLES (MOTIVATION, SOZIALE INTEGRATION, ERWARTETE NOTE)")
    print("================================================================================")

    merged["delta_motivation_A"] = merged["motivation_final_A"] - merged["motivation_initial"]
    merged["delta_motivation_C"] = merged["motivation_final_C"] - merged["motivation_initial"]
    merged["delta_soz_int_A"] = merged["soziale_integration_final_A"] - merged["soziale_integration_initial"]
    merged["delta_soz_int_C"] = merged["soziale_integration_final_C"] - merged["soziale_integration_initial"]
    merged["delta_erwartete_note_A"] = merged["hidden_erwartete_note_final_A"] - merged["hidden_erwartete_note_initial"]
    merged["delta_erwartete_note_C"] = merged["hidden_erwartete_note_final_C"] - merged["hidden_erwartete_note_initial"]

    hidden_df = merged.groupby("gruppe")[[
        "delta_motivation_A", "delta_motivation_C",
        "delta_soz_int_A", "delta_soz_int_C",
        "delta_erwartete_note_A", "delta_erwartete_note_C"
    ]].mean()
    print(hidden_df.to_string())

    # Detaillierter Blick auf Prüfungs-Ebene für G1_Geschaedigte
    print("\n================================================================================")
    print("DETAILED PRÜFUNGS-VERGLEICH FÜR G1_GESCHÄDIGTE (A vs C)")
    print("================================================================================")
    
    g1_ids = set(merged[merged["gruppe"] == "G1_Geschaedigte"]["studierenden_id"])
    pA_g1 = pruef_A[pruef_A["studierenden_id"].isin(g1_ids)]
    pC_g1 = pruef_C[pruef_C["studierenden_id"].isin(g1_ids)]

    merged_p_g1 = pA_g1.merge(pC_g1, on=["studierenden_id", "modul_id", "versuch"], suffixes=("_A", "_C"))
    print(f"Gematchte Prüfungen für G1 (Geschädigte): {len(merged_p_g1)}")
    
    with_supp_g1 = merged_p_g1[merged_p_g1["support_genutzt_A"].astype(str) == "True"]
    print(f"Davon Prüfungen mit fachlichem Support in A: {len(with_supp_g1)}")
    print(f"  Mittlere Note in A (mit Support): {with_supp_g1['note_A'].mean():.2f}")
    print(f"  Mittlere Note in C (ohne Support): {with_supp_g1['note_C'].mean():.2f}")
    print(f"  Bestehensquote in A (mit Support): {with_supp_g1['bestanden_A'].mean():.2%}")
    print(f"  Bestehensquote in C (ohne Support): {with_supp_g1['bestanden_C'].mean():.2%}")

    # 5. DML Discrepancy & Decision Tree Analysis
    print("\n================================================================================")
    print("DML DISKREPANZ: ARTEFAKT-BILDUNG UND INTERAKTIONSEFFEKTE")
    print("================================================================================")
    
    # Versuchen wir einen Decision Tree auf Universum A Daten zur Erwerbstätigkeit x Support
    X = merged[["erwerbstaetigkeit_std", "n_fach_support_A", "hzb_note", "motivation_initial"]].copy()
    y = merged["dropout_A"]

    dt = DecisionTreeClassifier(max_depth=3)
    dt.fit(X, y)
    tree_rules = export_text(dt, feature_names=list(X.columns))
    print("Decision Tree zum Dropout in Universum A (Max Depth 3):")
    print(tree_rules)

    # Export von Zusammenfassungen
    summary_export = merged.groupby("gruppe").agg(
        n_students=("studierenden_id", "count"),
        erwerbstaetigkeit_mean=("erwerbstaetigkeit_std", "mean"),
        hzb_note_mean=("hzb_note", "mean"),
        n_fach_support_mean=("n_fach_support_A", "mean"),
        failed_A_mean=("failed_A", "mean"),
        failed_C_mean=("failed_C", "mean"),
        delta_motivation_A_mean=("delta_motivation_A", "mean"),
        delta_motivation_C_mean=("delta_motivation_C", "mean"),
        delta_erwartete_note_A_mean=("delta_erwartete_note_A", "mean"),
        delta_erwartete_note_C_mean=("delta_erwartete_note_C", "mean")
    ).reset_index()
    
    summary_export.to_csv(output_dir / "overload_victims_summary.csv", index=False)
    print(f"\nZusammenfassung exportiert nach: {output_dir / 'overload_victims_summary.csv'}")

if __name__ == "__main__":
    main()
