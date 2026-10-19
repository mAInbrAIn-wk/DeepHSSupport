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
    print("1. ZEIT-AMORTISATION: LOHNEN SICH DIE 30H SUPPORT?")
    print("================================================================================")

    # 1. Daten laden
    stud_A = pd.read_csv(dir_A / "studierende.csv")
    stud_C = pd.read_csv(dir_C / "studierende.csv")
    abschl_A = pd.read_csv(dir_A / "abschluesse.csv")
    abschl_C = pd.read_csv(dir_C / "abschluesse.csv")
    pruef_A = pd.read_csv(dir_A / "pruefungen.csv")
    pruef_C = pd.read_csv(dir_C / "pruefungen.csv")

    # Modul-Workloads laden (approximiert aus config.py oder wir nehmen 150h als Standard für 5 ECTS, 
    # aber wir können es noch genauer machen. Die config hat oft 120-180h. Wir nehmen 150h als robusten Schätzer).
    WORKLOAD_H_PER_MODULE = 150
    SUPPORT_COST_H = 30

    # Gruppen definieren
    dropout_statuses = ["abgebrochen", "exmatrikuliert", "zeitueberschreitung"]
    last_status_A = abschl_A.sort_values("abschluss_semester_id").groupby("studierenden_id")["status"].last()
    last_status_C = abschl_C.sort_values("abschluss_semester_id").groupby("studierenden_id")["status"].last()

    stud_A["dropout_A"] = stud_A["studierenden_id"].map(last_status_A).isin(dropout_statuses)
    stud_A["dropout_C"] = stud_A["studierenden_id"].map(last_status_C).isin(dropout_statuses)

    g1_mask = (stud_A["dropout_A"] == True) & (stud_A["dropout_C"] == False)  # Geschädigte
    g2_mask = (stud_A["dropout_A"] == False) & (stud_A["dropout_C"] == True)  # Gerettete

    stud_A["gruppe"] = "Andere"
    stud_A.loc[g1_mask, "gruppe"] = "G1_Geschaedigte"
    stud_A.loc[g2_mask, "gruppe"] = "G2_Gerettete"

    # Prüfungs-Vergleich (Nur die Prüfungen, die MIT Support abgelegt wurden)
    pruef_A_fach_supp = pruef_A[pruef_A["support_genutzt"].astype(str) == "True"]
    
    # Merge mit C um den kontrafaktischen Ausgang zu sehen
    merged_pruef = pruef_A_fach_supp.merge(pruef_C, on=["studierenden_id", "semester_id", "modul_id", "versuch"], suffixes=("_A", "_C"))
    
    # Gerettete Prüfungen: In C durchgefallen (False), in A bestanden (True)
    merged_pruef["pruefung_gerettet"] = (merged_pruef["bestanden_A"] == True) & (merged_pruef["bestanden_C"] == False)
    
    # Aggregieren auf Studierenden-Ebene
    supp_stats = merged_pruef.groupby("studierenden_id").agg(
        n_support_teilnahmen=("modul_id", "count"),
        n_gerettete_pruefungen=("pruefung_gerettet", "sum")
    ).reset_index()

    # Zeitbilanz berechnen
    supp_stats["zeitkosten_h"] = supp_stats["n_support_teilnahmen"] * SUPPORT_COST_H
    supp_stats["zeitgewinn_h"] = supp_stats["n_gerettete_pruefungen"] * WORKLOAD_H_PER_MODULE
    supp_stats["netto_zeitbilanz"] = supp_stats["zeitgewinn_h"] - supp_stats["zeitkosten_h"]
    supp_stats["amortisiert"] = supp_stats["netto_zeitbilanz"] > 0

    # Mit Gruppen mergen
    final_df = stud_A[stud_A["gruppe"].isin(["G1_Geschaedigte", "G2_Gerettete"])].merge(supp_stats, on="studierenden_id", how="left").fillna(0)

    # Ergebnisse ausgeben
    print("\n--- Zeitbilanz für G1 (Geschädigte) vs G2 (Gerettete) ---")
    summary = final_df.groupby("gruppe").agg(
        n_students=("studierenden_id", "count"),
        avg_support_teilnahmen=("n_support_teilnahmen", "mean"),
        avg_gerettete_pruefungen=("n_gerettete_pruefungen", "mean"),
        avg_zeitkosten=("zeitkosten_h", "mean"),
        avg_zeitgewinn=("zeitgewinn_h", "mean"),
        avg_netto_bilanz=("netto_zeitbilanz", "mean"),
        pct_amortisiert=("amortisiert", "mean")
    )
    print(summary.to_string())

    print("\n================================================================================")
    print("2. DIE EXKLUSIVITÄTS-FRAGE: GRENZWERTE FÜR ERWERBSTÄTIGKEIT")
    print("================================================================================")
    
    g1_df = stud_A[stud_A["gruppe"] == "G1_Geschaedigte"]
    total_g1 = len(g1_df)
    
    thresholds = [10, 15, 17.5, 20, 25]
    for t in thresholds:
        count = len(g1_df[g1_df["erwerbstaetigkeit_std"] >= t])
        print(f"G1 Studierende mit >= {t}h Erwerbstätigkeit: {count} ({count/total_g1:.1%})")

    print("\nZum Vergleich für G2 (Gerettete):")
    g2_df = stud_A[stud_A["gruppe"] == "G2_Gerettete"]
    total_g2 = len(g2_df)
    for t in thresholds:
        count = len(g2_df[g2_df["erwerbstaetigkeit_std"] >= t])
        print(f"G2 Studierende mit >= {t}h Erwerbstätigkeit: {count} ({count/total_g2:.1%})")


    print("\n================================================================================")
    print("3. ML-DISKREPANZ: DIE 'BLIND' MODELLE")
    print("================================================================================")
    
    blind_file = base_dir / "metrics" / "keras_mlp_baseline_blind_metrics.json"
    reg_file = base_dir / "metrics" / "keras_mlp_baseline_metrics.json"

    if blind_file.exists() and reg_file.exists():
        with open(blind_file, "r") as f:
            blind_data = json.load(f)
        with open(reg_file, "r") as f:
            reg_data = json.load(f)
            
        print("MLP Baseline (Mit Noten):")
        print(f"  PR-AUC: {reg_data.get('PR-AUC_macro', 'N/A')}")
        
        print("\nMLP Baseline BLIND (Ohne Noten, keine Hidden Vars):")
        print(f"  PR-AUC: {blind_data.get('PR-AUC_macro', 'N/A')}")
        
        print("\n-> Wenn das BLIND-Modell viel schlechter performt, bedeutet das, dass das normale Modell")
        print("   seine 'Treffer' fast ausschließlich aus den Noten zieht (welche in der Simulation")
        print("   zwar durch Support künstlich verbessert werden, aber nicht vor Dropout schützen).")
    else:
        print("Baseline Metriken nicht gefunden. Überspringe BLIND-Vergleich.")

if __name__ == "__main__":
    main()
