import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    base_dir = Path(r"c:\GitHub_public\Abschlussprojekt\output_dl")
    if not (base_dir / "pruefungen.csv").exists():
        base_dir = Path(r"c:\GitHub_public\Abschlussprojekt\output_dl_old")
        
    output_dir = Path(r"c:\GitHub_public\Abschlussprojekt\output_dl\analysis")
    output_dir.mkdir(exist_ok=True, parents=True)

    print("================================================================================")
    print("REGRESSIONSANALYSE & NOTEN-TREATMENT-EFFEKT (ATT ON GRADES)")
    print("================================================================================")

    pruefungen_path = Path(r"c:\GitHub_public\Abschlussprojekt\output_dl\pruefungen.csv")
    if not pruefungen_path.exists():
        pruefungen_path = Path(r"c:\GitHub_public\Abschlussprojekt\output_dl_v2\pruefungen.csv")
    if not pruefungen_path.exists():
        print("Pruefungen.csv nicht gefunden in output_dl oder output_dl_v2.")
        return

    df_pruef = pd.read_csv(pruefungen_path)
    
    # Filtern auf Prüfungen, bei denen Support genutzt wurde
    treated = df_pruef[df_pruef["support_genutzt"].astype(str) == "True"].copy()
    
    print(f"Gesamt Prüfungen im Datensatz: {len(df_pruef)}")
    print(f"Behandelte Prüfungen (mit Support): {len(treated)} ({len(treated)/len(df_pruef)*100:.2f}%)")

    # Kausaler Noteneffekt (ATT on Grades): note (mit Support) - note_counterfactual (ohne Support)
    # Ein negativer Wert bedeutet Notenverbesserung (z.B. 2.0 statt 2.3)
    treated["note_gain"] = treated["note"] - treated["note_counterfactual"]
    
    avg_note_gain = treated["note_gain"].mean()
    median_note_gain = treated["note_gain"].median()
    
    # Bestanden-Raten Abgleich
    pass_rate_actual = treated["bestanden"].mean()
    pass_rate_cf = (treated["note_counterfactual"] <= 4.0).mean()
    pass_gain = (pass_rate_actual - pass_rate_cf) * 100

    print("\n--------------------------------------------------------------------------------")
    print("GROUND TRUTH NOTEN-EFFEKT (ATT ON TREATED EXAMS)")
    print("--------------------------------------------------------------------------------")
    print(f"Durchschnittlicher Notengewinn (Grade Gain): {avg_note_gain:.4f} Notenstufen")
    print(f"Median Notengewinn:                        {median_note_gain:.4f} Notenstufen")
    print(f"Tatsächliche Bestehensquote (mit Support):   {pass_rate_actual*100:.2f}%")
    print(f"Kontrafaktische Bestehensquote (ohne Supp):  {pass_rate_cf*100:.2f}%")
    print(f"Netto-Gewinn Bestehensquote:                +{pass_gain:.2f} %-Punkte")

    # Auswertung nach Versuch (1., 2., 3. Versuch)
    print("\n--------------------------------------------------------------------------------")
    print("NOTEN-GEWINN NACH PRÜFUNGSVERSUCH")
    print("--------------------------------------------------------------------------------")
    versuch_stats = treated.groupby("versuch").agg(
        n_pruefungen=("modul_id", "count"),
        avg_note_actual=("note", "mean"),
        avg_note_cf=("note_counterfactual", "mean"),
        avg_note_gain=("note_gain", "mean"),
        pass_rate_actual=("bestanden", "mean"),
        pass_rate_cf=("note_counterfactual", lambda x: (x <= 4.0).mean())
    )
    versuch_stats["pass_gain_pct"] = (versuch_stats["pass_rate_actual"] - versuch_stats["pass_rate_cf"]) * 100
    print(versuch_stats.to_string())

if __name__ == "__main__":
    main()
