"""
Berechnung des zugrundeliegenden Support-Effekts
=================================================
Liest die generierten `pruefungen.csv` Daten aus, um anhand der
`note_counterfactual` (Ground Truth ohne Support) den simulierten, 
reinen Treatment Effect on the Treated (ATT) zu berechnen.
"""

import pandas as pd
from pathlib import Path

def calculate_true_effect():
    print("=" * 70)
    print("   GROUND TRUTH EFFEKT-ANALYSE (SIMULATION LOGIC)")
    print("=" * 70)

    data_dir = Path('../output_dl') if Path('../output_dl').exists() else Path('output_dl')
    pruefungen_path = data_dir / "pruefungen.csv"
    
    if not pruefungen_path.exists():
        print(f"Fehler: {pruefungen_path} nicht gefunden.")
        return

    df = pd.read_csv(pruefungen_path)
    
    # Filtern auf Prüfungen, bei denen Support tatsächlich gewirkt hat
    treated = df[df["support_genutzt"] == True]
    
    if len(treated) == 0:
        print("Es wurden keine Prüfungen mit support_genutzt=True gefunden.")
        return

    # ATT (Average Treatment Effect on the Treated)
    # Beachte: Für Noten ist ein negativer Wert (z.B. -0.3) eine *Verbesserung*.
    treated["note_diff"] = treated["note"] - treated["note_counterfactual"]
    
    avg_note_diff = treated["note_diff"].mean()
    
    pass_rate_actual = treated["bestanden"].mean()
    pass_rate_cf = (treated["note_counterfactual"] <= 4.0).mean()
    pass_rate_diff = pass_rate_actual - pass_rate_cf

    print(f"Anzahl Support-Behandlungen in Modulprüfungen: {len(treated):,}")
    print(f"-> Durchschnittliche Notenverbesserung (ATT): {avg_note_diff:.3f} Notenpunkte")
    print(f"-> Tatsächliche Bestehensquote (mit Support): {pass_rate_actual:.2%}")
    print(f"-> Kontrafaktische Bestehensquote (ohne Support): {pass_rate_cf:.2%}")
    print(f"-> Absoluter Zuwachs in der Bestehensquote: {pass_rate_diff*100:.2f} Prozentpunkte")
    
    print("\nFazit:")
    print("Diese Werte repräsentieren den kausalen Grundmechanismus des Datengenerators.")
    print("Das Double Machine Learning (DML) Panel zielt darauf ab, diese durch")
    print("Confounding überlagerten Effekte in Form einer Hazard Ratio abzubilden.")
    
if __name__ == "__main__":
    calculate_true_effect()
