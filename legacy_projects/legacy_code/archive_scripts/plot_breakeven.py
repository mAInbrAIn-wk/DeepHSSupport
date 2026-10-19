import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    base_dir = Path(r"c:\GitHub_public\Abschlussprojekt\output_dl")
    dir_A = base_dir
    dir_C = base_dir / "universe_C"
    output_dir = base_dir / "analysis"
    output_dir.mkdir(exist_ok=True, parents=True)

    print("================================================================================")
    print("2. BREAK-EVEN PLOT (ERWERBSTÄTIGKEIT)")
    print("================================================================================")

    stud_A = pd.read_csv(dir_A / "studierende.csv")
    abschl_A = pd.read_csv(dir_A / "abschluesse.csv")
    abschl_C = pd.read_csv(dir_C / "abschluesse.csv")

    dropout_statuses = ["abgebrochen", "exmatrikuliert", "zeitueberschreitung"]
    
    last_A = abschl_A.sort_values("abschluss_semester_id").groupby("studierenden_id")["status"].last()
    last_C = abschl_C.sort_values("abschluss_semester_id").groupby("studierenden_id")["status"].last()

    stud_A["dropout_A"] = stud_A["studierenden_id"].map(last_A).isin(dropout_statuses)
    stud_A["dropout_C"] = stud_A["studierenden_id"].map(last_C).isin(dropout_statuses)

    # G1: Geschädigte (Dropout in A, Rettung in C)
    # G2: Gerettete (Rettung in A, Dropout in C)
    g1_mask = (stud_A["dropout_A"] == True) & (stud_A["dropout_C"] == False)
    g2_mask = (stud_A["dropout_A"] == False) & (stud_A["dropout_C"] == True)

    stud_A["is_g1"] = g1_mask.astype(int)
    stud_A["is_g2"] = g2_mask.astype(int)

    # Bins für Erwerbstätigkeit (0 bis 35 Stunden, 2-Stunden-Schritte)
    bins = np.arange(0, 36, 2)
    stud_A["erwerb_bin"] = pd.cut(stud_A["erwerbstaetigkeit_std"], bins=bins)

    grouped = stud_A.groupby("erwerb_bin", observed=False).agg(
        n_total=("studierenden_id", "count"),
        n_g1=("is_g1", "sum"),
        n_g2=("is_g2", "sum")
    ).reset_index()

    grouped["p_g1"] = grouped["n_g1"] / grouped["n_total"]
    grouped["p_g2"] = grouped["n_g2"] / grouped["n_total"]
    
    # Net Treatment Effect (NTE) = P(Gerettet) - P(Geschädigt)
    grouped["net_effect"] = grouped["p_g2"] - grouped["p_g1"]
    grouped["erwerb_mid"] = grouped["erwerb_bin"].apply(lambda x: x.mid if pd.notnull(x) else 0)

    # Plot erstellen
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(grouped["erwerb_mid"], grouped["net_effect"] * 100, marker='o', color='#1f77b4', linewidth=2.5, label="Netto-Effekt (Gerettete % - Geschädigte %)")
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label="Break-Even Line (Netto 0)")
    
    plt.title("Netto-Effekt des fachlichen Supports nach Erwerbstätigkeit", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Erwerbstätigkeit (Stunden / Woche)", fontsize=12)
    plt.ylabel("Netto-Effekt (%-Punkte)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    
    # Annotate Break-Even Point
    # Finde wo der Net-Effekt unter 0 fällt
    crossings = grouped[grouped["net_effect"] < 0]
    if not crossings.empty:
        be_val = crossings.iloc[0]["erwerb_mid"]
        plt.annotate(f'Kipppunkt: ~{be_val:.1f}h/W', xy=(be_val, 0), xytext=(be_val + 2, -0.5),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                     fontsize=11, fontweight='bold', color='darkred')

    plt.tight_layout()
    plot_path = output_dir / "breakeven_plot.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"Break-Even Plot erfolgreich gespeichert unter: {plot_path}")
    print("\n--- Netto-Effekte nach Erwerbstätigkeit ---")
    print(grouped[["erwerb_mid", "n_total", "n_g1", "n_g2", "net_effect"]].to_string())

if __name__ == "__main__":
    main()
