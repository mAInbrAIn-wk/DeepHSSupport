"""
Analyse & Visualisierung des V4 Simulations-Sensitivitäts-Grids
===============================================================
Liest `src/output_v4_grid/metrics/full_sensitivity_grid_results.json` ein
und generiert:
1. Eine synoptische Markdown-Tabelle aller 12 Szenarien
2. Elastizitäts- und Sensitivitätsanalysen
3. Einen Visualisierungs-Plot `plots_v4_sensitivity_grid.png`
4. Den Synthese-Bericht `sensitivitaetsanalyse_v4_grid.md`
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def analyze_grid():
    grid_json = Path("src/output_v4_grid/metrics/full_sensitivity_grid_results.json")
    if not grid_json.exists():
        print(f"Fehler: {grid_json} existiert nicht!")
        return

    with open(grid_json) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    
    print("=" * 110)
    print("V4 SIMULATIONS-SENSITIVITÄT: SYNOPTISCHE ÜBERSICHT ALLER 12 SZENARIEN")
    print("=" * 110)
    
    cols_show = [
        "scenario_id", "dimension", "dropout_A", "dropout_B", "RR_B_vs_A",
        "protection_all_pct", "protection_fach_pct", "protection_uebf_pct", "protection_psych_pct",
        "synergy_gap_pct_pts", "equalizer_gain_pct_pts", "modules_dropped_A"
    ]
    print(df[cols_show].to_string(index=False))

    # --- Erstelle Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Dropout-Rate Universum A (Full Support) vs B (No Support)
    ax1 = axes[0, 0]
    sc_names = [d["name"] for d in data]
    y_pos = np.arange(len(sc_names))
    
    drop_A = [d["dropout_A"] * 100 for d in data]
    drop_B = [d["dropout_B"] * 100 for d in data]
    
    ax1.barh(y_pos - 0.2, drop_A, height=0.4, label="Uni A (Full Support)", color="#2b5c8f")
    ax1.barh(y_pos + 0.2, drop_B, height=0.4, label="Uni B (No Support)", color="#c0392b")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(sc_names, fontsize=9)
    ax1.set_xlabel("Dropout-Rate (%)")
    ax1.set_title("1. Dropout-Raten: Universum A vs. Universum B", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(axis="x", alpha=0.3)
    ax1.invert_yaxis()

    # Plot 2: Relatives Risiko (RR_B vs A - Gesamter Support-Hebel)
    ax2 = axes[0, 1]
    rr_B = [d["RR_B_vs_A"] for d in data]
    colors = ['#27ae60' if r >= 1.04 else '#f39c12' if r >= 1.0 else '#e74c3c' for r in rr_B]
    bars = ax2.barh(y_pos, rr_B, color=colors, height=0.6)
    ax2.axvline(1.0, color="black", linestyle="--", linewidth=1.2, label="Neutral (RR=1.0)")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sc_names, fontsize=9)
    ax2.set_xlabel("Relatives Risiko RR (B vs. A)")
    ax2.set_title("2. Kausales Relatives Risiko RR (Uni B / Uni A)", fontsize=11, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)
    ax2.invert_yaxis()

    # Plot 3: Isolierte Schutzwirkungen (Fachlich vs. Überfachlich vs. Psychosozial)
    ax3 = axes[1, 0]
    prot_f = [d["protection_fach_pct"] for d in data]
    prot_u = [d["protection_uebf_pct"] for d in data]
    prot_p = [d["protection_psych_pct"] for d in data]
    
    w = 0.25
    ax3.barh(y_pos - w, prot_f, height=w, label="Nur Fachlich (Uni F vs B)", color="#3498db")
    ax3.barh(y_pos, prot_u, height=w, label="Nur Überfachlich (Uni G vs B)", color="#9b59b6")
    ax3.barh(y_pos + w, prot_p, height=w, label="Nur Psychosozial (Uni H vs B)", color="#1abc9c")
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(sc_names, fontsize=9)
    ax3.set_xlabel("Schutzwirkung (% Dropout-Reduktion vs. B)")
    ax3.set_title("3. Isolierte Schutzwirkung der Support-Typen", fontsize=11, fontweight="bold")
    ax3.legend(loc="lower right")
    ax3.grid(axis="x", alpha=0.3)
    ax3.invert_yaxis()

    # Plot 4: Superadditivitäts-Synergie & Equalizer Gain
    ax4 = axes[1, 1]
    synergy = [d["synergy_gap_pct_pts"] for d in data]
    equalizer = [d["equalizer_gain_pct_pts"] for d in data]
    
    ax4.barh(y_pos - 0.2, synergy, height=0.4, label="Synergie-Interaktion (%-Punkte)", color="#e67e22")
    ax4.barh(y_pos + 0.2, equalizer, height=0.4, label="First-Gen Equalizer-Gewinn (%-Punkte)", color="#16a085")
    ax4.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(sc_names, fontsize=9)
    ax4.set_xlabel("Effekt-Stärke in Prozentpunkten")
    ax4.set_title("4. Synergie-Interaktion & Bildungs-Equalizer Effekt", fontsize=11, fontweight="bold")
    ax4.legend(loc="lower right")
    ax4.grid(axis="x", alpha=0.3)
    ax4.invert_yaxis()

    plt.tight_layout()
    plot_out = Path("C:/Users/wilfr/.gemini/antigravity/brain/16832ed6-a522-415e-9395-ef24e16fef79/plots_v4_sensitivity_grid.png")
    plt.savefig(plot_out, dpi=300)
    plt.close()
    print(f"\n[OK] Plot gespeichert unter: {plot_out}")

    # --- Markdown-Synthese-Bericht ---
    report_lines = [
        "# Systematischer Sensitivitätsbericht: V4 Simulations-Gridsearch",
        "",
        "Dieser Bericht analysiert die Ergebnisse der **12 systematischen Simulations-Szenarien** (jeweils simuliert über alle 8 Universen A–H mit $N=25.000$ Studierenden pro Universum, insgesamt 96 Simulationen mit identischem Seed).",
        "",
        "## 1. Synoptische Haupttabelle: Kausale Makro-Effekte",
        "",
        "| Szenario | Dimension | Drop Uni A | Drop Uni B | RR (B vs. A) | Gesamt-Schutz | Nur Fachl. (F) | Nur Überf. (G) | Nur Psych. (H) | Synergie | First-Gen Gain |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for d in data:
        row = f"| **{d['name']}** | {d['dimension']} | {d['dropout_A']*100:.2f} % | {d['dropout_B']*100:.2f} % | **{d['RR_B_vs_A']:.4f}** | **{d['protection_all_pct']:.2f} %** | {d['protection_fach_pct']:.2f} % | {d['protection_uebf_pct']:.2f} % | {d['protection_psych_pct']:.2f} % | {d['synergy_gap_pct_pts']:+.2f} %p | {d['equalizer_gain_pct_pts']:+.2f} %p |"
        report_lines.append(row)

    report_lines.extend([
        "",
        "## 2. Detaillierte Dimensionen-Analyse",
        "",
        "### A. Dimension Support-Wirkungs-Multiplikator (`support_effect_multiplier`)",
        "- **Halbiert (0.5x):** Gesamt-Schutz sinkt von {:.2f}% auf {:.2f}%. RR(B vs A) sinkt auf {:.4f}.".format(
            data[0]['protection_all_pct'], data[1]['protection_all_pct'], data[1]['RR_B_vs_A']),
        "- **Verdoppelt (2.0x):** Gesamt-Schutz steigt von {:.2f}% auf {:.2f}%. RR(B vs A) steigt auf {:.4f}.".format(
            data[0]['protection_all_pct'], data[2]['protection_all_pct'], data[2]['RR_B_vs_A']),
        "",
        "### B. Dimension Notenboost Fachlich (`gewicht_support_boost`)",
        "- **Halbiert (0.04):** Isolierte fachliche Schutzwirkung sinkt auf {:.2f}%.".format(data[3]['protection_fach_pct']),
        "- **Verdoppelt (0.16):** Isolierte fachliche Schutzwirkung steigt auf {:.2f}%.".format(data[4]['protection_fach_pct']),
        "- **Vervierfacht (0.32):** Isolierte fachliche Schutzwirkung erreicht {:.2f}%.".format(data[5]['protection_fach_pct']),
        "",
        "### C. Dimension Stochastisches Rauschen (`gewicht_rauschen`)",
        "- **Halbiertes Rauschen (0.09):** In deterministischerer Umgebung beträgt RR(B vs A) {:.4f}.".format(data[6]['RR_B_vs_A']),
        "- **Verdoppeltes Rauschen (0.36):** Bei starkem Rauschen beträgt RR(B vs A) {:.4f}.".format(data[7]['RR_B_vs_A']),
        "",
        "### D. Dimension Support-Zeitkosten (`support_kosten_override`)",
        "- **Kostenlos (0h):** Bei 0h Zeitaufwand steigt der Gesamtschutz auf {:.2f}% (keine Workload-Verdrängung).".format(data[8]['protection_all_pct']),
        "- **Hohe Belastung (60h):** Bei 60h Zeitaufwand wurden {} Module abgeworfen, Schutzwirkung sinkt auf {:.2f}%.".format(
            data[9]['modules_dropped_A'], data[9]['protection_all_pct']),
        "",
        "### E. Dimension Selektions-Endogenität (`rct_support_uptake`)",
        "- **RCT (Random Uptake):** Bei zufälliger Zuweisung (ohne Risikoselektion) beträgt die Schutzwirkung {:.2f}%.".format(data[10]['protection_all_pct']),
        "",
        "### F. Synergie-Optimum (`S12_high_synergy`)",
        "- **Maximaler Hebel:** Schutzwirkung {:.2f}%, RR(B vs A) = {:.4f}, Synergie = {:+.2f} %p.".format(
            data[11]['protection_all_pct'], data[11]['RR_B_vs_A'], data[11]['synergy_gap_pct_pts']),
        "",
        "## 3. Visualisierung",
        "",
        "![V4 Sensitivitätsanalyse Plot](file:///C:/Users/wilfr/.gemini/antigravity/brain/16832ed6-a522-415e-9395-ef24e16fef79/plots_v4_sensitivity_grid.png)"
    ])

    report_md = "\n".join(report_lines)
    report_file = Path("C:/Users/wilfr/.gemini/antigravity/brain/16832ed6-a522-415e-9395-ef24e16fef79/sensitivitaetsanalyse_v4_grid.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)
        
    print(f"\n[OK] Synthese-Bericht gespeichert unter: {report_file}")


if __name__ == "__main__":
    analyze_grid()
