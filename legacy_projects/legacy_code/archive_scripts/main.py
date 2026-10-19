import numpy as np
from pathlib import Path
from config import CONFIG
from simulation import generiere_stammdaten, generiere_studierende, simuliere_verlaeufe
from export import as_dataframe, exportiere_csv
from aggregate import aggregiere_daten
from validate import validiere_und_dokumentiere

def main():
    print("=" * 70)
    print("SYNTHETISCHE STUDIENVERLAUFSDATEN – SIMULATION (DL EDITION)")
    print("=" * 70)
    
    rng = np.random.default_rng(CONFIG["seed"])
    
    print("\n[1/5] Stammdaten generieren ...")
    stammdaten = generiere_stammdaten()
    
    print("\n[2/5] Studierende generieren ...")
    studierende = generiere_studierende(stammdaten, rng)
    
    print("\n[3/5] Studienverläufe simulieren (Zeitkontenmodell & reaktiver Support) ...")
    studierende = simuliere_verlaeufe(studierende, stammdaten, rng)
    
    print("\n[4/5] Daten aufbereiten und exportieren ...")
    df_dict = stammdaten.copy()
    df_dict.update(as_dataframe(studierende, stammdaten))
    
    output_dir = Path(CONFIG["output_dir"])
    exportiere_csv(df_dict, output_dir)
    
    print("\n[5/5] Datenaggregation & Validierung ...")
    aggregiere_daten(output_dir)
    validiere_und_dokumentiere(output_dir)
    
    print("\n" + "=" * 70)
    print("SIMULATION & VALIDIERUNG ERFOLGREICH BEENDET")
    print(f"  Output-Verzeichnis: {output_dir.resolve()}")
    print("=" * 70)

if __name__ == "__main__":
    main()
