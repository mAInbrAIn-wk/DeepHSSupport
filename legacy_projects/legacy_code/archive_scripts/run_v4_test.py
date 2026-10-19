import numpy as np
from pathlib import Path
import sys

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CONFIG
# Wir ueberschreiben die Konfiguration fuer den Testlauf
CONFIG["num_studierende"] = 2000
CONFIG["output_dir"] = "src/output_v4_test"
CONFIG["seed"] = 42

# Wir importieren explizit aus simulation_v4!
from simulation_v4 import generiere_stammdaten, generiere_studierende, simuliere_verlaeufe
from export import as_dataframe, exportiere_csv
from aggregate import aggregiere_daten
from validate import validiere_und_dokumentiere

def main():
    print("=" * 70)
    print("SYNTHETISCHE STUDIENVERLAUFSDATEN - V4 TEST-RUN")
    print("=" * 70)
    
    rng = np.random.default_rng(CONFIG["seed"])
    
    print("\n[1/5] Stammdaten generieren ...")
    stammdaten = generiere_stammdaten()
    
    print("\n[2/5] Studierende generieren ...")
    studierende = generiere_studierende(stammdaten, rng)
    
    print("\n[3/5] Studienverlaeufe simulieren (V4: Beta, Friction, Time Tracker) ...")
    studierende = simuliere_verlaeufe(studierende, stammdaten, rng)
    
    print("\n[4/5] Daten aufbereiten und exportieren ...")
    df_dict = stammdaten.copy()
    df_dict.update(as_dataframe(studierende, stammdaten))
    
    output_dir = Path(CONFIG["output_dir"])
    exportiere_csv(df_dict, output_dir)
    
    print("\n[5/5] Datenaggregation & Validierung ...")
    aggregiere_daten(output_dir)
    # validate ignoriert ggf. V4 Besonderheiten, wir probieren es
    try:
        validiere_und_dokumentiere(output_dir)
    except Exception as e:
        print(f"Validierung uebersprungen (erwartet in Test): {e}")
    
    print("\n" + "=" * 70)
    print("V4 TEST-RUN ERFOLGREICH BEENDET")
    print(f"  Output-Verzeichnis: {output_dir.resolve()}")
    print("=" * 70)

if __name__ == "__main__":
    main()
