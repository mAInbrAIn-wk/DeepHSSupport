import os
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    base_dir = Path('src/output_dl_seed99999')
    dir_B = base_dir / 'universe_B'
    
    print("1. Lade Prfungen aus Welt A (mit Support) und Welt B (kein Support)...")
    pr_A = pd.read_csv(base_dir / 'pruefungen.csv')
    pr_B = pd.read_csv(dir_B / 'pruefungen.csv')
    
    # Finde Studierende, die in Welt A berhaupt Support genutzt haben
    # Ein Modul-Support zeigt sich an "support_genutzt" == True, aber wir wollen echte Zeit-Kosten sehen.
    # Wir nehmen alle Studenten, die in A Support hatten. (wir lden support_teilnahmen.csv)
    supp_A = pd.read_csv(base_dir / 'support_teilnahmen.csv')
    supp_users = supp_A['studierenden_id'].unique()
    print(f" -> {len(supp_users)} Studierende haben in Welt A Support genutzt.")
    
    pr_A_supp = pr_A[pr_A['studierenden_id'].isin(supp_users)]
    pr_B_supp = pr_B[pr_B['studierenden_id'].isin(supp_users)]
    
    # Um Module zu matchen, nehmen wir nur den *ersten* Versuch (versuch == 1) 
    # pro Modul und Student, um zu sehen, wann er es eingeplant hat.
    pr_A_first = pr_A_supp[pr_A_supp['versuch'] == 1].set_index(['studierenden_id', 'modul_id'])
    pr_B_first = pr_B_supp[pr_B_supp['versuch'] == 1].set_index(['studierenden_id', 'modul_id'])
    
    # Inner Join, um nur Module zu betrachten, die in beiden Welten (irgendwann) geschrieben wurden
    joined = pr_A_first.join(pr_B_first, lsuffix='_A', rsuffix='_B', how='inner')
    
    # Finde verzgerte Module: semester_id_B < semester_id_A (bzw. fachsemester_B < fachsemester_A)
    delayed = joined[joined['semester_id_B'] < joined['semester_id_A']]
    
    print(f"\n2. Analyse der zeitlichen Verschiebung:")
    print(f" -> Von {len(joined)} Erstablegungen bei Support-Nutzern wurden {len(delayed)} in Welt A wegen Zeitmangel auf ein spteres Semester verschoben!")
    
    if len(delayed) > 0:
        pass_B = delayed['bestanden_B'].mean() * 100
        pass_A = delayed['bestanden_A'].mean() * 100
        print(f"\n3. Noten-Vergleich der verschobenen Module:")
        print(f" -> Bestehensquote in Welt B (Frher geschrieben, OHNE Support-Effekt): {pass_B:.1f}%")
        print(f" -> Bestehensquote in Welt A (Spter geschrieben, MIT evtl. Support-Effekt): {pass_A:.1f}%")
        
        # Vielleicht sogar eine Verbesserung?
        improved = delayed[(delayed['bestanden_B'] == False) & (delayed['bestanden_A'] == True)]
        print(f" -> In {len(improved)} Fllen hat die Verschiebung (+ evt. Support in Welt A) ein Modul gerettet, das in Welt B durchgefallen wre.")
        
        # Oder Verschlechterung?
        worsened = delayed[(delayed['bestanden_B'] == True) & (delayed['bestanden_A'] == False)]
        print(f" -> In {len(worsened)} Fllen hat die Verschiebung dazu gefhrt, dass das Modul in Welt A pltzlich nicht mehr bestanden wurde (z.B. wegen schwindender Motivation).")
        
if __name__ == '__main__':
    main()
