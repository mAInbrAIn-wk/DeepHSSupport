"""
Vollständige kontrafaktische Supportanalyse & Migrationsanalyse
für V4 (korrigierte kappa-Werte), 8 Universen, N=50.000
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

BASE = Path("src/output_v4_universes")
UNIVERSES = ["A", "B", "C", "D", "E", "F", "G", "H"]

def load_universe(uni_key):
    uni_dir = BASE / f"universe_{uni_key}"
    stud = pd.read_csv(uni_dir / "studierende.csv")
    ab = pd.read_csv(uni_dir / "abschluesse.csv")
    pr = pd.read_csv(uni_dir / "pruefungen.csv")
    supp = pd.read_csv(uni_dir / "support_teilnahmen.csv")
    sa = pd.read_csv(uni_dir / "support_angebote.csv")
    return stud, ab, pr, supp, sa

print("=" * 80)
print("VOLLSTÄNDIGE KONTRAFAKTISCHE SUPPORTANALYSE V4 (korrigiert)")
print("=" * 80)

# ============================================================
# 1. GRUNDLEGENDE KOHORTEN-STATISTIK (Welt A)
# ============================================================
print("\n\n### 1. GRUNDLEGENDE KOHORTEN-STATISTIK (Welt A) ###\n")
stud_A, ab_A, pr_A, supp_A, sa_A = load_universe("A")

print(f"N = {len(stud_A):,}")
print(f"\nStatus-Verteilung:")
for status, pct in (ab_A["status"].value_counts(normalize=True) * 100).items():
    print(f"  {status:25s}: {pct:6.2f} %")

absolventen_A = ab_A[ab_A["status"] == "abgeschlossen"]
print(f"\nAbsolventen:")
print(f"  Notendurchschnitt:       {absolventen_A['abschlussnote'].mean():.3f}")
print(f"  Studiendauer (Sem):      {absolventen_A['studiendauer_semester'].mean():.2f}")

pass_rate = pr_A["bestanden"].mean() * 100
print(f"\nPrüfungen:")
print(f"  Gesamt:                  {len(pr_A):,}")
print(f"  Bestehensquote:          {pass_rate:.1f} %")
print(f"  Note (bestanden):        {pr_A[pr_A['bestanden']==True]['note'].mean():.2f}")

supp_typed_A = supp_A.merge(sa_A[["angebot_id", "typ"]], on="angebot_id")
print(f"\nSupport-Nutzung (pro Kopf): {len(supp_A)/len(stud_A):.2f}")
for typ, cnt in supp_typed_A["typ"].value_counts().items():
    print(f"  {typ:25s}: {cnt:,} ({cnt/len(stud_A):.2f} p.P.)")

# ============================================================
# 2. DROPOUT-RATEN ÜBER ALLE 8 UNIVERSEN
# ============================================================
print("\n\n### 2. DROPOUT-RATEN ÜBER ALLE 8 UNIVERSEN ###\n")

uni_data = {}
for u in UNIVERSES:
    _, ab_u, pr_u, supp_u, sa_u = load_universe(u)
    dropout_cnt = (ab_u["status"] != "abgeschlossen").sum()
    n = len(ab_u)
    uni_data[u] = {
        "ab": ab_u,
        "pr": pr_u,
        "supp": supp_u,
        "sa": sa_u,
        "dropout_rate": dropout_cnt / n,
        "dropout_cnt": dropout_cnt,
        "n": n,
        "note_mean": ab_u[ab_u["status"]=="abgeschlossen"]["abschlussnote"].mean(),
        "dauer_mean": ab_u[ab_u["status"]=="abgeschlossen"]["studiendauer_semester"].mean(),
        "pass_rate": pr_u["bestanden"].mean() * 100,
    }

labels = {
    "A": "Alle Support-Typen erlaubt",
    "B": "Kein Support (komplett blockiert)",
    "C": "Kein fachlicher Support",
    "D": "Kein ueberfachlicher Support",
    "E": "Kein psychosozialer Support",
    "F": "Nur fachlicher Support",
    "G": "Nur ueberfachlicher Support",
    "H": "Nur psychosozialer Support",
}

rate_A = uni_data["A"]["dropout_rate"]
rate_B = uni_data["B"]["dropout_rate"]

print(f"{'Uni':>3s} | {'Dropout':>8s} | {'RR vs A':>8s} | {'RR vs B':>8s} | {'Note':>5s} | {'Dauer':>5s} | {'Pass%':>5s} | Beschreibung")
print("-" * 110)
for u in UNIVERSES:
    d = uni_data[u]
    rr_a = d["dropout_rate"] / rate_A if rate_A > 0 else 0
    rr_b = d["dropout_rate"] / rate_B if rate_B > 0 else 0
    print(f"  {u} | {d['dropout_rate']*100:6.2f} % | {rr_a:8.4f} | {rr_b:8.4f} | {d['note_mean']:.2f} | {d['dauer_mean']:.1f}  | {d['pass_rate']:.1f} | {labels[u]}")

# ============================================================
# 3. SUPPORT-EFFEKTE (Partielle und Isolierte)
# ============================================================
print("\n\n### 3. SUPPORT-EFFEKTE ###\n")
print("PARTIELLE EFFEKTE (Was passiert, wenn man einen Typ WEGNIMMT?):")
for u, typ in [("C", "Fachlich"), ("D", "Überfachlich"), ("E", "Psychosozial")]:
    diff = uni_data[u]["dropout_rate"] - rate_A
    rr = uni_data[u]["dropout_rate"] / rate_A
    print(f"  {typ:20s} -> Dropout steigt um {diff*100:+.2f} pp (RR = {rr:.4f}, +{(rr-1)*100:.1f}%)")

print(f"\nGESAMTEFFEKT (B vs A): Dropout steigt um {(rate_B - rate_A)*100:+.2f} pp (RR = {rate_B/rate_A:.4f})")
additivitaet = sum(uni_data[u]["dropout_rate"] - rate_A for u in ["C", "D", "E"])
print(f"  Summe der Teileffekte:  {additivitaet*100:.2f} pp")
print(f"  Gesamteffekt (B-A):     {(rate_B - rate_A)*100:.2f} pp")
print(f"  Superadditivitaet:      {((rate_B - rate_A) - additivitaet)*100:.2f} pp (Interaktionseffekt)")

print("\nISOLIERTE EFFEKTE (Was bringt ein EINZELNER Typ alleine, vs. gar kein Support?):")
for u, typ in [("F", "Fachlich"), ("G", "Überfachlich"), ("H", "Psychosozial")]:
    diff = uni_data[u]["dropout_rate"] - rate_B
    rr = uni_data[u]["dropout_rate"] / rate_B
    print(f"  {typ:20s} -> Dropout sinkt um {diff*100:.2f} pp (RR = {rr:.4f}, {(1-rr)*100:.1f}% Reduktion)")

# ============================================================
# 4. MIGRATIONSANALYSE
# ============================================================
print("\n\n### 4. MIGRATIONSANALYSE (Alle 8 Universen) ###\n")

stud_all = pd.read_csv(BASE / "universe_A" / "studierende.csv")
mig_ids = set(stud_all[stud_all["migrationshintergrund"] == True]["studierenden_id"])
nomig_ids = set(stud_all[stud_all["migrationshintergrund"] == False]["studierenden_id"])

print(f"Kohorte: {len(mig_ids):,} mit Migrationshintergrund ({len(mig_ids)/len(stud_all)*100:.1f}%), {len(nomig_ids):,} ohne\n")

print(f"{'Uni':>3s} | {'Dropout MIG':>12s} | {'Dropout NOMIG':>13s} | {'Gap (pp)':>9s} | {'RR MIG/NOMIG':>13s}")
print("-" * 70)
for u in UNIVERSES:
    ab_u = uni_data[u]["ab"]
    mig = ab_u[ab_u["studierenden_id"].isin(mig_ids)]
    nomig = ab_u[ab_u["studierenden_id"].isin(nomig_ids)]
    drop_mig = (mig["status"] != "abgeschlossen").mean() * 100
    drop_nomig = (nomig["status"] != "abgeschlossen").mean() * 100
    gap = drop_mig - drop_nomig
    rr = drop_mig / drop_nomig if drop_nomig > 0 else 0
    print(f"  {u} | {drop_mig:10.2f} % | {drop_nomig:11.2f} % | {gap:+7.2f} | {rr:13.4f}")

# ============================================================
# 5. ERSTAKADEMIKER-ANALYSE
# ============================================================
print("\n\n### 5. ERSTAKADEMIKER-ANALYSE (Alle 8 Universen) ###\n")

erst_ids = set(stud_all[stud_all["erstakademiker"] == True]["studierenden_id"])
noerst_ids = set(stud_all[stud_all["erstakademiker"] == False]["studierenden_id"])

print(f"Kohorte: {len(erst_ids):,} Erstakademiker ({len(erst_ids)/len(stud_all)*100:.1f}%), {len(noerst_ids):,} Akademikerkinder\n")

print(f"{'Uni':>3s} | {'Dropout ERST':>13s} | {'Dropout NOERST':>14s} | {'Gap (pp)':>9s} | {'RR ERST/NOERST':>15s}")
print("-" * 75)
for u in UNIVERSES:
    ab_u = uni_data[u]["ab"]
    erst = ab_u[ab_u["studierenden_id"].isin(erst_ids)]
    noerst = ab_u[ab_u["studierenden_id"].isin(noerst_ids)]
    drop_erst = (erst["status"] != "abgeschlossen").mean() * 100
    drop_noerst = (noerst["status"] != "abgeschlossen").mean() * 100
    gap = drop_erst - drop_noerst
    rr = drop_erst / drop_noerst if drop_noerst > 0 else 0
    print(f"  {u} | {drop_erst:11.2f} % | {drop_noerst:12.2f} % | {gap:+7.2f} | {rr:15.4f}")

# ============================================================
# 6. INTERAKTION: SUPPORT-EFFEKT NACH MIGRATIONSGRUPPE
# ============================================================
print("\n\n### 6. INTERAKTION: SUPPORT-WIRKUNG NACH MIGRATIONSGRUPPE ###\n")
print("Gesamteffekt (B vs A) aufgeschlüsselt nach Migrationshintergrund:\n")

for group_name, group_ids in [("MIT Migration", mig_ids), ("OHNE Migration", nomig_ids)]:
    ab_A_g = uni_data["A"]["ab"][uni_data["A"]["ab"]["studierenden_id"].isin(group_ids)]
    ab_B_g = uni_data["B"]["ab"][uni_data["B"]["ab"]["studierenden_id"].isin(group_ids)]
    rate_A_g = (ab_A_g["status"] != "abgeschlossen").mean()
    rate_B_g = (ab_B_g["status"] != "abgeschlossen").mean()
    rr_g = rate_B_g / rate_A_g if rate_A_g > 0 else 0
    print(f"  {group_name:20s}: A={rate_A_g*100:.2f}%  B={rate_B_g*100:.2f}%  Diff={((rate_B_g-rate_A_g)*100):+.2f}pp  RR={rr_g:.4f}")

print("\nPartieller Effekt pro Support-Typ nach Migrationsstatus:\n")
for u, typ in [("C", "Fachlich"), ("D", "Überfachlich"), ("E", "Psychosozial")]:
    print(f"  {typ}:")
    for group_name, group_ids in [("  MIT Mig", mig_ids), ("  OHNE Mig", nomig_ids)]:
        ab_A_g = uni_data["A"]["ab"][uni_data["A"]["ab"]["studierenden_id"].isin(group_ids)]
        ab_U_g = uni_data[u]["ab"][uni_data[u]["ab"]["studierenden_id"].isin(group_ids)]
        rate_A_g = (ab_A_g["status"] != "abgeschlossen").mean()
        rate_U_g = (ab_U_g["status"] != "abgeschlossen").mean()
        rr = rate_U_g / rate_A_g if rate_A_g > 0 else 0
        print(f"    {group_name:15s}: A={rate_A_g*100:.2f}%  {u}={rate_U_g*100:.2f}%  Diff={((rate_U_g-rate_A_g)*100):+.2f}pp  RR={rr:.4f}")

# ============================================================
# 7. INTERAKTION: SUPPORT-EFFEKT NACH ERSTAKADEMIKER
# ============================================================
print("\n\n### 7. INTERAKTION: SUPPORT-WIRKUNG NACH ERSTAKADEMIKER ###\n")
print("Gesamteffekt (B vs A) aufgeschlüsselt nach Erstakademiker-Status:\n")

for group_name, group_ids in [("Erstakademiker", erst_ids), ("Akademikerkind", noerst_ids)]:
    ab_A_g = uni_data["A"]["ab"][uni_data["A"]["ab"]["studierenden_id"].isin(group_ids)]
    ab_B_g = uni_data["B"]["ab"][uni_data["B"]["ab"]["studierenden_id"].isin(group_ids)]
    rate_A_g = (ab_A_g["status"] != "abgeschlossen").mean()
    rate_B_g = (ab_B_g["status"] != "abgeschlossen").mean()
    rr_g = rate_B_g / rate_A_g if rate_A_g > 0 else 0
    print(f"  {group_name:20s}: A={rate_A_g*100:.2f}%  B={rate_B_g*100:.2f}%  Diff={((rate_B_g-rate_A_g)*100):+.2f}pp  RR={rr_g:.4f}")

# ============================================================
# 8. NOTEN-EFFEKTE ÜBER UNIVERSEN
# ============================================================
print("\n\n### 8. NOTEN-EFFEKTE (Nur Absolventen) ###\n")
print(f"{'Uni':>3s} | {'Note':>5s} | {'Dauer (Sem)':>11s} | {'N Absolventen':>13s}")
print("-" * 50)
for u in UNIVERSES:
    ab_u = uni_data[u]["ab"]
    absol = ab_u[ab_u["status"] == "abgeschlossen"]
    print(f"  {u} | {absol['abschlussnote'].mean():.3f} | {absol['studiendauer_semester'].mean():11.2f} | {len(absol):13,}")

# ============================================================
# 9. SUPPORT-NUTZUNG PRO UNIVERSUM
# ============================================================
print("\n\n### 9. SUPPORT-NUTZUNG PRO UNIVERSUM ###\n")
print(f"{'Uni':>3s} | {'Fachlich':>9s} | {'Überfachl.':>10s} | {'Psychosoz.':>10s} | {'Total':>7s} | {'Pro Kopf':>9s}")
print("-" * 65)
for u in UNIVERSES:
    supp_u = uni_data[u]["supp"]
    sa_u = uni_data[u]["sa"]
    if len(supp_u) == 0:
        print(f"  {u} | {'—':>9s} | {'—':>10s} | {'—':>10s} | {0:>7} | {0.0:>9.2f}")
        continue
    typed = supp_u.merge(sa_u[["angebot_id", "typ"]], on="angebot_id")
    counts = typed["typ"].value_counts()
    fach = counts.get("fachlich", 0)
    uebf = counts.get("ueberfachlich", 0)
    psych = counts.get("psychosozial", 0)
    total = len(supp_u)
    print(f"  {u} | {fach:>9,} | {uebf:>10,} | {psych:>10,} | {total:>7,} | {total/uni_data[u]['n']:>9.2f}")

print("\n\nAnalyse abgeschlossen!")
