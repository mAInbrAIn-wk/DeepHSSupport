import json
import sys
import copy
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from config import CONFIG
from models import Student

def as_dataframe(studierende: List[Student], stammdaten: Dict[str, pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    studierende_rows = []
    einschreibungen_rows = []
    pruefungen_rows = []
    support_teilnahmen_rows = []
    abschluesse_rows = []

    # Map modul_id -> modul name for Bachelorarbeit lookup
    ba_modul_ids = set()
    if stammdaten is not None and "module_df" in stammdaten:
        module_df = stammdaten["module_df"]
        ba_modul_ids = set(module_df[module_df["name"].str.lower().str.contains("bachelorarbeit")]["modul_id"])

    for s in studierende:
        studierende_rows.append({
            "studierenden_id": s.studierenden_id,
            "studiengang_id": s.studiengang_id,
            "kohorten_semester_id": s.kohorten_semester_id,
            "geschlecht": s.geschlecht,
            "alter_immatrikulation": s.alter_immatrikulation,
            "hzb_note": s.hzb_note,
            "hzb_typ": s.hzb_typ,
            "migrationshintergrund": s.migrationshintergrund,
            "erstakademiker": s.erstakademiker,
            "erwerbstaetigkeit_std": s.erwerbstaetigkeit_std,
            "motivation_initial": s.motivation_initial,
            "soziale_integration_initial": s.soziale_integration_initial,
            "motivation_final": round(s.motivation, 3),
            "soziale_integration_final": round(s.soziale_integration, 3),
            "hidden_erwartete_note_initial": s.erwartete_note_initial,
            "hidden_erwartete_note_final": round(s.erwartete_note, 3),
            "hidden_zeit_puffer": round(getattr(s, 'hidden_zeit_puffer', 60.0), 1),
        })
        
        for e in s.einschreibungen:
            e["studierenden_id"] = s.studierenden_id
            einschreibungen_rows.append(e)
            
        for st in s.support_teilnahmen:
            st["studierenden_id"] = s.studierenden_id
            support_teilnahmen_rows.append(st)
            
        for p in s.pruefungen:
            pruefungen_rows.append({
                "studierenden_id": s.studierenden_id,
                "semester_id": p.semester_id,
                "modul_id": p.modul_id,
                "versuch": p.versuch,
                "note": p.note,
                "bestanden": p.bestanden,
                "note_counterfactual": p.note_counterfactual,
                "support_genutzt": p.support_genutzt,
                "hidden_motivation": round(p.hidden_motivation, 3) if p.hidden_motivation is not None else None,
                "hidden_soziale_integration": round(p.hidden_soziale_integration, 3) if p.hidden_soziale_integration is not None else None,
                "hidden_erwartete_note": round(p.hidden_erwartete_note, 3) if p.hidden_erwartete_note is not None else None,
                "hidden_overload": round(p.hidden_overload, 1) if p.hidden_overload is not None else 0.0,
                "hidden_zeit_puffer": round(p.hidden_zeit_puffer, 1) if p.hidden_zeit_puffer is not None else 60.0,
                "hidden_penalty_capped": bool(p.hidden_penalty_capped) if p.hidden_penalty_capped is not None else False,
                "hidden_support_capped": bool(p.hidden_support_capped) if p.hidden_support_capped is not None else False,
            })
            
        status = "abgeschlossen" if s.abschluss_erreicht else ("exmatrikuliert" if s.exmatrikuliert else ("abgebrochen" if s.abgebrochen else "zeitueberschreitung"))
        letztes_sem = s.einschreibungen[-1] if s.einschreibungen else None
        
        # Abschlussnote & Bachelorarbeitsnote NUR bei abgeschlossenem Studium berechnen
        abschlussnote = None
        bachelorarbeitsnote = None
        
        if status == "abgeschlossen":
            bestandene = [p for p in s.pruefungen if p.bestanden]
            letzte_versuche = {}
            for p in bestandene:
                letzte_versuche[p.modul_id] = p.note
                if p.modul_id in ba_modul_ids:
                    bachelorarbeitsnote = p.note
                    
            if letzte_versuche:
                abschlussnote = round(sum(letzte_versuche.values()) / len(letzte_versuche), 2)
        
        abschluesse_rows.append({
            "studierenden_id": s.studierenden_id,
            "status": status,
            "abschluss_semester_id": letztes_sem["semester_id"] if letztes_sem else None,
            "studiendauer_semester": letztes_sem["fachsemester"] if letztes_sem else 0,
            "abschlussnote": abschlussnote,
            "bachelorarbeitsnote": bachelorarbeitsnote,
            "anomalie_typ": s.anomalie_typ
        })

    return {
        "studierende_df": pd.DataFrame(studierende_rows),
        "einschreibungen_df": pd.DataFrame(einschreibungen_rows),
        "pruefungen_df": pd.DataFrame(pruefungen_rows),
        "support_teilnahmen_df": pd.DataFrame(support_teilnahmen_rows, columns=["studierenden_id", "semester_id", "angebot_id"]),
        "abschluesse_df": pd.DataFrame(abschluesse_rows),
    }

def _get_git_info() -> Dict[str, Optional[str]]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=2).decode().strip()
    except Exception:
        commit = "unknown"
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL, timeout=2).decode().strip()
    except Exception:
        branch = "unknown"
    try:
        status_out = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, timeout=2).decode().strip()
        is_dirty = len(status_out) > 0
    except Exception:
        is_dirty = None
    return {
        "commit": commit,
        "branch": branch,
        "dirty": is_dirty
    }

def schreibe_generation_metadata(
    output_dir: Path, 
    cfg: Optional[Dict] = None, 
    generator_script: Optional[str] = None, 
    extra_info: Optional[Dict] = None
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "generation_metadata.json"
    
    config_dump = copy.deepcopy(cfg if cfg is not None else CONFIG)
    safe_config = {}
    for k, v in config_dump.items():
        if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
            safe_config[k] = v
        else:
            safe_config[k] = str(v)
            
    meta = {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": _get_git_info(),
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable
        },
        "generator": {
            "entry_script": sys.argv[0] if sys.argv else "unknown",
            "generator_script": generator_script or (sys.argv[0] if sys.argv else "unknown")
        },
        "config": safe_config
    }
    if extra_info:
        meta["extra_info"] = extra_info
        
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {'generation_metadata.json':<25} geschrieben")

def exportiere_csv(
    daten: Dict[str, pd.DataFrame], 
    output_dir: Path, 
    cfg: Optional[Dict] = None,
    generator_script: Optional[str] = None,
    extra_info: Optional[Dict] = None
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, df in daten.items():
        if df is not None:
            pfad = output_dir / f"{key.replace('_df', '')}.csv"
            df.to_csv(pfad, index=False, sep=",", decimal=".")
            print(f"  [OK] {pfad.name:<25} {len(df):>8} Zeilen")
    schreibe_generation_metadata(output_dir, cfg=cfg, generator_script=generator_script, extra_info=extra_info)
