from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class ModulState:
    modul_id: str
    status: str = "offen"  # "offen", "bestanden", "gescheitert"
    versuche: int = 0
    note: Optional[float] = None

@dataclass
class PruefungsErgebnis:
    semester_id: str
    modul_id: str
    versuch: int
    note: float
    bestanden: bool
    note_counterfactual: float  # Note ohne Support-Boost (Ground Truth)
    support_genutzt: bool
    # Hidden Ground Truth variables for control/validation
    hidden_motivation: Optional[float] = None
    hidden_soziale_integration: Optional[float] = None
    hidden_erwartete_note: Optional[float] = None
    hidden_overload: Optional[float] = None
    hidden_zeit_puffer: Optional[float] = None
    hidden_penalty_capped: Optional[bool] = None
    hidden_support_capped: Optional[bool] = None

@dataclass
class Student:
    studierenden_id: str
    studiengang_id: str
    kohorten_semester_id: str
    geschlecht: str
    alter_immatrikulation: int
    hzb_note: float
    hzb_typ: str
    migrationshintergrund: bool
    erstakademiker: bool
    erwerbstaetigkeit_std: int
    
    # Latente Variablen (dynamisch)
    motivation: float
    soziale_integration: float
    erwartete_note: float = 2.5
    hidden_zeit_puffer: float = 60.0
    
    # Initiale Latente Variablen (zum Speichern)
    motivation_initial: float = 0.5
    soziale_integration_initial: float = 0.5
    erwartete_note_initial: float = 2.5
    
    # Status
    abgebrochen: bool = False
    exmatrikuliert: bool = False
    abschluss_erreicht: bool = False
    anomalie_typ: Optional[str] = None
    
    # Tracking
    modul_states: Dict[str, ModulState] = field(default_factory=dict)
    pruefungen: List[PruefungsErgebnis] = field(default_factory=list)
    einschreibungen: List[Dict] = field(default_factory=list)
    support_teilnahmen: List[Dict] = field(default_factory=list)

    def cp_bestanden(self, modul_cp_dict: Dict[str, int]) -> int:
        return sum(modul_cp_dict[m] for m, state in self.modul_states.items() if state.status == "bestanden")

    def alle_pflicht_bestanden(self, pflicht_module: List[str]) -> bool:
        return all(self.modul_states[m].status == "bestanden" for m in pflicht_module)
