"""
DeepSupport PyTorch Data Layer
==============================
Stellt standardisierte, hochperformante PyTorch Dataset- und DataLoader-
Pipelines für Panel- und Sequenz-Modelle bereit.

Basiert direkt auf dem Data Backbone (feature_builder.py), erzwingt
einen strikten 3-Way Student-Group-Split (70% Train, 15% Val, 15% Test)
ohne Datenleckage und fittet alle Sklearn-Transformer ausschließlich auf Train.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import torch
from torch.utils.data import Dataset, DataLoader

import deepsupport.data_engine.feature_builder as fb


class StudyPanelDataset(Dataset):
    """
    PyTorch Dataset für 2D-Person-Semester- oder Prüfungs-Panels.
    
    Liefert (x, t_start, duration, event) bzw. (x, target) für Survival- und Regressionsmodelle.
    """
    def __init__(
        self,
        features: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        t_starts: Optional[np.ndarray] = None,
        competing_events: Optional[np.ndarray] = None,
        timesteps: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        student_ids: Optional[np.ndarray] = None,
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.durations = torch.tensor(durations, dtype=torch.float32)
        self.events = torch.tensor(events, dtype=torch.float32)
        if t_starts is not None:
            self.t_starts = torch.tensor(t_starts, dtype=torch.float32)
        else:
            self.t_starts = torch.clamp(self.durations - 1.0, min=0.0)
        self.competing_events = torch.tensor(competing_events, dtype=torch.int64) if competing_events is not None else None
        self.timesteps = torch.tensor(timesteps, dtype=torch.int64) if timesteps is not None else None
        self.weights = torch.tensor(weights, dtype=torch.float32) if weights is not None else None
        self.student_ids = student_ids

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "x": self.features[idx],
            "t_start": self.t_starts[idx],
            "duration": self.durations[idx],
            "event": self.events[idx],
        }
        if self.competing_events is not None:
            item["competing_event"] = self.competing_events[idx]
        if self.timesteps is not None:
            item["timestep"] = self.timesteps[idx]
        if self.weights is not None:
            item["weight"] = self.weights[idx]
        return item


class StudySequenceDataset(Dataset):
    """
    PyTorch Dataset für 3D-Sequenzen (N, SeqLen, FeatDim).
    
    Unterstützt Masking für gepaddete Zeitschritte und optionale statische Kontexte.
    """
    def __init__(
        self,
        sequences: np.ndarray,
        targets: Optional[np.ndarray] = None,
        targets_note: Optional[np.ndarray] = None,
        targets_pass: Optional[np.ndarray] = None,
        context: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        student_ids: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
    ):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        self.targets_note = torch.tensor(targets_note, dtype=torch.float32) if targets_note is not None else None
        self.targets_pass = torch.tensor(targets_pass, dtype=torch.float32) if targets_pass is not None else None
        self.context = torch.tensor(context, dtype=torch.float32) if context is not None else None
        self.sample_weights = torch.tensor(sample_weights, dtype=torch.float32) if sample_weights is not None else None
        
        if masks is not None:
            self.masks = torch.tensor(masks, dtype=torch.bool)
        else:
            # Automatisches Masking anhand des PADDING_VALUE (-99.0)
            self.masks = (self.sequences[:, :, 0] != fb.PADDING_VALUE)

        self.student_ids = student_ids

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "sequence": self.sequences[idx],
            "mask": self.masks[idx],
        }
        if self.targets is not None:
            item["target"] = self.targets[idx]
        if self.context is not None:
            item["context"] = self.context[idx]
        if self.targets_note is not None:
            item["target_note"] = self.targets_note[idx]
        if self.targets_pass is not None:
            item["target_pass"] = self.targets_pass[idx]
        if self.sample_weights is not None:
            item["sample_weight"] = self.sample_weights[idx]
        return item


def prepare_panel_dataloaders(
    data_dir: Union[str, Path] = Path("data_v4_grid/S01_baseline/universe_A"),
    mode: str = "standard",
    temporal: str = "prev",
    batch_size: int = 2048,
    num_workers: int = 0,
    seed: int = 42,
    panel_type: str = "semester",
) -> Dict[str, Any]:
    """
    Erstellt vollständig vorkonfigurierte PyTorch-DataLoader für Panel-Daten
    mit striktem Student-Group-Split (70% Train, 15% Val, 15% Test).
    """
    data_path = Path(data_dir)
    
    if panel_type == "semester":
        panel_df, feature_cols, target_col, meta = fb.build_semester_panel_df(
            data_path, mode=mode, temporal=temporal
        )
    elif panel_type == "exam":
        panel_df, feature_cols, target_col, meta = fb.build_exam_panel_df(
            data_path, mode=mode, temporal=temporal
        )
    else:
        raise ValueError(f"Unbekannter panel_type: '{panel_type}'. Erlaubt sind 'semester' oder 'exam'.")

    # Spaltentypen identifizieren
    cat_candidates = ["stg_name", "erstakademiker", "hzb_typ", "migrationshintergrund"]
    cat_cols = [c for c in cat_candidates if c in feature_cols]

    treatment_candidates = ["fach_supp_count", "uebf_supp_count", "psych_supp_count"]
    treatment_cols = [c for c in treatment_candidates if c in feature_cols]

    num_cols = [c for c in feature_cols if c not in cat_cols and c not in treatment_cols]

    # Strikter 3-Way Student Group Split
    unique_studis = np.array(panel_df["studierenden_id"].unique().tolist())
    train_ids, temp_ids = train_test_split(unique_studis, test_size=0.30, random_state=seed)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=seed)

    train_panel = panel_df[panel_df["studierenden_id"].isin(train_ids)].copy()
    val_panel = panel_df[panel_df["studierenden_id"].isin(val_ids)].copy()
    test_panel = panel_df[panel_df["studierenden_id"].isin(test_ids)].copy()

    # Preprocessing Pipeline ausschließlich auf Train-Panel fitten
    # Sicherstellen, dass alle numerischen und Treatment-Spalten als float32 vorliegen,
    # um Typ-Inkompatibilitäten beim SimpleImputer mit int64-Spalten zu vermeiden
    for col in num_cols + treatment_cols:
        train_panel[col] = pd.to_numeric(train_panel[col], errors="coerce").astype(np.float32)
        val_panel[col] = pd.to_numeric(val_panel[col], errors="coerce").astype(np.float32)
        test_panel[col] = pd.to_numeric(test_panel[col], errors="coerce").astype(np.float32)

    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            num_cols,
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
            ]),
            cat_cols,
        ))
    if treatment_cols:
        transformers.append((
            "treat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value=np.float32(0.0))),
                ("scaler", StandardScaler()),
            ]),
            treatment_cols,
        ))

    preprocessor = ColumnTransformer(transformers=transformers)
    X_train = preprocessor.fit_transform(train_panel[feature_cols]).astype(np.float32)
    X_val = preprocessor.transform(val_panel[feature_cols]).astype(np.float32)
    X_test = preprocessor.transform(test_panel[feature_cols]).astype(np.float32)

    # Durations, t_start und Events extrahieren
    # Duration = t_stop (Semester), t_start = Intervallbeginn, Event = target_col
    t_train = train_panel["t_stop"].values.astype(np.float32)
    t_start_train = train_panel["t_start"].values.astype(np.float32) if "t_start" in train_panel.columns else np.maximum(0.0, t_train - 1.0).astype(np.float32)
    e_train = train_panel[target_col].values.astype(np.float32)

    t_val = val_panel["t_stop"].values.astype(np.float32)
    t_start_val = val_panel["t_start"].values.astype(np.float32) if "t_start" in val_panel.columns else np.maximum(0.0, t_val - 1.0).astype(np.float32)
    e_val = val_panel[target_col].values.astype(np.float32)

    t_test = test_panel["t_stop"].values.astype(np.float32)
    t_start_test = test_panel["t_start"].values.astype(np.float32) if "t_start" in test_panel.columns else np.maximum(0.0, t_test - 1.0).astype(np.float32)
    e_test = test_panel[target_col].values.astype(np.float32)

    # Diskrete Zeitschritte (1 bis 16 Semester, 0-basiert: 0 bis 15 für Hazard-Bins)
    step_train = np.clip(train_panel["fachsemester"].values.astype(np.int64) - 1, 0, 15)
    step_val = np.clip(val_panel["fachsemester"].values.astype(np.int64) - 1, 0, 15)
    step_test = np.clip(test_panel["fachsemester"].values.astype(np.int64) - 1, 0, 15)

    # Competing events (0 = censored, 1 = dropout, 2 = graduation)
    if "competing_event" in train_panel.columns:
        ce_train = train_panel["competing_event"].values.astype(np.int64)
        ce_val = val_panel["competing_event"].values.astype(np.int64)
        ce_test = test_panel["competing_event"].values.astype(np.int64)
    else:
        ce_train = ce_val = ce_test = None

    # Datasets erstellen
    train_dataset = StudyPanelDataset(X_train, t_train, e_train, t_starts=t_start_train, competing_events=ce_train, timesteps=step_train, student_ids=train_panel["studierenden_id"].values)
    val_dataset = StudyPanelDataset(X_val, t_val, e_val, t_starts=t_start_val, competing_events=ce_val, timesteps=step_val, student_ids=val_panel["studierenden_id"].values)
    test_dataset = StudyPanelDataset(X_test, t_test, e_test, t_starts=t_start_test, competing_events=ce_test, timesteps=step_test, student_ids=test_panel["studierenden_id"].values)

    # Reproduzierbare Worker-Initialisierung
    def _seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        generator=g,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "input_dim": X_train.shape[1],
        "X_train": X_train,
        "t_start_train": t_start_train,
        "t_train": t_train,
        "e_train": e_train,
        "ce_train": ce_train,
        "X_val": X_val,
        "t_start_val": t_start_val,
        "t_val": t_val,
        "e_val": e_val,
        "ce_val": ce_val,
        "X_test": X_test,
        "t_start_test": t_start_test,
        "t_test": t_test,
        "e_test": e_test,
        "ce_test": ce_test,
        "train_panel": train_panel,
        "val_panel": val_panel,
        "test_panel": test_panel,
        "preprocessor": preprocessor,
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "treatment_cols": treatment_cols,
        "target_col": target_col,
    }


def prepare_sequence_dataloaders(
    data_dir: Union[str, Path] = Path("data_v4_grid/S01_baseline/universe_A"),
    mode: str = "standard",
    temporal: str = "prev",
    batch_size: int = 512,
    num_workers: int = 0,
    seed: int = 42,
    seq_type: str = "exam",
    max_len: int = 40,
) -> Dict[str, Any]:
    """
    Erstellt PyTorch DataLoader für 3D-Sequenzmodelle (Prüfungsverläufe oder Semesterfolgen).
    Group-Split erfolgt strikt auf Studierenden-Ebene.
    """
    data_path = Path(data_dir)
    df_abschluesse, _ = fb._load_raw_data(data_path)
    note_dict = df_abschluesse.set_index("studierenden_id")["abschlussnote"].to_dict()
    status_dict = df_abschluesse.set_index("studierenden_id")["status"].to_dict()

    if seq_type == "exam":
        studis, X_seq, _, _, feat_names, _ = fb.build_exam_sequence_tensor(
            data_path, max_exams=max_len, mode=mode, temporal=temporal, target_type="grade"
        )
    elif seq_type == "semester":
        studis, X_seq, _, _, feat_names, _ = fb.build_semester_sequence_tensor(
            data_path, max_semesters=16, mode=mode, temporal=temporal, target_type="gpa"
        )
    else:
        raise ValueError(f"Unbekannter seq_type: '{seq_type}'. Erlaubt sind 'exam' oder 'semester'.")

    # Target: Abschlussnote (kontinuierlich) und Dropout/Abschluss (binär)
    y_note_all = np.array([note_dict.get(s, np.nan) for s in studis], dtype=np.float32)
    # 1 für Dropout, 0 für Abschluss
    dropout_statuses = {"abgebrochen", "exmatrikuliert", "zeitueberschreitung"}
    y_drop_all = np.array([1.0 if str(status_dict.get(s, "")).strip().lower() in dropout_statuses else 0.0 for s in studis], dtype=np.float32)

    # Strikter Student-Level Split (70/15/15) auf allen Studierenden
    unique_studis = np.unique(studis)
    train_ids, temp_ids = train_test_split(unique_studis, test_size=0.30, random_state=seed)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=seed)

    train_mask = np.isin(studis, train_ids)
    val_mask = np.isin(studis, val_ids)
    test_mask = np.isin(studis, test_ids)

    train_seq = X_seq[train_mask]
    val_seq = X_seq[val_mask]
    test_seq = X_seq[test_mask]

    # Preprocessing: StandardScaler nur auf unpadded Werten des Train-Splits
    v_mask_tr = (train_seq[:, :, 0] != fb.PADDING_VALUE)
    scaler = StandardScaler()
    scaler.fit(train_seq[v_mask_tr])

    def _transform_seq(seq_arr: np.ndarray) -> np.ndarray:
        out = seq_arr.copy()
        vm = (seq_arr[:, :, 0] != fb.PADDING_VALUE)
        if np.any(vm):
            out[vm] = scaler.transform(seq_arr[vm])
        return out

    train_seq_norm = _transform_seq(train_seq).astype(np.float32)
    val_seq_norm = _transform_seq(val_seq).astype(np.float32)
    test_seq_norm = _transform_seq(test_seq).astype(np.float32)

    train_note = y_note_all[train_mask]
    val_note = y_note_all[val_mask]
    test_note = y_note_all[test_mask]

    train_drop = y_drop_all[train_mask]
    val_drop = y_drop_all[val_mask]
    test_drop = y_drop_all[test_mask]

    train_dataset = StudySequenceDataset(train_seq_norm, targets_note=train_note, targets_pass=train_drop, student_ids=studis[train_mask])
    val_dataset = StudySequenceDataset(val_seq_norm, targets_note=val_note, targets_pass=val_drop, student_ids=studis[val_mask])
    test_dataset = StudySequenceDataset(test_seq_norm, targets_note=test_note, targets_pass=test_drop, student_ids=studis[test_mask])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "seq_len": X_seq.shape[1],
        "feature_dim": X_seq.shape[2],
        "scaler": scaler,
        "feature_names": feat_names,
        "y_note_test": test_note,
        "y_drop_test": test_drop,
        "y_note_val": val_note,
        "y_drop_val": val_drop,
    }


def prepare_exam_regressor_dataloaders(
    data_dir: Union[str, Path] = Path("data_v4_grid/S01_baseline/universe_A"),
    mode: str = "gradeblind",
    temporal: str = "prev",
    batch_size: int = 256,
    seed: int = 42,
    max_len: int = 40,
) -> Dict[str, Any]:
    """
    Bereitet Daten für den PyTorchExamTransformerRegressor vor:
    - Ausschließlich Absolventen (Studierende mit gültiger Abschlussnote)
    - Default-Modus: gradeblind (strikt ohne Notenmerkmale)
    - Group-konsistenter Student-Split (70/15/15)
    - StandardScaler gefittet nur auf Train-Split (unpadded Schritte)
    """
    data_path = Path(data_dir)
    studis, X_seq, _, _, feat_names, _ = fb.build_exam_sequence_tensor(
        data_path, max_exams=max_len, mode=mode, temporal=temporal, target_type="grade"
    )
    df_abschluesse, _ = fb._load_raw_data(data_path)
    note_dict = df_abschluesse.set_index("studierenden_id")["abschlussnote"].to_dict()
    y_all = np.array([note_dict.get(s, np.nan) for s in studis], dtype=np.float32)

    # Filtere strikt auf Absolventen
    valid_grad = ~np.isnan(y_all)
    studis_clean = studis[valid_grad]
    X_clean = X_seq[valid_grad]
    y_clean = y_all[valid_grad]

    unique_studis = np.unique(studis_clean)
    train_ids, temp_ids = train_test_split(unique_studis, test_size=0.30, random_state=seed)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=seed)

    train_mask = np.isin(studis_clean, train_ids)
    val_mask = np.isin(studis_clean, val_ids)
    test_mask = np.isin(studis_clean, test_ids)

    X_train = X_clean[train_mask].copy()
    X_val = X_clean[val_mask].copy()
    X_test = X_clean[test_mask].copy()

    y_train = y_clean[train_mask]
    y_val = y_clean[val_mask]
    y_test = y_clean[test_mask]

    # Preprocessing: StandardScaler auf unpadded Train-Tokens
    vm_tr = (X_train[:, :, 0] != fb.PADDING_VALUE)
    scaler = StandardScaler()
    scaler.fit(X_train[vm_tr])

    def _transform(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        vm = (arr[:, :, 0] != fb.PADDING_VALUE)
        if np.any(vm):
            out[vm] = scaler.transform(arr[vm])
        return out.astype(np.float32)

    X_train_norm = _transform(X_train)
    X_val_norm = _transform(X_val)
    X_test_norm = _transform(X_test)

    train_ds = StudySequenceDataset(X_train_norm, targets=y_train, student_ids=studis_clean[train_mask])
    val_ds = StudySequenceDataset(X_val_norm, targets=y_val, student_ids=studis_clean[val_mask])
    test_ds = StudySequenceDataset(X_test_norm, targets=y_test, student_ids=studis_clean[test_mask])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "X_test": X_test_norm,
        "y_test": y_test,
        "test_mask": (X_test_norm[:, :, 0] != fb.PADDING_VALUE),
        "feat_names": feat_names,
        "feature_dim": X_clean.shape[2],
        "scaler": scaler,
        "n_graduates_test": len(y_test),
    }


def prepare_causal_survival_dataloaders(
    data_dir: Union[str, Path] = Path("data_v4_grid/S01_baseline/universe_A"),
    mode: str = "gradeblind",
    temporal: str = "prev",
    batch_size: int = 256,
    seed: int = 42,
    max_len: int = 40,
) -> Dict[str, Any]:
    """
    Bereitet Daten für PyTorchCausalExamTransformerSurvival vor:
    - Gesamte Kohorte (Absolventen + Abbrecher)
    - Kausales Prüfungsschritt-Target y_seq (N, max_len, 1)
    - 70/15/15 Split stratifiziert nach Endstatus (Dropout vs. Abschluss)
    - StandardScaler auf unpadded Train-Tokens
    """
    data_path = Path(data_dir)
    studis, X_seq, y_seq, studi_events, feat_names, _ = fb.build_exam_sequence_tensor(
        data_path, max_exams=max_len, mode=mode, temporal=temporal, target_type="dropout"
    )

    # 3-Way Stratified Split auf Student-Ebene
    unique_studis = np.unique(studis)
    train_ids, temp_ids = train_test_split(unique_studis, test_size=0.30, random_state=seed, stratify=studi_events)
    temp_mask = np.isin(studis, temp_ids)
    temp_events = studi_events[temp_mask]
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=seed, stratify=temp_events)

    train_mask = np.isin(studis, train_ids)
    val_mask = np.isin(studis, val_ids)
    test_mask = np.isin(studis, test_ids)

    X_train = X_seq[train_mask].copy()
    X_val = X_seq[val_mask].copy()
    X_test = X_seq[test_mask].copy()

    y_train = y_seq[train_mask].squeeze(-1)  # (N, S)
    y_val = y_seq[val_mask].squeeze(-1)
    y_test = y_seq[test_mask].squeeze(-1)

    # Preprocessing: StandardScaler auf unpadded Train-Tokens
    vm_tr = (X_train[:, :, 0] != fb.PADDING_VALUE)
    scaler = StandardScaler()
    scaler.fit(X_train[vm_tr])

    def _transform(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        vm = (arr[:, :, 0] != fb.PADDING_VALUE)
        if np.any(vm):
            out[vm] = scaler.transform(arr[vm])
        return out.astype(np.float32)

    X_train_norm = _transform(X_train)
    X_val_norm = _transform(X_val)
    X_test_norm = _transform(X_test)

    train_ds = StudySequenceDataset(X_train_norm, targets=y_train, student_ids=studis[train_mask])
    val_ds = StudySequenceDataset(X_val_norm, targets=y_val, student_ids=studis[val_mask])
    test_ds = StudySequenceDataset(X_test_norm, targets=y_test, student_ids=studis[test_mask])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "X_test": X_test_norm,
        "y_test": y_test,
        "test_mask": (X_test_norm[:, :, 0] != fb.PADDING_VALUE),
        "test_student_events": studi_events[test_mask],
        "feat_names": feat_names,
        "feature_dim": X_seq.shape[2],
        "scaler": scaler,
        "n_students_test": len(test_ids),
    }


def prepare_next_exam_dual_head_dataloaders(
    data_dir: Union[str, Path] = Path("data_v4_grid/S01_baseline/universe_A"),
    max_history_len: int = 30,
    batch_size: int = 256,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Bereitet Daten für autoregressive Next-Exam Dual-Head Multi-Task Modelle vor:
    - Eingabe 1: Prüfungshistorie 1..k (N, max_history_len, 12)
    - Eingabe 2: Nächster Prüfungskontext k+1 + Demographie (N, 14)
    - Target 1: Note k+1 (Regression)
    - Target 2: Bestanden k+1 (Klassifikation)
    - 3-Way Group-Consistent Split auf Student-ID-Ebene (70/15/15)
    - Skalierung via StandardScaler strikt auf Train-Tokens
    """
    data_path = Path(data_dir)
    from deepsupport.models.autoregressive_gru import prepare_next_exam_dataset

    X_hist, X_ctx, y_grade, y_pass, student_ids = prepare_next_exam_dataset(
        data_path, max_history_len=max_history_len
    )

    if max_samples is not None and max_samples < len(X_hist):
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(len(X_hist), size=max_samples, replace=False)
        X_hist = X_hist[sample_idx]
        X_ctx = X_ctx[sample_idx]
        y_grade = y_grade[sample_idx]
        y_pass = y_pass[sample_idx]
        student_ids = student_ids[sample_idx]

    # 3-Way Group-Consistent Split auf Student IDs
    unique_students = np.unique(student_ids)
    tr_students, temp_students = train_test_split(unique_students, test_size=0.30, random_state=seed)
    va_students, te_students = train_test_split(temp_students, test_size=0.50, random_state=seed)

    tr_idx = np.where(np.isin(student_ids, tr_students))[0]
    va_idx = np.where(np.isin(student_ids, va_students))[0]
    te_idx = np.where(np.isin(student_ids, te_students))[0]

    # Preprocessing: StandardScaler auf unpadded Train-Tokens und Context
    vm_tr = (X_hist[tr_idx, :, 0] != fb.PADDING_VALUE)
    scaler_seq = StandardScaler()
    scaler_seq.fit(X_hist[tr_idx][vm_tr])

    scaler_ctx = StandardScaler()
    scaler_ctx.fit(X_ctx[tr_idx])

    def _transform_seq(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        vm = (arr[:, :, 0] != fb.PADDING_VALUE)
        if np.any(vm):
            out[vm] = scaler_seq.transform(arr[vm])
        return out.astype(np.float32)

    X_hist_tr = _transform_seq(X_hist[tr_idx])
    X_hist_va = _transform_seq(X_hist[va_idx])
    X_hist_te = _transform_seq(X_hist[te_idx])

    X_ctx_tr = scaler_ctx.transform(X_ctx[tr_idx]).astype(np.float32)
    X_ctx_va = scaler_ctx.transform(X_ctx[va_idx]).astype(np.float32)
    X_ctx_te = scaler_ctx.transform(X_ctx[te_idx]).astype(np.float32)

    y_grade_tr = y_grade[tr_idx].astype(np.float32)
    y_grade_va = y_grade[va_idx].astype(np.float32)
    y_grade_te = y_grade[te_idx].astype(np.float32)

    y_pass_tr = y_pass[tr_idx].astype(np.float32)
    y_pass_va = y_pass[va_idx].astype(np.float32)
    y_pass_te = y_pass[te_idx].astype(np.float32)

    def _student_sample_weights(s_ids: np.ndarray) -> np.ndarray:
        uniq, cnts = np.unique(s_ids, return_counts=True)
        cmap = dict(zip(uniq, cnts))
        sw = np.array([1.0 / max(1, cmap[s]) for s in s_ids], dtype=np.float32)
        sw *= len(sw) / max(float(np.sum(sw)), 1e-8)
        return sw

    sw_tr = _student_sample_weights(student_ids[tr_idx])
    sw_va = _student_sample_weights(student_ids[va_idx])

    train_ds = StudySequenceDataset(
        X_hist_tr,
        context=X_ctx_tr,
        targets_note=y_grade_tr,
        targets_pass=y_pass_tr,
        student_ids=student_ids[tr_idx],
        sample_weights=sw_tr,
    )
    val_ds = StudySequenceDataset(
        X_hist_va,
        context=X_ctx_va,
        targets_note=y_grade_va,
        targets_pass=y_pass_va,
        student_ids=student_ids[va_idx],
        sample_weights=sw_va,
    )
    test_ds = StudySequenceDataset(
        X_hist_te,
        context=X_ctx_te,
        targets_note=y_grade_te,
        targets_pass=y_pass_te,
        student_ids=student_ids[te_idx],
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "X_hist_test": X_hist_te,
        "X_ctx_test": X_ctx_te,
        "y_grade_test": y_grade_te,
        "y_pass_test": y_pass_te,
        "student_ids_test": student_ids[te_idx],
        "seq_features": X_hist.shape[2],
        "context_features": X_ctx.shape[1],
        "scaler_seq": scaler_seq,
        "scaler_ctx": scaler_ctx,
        "n_samples_test": len(te_idx),
    }


def prepare_sequence_survival_dataloaders(
    data_dir: Union[str, Path] = Path("data_v4_grid/S01_baseline/universe_A"),
    sequence_type: str = "semester",
    max_len: int = 16,
    mode: str = "standard",
    temporal: str = "prev",
    batch_size: int = 256,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Bereitet Daten für sequentielle Survival-Modelle (Semester- oder Prüfungssequenzen) vor:
    - 3D-Tensor (N, max_len, feature_dim) mit Zeitschritt-Target y_seq (N, max_len, 1)
    - 3-Way Stratified Split auf Student-Ebene
    - Skalierung via StandardScaler auf aktiven Train-Tokens
    - Optionales Subsampling via max_samples für schnelle Integrationstests
    """
    data_path = Path(data_dir)
    if sequence_type == "semester":
        studis, X_seq, y_seq, studi_events, feat_names, _ = fb.build_semester_sequence_tensor(
            data_path, max_semesters=max_len, mode=mode, temporal=temporal
        )
    elif sequence_type == "exam":
        studis, X_seq, y_seq, studi_events, feat_names, _ = fb.build_exam_sequence_tensor(
            data_path, max_exams=max_len, mode=mode, temporal=temporal, target_type="dropout"
        )
    else:
        raise ValueError(f"Unbekannter sequence_type: {sequence_type}. Erwartet: 'semester' oder 'exam'.")

    n_samples = len(studis)
    if max_samples is not None and max_samples < n_samples:
        rng = np.random.default_rng(seed)
        sub_idx = rng.choice(n_samples, size=max_samples, replace=False)
        studis = studis[sub_idx]
        X_seq = X_seq[sub_idx]
        y_seq = y_seq[sub_idx]
        studi_events = studi_events[sub_idx]
        n_samples = len(studis)

    idx = np.arange(n_samples)
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=seed, stratify=studi_events)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=seed, stratify=studi_events[temp_idx])

    X_train, y_train = X_seq[train_idx].copy(), y_seq[train_idx].copy()
    X_val, y_val = X_seq[val_idx].copy(), y_seq[val_idx].copy()
    X_test, y_test = X_seq[test_idx].copy(), y_seq[test_idx].copy()

    # Preprocessing: StandardScaler auf unpadded Train-Tokens
    vm_tr = (X_train[:, :, 0] != fb.PADDING_VALUE)
    scaler = StandardScaler()
    scaler.fit(X_train[vm_tr])

    def _transform(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        vm = (arr[:, :, 0] != fb.PADDING_VALUE)
        if np.any(vm):
            out[vm] = scaler.transform(arr[vm])
        return out.astype(np.float32)

    X_train_norm = _transform(X_train)
    X_val_norm = _transform(X_val)
    X_test_norm = _transform(X_test)

    # y_seq: Squeeze last dimension falls nötig -> (N, S)
    if y_train.ndim == 3 and y_train.shape[2] == 1:
        y_train = y_train.squeeze(-1)
        y_val = y_val.squeeze(-1)
        y_test = y_test.squeeze(-1)

    train_ds = StudySequenceDataset(X_train_norm, targets=y_train, student_ids=studis[train_idx])
    val_ds = StudySequenceDataset(X_val_norm, targets=y_val, student_ids=studis[val_idx])
    test_ds = StudySequenceDataset(X_test_norm, targets=y_test, student_ids=studis[test_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "X_test": X_test_norm,
        "y_test": y_test,
        "test_mask": (X_test_norm[:, :, 0] != fb.PADDING_VALUE),
        "test_student_events": (studi_events[test_idx] == 1).astype(int),
        "feat_names": feat_names,
        "feature_dim": X_seq.shape[2],
        "max_len": max_len,
        "scaler": scaler,
        "n_students_test": len(test_idx),
    }

