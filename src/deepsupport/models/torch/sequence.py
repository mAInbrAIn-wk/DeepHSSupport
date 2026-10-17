"""
DeepSupport PyTorch Sequentielle Survival-Modelle
================================================
Modelliert dynamische Ausfallrisiken über den Zeitverlauf aus 3D-Sequenztensoren
(Semester- oder Prüfungsverläufe) mit diskreter Intervall-Hazard-Likelihood.

Architekturen:
1. PyTorchSemesterGRU:
   - Pre-LayerNorm 2-Stufen GRU (64 -> 32)
   - TimeDistributed Hazard-Head h(t | X_{1..t})
   - Masked Binary Cross-Entropy Loss mit Logits

2. PyTorchSemesterTransformer:
   - SinCos-Positional-Encoding + Kausale Attention (FlashAttention)
   - TimeDistributed Hazard-Head an jedem Semester-Schritt

3. PyTorchExamGRU:
   - Prüfungsfeine dynamische Sequenzmodellierung über bis zu 35 Prüfungsschritte

4. PyTorchDynamicDeepHit:
   - Diskretes Dynamic DeepHit für Longitudinalverläufe
   - Joint Loss aus PMF Negative Log-Likelihood + Cause-Specific Ranking Loss
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

from deepsupport.models.torch.transformer import (
    SinCosPositionalEncoding,
    TransformerEncoderBlock,
    AttentionPooling,
)
from deepsupport.evaluation.metrics_logger import (
    SurvivalEvaluator,
    save_metrics,
)
import deepsupport.data_engine.feature_builder as fb


class PyTorchSemesterGRU(nn.Module):
    """
    Sequentielles GRU-Survival-Modell über Personen-Semester.
    """
    def __init__(
        self,
        feat_dim: int,
        hidden_dim1: int = 64,
        hidden_dim2: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.feat_dim = feat_dim

        self.input_proj = nn.Linear(feat_dim, hidden_dim1)
        self.norm1 = nn.LayerNorm(hidden_dim1)

        self.gru1 = nn.GRU(hidden_dim1, hidden_dim1, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim1)

        self.gru2 = nn.GRU(hidden_dim1, hidden_dim2, batch_first=True)
        self.norm3 = nn.LayerNorm(hidden_dim2)

        self.hazard_head = nn.Sequential(
            nn.Linear(hidden_dim2, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward_logits(self, sequence: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        sequence: (N, S, feat_dim)
        mask: (N, S) True für valide Zeitschritte
        Rückgabe: (N, S) unnormalisierte Logits für Hazard h(t)
        """
        h = self.input_proj(sequence)
        h = self.norm1(h)
        h, _ = self.gru1(h)
        h = self.norm2(h)
        h, _ = self.gru2(h)
        h = self.norm3(h)

        logits = self.hazard_head(h).squeeze(-1)
        return logits

    def forward(self, sequence: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.forward_logits(sequence, mask=mask)
        return torch.sigmoid(logits)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        seq = batch["sequence"]
        y_true = batch["target"]  # (N, S) mit 1.0, 0.0 oder -99.0
        mask = (y_true != fb.PADDING_VALUE)

        logits = self.forward_logits(seq, mask=mask)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=seq.device, requires_grad=True)

        valid_logits = logits[mask]
        valid_targets = y_true[mask]
        return F.binary_cross_entropy_with_logits(valid_logits, valid_targets)


class PyTorchSemesterTransformer(nn.Module):
    """
    Kausales Transformer-Survival-Modell über Semestersequenzen.
    """
    def __init__(
        self,
        feat_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 20,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.d_model = d_model

        self.input_proj = nn.Linear(feat_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.pos_encoder = SinCosPositionalEncoding(d_model=d_model, max_len=max_seq_len + 10)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                is_causal=True,
            )
            for _ in range(n_layers)
        ])

        self.hazard_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward_logits(self, sequence: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.input_proj(sequence)
        h = self.norm(h)
        h = self.pos_encoder(h)

        for block in self.blocks:
            h = block(h, key_padding_mask=mask)

        logits = self.hazard_head(h).squeeze(-1)
        return logits

    def forward(self, sequence: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.forward_logits(sequence, mask=mask)
        return torch.sigmoid(logits)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        seq = batch["sequence"]
        y_true = batch["target"]
        mask = (y_true != fb.PADDING_VALUE)

        logits = self.forward_logits(seq, mask=mask)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=seq.device, requires_grad=True)

        valid_logits = logits[mask]
        valid_targets = y_true[mask]
        return F.binary_cross_entropy_with_logits(valid_logits, valid_targets)


class PyTorchExamGRU(nn.Module):
    """
    Prüfungsfeines GRU-Survival-Modell über Klausursequenzen (bis zu 35 Klausuren).
    """
    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.feat_dim = feat_dim

        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.hazard_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward_logits(self, sequence: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.input_proj(sequence)
        h = self.norm1(h)
        h, _ = self.gru(h)
        h = self.norm2(h)

        logits = self.hazard_head(h).squeeze(-1)
        return logits

    def forward(self, sequence: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.forward_logits(sequence, mask=mask)
        return torch.sigmoid(logits)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        seq = batch["sequence"]
        y_true = batch["target"]
        mask = (y_true != fb.PADDING_VALUE)

        logits = self.forward_logits(seq, mask=mask)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=seq.device, requires_grad=True)

        return F.binary_cross_entropy_with_logits(logits[mask], y_true[mask])


class PyTorchDynamicDeepHit(nn.Module):
    """
    Strikter kausaler Dynamic DeepHit für Longitudinalverläufe (Lee et al., 2018).
    Berechnet an jedem Semester-Zeitschritt t aus dem unidirektionalen (kausalen) GRU-Zustand h_t
    den bedingten Intervall-Hazard h(t | X_{1..t}) (ohne Future-Pooling über zukünftige Semester)
    und kombiniert die maskierte Intervall-Log-Likelihood mit dem paarweisen Konkordanz-Ranking-Loss
    auf der kumulativen Inzidenz F_i(T_i) vs. F_j(T_i) zum Ereigniszeitpunkt T_i (für Paare mit e_i=1 und T_i < T_j).
    """
    def __init__(
        self,
        feat_dim: int,
        max_duration: int = 16,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        alpha_rank: float = 0.1,
        sigma_rank: float = 0.1,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.max_duration = max_duration
        self.alpha_rank = alpha_rank
        self.sigma_rank = sigma_rank

        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.gru1 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim // 2, batch_first=True)
        self.norm3 = nn.LayerNorm(hidden_dim // 2)

        self.hazard_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward_logits(self, sequence: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Kausale Zeitschritt-Logits (N, S) ausschließlich aus der Historie X_{1..t}."""
        h = self.input_proj(sequence)
        h = self.norm1(h)
        h, _ = self.gru1(h)
        h = self.norm2(h)
        h, _ = self.gru2(h)
        h = self.norm3(h)
        return self.hazard_head(h).squeeze(-1)

    def forward(self, sequence: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Gibt den kausalen bedingten Intervall-Hazard h_t(X_{1..t}) in (0, 1) zurück (Shape: N, S)."""
        return torch.sigmoid(self.forward_logits(sequence, mask=mask))

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        seq = batch["sequence"]
        y_true = batch["target"]  # (N, S)
        valid_mask = (y_true != fb.PADDING_VALUE)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=seq.device, requires_grad=True)

        logits = self.forward_logits(seq)
        nll_loss = F.binary_cross_entropy_with_logits(logits[valid_mask], y_true[valid_mask])

        # Kausale kumulative Inzidenz F(t | X_{1..t}) = 1 - prod_{u <= t} (1 - h_u)
        hazards = torch.sigmoid(logits) * valid_mask.float()
        surv = torch.cumprod(1.0 - hazards + 1e-7, dim=-1)
        cif = 1.0 - surv  # (N, S)

        lengths = valid_mask.sum(dim=1).clamp(min=1)
        last_idx = (lengths - 1).clamp(max=seq.size(1) - 1)
        row_indices = torch.arange(len(y_true), device=seq.device)
        event = (y_true[row_indices, last_idx] == 1.0).float()

        # Paarweiser Konkordanz-Ranking-Loss nach Lee et al. (2018) zum Zeitpunkt T_i: F_i(T_i) vs. F_j(T_i)
        durations = last_idx.float()
        dur_diff = durations.unsqueeze(1) - durations.unsqueeze(0)  # T_i - T_j
        pair_mask = (event.unsqueeze(1) == 1.0) & (dur_diff < 0)    # e_i == 1 und T_i < T_j

        if pair_mask.sum() > 0:
            n_batch = len(y_true)
            cif_i_at_ti = cif[row_indices, last_idx].unsqueeze(1).expand(-1, n_batch)  # F_i(T_i)
            t_i_idx = last_idx.unsqueeze(1).expand(-1, n_batch)
            cif_j_at_ti = cif[torch.arange(n_batch, device=seq.device).unsqueeze(0).expand(n_batch, -1), t_i_idx]  # F_j(T_i)
            rank_loss = torch.exp(-(cif_i_at_ti - cif_j_at_ti) / self.sigma_rank)[pair_mask].mean()
        else:
            rank_loss = torch.tensor(0.0, device=seq.device)

        return (1.0 - self.alpha_rank) * nll_loss + self.alpha_rank * rank_loss


def train_torch_sequence_survival_model(
    model_type: str = "semester_gru",
    data_dir: Union[str, Path] = Path("data_v4_grid/S01_baseline/universe_A"),
    output_dir: Optional[Union[str, Path]] = None,
    mode: str = "standard",
    temporal: str = "prev",
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 0.0015,
    weight_decay: float = 1e-4,
    patience: int = 6,
    device: Optional[str] = None,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Trainiert und evaluiert ein sequentielles PyTorch Survival-Modell.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    data_path = Path(data_dir)
    out_path = Path(output_dir) if output_dir is not None else data_path

    from deepsupport.models.torch.data_loaders import prepare_sequence_survival_dataloaders

    seq_type = "exam" if "exam" in model_type.lower() else "semester"
    max_len = 35 if seq_type == "exam" else 16

    data = prepare_sequence_survival_dataloaders(
        data_dir=data_path,
        sequence_type=seq_type,
        max_len=max_len,
        mode=mode,
        temporal=temporal,
        batch_size=batch_size,
        seed=seed,
        max_samples=max_samples,
    )

    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]
    feat_dim = data["feature_dim"]

    if model_type.lower() in ["semester_gru", "recurrent_semester_gru"]:
        model = PyTorchSemesterGRU(feat_dim=feat_dim).to(device_obj)
        model_name = f"torch_semester_gru_{temporal}_{mode}"
    elif model_type.lower() in ["semester_transformer", "recurrent_semester_transformer"]:
        model = PyTorchSemesterTransformer(feat_dim=feat_dim, max_seq_len=max_len).to(device_obj)
        model_name = f"torch_semester_transformer_{temporal}_{mode}"
    elif model_type.lower() in ["exam_gru", "recurrent_exam_gru"]:
        model = PyTorchExamGRU(feat_dim=feat_dim).to(device_obj)
        model_name = f"torch_exam_gru_{temporal}_{mode}"
    elif model_type.lower() in ["dynamic_deephit", "torch_dynamic_deephit"]:
        model = PyTorchDynamicDeepHit(feat_dim=feat_dim, max_duration=max_len).to(device_obj)
        model_name = f"torch_dynamic_deephit_{temporal}_{mode}"
    else:
        raise ValueError(f"Unbekannter model_type: {model_type}.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    print(f"\n[TORCH TRAIN] Starte Training: {model_name} (Device: {device}, Epochen: {epochs})...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train_batches = 0

        for batch in train_loader:
            batch_dev = {k: v.to(device_obj) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model.compute_loss(batch_dev)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = train_loss / max(1, n_train_batches)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_dev = {k: v.to(device_obj) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                loss = model.compute_loss(batch_dev)
                val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = val_loss / max(1, n_val_batches)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 2 == 0 or epoch == epochs:
            print(f"  Epoche {epoch:02d}/{epochs:02d} | Train-Loss: {avg_train_loss:.4f} | Val-Loss: {avg_val_loss:.4f}")

        if patience_counter >= patience:
            print(f"  [EARLY STOPPING] Geduld ({patience} Epochen) erschöpft in Epoche {epoch}.")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Inferenz auf Testset
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            seq = batch["sequence"].to(device_obj)
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(device_obj)
            pred = model(seq, mask=mask)
            all_preds.append(pred.cpu().numpy())

    preds_test = np.concatenate(all_preds, axis=0)  # (N, max_len)
    y_test = data["y_test"]                          # (N, max_len)
    test_mask = (y_test != fb.PADDING_VALUE)

    # Alle Sequenz-Survival-Modelle (inkl. kausalem Dynamic DeepHit) liefern den bedingten Intervall-Hazard h_t
    h_test = preds_test
    y_flat = y_test[test_mask].flatten()
    p_flat = h_test[test_mask].flatten()

    # Aggregiertes Studierenden-Ausfallrisiko: 1 - prod(1 - h_t)
    raw_events = data["test_student_events"]
    test_student_events = (raw_events == 1).astype(int)
    surv_probs = np.ones(len(y_test))
    for i in range(len(y_test)):
        s_len = int(np.sum(test_mask[i]))
        if s_len > 0:
            h_i = np.clip(h_test[i, :s_len], 1e-6, 1.0 - 1e-6)
            surv_probs[i] = np.prod(1.0 - h_i)
    pred_student_risk = 1.0 - surv_probs

    p_flat = np.clip(p_flat, 0.0, 1.0)
    pred_student_risk = np.clip(pred_student_risk, 0.0, 1.0)

    # Metriken auf Zeitschritt-Ebene
    step_auc = float(roc_auc_score(y_flat, p_flat))
    step_pr_auc = float(average_precision_score(y_flat, p_flat))
    step_brier = float(brier_score_loss(y_flat, p_flat))

    # Metriken auf Studierenden-Ebene
    student_auc = float(roc_auc_score(test_student_events, pred_student_risk))
    student_pr_auc = float(average_precision_score(test_student_events, pred_student_risk))

    print("\n" + "=" * 74)
    print(f"   ERGEBNISSE {model_name.upper()} (TEST-SET)")
    print("=" * 74)
    print(f"  • Zeitschritt ROC-AUC        : {step_auc:.4f}")
    print(f"  • Zeitschritt PR-AUC (y=1)   : {step_pr_auc:.4f}")
    print(f"  • Zeitschritt Brier Score    : {step_brier:.4f}")
    print(f"  • Studierenden-Level ROC-AUC : {student_auc:.4f}")
    print(f"  • Studierenden-Level PR-AUC  : {student_pr_auc:.4f}")
    print("=" * 74)

    metrics_dict = {
        "Step_ROC_AUC": step_auc,
        "Step_PR_AUC": step_pr_auc,
        "Step_Brier_Score": step_brier,
        "Student_ROC_AUC": student_auc,
        "Student_PR_AUC": student_pr_auc,
        "temporal": temporal,
        "mode": mode,
    }

    evaluator = SurvivalEvaluator(base_dir=out_path, model_name=model_name, mode=mode)
    evaluator.evaluate_and_log(
        y_true=y_flat,
        y_prob=p_flat,
        model=model,
        mode=mode,
        temporal_type=temporal,
        extra_metrics=metrics_dict,
    )
    save_metrics(model_name, metrics_dict, out_path)

    # Modellgewichte speichern
    models_dir = out_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_file = models_dir / f"{model_name}.pt"
    torch.save(model.state_dict(), model_file)
    print(f"[OK] Modellgewichte gespeichert unter: {model_file}")

    return {
        "model": model,
        "metrics": metrics_dict,
        "model_file": str(model_file),
    }
