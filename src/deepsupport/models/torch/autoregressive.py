"""
DeepSupport PyTorch Hybride Autoregressoren
===========================================
Dual-Head Multi-Task Modelle zur gemeinsamen Vorhersage der Note
und Bestehenswahrscheinlichkeit der NÄCHSTEN Prüfung (k+1) aus der
Prüfungshistorie (1..k) und dem nächsten Prüfungskontext (Late Fusion).

Architekturen:
1. PyTorchAutoregressiveNextExamGRU:
   - Pre-LayerNorm GRU über Prüfungshistorie mit variabler Länge
   - Kontext-Projektion für Prüfung k+1 + statische Demographien
   - Late Fusion + Shared Pre-LN Dense
   - Head 1: Notenregression (MSE-Loss, Gewicht 1.0)
   - Head 2: Bestehens-Klassifikation (BCE-Loss, Gewicht 0.8)

2. PyTorchAutoregressiveNextExamTransformer:
   - SinCos-Positional-Encoding + Pre-LN TransformerEncoder (FlashAttention)
   - AttentionPooling über Historien-Tokens
   - Kontext-Projektion + Late Fusion + Dual-Head Multi-Task
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
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
    DualHeadEvaluator,
    save_metrics,
)


class PyTorchAutoregressiveNextExamGRU(nn.Module):
    """
    Hybrider GRU-Autoregressor für Next-Exam Dual-Head Multi-Task Inferenz.
    """
    def __init__(
        self,
        seq_features: int,
        context_features: int,
        hidden_dim: int = 64,
        ctx_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.seq_features = seq_features
        self.context_features = context_features
        self.hidden_dim = hidden_dim

        # Sequenz-Zweig (Prüfungshistorie 1..k)
        self.seq_proj = nn.Linear(seq_features, hidden_dim)
        self.seq_norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.gru_norm = nn.LayerNorm(hidden_dim)

        # Kontext-Zweig (Nächste Prüfung k+1 + Demographie)
        self.ctx_proj = nn.Sequential(
            nn.Linear(context_features, ctx_dim),
            nn.LayerNorm(ctx_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Late-Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + ctx_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Dual-Task Heads
        self.head_grade = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.head_pass = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        sequence: (N, S, seq_features)
        context: (N, context_features)
        mask: (N, S) True für valide Schritte
        Rückgabe: (pred_grade, logit_pass)
        """
        h_seq = self.seq_proj(sequence)
        h_seq = self.seq_norm(h_seq)
        out_gru, _ = self.gru(h_seq)

        if mask is not None:
            lengths = mask.sum(dim=1).clamp(min=1)  # (N,)
            idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out_gru.size(-1))
            seq_rep = out_gru.gather(1, idx).squeeze(1)  # (N, hidden_dim)
        else:
            seq_rep = out_gru[:, -1, :]

        seq_rep = self.gru_norm(seq_rep)
        ctx_rep = self.ctx_proj(context)

        merged = torch.cat([seq_rep, ctx_rep], dim=-1)
        shared = self.fusion(merged)

        pred_grade = self.head_grade(shared).squeeze(-1)
        logit_pass = self.head_pass(shared).squeeze(-1)

        return pred_grade, logit_pass

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        weight_pass: float = 0.8,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        seq = batch["sequence"]
        ctx = batch["context"]
        mask = batch.get("mask")
        y_grade = batch["target_note"]
        y_pass = batch["target_pass"]
        sw = batch.get("sample_weight")

        pred_grade, logit_pass = self.forward(seq, ctx, mask=mask)

        if sw is not None:
            sw = sw / (sw.mean() + 1e-8)
            loss_grade = torch.mean(sw * (pred_grade - y_grade) ** 2)
            loss_pass = torch.mean(sw * F.binary_cross_entropy_with_logits(logit_pass, y_pass, reduction="none"))
        else:
            loss_grade = F.mse_loss(pred_grade, y_grade)
            loss_pass = F.binary_cross_entropy_with_logits(logit_pass, y_pass)
        total_loss = loss_grade + weight_pass * loss_pass

        return total_loss, {
            "loss_grade": loss_grade.item(),
            "loss_pass": loss_pass.item(),
            "loss_total": total_loss.item(),
        }

    @torch.no_grad()
    def predict_dual(
        self,
        sequence: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.eval()
        pred_grade, logit_pass = self.forward(sequence, context, mask=mask)
        prob_pass = torch.sigmoid(logit_pass)
        return pred_grade.cpu().numpy(), prob_pass.cpu().numpy()


class PyTorchAutoregressiveNextExamTransformer(nn.Module):
    """
    Hybrider Transformer-Autoregressor für Next-Exam Dual-Head Multi-Task Inferenz.
    """
    def __init__(
        self,
        seq_features: int,
        context_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 128,
        n_layers: int = 2,
        ctx_dim: int = 32,
        dropout: float = 0.1,
        max_seq_len: int = 40,
    ):
        super().__init__()
        self.seq_features = seq_features
        self.context_features = context_features
        self.d_model = d_model

        # Sequenz-Zweig (Transformer mit SinCos-PE und FlashAttention)
        self.input_proj = nn.Linear(seq_features, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_encoder = SinCosPositionalEncoding(d_model=d_model, max_len=max_seq_len + 10)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                is_causal=False,
            )
            for _ in range(n_layers)
        ])

        self.pooler = AttentionPooling(d_model=d_model)
        self.seq_dense = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Kontext-Zweig
        self.ctx_proj = nn.Sequential(
            nn.Linear(context_features, ctx_dim),
            nn.LayerNorm(ctx_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Late-Fusion
        self.fusion = nn.Sequential(
            nn.Linear(64 + ctx_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Dual-Task Heads
        self.head_grade = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.head_pass = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(sequence)
        h = self.input_norm(h)
        h = self.pos_encoder(h)

        for block in self.blocks:
            h = block(h, key_padding_mask=mask)

        pooled = self.pooler(h, mask=mask)
        seq_rep = self.seq_dense(pooled)

        ctx_rep = self.ctx_proj(context)
        merged = torch.cat([seq_rep, ctx_rep], dim=-1)
        shared = self.fusion(merged)

        pred_grade = self.head_grade(shared).squeeze(-1)
        logit_pass = self.head_pass(shared).squeeze(-1)

        return pred_grade, logit_pass

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        weight_pass: float = 0.8,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        seq = batch["sequence"]
        ctx = batch["context"]
        mask = batch.get("mask")
        y_grade = batch["target_note"]
        y_pass = batch["target_pass"]
        sw = batch.get("sample_weight")

        pred_grade, logit_pass = self.forward(seq, ctx, mask=mask)

        if sw is not None:
            sw = sw / (sw.mean() + 1e-8)
            loss_grade = torch.mean(sw * (pred_grade - y_grade) ** 2)
            loss_pass = torch.mean(sw * F.binary_cross_entropy_with_logits(logit_pass, y_pass, reduction="none"))
        else:
            loss_grade = F.mse_loss(pred_grade, y_grade)
            loss_pass = F.binary_cross_entropy_with_logits(logit_pass, y_pass)
        total_loss = loss_grade + weight_pass * loss_pass

        return total_loss, {
            "loss_grade": loss_grade.item(),
            "loss_pass": loss_pass.item(),
            "loss_total": total_loss.item(),
        }

    @torch.no_grad()
    def predict_dual(
        self,
        sequence: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.eval()
        pred_grade, logit_pass = self.forward(sequence, context, mask=mask)
        prob_pass = torch.sigmoid(logit_pass)
        return pred_grade.cpu().numpy(), prob_pass.cpu().numpy()


def train_autoregressive_dual_head_model(
    model_type: str = "transformer",
    data_dir: Union[str, Path] = Path("data_v4_grid/S01_baseline/universe_A"),
    output_dir: Optional[Union[str, Path]] = None,
    epochs: int = 15,
    batch_size: int = 256,
    max_samples: Optional[int] = None,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    weight_pass: float = 0.8,
    patience: int = 5,
    device: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Trainiert und evaluiert ein hybrides Autoregressor-Modell (GRU oder Transformer).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    data_path = Path(data_dir)
    out_path = Path(output_dir) if output_dir is not None else data_path

    from deepsupport.models.torch.data_loaders import prepare_next_exam_dual_head_dataloaders
    data = prepare_next_exam_dual_head_dataloaders(
        data_dir=data_path, batch_size=batch_size, max_samples=max_samples, seed=seed
    )

    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]
    seq_features = data["seq_features"]
    context_features = data["context_features"]

    if model_type.lower() in ["gru", "autoregressive_gru"]:
        model = PyTorchAutoregressiveNextExamGRU(
            seq_features=seq_features,
            context_features=context_features,
            hidden_dim=64,
            ctx_dim=32,
            dropout=0.2,
        ).to(device_obj)
        model_name = "torch_autoregressive_next_exam_gru"
    elif model_type.lower() in ["transformer", "autoregressive_transformer"]:
        model = PyTorchAutoregressiveNextExamTransformer(
            seq_features=seq_features,
            context_features=context_features,
            d_model=64,
            n_heads=4,
            d_ff=128,
            n_layers=2,
            ctx_dim=32,
            dropout=0.1,
        ).to(device_obj)
        model_name = "torch_autoregressive_next_exam_transformer"
    else:
        raise ValueError(f"Unbekannter model_type: {model_type}. Erwartet 'gru' oder 'transformer'.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    print(f"\n[TORCH TRAIN] Starte Training: {model_name} (Device: {device}, Epochen: {epochs})...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_grade_loss = 0.0
        train_pass_loss = 0.0
        n_train_batches = 0

        for batch in train_loader:
            batch_dev = {k: v.to(device_obj) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()
            loss, loss_dict = model.compute_loss(batch_dev, weight_pass=weight_pass)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_grade_loss += loss_dict["loss_grade"]
            train_pass_loss += loss_dict["loss_pass"]
            n_train_batches += 1

        avg_train_loss = train_loss / max(1, n_train_batches)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_dev = {k: v.to(device_obj) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                loss, _ = model.compute_loss(batch_dev, weight_pass=weight_pass)
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
            print(f"  Epoche {epoch:02d}/{epochs:02d} | Train-Loss: {avg_train_loss:.4f} (Grade: {train_grade_loss/n_train_batches:.4f}, Pass: {train_pass_loss/n_train_batches:.4f}) | Val-Loss: {avg_val_loss:.4f}")

        if patience_counter >= patience:
            print(f"  [EARLY STOPPING] Geduld ({patience} Epochen) erschöpft in Epoche {epoch}.")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Inferenz auf Testset
    model.eval()
    all_pred_grade, all_prob_pass = [], []
    with torch.no_grad():
        for batch in test_loader:
            seq = batch["sequence"].to(device_obj)
            ctx = batch["context"].to(device_obj)
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(device_obj)
            pred_g, logit_p = model(seq, ctx, mask=mask)
            all_pred_grade.append(pred_g.cpu().numpy())
            all_prob_pass.append(torch.sigmoid(logit_p).cpu().numpy())

    preds_grade = np.concatenate(all_pred_grade)
    preds_pass = np.concatenate(all_prob_pass)

    y_te_grade = data["y_grade_test"]
    y_te_pass = data["y_pass_test"]

    # Metriken Grade
    mse = float(mean_squared_error(y_te_grade, preds_grade))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_te_grade, preds_grade))
    r2 = float(r2_score(y_te_grade, preds_grade))

    # Metriken Pass & Fail (beide Klassen)
    auc_pass = float(roc_auc_score(y_te_pass, preds_pass))
    pr_pass = float(average_precision_score(y_te_pass, preds_pass))
    pr_fail = float(average_precision_score(1.0 - y_te_pass, 1.0 - preds_pass))
    brier_pass = float(brier_score_loss(y_te_pass, preds_pass))
    pi0_pass = float(np.mean(y_te_pass))
    pi0_fail = float(1.0 - pi0_pass)
    lift_pass = float(pr_pass / max(pi0_pass, 1e-9))
    lift_fail = float(pr_fail / max(pi0_fail, 1e-9))

    print("\n" + "=" * 74)
    print(f"   ERGEBNISSE {model_name.upper()} (TEST-SET)")
    print("=" * 74)
    print(f"  • Note (k+1) R2 Score        : {r2:.4f}")
    print(f"  • Note (k+1) RMSE            : {rmse:.4f}")
    print(f"  • Note (k+1) MAE             : {mae:.4f}")
    print(f"  • Bestanden (k+1) ROC-AUC    : {auc_pass:.4f}")
    print(f"  • Pass (y=1) PR-AUC          : {pr_pass:.4f} (Baseline pi0={pi0_pass:.3f}, Lift: {lift_pass:.2f}x)")
    print(f"  • Fail (y=0) PR-AUC          : {pr_fail:.4f} (Baseline pi0={pi0_fail:.3f}, Lift: {lift_fail:.2f}x)")
    print(f"  • Bestanden (k+1) Brier Score: {brier_pass:.4f}")
    print("=" * 74)

    # Logging über standardisierten DualHeadEvaluator
    metrics_dict = {
        "Next_Exam_Grade_R2": r2,
        "Next_Exam_Grade_RMSE": rmse,
        "Next_Exam_Grade_MAE": mae,
        "Next_Exam_Pass_ROC_AUC": auc_pass,
        "Next_Exam_Pass_PR_AUC": pr_pass,
        "Next_Exam_Fail_PR_AUC": pr_fail,
        "Next_Exam_Pass_pi0": pi0_pass,
        "Next_Exam_Fail_pi0": pi0_fail,
        "Next_Exam_Pass_Lift": lift_pass,
        "Next_Exam_Fail_Lift": lift_fail,
        "Next_Exam_Pass_Brier_Score": brier_pass,
    }

    evaluator = DualHeadEvaluator(base_dir=out_path, model_name=model_name)
    evaluator.evaluate_and_log(
        y_grade_true=y_te_grade,
        y_grade_pred=preds_grade,
        y_pass_true=y_te_pass,
        y_pass_prob=preds_pass,
        model=model,
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
