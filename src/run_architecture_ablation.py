"""
==============================================================================
DeepSupport Architecture Ablation Study: 2^4 Factorial Benchmark
Empirische Pruefung der Keras-vs-PyTorch Hypothesen (H1 bis H4)
==============================================================================
Run-Plan:
- R0 (Keras Baseline):  Padding-Diffusion, Post-LN, Sigmoid+BCE, Adam (fix)
- R1 (Keras + Gather):   Index-Gathering,   Post-LN, Sigmoid+BCE, Adam (fix)
- R2 (Keras + Pre-LN):   Index-Gathering,   Pre-LN,  Sigmoid+BCE, Adam (fix)
- R3 (Keras + Logits):   Index-Gathering,   Pre-LN,  Logits Loss, Adam (fix)
- R4 (Keras Parity):     Index-Gathering,   Pre-LN,  Logits Loss, AdamW + Cosine
- R5 (PyTorch Baseline): Index-Gathering,   Pre-LN,  Logits Loss, AdamW + Cosine
- R6 (PyTorch Decay):    Padding-Diffusion, Pre-LN,  Logits Loss, AdamW + Cosine
- R7 (PyTorch Post-LN):  Index-Gathering,   Post-LN, Logits Loss, AdamW + Cosine
- R8 (PyTorch Adam):     Index-Gathering,   Pre-LN,  Logits Loss, Adam (fix)
==============================================================================
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses

# Projekt-Imports
from deepsupport.models.autoregressive_gru import prepare_next_exam_dataset
from deepsupport.evaluation.metrics_logger import DualHeadEvaluator

PADDING_VALUE = -99.0


@dataclass
class AblationConfig:
    run_id: str
    framework: str        # 'keras' oder 'pytorch'
    state_agg: str        # 'gather' oder 'diffusion'
    norm_type: str        # 'pre_ln' oder 'post_ln'
    loss_type: str        # 'logits' oder 'sigmoid'
    optimizer_type: str   # 'adamw_cosine' oder 'adam_fix'
    description: str


EXPERIMENT_RUNS = [
    AblationConfig("R0", "keras",   "diffusion", "post_ln", "sigmoid", "adam_fix",     "Keras Original Baseline"),
    AblationConfig("R1", "keras",   "gather",    "post_ln", "sigmoid", "adam_fix",     "Keras + Index-Gathering (H1)"),
    AblationConfig("R2", "keras",   "gather",    "pre_ln",  "sigmoid", "adam_fix",     "Keras + Pre-LN (H1+H2)"),
    AblationConfig("R3", "keras",   "gather",    "pre_ln",  "logits",  "adam_fix",     "Keras + Logits Loss (H1+H2+H3)"),
    AblationConfig("R4", "keras",   "gather",    "pre_ln",  "logits",  "adamw_cosine", "Keras Full-Parity (H1+H2+H3+H4)"),
    AblationConfig("R5", "pytorch", "gather",    "pre_ln",  "logits",  "adamw_cosine", "PyTorch Original Baseline"),
    AblationConfig("R6", "pytorch", "diffusion", "pre_ln",  "logits",  "adamw_cosine", "PyTorch Zero-Padding Decay (H1 Downgrade)"),
    AblationConfig("R7", "pytorch", "gather",    "post_ln", "logits",  "adamw_cosine", "PyTorch Post-LN (H2 Downgrade)"),
    AblationConfig("R8", "pytorch", "gather",    "pre_ln",  "logits",  "adam_fix",     "PyTorch Standard Adam (H4 Downgrade)"),
]


# ==============================================================================
# KERAS ABLATION MODEL BUILDER
# ==============================================================================

class IndexGatherLayer(layers.Layer):
    """Greift den verborgenen Zustand am realen Sequenzende (len_input - 1) ab."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs):
        seq, lens = inputs
        batch_sz = tf.shape(seq)[0]
        last_idx = tf.maximum(0, tf.squeeze(lens, axis=-1) - 1)
        batch_idx = tf.range(batch_sz, dtype=tf.int32)
        indices = tf.stack([batch_idx, last_idx], axis=1)
        return tf.gather_nd(seq, indices)

    def compute_output_shape(self, input_shape):
        seq_shape, _ = input_shape
        return (seq_shape[0], seq_shape[2])


def build_keras_ablation_model(
    seq_timesteps: int,
    seq_features: int,
    context_features: int,
    config: AblationConfig,
) -> tf.keras.Model:
    seq_input = layers.Input(shape=(seq_timesteps, seq_features), name="seq_input")
    ctx_input = layers.Input(shape=(context_features,), name="ctx_input")
    len_input = layers.Input(shape=(1,), dtype=tf.int32, name="len_input")

    masked_seq = layers.Masking(mask_value=PADDING_VALUE)(seq_input)

    # 1. State Aggregation: GRU + Gathering vs. Diffusion
    if config.state_agg == "gather":
        gru_seq = layers.GRU(64, return_sequences=True, dropout=0.2)(masked_seq)
        gru_out = IndexGatherLayer(name="index_gather")([gru_seq, len_input])
    else:  # 'diffusion' (greift Zeitschritt 29 nach Masking ab)
        gru_out = layers.GRU(64, return_sequences=False, dropout=0.2)(masked_seq)

    # 2. Normalisierung (Pre-LN vs. Post-LN)
    if config.norm_type == "post_ln":
        gru_feat = layers.LayerNormalization()(gru_out)
        ctx_dense = layers.Dense(32, activation="relu")(ctx_input)
        ctx_dense = layers.LayerNormalization()(ctx_dense)
        merged = layers.Concatenate()([gru_feat, ctx_dense])
        shared = layers.Dense(64, activation="relu")(merged)
        shared = layers.LayerNormalization()(shared)
        shared = layers.Dropout(0.2)(shared)
    else:  # 'pre_ln'
        gru_feat = layers.LayerNormalization()(gru_out)
        ctx_norm = layers.LayerNormalization()(ctx_input)
        ctx_dense = layers.Dense(32, activation="relu")(ctx_norm)
        ctx_dense = layers.Dropout(0.2)(ctx_dense)
        merged = layers.Concatenate()([gru_feat, ctx_dense])
        merged_norm = layers.LayerNormalization()(merged)
        shared = layers.Dense(64, activation="relu")(merged_norm)
        shared = layers.Dropout(0.2)(shared)

    # 3. Heads
    h_grade = layers.Dense(32, activation="relu")(shared)
    out_grade = layers.Dense(1, activation="linear", name="out_grade")(h_grade)

    h_pass = layers.Dense(32, activation="relu")(shared)
    if config.loss_type == "logits":
        out_pass = layers.Dense(1, activation="linear", name="out_pass")(h_pass)
    else:
        out_pass = layers.Dense(1, activation="sigmoid", name="out_pass")(h_pass)

    inputs = [seq_input, ctx_input, len_input] if config.state_agg == "gather" else [seq_input, ctx_input]
    model = tf.keras.Model(inputs=inputs, outputs=[out_grade, out_pass], name=f"keras_ablation_{config.run_id}")
    return model


# ==============================================================================
# PYTORCH ABLATION MODEL
# ==============================================================================

class PyTorchAblationGRU(nn.Module):
    def __init__(self, seq_features: int, context_features: int, config: AblationConfig):
        super().__init__()
        self.config = config
        hidden_dim = 64
        ctx_dim = 32
        dropout = 0.2

        self.seq_proj = nn.Linear(seq_features, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        if config.norm_type == "pre_ln":
            self.seq_norm = nn.LayerNorm(hidden_dim)
            self.gru_norm = nn.LayerNorm(hidden_dim)
            self.ctx_proj = nn.Sequential(
                nn.LayerNorm(context_features),
                nn.Linear(context_features, ctx_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.fusion = nn.Sequential(
                nn.LayerNorm(hidden_dim + ctx_dim),
                nn.Linear(hidden_dim + ctx_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:  # 'post_ln'
            self.seq_norm = nn.Identity()
            self.gru_norm = nn.LayerNorm(hidden_dim)
            self.ctx_proj = nn.Sequential(
                nn.Linear(context_features, ctx_dim),
                nn.LayerNorm(ctx_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim + ctx_dim, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.head_grade = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.head_pass = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, sequence: torch.Tensor, context: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_seq = self.seq_norm(self.seq_proj(sequence))
        out_gru, _ = self.gru(h_seq)

        if self.config.state_agg == "gather":
            idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, out_gru.size(-1))
            seq_rep = out_gru.gather(1, idx).squeeze(1)
        else:  # 'diffusion' (letzter Zeitschritt 29)
            seq_rep = out_gru[:, -1, :]

        seq_rep = self.gru_norm(seq_rep)
        ctx_rep = self.ctx_proj(context)
        merged = torch.cat([seq_rep, ctx_rep], dim=-1)
        shared = self.fusion(merged)

        pred_grade = self.head_grade(shared).squeeze(-1)
        pred_pass = self.head_pass(shared).squeeze(-1)
        return pred_grade, pred_pass


# ==============================================================================
# TRAININGS-ROUTINEN PRO FRAMEWORK
# ==============================================================================

def train_keras_run(
    config: AblationConfig,
    data_splits: Dict[str, Any],
    out_dir: Path,
    epochs: int = 8,
    batch_size: int = 256,
) -> Dict[str, Any]:
    X_hist_tr, X_ctx_tr, len_tr, y_grade_tr, y_pass_tr = data_splits["train"]
    X_hist_va, X_ctx_va, len_va, y_grade_va, y_pass_va = data_splits["val"]
    X_hist_te, X_ctx_te, len_te, y_grade_te, y_pass_te = data_splits["test"]

    seq_steps, seq_feats = X_hist_tr.shape[1], X_hist_tr.shape[2]
    ctx_feats = X_ctx_tr.shape[1]

    model = build_keras_ablation_model(seq_steps, seq_feats, ctx_feats, config)

    # Optimizer-Konfiguration
    steps_per_epoch = len(X_hist_tr) // batch_size
    total_steps = steps_per_epoch * epochs

    if config.optimizer_type == "adamw_cosine":
        lr_schedule = optimizers.schedules.CosineDecay(
            initial_learning_rate=0.001,
            decay_steps=total_steps,
            alpha=0.05,
        )
        opt = optimizers.AdamW(learning_rate=lr_schedule, weight_decay=0.01)
    else:
        opt = optimizers.Adam(learning_rate=0.001)

    # Loss-Konfiguration
    loss_pass = losses.BinaryCrossentropy(from_logits=True) if config.loss_type == "logits" else "binary_crossentropy"
    model.compile(
        optimizer=opt,
        loss={"out_grade": "mse", "out_pass": loss_pass},
        loss_weights={"out_grade": 1.0, "out_pass": 0.8},
    )

    # Trainings-Input
    in_tr = [X_hist_tr, X_ctx_tr, len_tr] if config.state_agg == "gather" else [X_hist_tr, X_ctx_tr]
    in_va = [X_hist_va, X_ctx_va, len_va] if config.state_agg == "gather" else [X_hist_va, X_ctx_va]
    in_te = [X_hist_te, X_ctx_te, len_te] if config.state_agg == "gather" else [X_hist_te, X_ctx_te]

    t0 = time.time()
    model.fit(
        in_tr,
        {"out_grade": y_grade_tr, "out_pass": y_pass_tr},
        validation_data=(in_va, {"out_grade": y_grade_va, "out_pass": y_pass_va}),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )
    duration_s = time.time() - t0

    preds = model.predict(in_te, batch_size=batch_size, verbose=0)
    pred_grade = preds[0].flatten()
    raw_pass = preds[1].flatten()
    prob_pass = tf.nn.sigmoid(raw_pass).numpy() if config.loss_type == "logits" else raw_pass

    evaluator = DualHeadEvaluator(base_dir=out_dir, model_name=f"ablation_{config.run_id}")
    metrics = evaluator.evaluate_and_log(
        y_grade_true=y_grade_te,
        y_grade_pred=pred_grade,
        y_pass_true=y_pass_te.astype(int),
        y_pass_prob=prob_pass,
        mode="standard",
        temporal_type="prev",
    )
    metrics["training_time_s"] = duration_s
    metrics["run_id"] = config.run_id
    metrics["framework"] = config.framework
    return metrics


def train_pytorch_run(
    config: AblationConfig,
    data_splits: Dict[str, Any],
    out_dir: Path,
    epochs: int = 8,
    batch_size: int = 256,
    device: str = "cpu",
) -> Dict[str, Any]:
    X_hist_tr, X_ctx_tr, len_tr, y_grade_tr, y_pass_tr = data_splits["train"]
    X_hist_va, X_ctx_va, len_va, y_grade_va, y_pass_va = data_splits["val"]
    X_hist_te, X_ctx_te, len_te, y_grade_te, y_pass_te = data_splits["test"]

    seq_steps, seq_feats = X_hist_tr.shape[1], X_hist_tr.shape[2]
    ctx_feats = X_ctx_tr.shape[1]

    # DataLoader
    def to_loader(xh, xc, xl, yg, yp, shuffle=False):
        ds = TensorDataset(
            torch.tensor(xh, dtype=torch.float32),
            torch.tensor(xc, dtype=torch.float32),
            torch.tensor(xl, dtype=torch.long),
            torch.tensor(yg, dtype=torch.float32),
            torch.tensor(yp, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    tr_loader = to_loader(X_hist_tr, X_ctx_tr, len_tr, y_grade_tr, y_pass_tr, shuffle=True)
    va_loader = to_loader(X_hist_va, X_ctx_va, len_va, y_grade_va, y_pass_va, shuffle=False)
    te_loader = to_loader(X_hist_te, X_ctx_te, len_te, y_grade_te, y_pass_te, shuffle=False)

    model = PyTorchAblationGRU(seq_feats, ctx_feats, config).to(device)

    # Optimizer
    if config.optimizer_type == "adamw_cosine":
        opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=5e-5)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = None

    loss_mse = nn.MSELoss()
    loss_bce = nn.BCEWithLogitsLoss()

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for bx_h, bx_c, b_l, by_g, by_p in tr_loader:
            bx_h, bx_c, b_l = bx_h.to(device), bx_c.to(device), b_l.to(device)
            by_g, by_p = by_g.to(device), by_p.to(device)

            opt.zero_grad()
            pred_g, pred_p = model(bx_h, bx_c, b_l)
            l_g = loss_mse(pred_g, by_g)
            l_p = loss_bce(pred_p, by_p)
            loss = l_g + 0.8 * l_p
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        # Validierung
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx_h, bx_c, b_l, by_g, by_p in va_loader:
                bx_h, bx_c, b_l = bx_h.to(device), bx_c.to(device), b_l.to(device)
                by_g, by_p = by_g.to(device), by_p.to(device)
                pred_g, pred_p = model(bx_h, bx_c, b_l)
                val_loss += (loss_mse(pred_g, by_g) + 0.8 * loss_bce(pred_p, by_p)).item()

        print(f"  [Epoch {ep+1}/{epochs}] Train-Loss: {total_loss/len(tr_loader):.4f} | Val-Loss: {val_loss/len(va_loader):.4f}")

    duration_s = time.time() - t0

    # Test-Evaluation
    model.eval()
    preds_g, preds_p = [], []
    with torch.no_grad():
        for bx_h, bx_c, b_l, by_g, by_p in te_loader:
            bx_h, bx_c, b_l = bx_h.to(device), bx_c.to(device), b_l.to(device)
            pg, pp = model(bx_h, bx_c, b_l)
            preds_g.extend(pg.cpu().numpy().tolist())
            preds_p.extend(torch.sigmoid(pp).cpu().numpy().tolist())

    evaluator = DualHeadEvaluator(base_dir=out_dir, model_name=f"ablation_{config.run_id}")
    metrics = evaluator.evaluate_and_log(
        y_grade_true=y_grade_te,
        y_grade_pred=np.array(preds_g),
        y_pass_true=y_pass_te.astype(int),
        y_pass_prob=np.array(preds_p),
        mode="standard",
        temporal_type="prev",
    )
    metrics["training_time_s"] = duration_s
    metrics["run_id"] = config.run_id
    metrics["framework"] = config.framework
    return metrics


# ==============================================================================
# SYNOPSIS & HYPOTHESEN-AUSWERTUNG
# ==============================================================================

def generate_ablation_summary(results: Dict[str, Dict[str, Any]], out_dir: Path):
    lines = [
        "---",
        f"created: {time.strftime('%Y-%m-%d')}",
        f"last_updated: {time.strftime('%Y-%m-%d')}",
        "status: abgeschlossen",
        "tags: [ablation-study, keras-vs-pytorch, architecture-hypotheses, state-decay, pre-ln, logits-loss, adamw]",
        "---",
        "",
        "# Synopse: Faktorielle $2^4$-Ablationsstudie (Keras vs. PyTorch)",
        "",
        "## 1. Übersicht & Versuchsmatrix",
        "",
        "| Run-ID | Framework | State Agg | Normalisierung | Loss | Optimizer | Noten $R^2$ | Noten RMSE | Bestehen ROC-AUC | Nichtbest. PR-AUC ($y=0$) | Brier Score | Zeit (s) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for cfg in EXPERIMENT_RUNS:
        rid = cfg.run_id
        if rid not in results:
            continue
        m = results[rid]
        
        r2 = m.get("grade_r2_score", m.get("grade_r2", 0.0))
        rmse = m.get("grade_rmse", 0.0)
        roc = m.get("pass_roc_auc", 0.0)
        pr_fail = m.get("pass_pr_auc_fail", 0.0)
        brier = m.get("pass_brier_score", 0.0)
        t_s = m.get("training_time_s", 0.0)

        lines.append(
            f"| **{rid}** | `{cfg.framework}` | `{cfg.state_agg}` | `{cfg.norm_type}` | `{cfg.loss_type}` | `{cfg.optimizer_type}` | "
            f"**{r2:.4f}** | {rmse:.4f} | {roc:.4f} | {pr_fail:.4f} | {brier:.4f} | {t_s:.1f} |"
        )

    # Hypothesen-Auswertung
    lines.extend([
        "",
        "---",
        "",
        "## 2. Quantitative Hypothesen-Prüfung",
        "",
    ])

    if "R0" in results and "R1" in results and "R5" in results and "R6" in results:
        r2_r0 = results["R0"].get("grade_r2_score", results["R0"].get("grade_r2", 0.0))
        r2_r1 = results["R1"].get("grade_r2_score", results["R1"].get("grade_r2", 0.0))
        r2_r5 = results["R5"].get("grade_r2_score", results["R5"].get("grade_r2", 0.0))
        r2_r6 = results["R6"].get("grade_r2_score", results["R6"].get("grade_r2", 0.0))
        delta_k_h1 = r2_r1 - r2_r0
        delta_p_h1 = r2_r5 - r2_r6
        lines.extend([
            "### Hypothese 1 (State Gathering vs. Padding Diffusion):",
            f"- **Keras Gewinn (R1 - R0):** $\\Delta R^2 = {delta_k_h1:+.4f}$",
            f"- **PyTorch Verlust durch Downgrade (R5 - R6):** $\\Delta R^2 = {delta_p_h1:+.4f}$",
            f"- **Befund:** {'Bestätigt' if delta_k_h1 >= 0.05 or delta_p_h1 >= 0.05 else 'Falsifiziert'} (Schwellenwert $\\Delta R^2 \\ge 0.05$).",
            "",
        ])

    if "R1" in results and "R2" in results and "R5" in results and "R7" in results:
        r2_r1 = results["R1"].get("grade_r2_score", results["R1"].get("grade_r2", 0.0))
        r2_r2 = results["R2"].get("grade_r2_score", results["R2"].get("grade_r2", 0.0))
        r2_r5 = results["R5"].get("grade_r2_score", results["R5"].get("grade_r2", 0.0))
        r2_r7 = results["R7"].get("grade_r2_score", results["R7"].get("grade_r2", 0.0))
        delta_k_h2 = r2_r2 - r2_r1
        delta_p_h2 = r2_r5 - r2_r7
        lines.extend([
            "### Hypothese 2 (Pre-LayerNorm vs. Post-LayerNorm):",
            f"- **Keras Pre-LN Effekt (R2 - R1):** $\\Delta R^2 = {delta_k_h2:+.4f}$",
            f"- **PyTorch Pre-LN Effekt (R5 - R7):** $\\Delta R^2 = {delta_p_h2:+.4f}$",
            f"- **Befund:** {'Bestätigt' if abs(delta_k_h2) >= 0.005 or abs(delta_p_h2) >= 0.005 else 'Vernachlässigbar'}.",
            "",
        ])

    if "R2" in results and "R3" in results:
        br_r2 = results["R2"].get("pass_brier_score", 0.0)
        br_r3 = results["R3"].get("pass_brier_score", 0.0)
        delta_brier = ((br_r2 - br_r3) / br_r2 * 100.0) if br_r2 > 0 else 0.0
        lines.extend([
            "### Hypothese 3 (Logits Loss Numerik):",
            f"- **Brier Score Reduktion (R2 -> R3):** {delta_brier:+.2f} %",
            f"- **Befund:** {'Bestätigt' if delta_brier >= 2.0 else 'Falsifiziert'}.",
            "",
        ])

    if "R3" in results and "R4" in results and "R8" in results and "R5" in results:
        r2_r3 = results["R3"].get("grade_r2_score", results["R3"].get("grade_r2", 0.0))
        r2_r4 = results["R4"].get("grade_r2_score", results["R4"].get("grade_r2", 0.0))
        r2_r8 = results["R8"].get("grade_r2_score", results["R8"].get("grade_r2", 0.0))
        r2_r5 = results["R5"].get("grade_r2_score", results["R5"].get("grade_r2", 0.0))
        lines.extend([
            "### Hypothese 4 (AdamW + Cosine Decay vs. Standard Adam):",
            f"- **Keras Optimierer-Effekt (R4 - R3):** $\\Delta R^2 = {r2_r4 - r2_r3:+.4f}$",
            f"- **PyTorch Optimierer-Effekt (R5 - R8):** $\\Delta R^2 = {r2_r5 - r2_r8:+.4f}$",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## 3. Verwandte Dokumente",
        "",
        "| Dokument | Pfad / Referenz | Kerninhalt |",
        "| :--- | :--- | :--- |",
        "| **Ablations-Masterplan** | [`../01_master_plans/ablationsplan_keras_vs_pytorch_architekturhypothesen.md`](../01_master_plans/ablationsplan_keras_vs_pytorch_architekturhypothesen.md) | Theoretische Herleitung & Hypothesen |",
        "| **LXC Benchmark Evaluation V4.2** | [`../03_evaluations_and_benchmarks/pytorch_lxc_benchmark_evaluation_v42.md`](../03_evaluations_and_benchmarks/pytorch_lxc_benchmark_evaluation_v42.md) | Ausgangsbefund des Keras-vs-PyTorch-Vergleichs |",
        "",
    ])

    out_md = out_dir / "ablation_benchmark_summary.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[INFO] Zusammenfassender Markdown-Bericht gespeichert: {out_md}")


# ==============================================================================
# MAIN BATCH RUNNER
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="DeepSupport 2^4 Architecture Ablation Runner")
    parser.add_argument("--data_dir", type=str, default="data_v4_grid/S01_baseline/universe_A", help="Pfad zum Szenario")
    parser.add_argument("--output_dir", type=str, default="outputs/output_ablation_study", help="Ausgabeverzeichnis")
    parser.add_argument("--epochs", type=int, default=8, help="Anzahl Epochen pro Run (Standard: 8)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch Size (Standard: 256)")
    parser.add_argument("--runs", nargs="+", default=None, help="Spezifische Runs ausführen (z.B. R0 R1)")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch Device")
    parser.add_argument("--smoke_test", action="store_true", help="1-Epochen Testlauf auf 5% Subset")
    parser.add_argument("--overwrite", action="store_true", help="Bereits gerechnete Runs neu berechnen")
    args = parser.parse_args()

    # CPU-Optimierung für Host
    cpu_cores = os.cpu_count() or 4
    torch_threads = max(1, cpu_cores - 1)
    torch.set_num_threads(torch_threads)

    print("\n" + "#" * 88)
    print("   DEEPSUPPORT ARCHITECTURE ABLATION RUNNER (2^4 FACTORIAL)")
    print(f"   CPU-Kerne: {cpu_cores} | PyTorch-Threads: {torch_threads} | Device: {args.device}")
    print(f"   Daten-Pfad: {args.data_dir}")
    print("#" * 88)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Daten laden & vorbereiten
    print("\n[1/3] Lade Daten und erstelle Next-Exam Paare ...")
    X_hist, X_ctx, y_grade, y_pass, student_ids = prepare_next_exam_dataset(data_dir, max_history_len=30)
    lengths = (X_hist[:, :, 0] != PADDING_VALUE).sum(axis=1, keepdims=True).astype(np.int32)

    if args.smoke_test:
        print("[SMOKE TEST] Reduziere Datensatz auf 10.000 Samples für schnellen Integritätstest ...")
        subset_idx = np.random.RandomState(42).choice(len(X_hist), 10000, replace=False)
        X_hist = X_hist[subset_idx]
        X_ctx = X_ctx[subset_idx]
        lengths = lengths[subset_idx]
        y_grade = y_grade[subset_idx]
        y_pass = y_pass[subset_idx]
        student_ids = student_ids[subset_idx]
        args.epochs = 1

    # 2. Group-Consistent 70/15/15 Split
    print("\n[2/3] Erstelle group-consistent 70/15/15 Train/Val/Test Split ...")
    unique_students = np.unique(student_ids)
    tr_stud, temp_stud = train_test_split(unique_students, test_size=0.30, random_state=42)
    va_stud, te_stud = train_test_split(temp_stud, test_size=0.50, random_state=42)

    tr_mask = np.isin(student_ids, tr_stud)
    va_mask = np.isin(student_ids, va_stud)
    te_mask = np.isin(student_ids, te_stud)

    # Standardisierung Sequenz (auf unpadded Train-Tokens)
    vm_tr = (X_hist[tr_mask, :, 0] != PADDING_VALUE)
    scaler_seq = StandardScaler()
    scaler_seq.fit(X_hist[tr_mask][vm_tr])

    def _transform_seq(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        vm = (arr[:, :, 0] != PADDING_VALUE)
        if np.any(vm):
            out[vm] = scaler_seq.transform(arr[vm])
        return out.astype(np.float32)

    X_hist_tr = _transform_seq(X_hist[tr_mask])
    X_hist_va = _transform_seq(X_hist[va_mask])
    X_hist_te = _transform_seq(X_hist[te_mask])

    # Standardisierung Kontext
    scaler_ctx = StandardScaler()
    X_ctx_tr = scaler_ctx.fit_transform(X_ctx[tr_mask]).astype(np.float32)
    X_ctx_va = scaler_ctx.transform(X_ctx[va_mask]).astype(np.float32)
    X_ctx_te = scaler_ctx.transform(X_ctx[te_mask]).astype(np.float32)

    data_splits = {
        "train": (X_hist_tr, X_ctx_tr, lengths[tr_mask], y_grade[tr_mask], y_pass[tr_mask]),
        "val":   (X_hist_va, X_ctx_va, lengths[va_mask], y_grade[va_mask], y_pass[va_mask]),
        "test":  (X_hist_te, X_ctx_te, lengths[te_mask], y_grade[te_mask], y_pass[te_mask]),
    }

    # 3. Iteration über Ablations-Runs
    runs_to_run = [c for c in EXPERIMENT_RUNS if (args.runs is None or c.run_id in args.runs)]
    print(f"\n[3/3] Starte {len(runs_to_run)} Ablations-Runs ...")

    all_results = {}
    summary_json = out_dir / "ablation_benchmark_summary.json"
    if summary_json.exists():
        try:
            with open(summary_json, "r", encoding="utf-8") as f:
                all_results = json.load(f)
        except Exception:
            all_results = {}

    for cfg in runs_to_run:
        if cfg.run_id in all_results and not args.overwrite:
            print(f"\n>>> [RESUME] Überspringe {cfg.run_id} (bereits in {summary_json} vorhanden).")
            continue

        print("\n" + "=" * 80)
        print(f">>> [ABLATION RUN: {cfg.run_id}] Framework: {cfg.framework.upper()} | {cfg.description}")
        print(f"    State: {cfg.state_agg} | Norm: {cfg.norm_type} | Loss: {cfg.loss_type} | Opt: {cfg.optimizer_type}")
        print("=" * 80)

        if cfg.framework == "keras":
            metrics = train_keras_run(cfg, data_splits, out_dir=out_dir, epochs=args.epochs, batch_size=args.batch_size)
        else:
            metrics = train_pytorch_run(cfg, data_splits, out_dir=out_dir, epochs=args.epochs, batch_size=args.batch_size, device=args.device)

        metrics["config"] = asdict(cfg)
        all_results[cfg.run_id] = metrics

        # Einzelmetrik speichern
        m_file = metrics_dir / f"ablation_{cfg.run_id}_metrics.json"
        with open(m_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)

        # Gesamtergebnisse inkrementell speichern
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)

        r2 = metrics.get("grade_r2_score", metrics.get("grade_r2", 0.0))
        rmse = metrics.get("grade_rmse", 0.0)
        roc = metrics.get("pass_roc_auc", 0.0)
        print(f"\n>>> [ERGEBNIS {cfg.run_id}] Noten-R2: {r2:.4f} | RMSE: {rmse:.4f} | Bestehen ROC-AUC: {roc:.4f} | Zeit: {metrics.get('training_time_s', 0.0):.1f}s")

    # Gesamtreport erzeugen
    generate_ablation_summary(all_results, out_dir)
    print("\n" + "#" * 88)
    print("   ABLATIONSSTUDIE ERFOLGREICH BEENDET")
    print(f"   Ergebnisse archiviert in: {out_dir}")
    print("#" * 88)


if __name__ == "__main__":
    main()
