"""
DeepSupport PyTorch & PyCox LXC Batch Runner
============================================
Headless, robuster Batch-Runner für LXC-Container und Linux-Serverumgebungen.
Führt die PyTorch/PyCox-Suite über multiple Sensitivitätsszenarien und Universen aus.

Features:
- Automatische CPU-Core-Allokation (torch.set_num_threads)
- GPU/CUDA-Autodetect mit Fallback auf Multi-Core CPU
- Unterstützung für V4-Grid-Szenarien (S01, S02, S03, S09, S10) und V3.6-Historie
- Strukturierte JSON- und Markdown-Berichterstattung in output_LXC/
- Gradeblind-Policy für GPA-Regression, Standard-Features für Survival
- Keine interaktiven GUI-Abhängigkeiten, sauberes Headless-Logging
"""

import sys
import os
import time
import json
import argparse

# Ungepuffertes Logging erzwingen (damit 'nohup ... > log' Zeilen sofort flusht)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import torch

from deepsupport.models.torch.data_loaders import (
    prepare_panel_dataloaders,
    prepare_sequence_dataloaders,
    prepare_exam_regressor_dataloaders,
    prepare_causal_survival_dataloaders,
)
from deepsupport.models.torch.survival import (
    PyTorchLogisticHazard,
    PyTorchDeepHit,
    PyTorchCoxPH,
    PyTorchCoxTime,
    PyTorchDeepHitCompetingRisks,
)
from deepsupport.models.torch.transformer import (
    PyTorchExamTransformerRegressor,
    PyTorchCausalExamTransformerSurvival,
)
from deepsupport.models.torch.autoregressive import (
    PyTorchAutoregressiveNextExamGRU,
    PyTorchAutoregressiveNextExamTransformer,
    train_autoregressive_dual_head_model,
)
from deepsupport.models.torch.sequence import (
    PyTorchSemesterGRU,
    PyTorchSemesterTransformer,
    PyTorchExamGRU,
    PyTorchDynamicDeepHit,
    train_torch_sequence_survival_model,
)
from deepsupport.models.torch.trainer import (
    ModelTrainer,
)
from deepsupport.evaluation.metrics_logger import (
    SurvivalEvaluator,
    RegressionEvaluator,
    DualHeadEvaluator,
    save_metrics,
)


def run_scenario_universe(
    data_dir: Path,
    output_dir: Path,
    scenario_name: str,
    universe_name: str,
    mode_surv: str = "standard",
    mode_reg: str = "gradeblind",
    temporal: str = "prev",
    batch_size: int = 2048,
    epochs_lh: int = 20,
    epochs_dh: int = 20,
    epochs_cox: int = 20,
    epochs_ct: int = 20,
    epochs_dh_cr: int = 20,
    epochs_trans: int = 15,
    epochs_ar: int = 15,
    epochs_seq: int = 20,
    include_ar: bool = False,
    include_seq: bool = False,
    seed: int = 42,
    skip_trans: bool = False,
    skip_surv: bool = False,
) -> Dict[str, Any]:
    print("\n" + "=" * 88)
    print(f">>> [LXC EXECUTION] Szenario: {scenario_name} | Universum: {universe_name}")
    print(f"    Pfad: {data_dir}")
    print(f"    Survival-Mode: {mode_surv} | Regressor-Mode: {mode_reg} | Temporal: {temporal}")
    if include_ar:
        print("    [Option] Inklusive Hybride Autoregressoren (Dual-Head Multi-Task)")
    if include_seq:
        print("    [Option] Inklusive Sequentielle Survival-Modelle (Semester- & Prüfungsverläufe)")
    print("=" * 88)

    scenario_out = output_dir / scenario_name / universe_name
    scenario_out.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "scenario": scenario_name,
        "universe": universe_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {},
    }

    # 1. Survival Suite
    if not skip_surv:
        print("\n--- [LXC] Phase 1: Survival Modeling Suite (5 Architekturen) ---")
        bundle = prepare_panel_dataloaders(
            data_dir=data_dir,
            mode=mode_surv,
            temporal=temporal,
            batch_size=batch_size,
            seed=seed,
            panel_type="semester",
        )
        train_loader = bundle["train_loader"]
        val_loader = bundle["val_loader"]
        test_loader = bundle["test_loader"]
        X_train = bundle["X_train"]
        t_train = bundle["t_train"]
        e_train = bundle["e_train"]
        X_test = bundle["X_test"]
        t_test = bundle["t_test"]
        e_test = bundle["e_test"]
        ce_test = bundle.get("ce_test")
        test_panel = bundle["test_panel"]
        in_dim = bundle["input_dim"]
        step_test_np = np.clip(test_panel["fachsemester"].values.astype(np.int64) - 1, 0, 15)
        step_test = torch.tensor(step_test_np)

        # 1.1 LogisticHazard
        print("\n[1/5] Training PyTorchLogisticHazard...")
        lh_model = PyTorchLogisticHazard(in_features=in_dim, num_durations=16, hidden_dims=[128, 64, 32], dropout=0.2)
        opt_lh = torch.optim.AdamW(lh_model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched_lh = torch.optim.lr_scheduler.CosineAnnealingLR(opt_lh, T_max=epochs_lh)
        trainer_lh = ModelTrainer(lh_model, opt_lh, scheduler=sched_lh)
        t0 = time.time()
        trainer_lh.fit(train_loader, val_loader, epochs=epochs_lh, patience=5, verbose=False)
        t_fit_lh = time.time() - t0

        lh_model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test, dtype=torch.float32, device=trainer_lh.device)
            lh_model.to(trainer_lh.device)
            step_dev = step_test.to(trainer_lh.device)
            risk_lh = lh_model.predict_risk(X_test_t, step_dev).cpu().numpy()
            instant_h_lh = lh_model.predict_step_hazard(X_test_t, step_dev).cpu().numpy()

        extra_lh = {
            "instant_hazard_roc_auc": float(roc_auc_score(e_test, instant_h_lh)),
            "instant_hazard_pr_auc_dropout": float(average_precision_score(e_test, instant_h_lh)),
            "instant_hazard_brier": float(brier_score_loss(e_test, instant_h_lh)),
            "cumulative_risk_roc_auc": float(roc_auc_score(e_test, risk_lh)),
        }
        print(f"    -> PyTorch LogisticHazard Paritaet: Instant-Hazard ROC-AUC = {extra_lh['instant_hazard_roc_auc']:.4f} (vs. Keras 0.8002)")

        ev_lh = SurvivalEvaluator(base_dir=scenario_out, model_name="torch_logistic_hazard")
        m_lh = ev_lh.evaluate_and_log(e_test, instant_h_lh, t_stop=t_test, mode=mode_surv, temporal_type=temporal, fit_time_s=t_fit_lh, extra_metrics=extra_lh)
        results["models"]["torch_logistic_hazard"] = m_lh

        # 1.2 DeepHit (Single Event)
        print("[2/5] Training PyTorchDeepHit (Single Event)...")
        dh_model = PyTorchDeepHit(in_features=in_dim, num_durations=16, hidden_dims=[128, 64, 32], dropout=0.2, alpha_ranking=0.5)
        opt_dh = torch.optim.AdamW(dh_model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched_dh = torch.optim.lr_scheduler.CosineAnnealingLR(opt_dh, T_max=epochs_dh)
        trainer_dh = ModelTrainer(dh_model, opt_dh, scheduler=sched_dh)
        t0 = time.time()
        trainer_dh.fit(train_loader, val_loader, epochs=epochs_dh, patience=5, verbose=False)
        t_fit_dh = time.time() - t0

        dh_model.eval()
        with torch.no_grad():
            dh_model.to(trainer_dh.device)
            step_dh_dev = step_test.to(trainer_dh.device)
            risk_dh = dh_model.predict_risk(X_test_t, step_dh_dev).cpu().numpy()
            instant_h_dh = dh_model.predict_step_hazard(X_test_t, step_dh_dev).cpu().numpy()

        extra_dh = {"cumulative_risk_roc_auc": float(roc_auc_score(e_test, risk_dh))}
        ev_dh = SurvivalEvaluator(base_dir=scenario_out, model_name="torch_deephit")
        m_dh = ev_dh.evaluate_and_log(e_test, instant_h_dh, t_stop=t_test, mode=mode_surv, temporal_type=temporal, fit_time_s=t_fit_dh, extra_metrics=extra_dh)
        results["models"]["torch_deephit"] = m_dh

        # 1.3 CoxPH (Extended DeepSurv mit Breslow)
        print("[3/5] Training PyTorchCoxPH (Extended DeepSurv mit Breslow)...")
        cox_model = PyTorchCoxPH(in_features=in_dim, hidden_dims=[128, 64, 32], dropout=0.2)
        opt_cox = torch.optim.AdamW(cox_model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched_cox = torch.optim.lr_scheduler.CosineAnnealingLR(opt_cox, T_max=epochs_cox)
        trainer_cox = ModelTrainer(cox_model, opt_cox, scheduler=sched_cox)
        t0 = time.time()
        trainer_cox.fit(train_loader, val_loader, epochs=epochs_cox, patience=5, verbose=False)
        t_fit_cox = time.time() - t0

        cox_model.compute_baseline_hazard(X_train, t_train, e_train)
        risk_cox = cox_model.predict_risk(X_test, step_test_np)
        instant_h_cox = cox_model.predict_step_hazard(X_test, t_test)

        extra_cox = {"cumulative_risk_roc_auc": float(roc_auc_score(e_test, risk_cox))}
        ev_cox = SurvivalEvaluator(base_dir=scenario_out, model_name="torch_coxph_extended")
        m_cox = ev_cox.evaluate_and_log(e_test, instant_h_cox, t_stop=t_test, mode=mode_surv, temporal_type=temporal, fit_time_s=t_fit_cox, extra_metrics=extra_cox)
        results["models"]["torch_coxph_extended"] = m_cox

        # 1.4 CoxTime
        print("[4/5] Training PyTorchCoxTime (Non-Proportional Hazards)...")
        ct_model = PyTorchCoxTime(in_features=in_dim, hidden_dims=[128, 64, 32], dropout=0.2)
        opt_ct = torch.optim.AdamW(ct_model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched_ct = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ct, T_max=epochs_ct)
        trainer_ct = ModelTrainer(ct_model, opt_ct, scheduler=sched_ct)
        t0 = time.time()
        trainer_ct.fit(train_loader, val_loader, epochs=epochs_ct, patience=5, verbose=False)
        t_fit_ct = time.time() - t0

        ct_model.compute_baseline_hazard(X_train, t_train, e_train)
        risk_ct = ct_model.predict_risk(X_test, step_test_np)
        instant_h_ct = ct_model.predict_step_hazard(X_test, t_test)

        extra_ct = {"cumulative_risk_roc_auc": float(roc_auc_score(e_test, risk_ct))}
        ev_ct = SurvivalEvaluator(base_dir=scenario_out, model_name="torch_coxtime")
        m_ct = ev_ct.evaluate_and_log(e_test, instant_h_ct, t_stop=t_test, mode=mode_surv, temporal_type=temporal, fit_time_s=t_fit_ct, extra_metrics=extra_ct)
        results["models"]["torch_coxtime"] = m_ct

        # 1.5 DeepHit Competing Risks
        print("[5/5] Training PyTorchDeepHitCompetingRisks (Multi-Event)...")
        dh_cr = PyTorchDeepHitCompetingRisks(in_features=in_dim, num_risks=2, num_durations=16, hidden_dims=[128, 64, 32], dropout=0.2)
        opt_cr = torch.optim.AdamW(dh_cr.parameters(), lr=1e-3, weight_decay=1e-4)
        sched_cr = torch.optim.lr_scheduler.CosineAnnealingLR(opt_cr, T_max=epochs_dh_cr)
        trainer_cr = ModelTrainer(dh_cr, opt_cr, scheduler=sched_cr)
        t0 = time.time()
        trainer_cr.fit(train_loader, val_loader, epochs=epochs_dh_cr, patience=5, verbose=False)
        t_fit_cr = time.time() - t0

        dh_cr.eval()
        with torch.no_grad():
            dh_cr.to(trainer_cr.device)
            step_cr_dev = step_test.to(trainer_cr.device)
            risk_cr_do = dh_cr.predict_risk(X_test_t, step_cr_dev, risk_idx=0).cpu().numpy()
            risk_cr_gr = dh_cr.predict_risk(X_test_t, step_cr_dev, risk_idx=1).cpu().numpy()
            instant_h_cr_do = dh_cr.predict_step_hazard(X_test_t, step_cr_dev, risk_idx=0).cpu().numpy()
            instant_h_cr_gr = dh_cr.predict_step_hazard(X_test_t, step_cr_dev, risk_idx=1).cpu().numpy()

        extra_cr = {"cumulative_risk_roc_auc": float(roc_auc_score(e_test, risk_cr_do))}
        if ce_test is not None:
            e_grad_test = (ce_test == 2).astype(float)
            extra_cr["grad_roc_auc"] = float(roc_auc_score(e_grad_test, instant_h_cr_gr))
            extra_cr["grad_pr_auc"] = float(average_precision_score(e_grad_test, instant_h_cr_gr))
            extra_cr["grad_cumulative_roc_auc"] = float(roc_auc_score(e_grad_test, risk_cr_gr))

        ev_cr = SurvivalEvaluator(base_dir=scenario_out, model_name="torch_deephit_competing_risks")
        m_cr = ev_cr.evaluate_and_log(e_test, instant_h_cr_do, t_stop=t_test, mode=mode_surv, temporal_type=temporal, fit_time_s=t_fit_cr, extra_metrics=extra_cr)
        results["models"]["torch_deephit_competing_risks"] = m_cr

    # 2. Transformer Suite
    if not skip_trans:
        print("\n--- [LXC] Phase 2: Transformer Suite (Gradeblind Regressor & Causal Survival) ---")
        
        # 2.1 Exam Regressor
        print("\n[1/2] Training PyTorchExamTransformerRegressor (Gradeblind)...")
        reg_bundle = prepare_exam_regressor_dataloaders(
            data_dir=data_dir,
            mode=mode_reg,
            temporal=temporal,
            batch_size=256,
            seed=seed,
            max_len=40,
        )
        r_tr = reg_bundle["train_loader"]
        r_va = reg_bundle["val_loader"]
        r_te = reg_bundle["test_loader"]
        y_test_reg = reg_bundle["y_test"]
        feat_dim = reg_bundle["feature_dim"]

        reg_model = PyTorchExamTransformerRegressor(
            feat_dim=feat_dim,
            d_model=64,
            n_heads=4,
            d_ff=128,
            n_layers=2,
            dropout=0.1,
            max_seq_len=40,
        )
        opt_reg = torch.optim.AdamW(reg_model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched_reg = torch.optim.lr_scheduler.CosineAnnealingLR(opt_reg, T_max=epochs_trans)
        tr_reg = ModelTrainer(reg_model, opt_reg, scheduler=sched_reg)
        t0 = time.time()
        tr_reg.fit(r_tr, r_va, epochs=epochs_trans, patience=5, verbose=False)
        t_fit_reg = time.time() - t0

        reg_model.eval()
        preds_list = []
        with torch.no_grad():
            reg_model.to(tr_reg.device)
            for b in r_te:
                s = b["sequence"].to(tr_reg.device)
                m = b["mask"].to(tr_reg.device)
                preds_list.append(reg_model(s, mask=m).cpu().numpy())
        pred_notes = np.concatenate(preds_list)

        ev_reg = RegressionEvaluator(base_dir=scenario_out, model_name="torch_exam_transformer_regressor")
        m_reg = ev_reg.evaluate_and_log(y_test_reg, pred_notes, mode=mode_reg, temporal_type=temporal, fit_time_s=t_fit_reg)
        results["models"]["torch_exam_transformer_regressor"] = m_reg

        # 2.2 Causal Exam Survival
        print("[2/2] Training PyTorchCausalExamTransformerSurvival...")
        sv_bundle = prepare_causal_survival_dataloaders(
            data_dir=data_dir,
            mode=mode_reg,
            temporal=temporal,
            batch_size=256,
            seed=seed,
            max_len=40,
        )
        sv_tr = sv_bundle["train_loader"]
        sv_va = sv_bundle["val_loader"]
        sv_te = sv_bundle["test_loader"]
        y_test_sv = sv_bundle["y_test"]
        test_mask = sv_bundle["test_mask"]
        test_student_events = sv_bundle["test_student_events"]
        feat_dim_sv = sv_bundle["feature_dim"]

        sv_model = PyTorchCausalExamTransformerSurvival(
            feat_dim=feat_dim_sv,
            d_model=64,
            n_heads=4,
            d_ff=128,
            n_layers=2,
            dropout=0.1,
            max_seq_len=40,
        )
        opt_sv = torch.optim.AdamW(sv_model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched_sv = torch.optim.lr_scheduler.CosineAnnealingLR(opt_sv, T_max=epochs_trans)
        tr_sv = ModelTrainer(sv_model, opt_sv, scheduler=sched_sv)
        t0 = time.time()
        tr_sv.fit(sv_tr, sv_va, epochs=epochs_trans, patience=5, verbose=False)
        t_fit_sv = time.time() - t0

        sv_model.eval()
        hazards_list = []
        with torch.no_grad():
            sv_model.to(tr_sv.device)
            for b in sv_te:
                s = b["sequence"].to(tr_sv.device)
                m = b["mask"].to(tr_sv.device)
                hazards_list.append(sv_model(s, mask=m).cpu().numpy())
        hazards_all = np.concatenate(hazards_list, axis=0)

        y_test_flat = y_test_sv[test_mask].flatten()
        preds_flat = hazards_all[test_mask].flatten()

        surv_probs = np.ones(len(y_test_sv), dtype=np.float32)
        for i in range(len(y_test_sv)):
            s_len = int(np.sum(test_mask[i]))
            if s_len > 0:
                h_k = np.clip(hazards_all[i, :s_len], 1e-6, 1.0 - 1e-6)
                surv_probs[i] = np.prod(1.0 - h_k)
        risk_student = 1.0 - surv_probs
        student_auc = float(roc_auc_score(test_student_events, risk_student))
        student_pr_auc = float(average_precision_score(test_student_events, risk_student))

        ev_sv = SurvivalEvaluator(base_dir=scenario_out, model_name="torch_causal_exam_transformer_survival")
        m_sv = ev_sv.evaluate_and_log(
            y_true=y_test_flat,
            y_prob=preds_flat,
            mode=mode_reg,
            temporal_type=temporal,
            fit_time_s=t_fit_sv,
            extra_metrics={
                "student_roc_auc": student_auc,
                "student_pr_auc": student_pr_auc,
                "n_test_exam_steps": len(y_test_flat),
                "step_prevalence_pi0": float(np.mean(y_test_flat)),
            }
        )
        results["models"]["torch_causal_exam_transformer_survival"] = m_sv

    # 3. Phase 3: Hybride Autoregressoren (Dual-Head Multi-Task)
    if include_ar:
        print("\n--- [LXC] Phase 3: Hybride Autoregressoren (Dual-Head Multi-Task) ---")
        print("\n[1/2] Training PyTorchAutoregressiveNextExamTransformer...")
        res_ar_trans = train_autoregressive_dual_head_model(
            model_type="transformer",
            data_dir=data_dir,
            output_dir=scenario_out,
            epochs=epochs_ar,
            batch_size=512,
            seed=seed,
        )
        results["models"]["torch_autoregressive_next_exam_transformer"] = res_ar_trans["metrics"]

        print("\n[2/2] Training PyTorchAutoregressiveNextExamGRU...")
        res_ar_gru = train_autoregressive_dual_head_model(
            model_type="gru",
            data_dir=data_dir,
            output_dir=scenario_out,
            epochs=epochs_ar,
            batch_size=512,
            seed=seed,
        )
        results["models"]["torch_autoregressive_next_exam_gru"] = res_ar_gru["metrics"]

    # 4. Phase 4: Sequentielle Survival-Modelle (Semester- & Prüfungsverläufe)
    if include_seq:
        print("\n--- [LXC] Phase 4: Sequentielle Survival-Modelle (Semester- & Prüfungsverläufe) ---")
        print("\n[1/4] Training PyTorchSemesterGRU...")
        res_sem_gru = train_torch_sequence_survival_model(
            model_type="semester_gru",
            data_dir=data_dir,
            output_dir=scenario_out,
            mode=mode_surv,
            temporal=temporal,
            epochs=epochs_seq,
            batch_size=256,
            seed=seed,
        )
        results["models"]["torch_semester_gru"] = res_sem_gru["metrics"]

        print("\n[2/4] Training PyTorchSemesterTransformer...")
        res_sem_trans = train_torch_sequence_survival_model(
            model_type="semester_transformer",
            data_dir=data_dir,
            output_dir=scenario_out,
            mode=mode_surv,
            temporal=temporal,
            epochs=epochs_seq,
            batch_size=256,
            seed=seed,
        )
        results["models"]["torch_semester_transformer"] = res_sem_trans["metrics"]

        print("\n[3/4] Training PyTorchExamGRU...")
        res_exam_gru = train_torch_sequence_survival_model(
            model_type="exam_gru",
            data_dir=data_dir,
            output_dir=scenario_out,
            mode=mode_surv,
            temporal=temporal,
            epochs=epochs_seq,
            batch_size=256,
            seed=seed,
        )
        results["models"]["torch_exam_gru"] = res_exam_gru["metrics"]

        print("\n[4/4] Training PyTorchDynamicDeepHit...")
        res_ddh = train_torch_sequence_survival_model(
            model_type="dynamic_deephit",
            data_dir=data_dir,
            output_dir=scenario_out,
            mode=mode_surv,
            temporal=temporal,
            epochs=epochs_seq,
            batch_size=256,
            seed=seed,
        )
        results["models"]["torch_dynamic_deephit"] = res_ddh["metrics"]

    # Speichern des Ergebnis-JSON für dieses Universum
    res_path = scenario_out / "metrics" / f"lxc_results_{scenario_name}_{universe_name}.json"
    res_path.parent.mkdir(parents=True, exist_ok=True)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\n[LXC INFO] Ergebnisse gespeichert unter: {res_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="DeepSupport PyTorch LXC Runner")
    parser.add_argument("--data_root", type=str, default="data_v4_grid", help="Wurzelverzeichnis der Daten")
    parser.add_argument("--output_dir", type=str, default="outputs/output_LXC", help="Ausgabeverzeichnis")
    parser.add_argument(
        "--scenarios",
        type=str,
        default="S01_baseline,S02_supp_half,S03_supp_double,S07_noise_half,S08_noise_double,S11_rct_calibrated",
        help="Komma-separierte Liste der Szenarien oder 'all'"
    )
    parser.add_argument("--universes", type=str, default="universe_A", help="Komma-separierte Liste der Universen (z. B. universe_A) oder 'all'")
    parser.add_argument("--mode_surv", type=str, default="standard", help="Feature-Modus für Survival")
    parser.add_argument("--mode_reg", type=str, default="gradeblind", help="Feature-Modus für GPA-Regression")
    parser.add_argument("--temporal", type=str, default="prev")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--n_threads", type=int, default=None, help="Anzahl CPU-Threads (Standard: Autodetect)")
    parser.add_argument("--dry_run", action="store_true", help="Schneller Probelauf mit 2 Epochen")
    parser.add_argument("--skip_trans", action="store_true", help="Transformer überspringen")
    parser.add_argument("--skip_surv", action="store_true", help="Survival überspringen")
    parser.add_argument("--include_ar", action="store_true", help="Hybride Autoregressoren (Dual-Head) trainieren")
    parser.add_argument("--include_seq", action="store_true", help="Sequentielle Survival-Modelle (Semester/Exam) trainieren")
    parser.add_argument("--epochs_ar", type=int, default=15, help="Epochen für Autoregressoren")
    parser.add_argument("--epochs_seq", type=int, default=20, help="Epochen für Sequenz-Survival")

    args = parser.parse_args()

    # CPU-Thread-Allokation
    if args.n_threads is not None:
        torch.set_num_threads(args.n_threads)
        print(f"[LXC INIT] Manuelle CPU-Thread-Begrenzung: {args.n_threads}")
    else:
        num_cores = os.cpu_count() or 4
        alloc_threads = max(1, num_cores - 1)
        torch.set_num_threads(alloc_threads)
        print(f"[LXC INIT] CPU-Kerne erkannt: {num_cores} | PyTorch-Threads gesetzt: {alloc_threads}")

    device_info = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[LXC INIT] Rechendevice: {device_info}")

    epochs_lh = 2 if args.dry_run else 20
    epochs_dh = 2 if args.dry_run else 20
    epochs_cox = 2 if args.dry_run else 20
    epochs_ct = 2 if args.dry_run else 20
    epochs_dh_cr = 2 if args.dry_run else 20
    epochs_trans = 2 if args.dry_run else 15
    epochs_ar = 2 if args.dry_run else args.epochs_ar
    epochs_seq = 2 if args.dry_run else args.epochs_seq

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    # Szenarien ermitteln
    if args.scenarios.lower() == "all":
        scenarios = [d.name for d in data_root.iterdir() if d.is_dir() and d.name.startswith("S")]
        scenarios = sorted(scenarios)
    else:
        scenarios = [s.strip() for s in args.scenarios.split(",")]

    all_runs = []

    for sc in scenarios:
        sc_path = data_root / sc
        if not sc_path.exists():
            # Robustes Prefix-Matching (z. B. S11_rct_uptake -> S11_rct_calibrated, S09 -> S09_cost_zero)
            prefix = sc.split("_")[0]
            candidates = [d for d in data_root.iterdir() if d.is_dir() and d.name.startswith(prefix)]
            if candidates:
                sc_path = candidates[0]
                print(f"[LXC MATCH] Szenario '{sc}' aufgelöst zu '{sc_path.name}'.")
                sc = sc_path.name
            else:
                print(f"[LXC WARN] Szenario {sc} nicht gefunden unter {sc_path}. Überspringe.")
                continue

        if args.universes.lower() == "all":
            universes = [d.name for d in sc_path.iterdir() if d.is_dir() and d.name.startswith("universe_")]
            universes = sorted(universes)
        else:
            universes = [u.strip() for u in args.universes.split(",")]

        for un in universes:
            un_path = sc_path / un
            if not un_path.exists():
                print(f"[LXC WARN] Universum {un} in {sc} nicht gefunden unter {un_path}. Überspringe.")
                continue

            try:
                run_res = run_scenario_universe(
                    data_dir=un_path,
                    output_dir=output_dir,
                    scenario_name=sc,
                    universe_name=un,
                    mode_surv=args.mode_surv,
                    mode_reg=args.mode_reg,
                    temporal=args.temporal,
                    batch_size=args.batch_size,
                    epochs_lh=epochs_lh,
                    epochs_dh=epochs_dh,
                    epochs_cox=epochs_cox,
                    epochs_ct=epochs_ct,
                    epochs_dh_cr=epochs_dh_cr,
                    epochs_trans=epochs_trans,
                    epochs_ar=epochs_ar,
                    epochs_seq=epochs_seq,
                    include_ar=args.include_ar,
                    include_seq=args.include_seq,
                    skip_trans=args.skip_trans,
                    skip_surv=args.skip_surv,
                )
                all_runs.append(run_res)
            except Exception as exc:
                print(f"\n[LXC ERROR] Fehler während Ausführung von {sc} / {un}: {exc}")
                import traceback
                traceback.print_exc()
                print(f"[LXC ERROR] Fahre mit verbleibenden Szenarien fort...\n")

    # Synopse über alle Läufe speichern
    synopsis_path = output_dir / "lxc_batch_synopsis.json"
    synopsis_path.parent.mkdir(parents=True, exist_ok=True)
    with open(synopsis_path, "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=4)

    print("\n" + "=" * 98)
    print("   LXC BATCH RUN COMPLETE: ALL SCENARIOS AND UNIVERSES PROCESSED")
    print(f"   Ergebnisse archiviert in: {synopsis_path}")
    print("=" * 98 + "\n")


if __name__ == "__main__":
    main()
