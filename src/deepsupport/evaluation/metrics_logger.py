import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def ensure_dir(directory: Path):
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

def get_output_dirs(base_dir: Path):
    base_dir = Path(base_dir)
    if 'OUTPUT_DIR' in os.environ and os.environ['OUTPUT_DIR']:
        env_out = Path(os.environ['OUTPUT_DIR'])
        base_dir_str = str(base_dir).lower()
        if 'data_' in base_dir_str or ('DATA_DIR' in os.environ and str(base_dir) == os.environ['DATA_DIR']):
            base_dir = env_out
    metrics_dir = base_dir / 'metrics'
    plots_dir = base_dir / 'plots'
    models_dir = base_dir / 'models'
    ensure_dir(metrics_dir)
    ensure_dir(plots_dir)
    ensure_dir(models_dir)
    return metrics_dir, plots_dir, models_dir

def _clean_numeric(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, dict):
        return {str(k): _clean_numeric(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean_numeric(val) for val in v]
    if isinstance(v, str):
        return v
    try:
        val = float(v)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    except Exception:
        return str(v)

def _strip_mode_temporal_suffix(model_name: str, mode: str, temporal_type: str) -> str:
    name = str(model_name)
    for suffix in (f"_{temporal_type}_{mode}", f"_{mode}_{temporal_type}"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    if mode and name.endswith(f"_{mode}"):
        return name[:-len(f"_{mode}")]
    return name

def save_metrics(
    model_name: str = None,
    metrics: dict = None,
    base_dir: Path = Path('src/output_dl'),
    mode: str = 'standard',
    temporal_type: str = 'flat',
    report_str: str = None,
    **kwargs,
):
    if metrics is None and 'metrics_dict' in kwargs:
        metrics = kwargs['metrics_dict']
    metrics_dir, _, _ = get_output_dirs(base_dir)
    clean_metrics = {k: _clean_numeric(v) for k, v in (metrics or {}).items()}
    clean_name = _strip_mode_temporal_suffix(model_name, mode, temporal_type)
    json_path = metrics_dir / f"{clean_name}_{mode}_{temporal_type}_metrics.json"
    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            if isinstance(existing, dict) and 'model_name' in existing:
                # Preserve standardized OOP Evaluator fields while keeping legacy keys
                clean_metrics = {**clean_metrics, **existing}
        except Exception:
            pass
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(clean_metrics, f, indent=4, ensure_ascii=False)

def plot_roc_curve(y_true, y_score, model_name: str, base_dir: Path):
    """Plottet und speichert die ROC-Kurve."""
    _, plots_dir, _ = get_output_dirs(base_dir)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic: {model_name}')
    plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
    plot_path = plots_dir / f"{model_name}_roc_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] ROC-Kurve für {model_name} in {plots_dir} gespeichert.")

def plot_pr_curve(y_true, y_score, model_name: str, base_dir: Path):
    """Plottet und speichert die Precision-Recall-Kurve."""
    _, plots_dir, _ = get_output_dirs(base_dir)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve: {model_name}')
    plt.legend(loc="lower left"); plt.grid(True, alpha=0.3)
    plot_path = plots_dir / f"{model_name}_pr_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] PR-Kurve für {model_name} in {plots_dir} gespeichert.")

def plot_learning_curve(history_dict, model_name: str, base_dir: Path, metric_name='loss'):
    """Plottet und speichert die Keras Lernkurve."""
    if hasattr(history_dict, 'history'):
        history_dict = history_dict.history
    if not isinstance(history_dict, dict) or 'loss' not in history_dict or len(history_dict['loss']) == 0:
        return
    _, plots_dir, _ = get_output_dirs(base_dir)
    epochs = range(1, len(history_dict['loss']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_dict['loss'], 'b-', label='Training Loss')
    if 'val_loss' in history_dict:
        plt.plot(epochs, history_dict['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss'); plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True, alpha=0.3)
    if metric_name in history_dict:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history_dict[metric_name], 'b-', label=f'Training {metric_name}')
        val_metric = f'val_{metric_name}'
        if val_metric in history_dict:
            plt.plot(epochs, history_dict[val_metric], 'r-', label=f'Validation {metric_name}')
        plt.title(f'Training and Validation {metric_name}'); plt.xlabel('Epochs')
        plt.ylabel(metric_name.capitalize()); plt.legend(); plt.grid(True, alpha=0.3)
    plt.suptitle(f'Learning Curves: {model_name}'); plt.tight_layout()
    plot_path = plots_dir / f"{model_name}_learning_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Lernkurve für {model_name} in {plots_dir} gespeichert.")

def save_keras_model(model, model_name: str, base_dir: Path):
    """Speichert ein trainiertes Keras- (.keras) oder PyTorch-Modell (.pt)."""
    _, _, models_dir = get_output_dirs(base_dir)
    if hasattr(model, 'save'):
        model_path = models_dir / f"{model_name}.keras"
        model.save(model_path)
        print(f"[INFO] Modell {model_name} unter {model_path} gespeichert.")
    elif hasattr(model, 'state_dict'):
        import torch
        model_path = models_dir / f"{model_name}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"[INFO] PyTorch-Modell {model_name} unter {model_path} gespeichert.")

def plot_parity_plot(y_true, y_pred, model_name: str, base_dir: Path):
    """Plottet und speichert einen Parity-Plot für Regressionsmodelle."""
    from sklearn.metrics import r2_score, mean_squared_error
    _, plots_dir, _ = get_output_dirs(base_dir)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.3, color='royalblue', edgecolors='none', s=20)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal (1:1 Linie)')
    plt.xlabel('Tatsächlicher Wert (y_true)'); plt.ylabel('Vorhergesagter Wert (y_pred)')
    plt.title(f'Parity Plot: {model_name}\n(R² = {r2:.4f}, RMSE = {rmse:.4f})')
    plt.legend(loc='upper left'); plt.grid(True, alpha=0.3)
    plot_path = plots_dir / f"{model_name}_parity_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Parity-Plot für {model_name} in {plots_dir} gespeichert.")

def plot_confusion_matrix(y_true, y_pred_binary, model_name: str, base_dir: Path, labels=None):
    """Plottet und speichert eine Konfusionsmatrix für Klassifikationsmodelle."""
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    _, plots_dir, _ = get_output_dirs(base_dir)
    cm = confusion_matrix(y_true, y_pred_binary)
    display_labels = labels if (labels and len(labels) == cm.shape[0]) else [str(c) for c in np.unique(y_true)]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    plt.title(f'Confusion Matrix: {model_name}')
    plot_path = plots_dir / f"{model_name}_confusion_matrix.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Confusion Matrix für {model_name} in {plots_dir} gespeichert.")


# =============================================================================
# OOP EVALUATOR-KLASSEN (V4.2+)
# Alle bestehenden Hilfsfunktionen bleiben erhalten (Rückwärtskompatibilität).
# =============================================================================

class _BaseEvaluator:
    """Basisklasse für alle Evaluatoren mit robuster, flexibler Parameterauflösung."""
    def __init__(self, *args, **kwargs):
        base_dir = kwargs.get('base_dir') or kwargs.get('output_dir')
        model_name = kwargs.get('model_name')

        if len(args) == 1:
            arg = args[0]
            if base_dir is None and (isinstance(arg, Path) or (isinstance(arg, str) and ('/' in arg or '\\' in arg))):
                base_dir = arg
            elif model_name is None:
                model_name = str(arg)
            elif base_dir is None:
                base_dir = arg
        elif len(args) >= 2:
            arg0, arg1 = args[0], args[1]
            if isinstance(arg0, Path) or (isinstance(arg0, str) and ('/' in str(arg0) or '\\' in str(arg0))):
                base_dir = arg0
                model_name = str(arg1)
            else:
                model_name = str(arg0)
                base_dir = arg1

        if base_dir is None:
            base_dir = Path('src/output_dl')
        if model_name is None:
            model_name = 'model'

        self.base_dir = Path(base_dir)
        if 'OUTPUT_DIR' in os.environ and os.environ['OUTPUT_DIR']:
            env_out = Path(os.environ['OUTPUT_DIR'])
            base_dir_str = str(self.base_dir).lower()
            if 'data_' in base_dir_str or ('DATA_DIR' in os.environ and str(self.base_dir) == os.environ['DATA_DIR']):
                self.base_dir = env_out
        self.model_name = str(model_name)
        self.temporal = kwargs.get('temporal') or kwargs.get('temporal_type') or 'flat'
        self.mode = kwargs.get('mode', 'standard')
        self.extra_init_kwargs = kwargs

    def _clean_model_name(self, mode: str, temporal_type: str) -> str:
        """Entfernt versehentlich doppelt angehängte Mode/Temporal-Suffixe aus model_name."""
        name = self.model_name
        for suffix in (f"_{temporal_type}_{mode}", f"_{mode}_{temporal_type}"):
            if name.endswith(suffix):
                return name[:-len(suffix)]
        if mode and name.endswith(f"_{mode}"):
            return name[:-len(f"_{mode}")]
        return name

    def _write_merged_json(self, clean: dict, mode: str, temporal_type: str) -> Path:
        metrics_dir, _, _ = get_output_dirs(self.base_dir)
        clean_name = self._clean_model_name(mode, temporal_type)
        path = metrics_dir / f"{clean_name}_{mode}_{temporal_type}_metrics.json"
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                if isinstance(existing, dict):
                    clean = {**existing, **clean}
            except Exception:
                pass
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(clean, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Metriken gespeichert: {path}")
        return path


class SurvivalEvaluator(_BaseEvaluator):
    """
    Einheitlicher Evaluator für binäre Survival/Dropout-Modelle.

    Berechnet und speichert:
      - ROC-AUC (global)
      - PR-AUC für ALLE Klassen: Dropout (y=1) UND Non-Dropout (y=0)
      - pr_auc_baseline = π₀ (Dropout-Prävalenz im Testset)
      - Brier Score + Brier Skill Score (BSS = 1 - B/B_ref)
      - Balanced Accuracy, F1-Score (bei τ=0.5)
      - Harrell's C-Index (optional, erfordert t_stop)
    Plots: roc_curve, pr_curve (mit π₀-Baseline-Linie), learning_curve
    """

    def __init__(self, *args, label_pos: str = None, label_neg: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        if label_pos is not None:
            self.label_pos = str(label_pos)
            self.label_neg = str(label_neg or f"Non-{label_pos}")
        elif any(term in self.model_name.lower() for term in ["pass", "bestehen"]):
            self.label_pos = "Pass"
            self.label_neg = "Fail"
        else:
            self.label_pos = "Dropout"
            self.label_neg = "Non-Drop"

    def evaluate_and_log(
        self,
        y_true: np.ndarray = None,
        y_prob: np.ndarray = None,
        t_stop: np.ndarray = None,
        history: dict = None,
        model=None,
        mode: str = None,
        temporal_type: str = None,
        fit_time_s: float = None,
        extra_metrics: dict = None,
        hr_estimates: dict = None,
        **kwargs
    ) -> dict:
        """
        Parameters
        ----------
        y_true       : Binary labels (0/1), Dropout = 1 (Minderheitsklasse)
        y_prob       : Predicted dropout probabilities (float 0–1)
        t_stop       : Beobachtungszeiten/Semester (für C-Index; optional)
        history      : Keras history.history dict (für Lernkurve; optional)
        model        : Keras-Modell-Objekt (für save_keras_model; optional)
        mode         : Feature-Modus ('standard', 'oracle', etc.)
        temporal_type: Temporal-Modus ('prev', 'cum', 'flat')
        extra_metrics: Zusätzliche Metriken (z. B. student-level AUC)
        """
        mode = mode or self.mode
        temporal_type = temporal_type or self.temporal

        if y_true is None and hr_estimates is not None:
            causal_ev = CausalEvaluator(base_dir=self.base_dir, model_name=self.model_name, temporal=temporal_type, mode=mode)
            return causal_ev.evaluate_and_log(
                hr_estimates=hr_estimates,
                hr_se=kwargs.get('hr_se'),
                bootstrap_samples=kwargs.get('bootstrap_samples'),
                mode=mode,
                temporal_type=temporal_type,
                extra_metrics=extra_metrics,
                **kwargs
            )
        from sklearn.metrics import (
            roc_auc_score, brier_score_loss, average_precision_score,
            f1_score, balanced_accuracy_score,
        )

        y_true = np.asarray(y_true).flatten()
        y_prob = np.asarray(y_prob).flatten()

        # --- Kernmetriken ---
        roc_auc = float(roc_auc_score(y_true, y_prob))

        # PR-AUC für Dropout-Klasse (y=1, Minderheitsklasse)
        pr_auc_dropout = float(average_precision_score(y_true, y_prob, pos_label=1))
        # PR-AUC für Non-Dropout-Klasse (y=0) — invertierte Scores
        pr_auc_nondropout = float(average_precision_score(1 - y_true, 1.0 - y_prob, pos_label=1))

        # Prävalenz π₀ als Baseline für PR-AUC
        pr_auc_baseline = float(np.mean(y_true))

        # Brier Score & Brier Skill Score (nur bei Wahrscheinlichkeiten in [0, 1]; DeepSurv liefert unbeschränkte Hazard-Raten)
        brier = None
        brier_skill = None
        if np.all((y_prob >= 0.0) & (y_prob <= 1.0)):
            try:
                brier = float(brier_score_loss(y_true, y_prob))
                brier_ref = pr_auc_baseline * (1.0 - pr_auc_baseline)
                brier_skill = float(1.0 - brier / brier_ref) if brier_ref > 1e-9 else None
            except Exception:
                brier = None
                brier_skill = None

        # F1, Balanced Accuracy bei τ=0.5
        y_pred_binary = (y_prob >= 0.5).astype(int)
        f1 = float(f1_score(y_true, y_pred_binary, zero_division=0))
        balanced_acc = float(balanced_accuracy_score(y_true, y_pred_binary))

        # Harrell's C-Index (optional)
        c_index = None
        if t_stop is not None:
            try:
                from lifelines.utils import concordance_index
                c_index = float(concordance_index(t_stop, -y_prob, y_true))
            except ImportError:
                print("[WARN] lifelines nicht installiert — C-Index wird übersprungen.")
            except Exception as e:
                print(f"[WARN] C-Index Berechnung fehlgeschlagen: {e}")

        pi0_pos = float(pr_auc_baseline)
        pi0_neg = float(1.0 - pr_auc_baseline)
        lift_pos = float(pr_auc_dropout / max(pi0_pos, 1e-9))
        lift_neg = float(pr_auc_nondropout / max(pi0_neg, 1e-9))

        metrics_dict = {
            "model_name": self.model_name,
            "mode": mode,
            "temporal_type": temporal_type,
            "n_samples": int(len(y_true)),
            "label_class_1": self.label_pos,
            "label_class_0": self.label_neg,
            "dropout_prevalence_pi0": pi0_pos,
            "roc_auc": roc_auc,
            "pr_auc_dropout": pr_auc_dropout,
            "pr_auc_nondropout": pr_auc_nondropout,
            "pr_auc_baseline_pi0": pi0_pos,
            "pr_auc_class_1": pr_auc_dropout,
            "pr_auc_class_0": pr_auc_nondropout,
            "pr_auc_baseline_class_1": pi0_pos,
            "pr_auc_baseline_class_0": pi0_neg,
            "pr_auc_lift_class_1": lift_pos,
            "pr_auc_lift_class_0": lift_neg,
            "brier_score": brier,
            "brier_skill_score": brier_skill,
            "f1_score": f1,
            "balanced_accuracy": balanced_acc,
            "c_index": c_index,
        }

        if fit_time_s is not None:
            metrics_dict["training_time_s"] = _clean_numeric(fit_time_s)

        if extra_metrics:
            metrics_dict.update(extra_metrics)

        self._save(metrics_dict, mode, temporal_type)
        self._plot_roc(y_true, y_prob, roc_auc)
        self._plot_pr(y_true, y_prob, pr_auc_dropout, pr_auc_baseline)
        if history is not None:
            plot_learning_curve(history, self.model_name, self.base_dir)
        if model is not None:
            save_keras_model(model, self.model_name, self.base_dir)

        self._print_summary(metrics_dict)
        return metrics_dict

    def _save(self, metrics_dict, mode, temporal_type):
        clean = {k: _clean_numeric(v) for k, v in metrics_dict.items()}
        self._write_merged_json(clean, mode, temporal_type)

    def _plot_roc(self, y_true, y_prob, roc_auc_val):
        _, plots_dir, _ = get_output_dirs(self.base_dir)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC AUC = {roc_auc_val:.4f}')
        plt.plot([0, 1], [0, 1], 'navy', lw=1.5, linestyle='--', label='Zufall')
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve: {self.model_name}')
        plt.legend(loc='lower right'); plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / f"{self.model_name}_roc_curve.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_pr(self, y_true, y_prob, pr_auc_val, baseline):
        _, plots_dir, _ = get_output_dirs(self.base_dir)
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        plt.figure(figsize=(8, 6))
        lbl = getattr(self, 'label_pos', 'Dropout')
        plt.plot(rec, prec, color='steelblue', lw=2,
                 label=f'PR AUC ({lbl} y=1) = {pr_auc_val:.4f}')
        plt.axhline(baseline, color='red', linestyle='--', lw=1.5,
                    label=f'Baseline π₀ = {baseline:.3f}')
        plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve: {self.model_name}')
        plt.legend(loc='upper right'); plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / f"{self.model_name}_pr_curve.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _print_summary(self, m):
        w = 70
        lbl_pos = m.get('label_class_1', getattr(self, 'label_pos', 'Dropout'))
        lbl_neg = m.get('label_class_0', getattr(self, 'label_neg', 'Non-Drop'))
        print(f"\n{'='*w}")
        print(f"  SurvivalEvaluator -- {m['model_name']} [{m['mode']}/{m['temporal_type']}]")
        print(f"{'='*w}")
        roc = m.get('roc_auc')
        print(f"  ROC-AUC                : {roc:.4f}" if roc is not None else "  ROC-AUC                : N/A")
        
        pr_1 = m.get('pr_auc_class_1', m.get('pr_auc_dropout'))
        pi0_1 = m.get('pr_auc_baseline_class_1', m.get('pr_auc_baseline_pi0'))
        lift_1 = m.get('pr_auc_lift_class_1')
        if pr_1 is not None:
            lift_str = f", Lift: {lift_1:.2f}x" if lift_1 is not None else ""
            pi0_str = f"  (Baseline pi0={pi0_1:.3f}{lift_str})" if pi0_1 is not None else ""
            print(f"  PR-AUC ({lbl_pos:<8} y=1)  : {pr_1:.4f}{pi0_str}")
        else:
            print(f"  PR-AUC ({lbl_pos:<8} y=1)  : N/A")
            
        pr_0 = m.get('pr_auc_class_0', m.get('pr_auc_nondropout'))
        pi0_0 = m.get('pr_auc_baseline_class_0', (1.0 - pi0_1) if pi0_1 is not None else None)
        lift_0 = m.get('pr_auc_lift_class_0')
        if pr_0 is not None:
            lift_str = f", Lift: {lift_0:.2f}x" if lift_0 is not None else ""
            pi0_str = f"  (Baseline pi0={pi0_0:.3f}{lift_str})" if pi0_0 is not None else ""
            print(f"  PR-AUC ({lbl_neg:<8} y=0)  : {pr_0:.4f}{pi0_str}")
        else:
            print(f"  PR-AUC ({lbl_neg:<8} y=0)  : N/A")
            
        bss = m.get('brier_skill_score')
        brier = m.get('brier_score')
        brier_str = f"{brier:.4f}" if brier is not None else "N/A"
        bss_str = f"  (BSS={bss:.3f})" if bss is not None else ""
        print(f"  Brier Score            : {brier_str}{bss_str}")
        f1 = m.get('f1_score')
        print(f"  F1 Score (tau=0.5)     : {f1:.4f}" if f1 is not None else "  F1 Score (tau=0.5)     : N/A")
        ba = m.get('balanced_accuracy')
        print(f"  Balanced Accuracy      : {ba:.4f}" if ba is not None else "  Balanced Accuracy      : N/A")
        if m.get('c_index') is not None:
            print(f"  Harrell C-Index        : {m['c_index']:.4f}")
        if m.get('training_time_s') is not None:
            print(f"  Training Time          : {m['training_time_s']:.2f}s")
        print(f"{'='*w}\n")



class RegressionEvaluator(_BaseEvaluator):
    """
    Einheitlicher Evaluator für kontinuierliche Noten-/GPA-Regressoren.

    Metriken: R², adj. R², RMSE, MAE, MedianAE, ExplainedVariance, MaxError
    Plots: parity_plot, residuals_hist, learning_curve (optional)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate_and_log(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_features: int = None,
        history: dict = None,
        model=None,
        mode: str = None,
        temporal_type: str = None,
        fit_time_s: float = None,
        extra_metrics: dict = None,
        **kwargs
    ) -> dict:
        mode = mode or self.mode
        temporal_type = temporal_type or self.temporal
        from sklearn.metrics import (
            r2_score, mean_squared_error, mean_absolute_error,
            median_absolute_error, explained_variance_score, max_error,
        )

        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        n = len(y_true)

        r2 = float(r2_score(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        med_ae = float(median_absolute_error(y_true, y_pred))
        expl_var = float(explained_variance_score(y_true, y_pred))
        max_err = float(max_error(y_true, y_pred))

        adj_r2 = None
        if n_features is not None and n > n_features + 1:
            adj_r2 = float(1.0 - (1.0 - r2) * (n - 1) / (n - n_features - 1))

        metrics_dict = {
            "model_name": self.model_name,
            "mode": mode,
            "temporal_type": temporal_type,
            "n_samples": n,
            "r2_score": r2,
            "adj_r2": adj_r2,
            "rmse": rmse,
            "mae": mae,
            "median_ae": med_ae,
            "explained_variance": expl_var,
            "max_error": max_err,
        }

        if fit_time_s is not None:
            metrics_dict["training_time_s"] = _clean_numeric(fit_time_s)

        if extra_metrics:
            metrics_dict.update(extra_metrics)

        self._save(metrics_dict, mode, temporal_type)
        self._plot_parity(y_true, y_pred, r2, rmse)
        self._plot_residuals(y_true, y_pred)
        if history is not None:
            plot_learning_curve(history, self.model_name, self.base_dir)
        if model is not None:
            save_keras_model(model, self.model_name, self.base_dir)

        self._print_summary(metrics_dict)
        return metrics_dict

    def _save(self, metrics_dict, mode, temporal_type):
        clean = {k: _clean_numeric(v) for k, v in metrics_dict.items()}
        self._write_merged_json(clean, mode, temporal_type)

    def _plot_parity(self, y_true, y_pred, r2, rmse):
        _, plots_dir, _ = get_output_dirs(self.base_dir)
        min_v = min(float(y_true.min()), float(y_pred.min()))
        max_v = max(float(y_true.max()), float(y_pred.max()))
        plt.figure(figsize=(7, 7))
        plt.scatter(y_true, y_pred, alpha=0.3, color='royalblue', s=15, edgecolors='none')
        plt.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='Ideal (1:1)')
        plt.xlabel('Tatsächlicher Wert'); plt.ylabel('Vorhergesagter Wert')
        plt.title(f'Parity Plot: {self.model_name}\nR²={r2:.4f}  RMSE={rmse:.4f}')
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / f"{self.model_name}_parity_plot.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_residuals(self, y_true, y_pred):
        _, plots_dir, _ = get_output_dirs(self.base_dir)
        residuals = y_true - y_pred
        plt.figure(figsize=(8, 5))
        plt.hist(residuals, bins=60, color='steelblue', alpha=0.75, edgecolor='white')
        plt.axvline(0, color='red', linestyle='--', lw=2, label='Kein Fehler')
        plt.xlabel('Residuum (y_true − y_pred)'); plt.ylabel('Häufigkeit')
        plt.title(f'Residuenverteilung: {self.model_name}')
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / f"{self.model_name}_residuals_hist.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _print_summary(self, m):
        w = 70
        print(f"\n{'='*w}")
        print(f"  RegressionEvaluator -- {m['model_name']} [{m['mode']}/{m['temporal_type']}]")
        print(f"{'='*w}")
        adj = f"  (adj. R2={m['adj_r2']:.4f})" if m['adj_r2'] is not None else ""
        print(f"  R2                     : {m['r2_score']:.4f}{adj}")
        print(f"  RMSE                   : {m['rmse']:.4f}")
        print(f"  MAE                    : {m['mae']:.4f}  (MedianAE={m['median_ae']:.4f})")
        print(f"  Max Error              : {m['max_error']:.4f}")
        if m.get('training_time_s') is not None:
            print(f"  Training Time          : {m['training_time_s']:.2f}s")
        print(f"{'='*w}\n")


class MulticlassEvaluator(_BaseEvaluator):
    """
    Einheitlicher Evaluator für Mehrklassen-Klassifikatoren.
    (z. B. 4-Klassen Landmark-Status-Prognose)

    Metriken: ROC-AUC OvR Macro, F1-Macro, F1-Weighted, PR-AUC per class, per_class dict
    Plots: confusion_matrix (normalisiert, Zeilen-normalisiert)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate_and_log(
        self,
        y_true: np.ndarray,
        y_pred_classes: np.ndarray = None,
        y_prob_matrix: np.ndarray = None,
        class_names: list = None,
        model=None,
        mode: str = None,
        temporal_type: str = None,
        extra_metrics: dict = None,
        **kwargs
    ) -> dict:
        mode = mode or self.mode
        temporal_type = temporal_type or self.temporal
        from sklearn.metrics import (
            f1_score, balanced_accuracy_score, classification_report,
            roc_auc_score, average_precision_score,
        )

        if y_prob_matrix is None and 'y_prob' in kwargs:
            y_prob_matrix = kwargs['y_prob']
        if y_pred_classes is None and 'y_pred' in kwargs:
            y_pred_classes = kwargs['y_pred']
        if y_pred_classes is None and 'predictions' in kwargs:
            y_pred_classes = kwargs['predictions']
        if y_pred_classes is None and y_prob_matrix is not None:
            y_prob_matrix_arr = np.asarray(y_prob_matrix)
            if y_prob_matrix_arr.ndim > 1:
                y_pred_classes = np.argmax(y_prob_matrix_arr, axis=1)
            else:
                y_pred_classes = (y_prob_matrix_arr >= 0.5).astype(int)

        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred_classes).flatten() if y_pred_classes is not None else np.zeros_like(y_true)
        unique_classes = sorted(np.unique(y_true))
        classes = class_names or [str(c) for c in unique_classes]

        f1_macro = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))

        roc_auc_ovr = None
        if y_prob_matrix is not None:
            try:
                roc_auc_ovr = float(roc_auc_score(
                    y_true, y_prob_matrix, multi_class='ovr', average='macro'
                ))
            except Exception as e:
                print(f"[WARN] ROC-AUC OvR: {e}")

        # PR-AUC & Baseline-Prävalenz pro Klasse (One-vs-Rest)
        per_class_pr_auc = {}
        per_class_prevalence = {}
        for i, cls_name in enumerate(classes):
            y_bin = (y_true == unique_classes[i]).astype(int)
            per_class_prevalence[cls_name] = float(np.mean(y_bin))
            if y_prob_matrix is not None and i < y_prob_matrix.shape[1] and y_bin.sum() > 0:
                per_class_pr_auc[f"pr_auc_{cls_name}"] = float(
                    average_precision_score(y_bin, y_prob_matrix[:, i])
                )

        report = classification_report(
            y_true, y_pred, target_names=classes,
            output_dict=True, zero_division=0,
        )
        per_class = {}
        for cls in classes:
            if cls in report:
                cls_entry = {k: round(float(v), 4) for k, v in report[cls].items()}
                cls_entry["prevalence_baseline"] = round(per_class_prevalence.get(cls, 0.0), 4)
                cls_entry["pr_auc"] = (
                    round(per_class_pr_auc[f"pr_auc_{cls}"], 4)
                    if f"pr_auc_{cls}" in per_class_pr_auc
                    else None
                )
                per_class[cls] = cls_entry

        metrics_dict = {
            "model_name": self.model_name,
            "mode": mode,
            "temporal_type": temporal_type,
            "n_samples": int(len(y_true)),
            "n_classes": len(classes),
            "roc_auc_ovr_macro": roc_auc_ovr,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "balanced_accuracy": bal_acc,
            **per_class_pr_auc,
            "per_class": per_class,
            "per_class_metrics": per_class,
        }

        if extra_metrics:
            metrics_dict.update(extra_metrics)

        self._save(metrics_dict, mode, temporal_type)
        self._plot_cm(y_true, y_pred, classes)
        if model is not None:
            save_keras_model(model, self.model_name, self.base_dir)

        self._print_summary(metrics_dict, per_class_pr_auc)
        return metrics_dict

    def _save(self, metrics_dict, mode, temporal_type):
        clean = {k: _clean_numeric(v) for k, v in metrics_dict.items()}
        self._write_merged_json(clean, mode, temporal_type)

    def _plot_cm(self, y_true, y_pred, class_names):
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        _, plots_dir, _ = get_output_dirs(self.base_dir)
        cm = confusion_matrix(y_true, y_pred)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = np.where(cm.sum(axis=1, keepdims=True) > 0,
                               cm.astype(float) / cm.sum(axis=1, keepdims=True), 0.0)
        disp = ConfusionMatrixDisplay(cm_norm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(8, 7))
        disp.plot(cmap='Blues', ax=ax, values_format='.2f')
        plt.title(f'Confusion Matrix (normalisiert): {self.model_name}')
        plt.tight_layout()
        plt.savefig(plots_dir / f"{self.model_name}_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _print_summary(self, m, pr_aucs):
        w = 70
        print(f"\n{'='*w}")
        print(f"  MulticlassEvaluator -- {m['model_name']} [{m['mode']}/{m['temporal_type']}]")
        print(f"{'='*w}")
        if m['roc_auc_ovr_macro']:
            print(f"  ROC-AUC OvR Macro      : {m['roc_auc_ovr_macro']:.4f}")
        print(f"  F1-Macro               : {m['f1_macro']:.4f}")
        print(f"  F1-Weighted            : {m['f1_weighted']:.4f}")
        print(f"  Balanced Accuracy      : {m['balanced_accuracy']:.4f}")
        for k, v in pr_aucs.items():
            print(f"  {k:<30}: {v:.4f}")
        if m.get('training_time_s') is not None:
            print(f"  Training Time          : {m['training_time_s']:.2f}s")
        print(f"{'='*w}\n")


class CausalEvaluator(_BaseEvaluator):
    """
    Evaluator für kausale Effektschätzer (HR, OR, RR, ATE, DML, kontrafaktische Analysen).

    Unterstützt explizite Estimand-Typen (estimand_type: 'HR', 'OR', 'RR', 'ATE')
    und Standardfehler-Skalen (se_scale: 'log' oder 'natural').
    Berechnet und vergleicht BEIDE CI-Methoden:
      1. Asymptotische CIs (Log-Skala für Ratio-Estimands HR/OR/RR inkl. Delta-Methode bei se_scale='natural';
         additive Skala für Differenz-Estimands ATE/ARR)
      2. Bootstrap CIs (Percentile-Methode)
    Plot: Forest-Plot aller Treatment-Effekte mit CI-Fehlerbalken
    """

    def __init__(self, *args, estimand_type: str = "HR", se_scale: str = "log", **kwargs):
        super().__init__(*args, **kwargs)
        self.estimand_type = str(estimand_type).upper()
        self.se_scale = str(se_scale).lower()

    @staticmethod
    def _is_additive_key(k: str, default_estimand: str) -> bool:
        kl = k.lower()
        if default_estimand.upper() in ("ATE", "ARR", "DIFF"):
            return True
        return any(tok in kl for tok in ("_ate", "ate_", "_arr", "arr_"))

    def evaluate_and_log(
        self,
        hr_estimates: dict,
        hr_se: dict = None,
        bootstrap_samples: np.ndarray = None,
        mode: str = None,
        temporal_type: str = None,
        extra_metrics: dict = None,
        estimand_type: str = None,
        se_scale: str = None,
        **kwargs
    ) -> dict:
        """
        Parameters
        ----------
        hr_estimates     : {'hr_fachlich': 0.88, 'hr_ueberfachlich': 0.95, ...}
        hr_se            : Standard-Fehler für asymptotische CIs (Alias: asymptotic_se)
        bootstrap_samples: Array shape (n_bootstrap, n_treatments) für Bootstrap-CIs
        estimand_type    : 'HR' (Default), 'OR', 'RR' oder 'ATE'
        se_scale         : 'log' (Default: SE auf log(HR)-Skala) oder 'natural' (SE auf Originalskala)
        """
        mode = mode or self.mode
        temporal_type = temporal_type or self.temporal
        est_type = (estimand_type or self.estimand_type or "HR").upper()
        scale_mode = (se_scale or self.se_scale or "log").lower()
        if hr_se is None and 'asymptotic_se' in kwargs:
            hr_se = kwargs['asymptotic_se']

        metrics_dict = {
            "model_name": self._clean_model_name(mode, temporal_type),
            "mode": mode,
            "temporal_type": temporal_type,
            "estimand_type": est_type,
            "se_scale": scale_mode,
        }
        metrics_dict.update({k: _clean_numeric(v) for k, v in hr_estimates.items()})

        # Risk Reduction % nur für Ratio-Estimands (HR, OR, RR), nicht für additive Größen (ATE, ARR)
        for k, v in hr_estimates.items():
            if v is not None and not self._is_additive_key(k, est_type):
                try:
                    metrics_dict[f"risk_reduction_pct_{k}"] = _clean_numeric((1.0 - float(v)) * 100.0)
                except (TypeError, ValueError):
                    pass

        # Asymptotische CIs (respektiert estimand_type und se_scale)
        if hr_se is not None:
            for k, se in hr_se.items():
                hr_val = hr_estimates.get(k)
                if hr_val is not None and se is not None:
                    val_f = float(hr_val)
                    se_f = float(se)
                    if self._is_additive_key(k, est_type):
                        metrics_dict[f"ci_lower_95_asym_{k}"] = _clean_numeric(val_f - 1.96 * se_f)
                        metrics_dict[f"ci_upper_95_asym_{k}"] = _clean_numeric(val_f + 1.96 * se_f)
                    else:
                        se_log = (se_f / max(abs(val_f), 1e-9)) if scale_mode == "natural" else se_f
                        log_hr = np.log(max(val_f, 1e-9))
                        metrics_dict[f"ci_lower_95_asym_{k}"] = _clean_numeric(np.exp(log_hr - 1.96 * se_log))
                        metrics_dict[f"ci_upper_95_asym_{k}"] = _clean_numeric(np.exp(log_hr + 1.96 * se_log))

        # Bootstrap CIs (Percentile-Methode)
        if bootstrap_samples is not None:
            keys = list(hr_estimates.keys())
            n_boot_cols = bootstrap_samples.shape[1] if bootstrap_samples.ndim > 1 else 1
            for i, k in enumerate(keys[:n_boot_cols]):
                col = bootstrap_samples[:, i] if bootstrap_samples.ndim > 1 else bootstrap_samples
                metrics_dict[f"ci_lower_95_boot_{k}"] = _clean_numeric(float(np.percentile(col, 2.5)))
                metrics_dict[f"ci_upper_95_boot_{k}"] = _clean_numeric(float(np.percentile(col, 97.5)))
                metrics_dict[f"ci_boot_mean_{k}"] = _clean_numeric(float(np.mean(col)))
                metrics_dict[f"ci_boot_std_{k}"] = _clean_numeric(float(np.std(col)))

        if extra_metrics:
            metrics_dict.update(extra_metrics)

        self._save(metrics_dict, mode, temporal_type)
        self._plot_forest(hr_estimates, hr_se, bootstrap_samples, est_type=est_type, scale_mode=scale_mode)

        # Optionale Keras-Artefakte
        k_model = kwargs.get('keras_model') or kwargs.get('model')
        if k_model is not None:
            save_keras_model(k_model, self.model_name, self.base_dir)
        k_hist = kwargs.get('history')
        if k_hist is not None:
            plot_learning_curve(k_hist, self.model_name, self.base_dir)

        self._print_summary(metrics_dict, hr_estimates)
        return metrics_dict

    def _save(self, metrics_dict, mode, temporal_type):
        clean = {k: _clean_numeric(v) for k, v in metrics_dict.items()}
        self._write_merged_json(clean, mode, temporal_type)

    def _plot_forest(self, hr_estimates, hr_se, bootstrap_samples, est_type: str = "HR", scale_mode: str = "log"):
        _, plots_dir, _ = get_output_dirs(self.base_dir)
        labels = [k for k, v in hr_estimates.items() if v is not None]
        values = [float(hr_estimates[k]) for k in labels]
        n = len(labels)
        if n == 0:
            return

        all_additive = all(self._is_additive_key(lbl, est_type) for lbl in labels)
        null_line = 0.0 if all_additive else 1.0

        y_pos = np.arange(n)
        fig, ax = plt.subplots(figsize=(10, max(4, n * 1.3)))
        ax.axvline(null_line, color='grey', linestyle='--', lw=1.5, label=f'Kein Effekt ({est_type}={null_line:g})')

        for i, (lbl, val) in enumerate(zip(labels, values)):
            lo, hi = None, None
            is_add = self._is_additive_key(lbl, est_type)
            if bootstrap_samples is not None and i < (bootstrap_samples.shape[1] if bootstrap_samples.ndim > 1 else 1):
                col = bootstrap_samples[:, i] if bootstrap_samples.ndim > 1 else bootstrap_samples
                lo, hi = float(np.percentile(col, 2.5)), float(np.percentile(col, 97.5))
            elif hr_se is not None and lbl in hr_se and hr_se[lbl] is not None:
                se_f = float(hr_se[lbl])
                if is_add:
                    lo, hi = val - 1.96 * se_f, val + 1.96 * se_f
                else:
                    se_log = (se_f / max(abs(val), 1e-9)) if scale_mode == "natural" else se_f
                    log_v = np.log(max(val, 1e-9))
                    lo = float(np.exp(log_v - 1.96 * se_log))
                    hi = float(np.exp(log_v + 1.96 * se_log))

            thresh = 0.0 if is_add else 1.0
            color = 'steelblue' if val < thresh else 'tomato'
            ax.plot(val, y_pos[i], 'o', color=color, markersize=10, zorder=5)
            if lo is not None and hi is not None:
                ax.plot([lo, hi], [y_pos[i], y_pos[i]], '-', color=color, lw=2.5, zorder=4)
            ax.text(val, y_pos[i] + 0.25, f'{val:.3f}', ha='center', fontsize=8, color=color)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([lbl.replace('_', ' ') for lbl in labels], fontsize=9)
        ax.set_xlabel(f'Causal Effect ({est_type})')
        ax.set_title(f'Forest Plot: {self.model_name}')
        ax.legend(); ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(plots_dir / f"{self.model_name}_forest_plot.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _print_summary(self, m, hr_estimates):
        w = 70
        est_type = m.get('estimand_type', 'HR')
        print(f"\n{'='*w}")
        print(f"  CausalEvaluator -- {m['model_name']} [{m['mode']}/{m['temporal_type']}] ({est_type})")
        print(f"{'='*w}")
        for k, v in hr_estimates.items():
            if v is None:
                continue
            rr = m.get(f"risk_reduction_pct_{k}")
            lo_b = m.get(f"ci_lower_95_boot_{k}")
            hi_b = m.get(f"ci_upper_95_boot_{k}")
            lo_a = m.get(f"ci_lower_95_asym_{k}")
            hi_a = m.get(f"ci_upper_95_asym_{k}")
            ci_str = ""
            if lo_b is not None and hi_b is not None:
                ci_str = f"  Boot-95%-CI=[{lo_b:.3f}, {hi_b:.3f}]"
            elif lo_a is not None and hi_a is not None:
                ci_str = f"  Asym-95%-CI=[{lo_a:.3f}, {hi_a:.3f}]"
            rr_str = f"  (RR-Senkung: {rr:.1f}%)" if rr is not None else ""
            lbl_est = "ATE" if self._is_additive_key(k, est_type) else est_type
            print(f"  {k:<35}: {lbl_est}={float(v):.4f}{rr_str}{ci_str}")
        if m.get('training_time_s') is not None:
            print(f"  Training Time                      : {m['training_time_s']:.2f}s")
        print(f"{'='*w}\n")


class DualHeadEvaluator(_BaseEvaluator):
    """
    Kombinierter Evaluator für Dual-Head Multi-Task Modelle
    (Autoregressive GRU/Transformer: Note + Bestehens-Wahrscheinlichkeit).

    Delegiert:
      Head 1 (Noten-Regression) → RegressionEvaluator
      Head 2 (Bestehen/Survival) → SurvivalEvaluator

    Speichert eine kombinierte JSON-Datei mit allen Metriken beider Heads
    (Felder mit Präfix 'grade_' bzw. 'pass_').
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reg = RegressionEvaluator(self.base_dir, f"{self.model_name}_grade")
        self._surv = SurvivalEvaluator(self.base_dir, f"{self.model_name}_pass", label_pos="Pass", label_neg="Fail")

    def evaluate_and_log(
        self,
        y_grade_true: np.ndarray,
        y_grade_pred: np.ndarray,
        y_pass_true: np.ndarray,
        y_pass_prob: np.ndarray,
        n_features_grade: int = None,
        history: dict = None,
        model=None,
        mode: str = None,
        temporal_type: str = None,
        extra_metrics: dict = None,
        **kwargs
    ) -> dict:
        """
        Parameters
        ----------
        y_grade_true     : True grades (continuous, z. B. 1.0–5.0)
        y_grade_pred     : Predicted grades
        y_pass_true      : True pass labels (0=nicht bestanden, 1=bestanden)
        y_pass_prob      : Predicted pass probabilities
        n_features_grade : Anzahl Features (für adj. R²; optional)
        history          : Keras history.history (für gemeinsame Lernkurve; optional)
        model            : Keras-Modell-Objekt (wird einmalig gespeichert; optional)
        """
        mode = mode or self.mode
        temporal_type = temporal_type or self.temporal
        reg_metrics = self._reg.evaluate_and_log(
            y_grade_true, y_grade_pred,
            n_features=n_features_grade,
            history=history,
            mode=mode,
            temporal_type=temporal_type,
        )
        surv_metrics = self._surv.evaluate_and_log(
            y_pass_true, y_pass_prob,
            history=history,
            mode=mode,
            temporal_type=temporal_type,
        )

        # Kombinierte JSON mit Präfix
        combined = {
            "model_name": self._clean_model_name(mode, temporal_type),
            "mode": mode,
            "temporal_type": temporal_type,
        }
        for k, v in reg_metrics.items():
            if k not in ("model_name", "mode", "temporal_type"):
                combined[f"grade_{k}"] = v
        for k, v in surv_metrics.items():
            if k not in ("model_name", "mode", "temporal_type"):
                combined[f"pass_{k}"] = v

        combined["pass_pr_auc_pass"] = surv_metrics.get("pr_auc_class_1")
        combined["pass_pr_auc_fail"] = surv_metrics.get("pr_auc_class_0")
        combined["pass_baseline_pass"] = surv_metrics.get("pr_auc_baseline_class_1")
        combined["pass_baseline_fail"] = surv_metrics.get("pr_auc_baseline_class_0")
        combined["pass_lift_fail"] = surv_metrics.get("pr_auc_lift_class_0")

        if extra_metrics:
            combined.update(extra_metrics)

        clean = {k: _clean_numeric(v) for k, v in combined.items()}
        path = self._write_merged_json(clean, mode, temporal_type)

        if model is not None:
            save_keras_model(model, self.model_name, self.base_dir)

        print(f"\n[OK] DualHeadEvaluator: Kombinierte Metriken gespeichert -> {path}")
        return combined

