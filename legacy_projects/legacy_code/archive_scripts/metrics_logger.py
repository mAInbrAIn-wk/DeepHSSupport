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
    metrics_dir = base_dir / 'metrics'
    plots_dir = base_dir / 'plots'
    models_dir = base_dir / 'models'
    
    ensure_dir(metrics_dir)
    ensure_dir(plots_dir)
    ensure_dir(models_dir)
    
    return metrics_dir, plots_dir, models_dir

def save_metrics(model_name: str, metrics: dict, base_dir: Path, report_str: str = None):
    """Speichert Metriken als JSON und Markdown-Tabelle (inklusive optionalem Classification Report)."""
    metrics_dir, _, _ = get_output_dirs(base_dir)
    
    # Clean up non-serializable types for JSON (like np.float32)
    clean_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, float)):
            clean_metrics[k] = float(v)
        elif isinstance(v, (np.integer, int)):
            clean_metrics[k] = int(v)
        else:
            clean_metrics[k] = str(v)
            
    # Save JSON
    json_path = metrics_dir / f"{model_name}_metrics.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(clean_metrics, f, indent=4, ensure_ascii=False)
        
    # Save Markdown
    md_path = metrics_dir / f"{model_name}_metrics.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Metriken: {model_name}\n\n")
        f.write("| Metrik | Wert |\n")
        f.write("| :--- | :---: |\n")
        for k, v in clean_metrics.items():
            if isinstance(v, float):
                f.write(f"| **{k}** | {v:.4f} |\n")
            else:
                f.write(f"| **{k}** | {v} |\n")
                
        if report_str:
            f.write("\n## Classification Report\n\n```text\n")
            f.write(report_str)
            f.write("\n```\n")
                
    print(f"[INFO] Metriken für {model_name} in {metrics_dir} gespeichert.")

def plot_roc_curve(y_true, y_score, model_name: str, base_dir: Path):
    """Plottet und speichert die ROC-Kurve."""
    _, plots_dir, _ = get_output_dirs(base_dir)
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic: {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
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
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve: {model_name}')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    plot_path = plots_dir / f"{model_name}_pr_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] PR-Kurve für {model_name} in {plots_dir} gespeichert.")

def plot_learning_curve(history_dict, model_name: str, base_dir: Path, metric_name='loss'):
    """Plottet und speichert die Keras Lernkurve (Loss und eine weitere Metrik, z.B. Accuracy oder MAE)."""
    _, plots_dir, _ = get_output_dirs(base_dir)
    
    epochs = range(1, len(history_dict['loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_dict['loss'], 'b-', label='Training Loss')
    if 'val_loss' in history_dict:
        plt.plot(epochs, history_dict['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Metric Plot
    if metric_name in history_dict:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history_dict[metric_name], 'b-', label=f'Training {metric_name}')
        val_metric = f'val_{metric_name}'
        if val_metric in history_dict:
            plt.plot(epochs, history_dict[val_metric], 'r-', label=f'Validation {metric_name}')
        plt.title(f'Training and Validation {metric_name}')
        plt.xlabel('Epochs')
        plt.ylabel(metric_name.capitalize())
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    plt.suptitle(f'Learning Curves: {model_name}')
    plt.tight_layout()
    
    plot_path = plots_dir / f"{model_name}_learning_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Lernkurve für {model_name} in {plots_dir} gespeichert.")

def save_keras_model(model, model_name: str, base_dir: Path):
    """Speichert ein trainiertes Keras-Modell (.keras Format)."""
    _, _, models_dir = get_output_dirs(base_dir)
    
    model_path = models_dir / f"{model_name}.keras"
    model.save(model_path)
    print(f"[INFO] Modell {model_name} unter {model_path} gespeichert.")

def plot_parity_plot(y_true, y_pred, model_name: str, base_dir: Path):
    """Plottet und speichert einen Parity-Plot (Ist vs. Soll) für Regressionsmodelle."""
    from sklearn.metrics import r2_score, mean_squared_error
    _, plots_dir, _ = get_output_dirs(base_dir)
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.3, color='royalblue', edgecolors='none', s=20)
    
    # Ideal 1:1 Line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal (1:1 Linie)')
    
    plt.xlabel('Tatsächlicher Wert (y_true)')
    plt.ylabel('Vorhergesagter Wert (y_pred)')
    plt.title(f'Parity Plot: {model_name}\n(R² = {r2:.4f}, RMSE = {rmse:.4f})')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plot_path = plots_dir / f"{model_name}_parity_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Parity-Plot für {model_name} in {plots_dir} gespeichert.")

def plot_confusion_matrix(y_true, y_pred_binary, model_name: str, base_dir: Path, labels=None):
    """Plottet und speichert eine Konfusionsmatrix für Klassifikationsmodelle."""
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    _, plots_dir, _ = get_output_dirs(base_dir)
    
    cm = confusion_matrix(y_true, y_pred_binary)
    if labels is not None and len(labels) == cm.shape[0]:
        display_labels = labels
    else:
        display_labels = [str(c) for c in np.unique(y_true)]
        
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    plt.title(f'Confusion Matrix: {model_name}')
    
    plot_path = plots_dir / f"{model_name}_confusion_matrix.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Confusion Matrix für {model_name} in {plots_dir} gespeichert.")

