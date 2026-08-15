# metrics_and_plots.py
"""
Metrics + Plotting utilities for Q1-grade Emotion Classification.

Provides:
 - compute_classification_metrics(...)
 - save_classification_report(...)
 - save_confusion_matrix(...)
 - plot_confusion_heatmap(...)
 - compute_roc_pr_curves(...)
 - plot_roc_curves(...)
 - plot_pr_curves(...)
 - plot_learning_curves(...)
 - plot_model_comparison(...)
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

# Set professional scientific publication plotting parameters
# Single-column journal width ≈ 3.5 in (IEEE/Springer); double-column ≈ 7.0 in
SINGLE_COL_WIDTH = 3.5
DOUBLE_COL_WIDTH = 7.0

PUBLICATION_RC = {
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
}

plt.rcParams.update(PUBLICATION_RC)


def pub_figsize(aspect: float = 1.0, single_column: bool = True):
    """Return (width, height) for one-column or two-column journal figures."""
    w = SINGLE_COL_WIDTH if single_column else DOUBLE_COL_WIDTH
    return (w, w * aspect)


def apply_publication_style(single_column: bool = True):
    """Apply publication rcParams; use before saving paper figures."""
    plt.rcParams.update(PUBLICATION_RC)
    if single_column:
        plt.rcParams["figure.figsize"] = pub_figsize(1.0, single_column=True)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
try:
    import umap
    HAVE_UMAP = True
except ImportError:
    HAVE_UMAP = False

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ImageNet normalisation constants for inverse-transform
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _prepare_display_image(image):
    """
    Convert a tensor or array to a float32 image in [0, 1].

    If the input is an ImageNet-normalised tensor (values roughly in [-2, 2])
    it is inverse-normalised so the resulting image has correct colours.
    Returns float32 HWC array in [0, 1] — safe to pass directly to plt.imshow().
    DO NOT cast the return value to uint8; matplotlib handles float [0,1] natively.
    """
    if isinstance(image, torch.Tensor):
        img = image.detach().cpu()
        if img.ndim == 4:
            img = img[0]
        img = img.permute(1, 2, 0).numpy().astype(np.float32)
        # Detect ImageNet-normalised tensors (values outside [0, 1])
        if img.min() < -0.1 or img.max() > 1.1:
            img = img * _IMAGENET_STD + _IMAGENET_MEAN  # inverse normalise
    else:
        img = np.array(image, dtype=np.float32)
        if img.max() > 1.5:          # uint8 [0-255] input
            img = img / 255.0
    # Final clip + rescale to [0, 1]
    img = np.clip(img, 0.0, 1.0)
    lo, hi = img.min(), img.max()
    if hi - lo > 1e-6:
        img = (img - lo) / (hi - lo)
    return img.astype(np.float32)


def _create_gradcam_overlay(image, cam):
    """Blend Grad-CAM heatmap over original image. Returns float [0,1] arrays."""
    base = _prepare_display_image(image)                     # float [0,1] HWC
    if cam is None:
        cam = np.ones((base.shape[0], base.shape[1]), dtype=np.float32) * 0.5
    cam_uint8 = (np.clip(cam, 0, 1) * 255).astype(np.uint8)
    cam_resized = np.array(
        Image.fromarray(cam_uint8).resize((base.shape[1], base.shape[0]),
                                          resample=Image.BILINEAR)
    )
    cam_norm = cam_resized.astype(np.float32) / 255.0
    # Use plt.colormaps instead of deprecated cm.get_cmap (matplotlib >= 3.7)
    try:
        cmap_jet = plt.colormaps["jet"]
    except AttributeError:
        cmap_jet = cm.get_cmap("jet")
    heatmap = cmap_jet(cam_norm)[..., :3].astype(np.float32)
    overlay = np.clip(0.55 * heatmap + 0.45 * base, 0.0, 1.0)
    return base, overlay   # both float [0,1] — pass directly to plt.imshow()


def _locate_deep_feature_block(model):
    if hasattr(model, "stages"):
        stage = model.stages[-1]
        try:
            return stage[-1]
        except Exception:
            return stage
    if hasattr(model, "features"):
        try:
            return model.features[-1]
        except Exception:
            return None
    if hasattr(model, "layer4"):
        block = model.layer4[-1]
        return block
    return None


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + 1e-12) / (m + 1e-12)))
    kl_qm = np.sum(q * np.log((q + 1e-12) / (m + 1e-12)))
    return 0.5 * (kl_pm + kl_qm)

# -------------------------------------------------------------------
#  Basic metrics
# -------------------------------------------------------------------
def compute_classification_metrics(y_true, y_pred, y_prob=None, classes=None):
    """
    Compute standard metrics for a classifier.
    y_true, y_pred: (N,)
    y_prob: (N, C) probabilities
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    }

    # AUC metrics
    if y_prob is not None and classes is not None:
        try:
            y_true_bin = label_binarize(y_true, classes=list(range(len(classes))))
            roc_macro = roc_auc_score(y_true_bin, y_prob, average="macro")
            roc_micro = roc_auc_score(y_true_bin, y_prob, average="micro")
            metrics["roc_auc_macro"] = float(roc_macro)
            metrics["roc_auc_micro"] = float(roc_micro)
        except Exception:
            pass

    return metrics


# -------------------------------------------------------------------
#  Classification Report Saving
# -------------------------------------------------------------------
def save_classification_report(y_true, y_pred, classes, out_path: Path):
    """Save sklearn classification report as CSV."""
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(out_path, index=True)


# -------------------------------------------------------------------
#  Confusion Matrix
# -------------------------------------------------------------------
def save_confusion_matrix(y_true, y_pred, classes, out_dir: Path, filename="confusion", normalize=False, epoch=None, model_name=None, single_column=True):
    """Save confusion matrix with optional epoch number in filename."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cmn = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    else:
        cmn = cm

    out_dir.mkdir(parents=True, exist_ok=True)
    apply_publication_style(single_column=single_column)

    # Single-column: square figure with readable annotations (Reviewer 1 Comment 1)
    fig_w, fig_h = pub_figsize(1.05, single_column=single_column)
    plt.figure(figsize=(fig_w, fig_h))
    annot_size = 8 if single_column else 10
    sns.heatmap(
        cmn, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
        xticklabels=classes, yticklabels=classes,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
        annot_kws={"size": annot_size},
    )
    plt.title(
        f"{model_name if model_name else ''} Confusion Matrix {'(Normalized)' if normalize else ''}"
        + (f" - Epoch {epoch}" if epoch is not None else ""),
        fontsize=13, fontweight='bold',
    )
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Generate filename with epoch
    if epoch is not None and model_name:
        filename_final = f"{model_name}_epoch{epoch:03d}_confusion{'_norm' if normalize else ''}"
    else:
        filename_final = filename
    
    plt.savefig(out_dir / f"{filename_final}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_heatmap_unified(cm: np.ndarray, classes: List[str], out_path: Path, title: str = "Confusion Matrix Heatmap (Unified Dataset)", single_column: bool = True):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = cm.astype(np.float32)
    matrix = matrix / (matrix.sum(axis=1, keepdims=True) + 1e-8)
    apply_publication_style(single_column=single_column)
    plt.figure(figsize=pub_figsize(1.05, single_column=single_column))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="rocket", xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 8})
    plt.title(title, fontsize=13)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cooccurrence_heatmap(cm: np.ndarray, classes: List[str], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    coocc = cm.astype(np.float32)
    coocc = coocc + coocc.T
    coocc = coocc / (coocc.sum() + 1e-8)
    plt.figure(figsize=(10, 8))
    sns.heatmap(coocc, annot=True, fmt=".3f", cmap="mako", xticklabels=classes, yticklabels=classes)
    plt.title("Correlation / Co-occurrence Heatmap")
    plt.xlabel("Class")
    plt.ylabel("Class")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# -------------------------------------------------------------------
#  ROC + PR curves
# -------------------------------------------------------------------
def compute_roc_pr_curves(y_true, y_prob, classes):
    """
    Compute ROC & PR curves for each class.
    Returns dicts containing fpr, tpr, precision, recall, auc values.
    """
    n = len(classes)
    y_bin = label_binarize(y_true, classes=list(range(n)))

    roc_data = {}
    pr_data = {}

    # ROC for each class
    for i in range(n):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc_c = auc(fpr, tpr)
        roc_data[i] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc_c}

    # micro-average ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
    roc_data["micro"] = {
        "fpr": fpr_micro,
        "tpr": tpr_micro,
        "auc": auc(fpr_micro, tpr_micro)
    }

    # PR curves
    for i in range(n):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        pr_auc_c = auc(rec, prec)
        pr_data[i] = {"precision": prec, "recall": rec, "auc": pr_auc_c}

    prec_micro, rec_micro, _ = precision_recall_curve(y_bin.ravel(), y_prob.ravel())
    pr_data["micro"] = {
        "precision": prec_micro,
        "recall": rec_micro,
        "auc": auc(rec_micro, prec_micro)
    }

    return roc_data, pr_data


def plot_roc_curves(roc_data, classes, out_dir: Path, filename="roc_curves"):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))

    for i, c in enumerate(classes):
        d = roc_data[i]
        plt.plot(d["fpr"], d["tpr"], lw=1.5, label=f"{c} (AUC={d['auc']:.3f})")

    # micro
    d = roc_data["micro"]
    plt.plot(d["fpr"], d["tpr"], '--', color="black", label=f"micro (AUC={d['auc']:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_pr_curves(pr_data, classes, out_dir: Path, filename="pr_curves"):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))

    for i, c in enumerate(classes):
        d = pr_data[i]
        plt.plot(d["recall"], d["precision"], lw=1.5, label=f"{c} (AUC={d['auc']:.3f})")

    d = pr_data["micro"]
    plt.plot(d["recall"], d["precision"], '--', color="black", label=f"micro (AUC={d['auc']:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curves")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_pr_roc_heatmaps(roc_data: Dict, pr_data: Dict, classes: List[str], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    roc_scores = [roc_data[i]["auc"] for i in range(len(classes))]
    pr_scores = [pr_data[i]["auc"] for i in range(len(classes))]

    plt.figure(figsize=(max(8, len(classes)), 3))
    sns.heatmap(np.array([roc_scores]), annot=True, fmt=".3f", cmap="viridis",
                xticklabels=classes, yticklabels=["ROC AUC"])
    plt.title("ROC AUC Heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_auc_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(max(8, len(classes)), 3))
    sns.heatmap(np.array([pr_scores]), annot=True, fmt=".3f", cmap="plasma",
                xticklabels=classes, yticklabels=["PR AUC"])
    plt.title("Precision–Recall AUC Heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "pr_auc_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


# -------------------------------------------------------------------
#  Learning curves
# -------------------------------------------------------------------
def plot_learning_curves(history: Dict[str, List[float]], out_dir: Path, model_name: str, epoch: Optional[int] = None, single_column: bool = True):
    """
    Learning curves in single-column journal format (stacked loss + accuracy).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_publication_style(single_column=single_column)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=pub_figsize(1.6, single_column=single_column))

    ax1.plot(epochs, history["train_loss"], 'b-', label="Train", linewidth=1.5, marker='o', markersize=3)
    ax1.plot(epochs, history["val_loss"], 'r-', label="Val", linewidth=1.5, marker='s', markersize=3)
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.set_title("Loss", fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], 'b-', label="Train", linewidth=1.5, marker='o', markersize=3)
    ax2.plot(epochs, history["val_acc"], 'r-', label="Val", linewidth=1.5, marker='s', markersize=3)
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy", fontsize=10)
    ax2.set_title("Accuracy", fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"{model_name} — Training Curves", fontsize=12, fontweight='bold')
    plt.tight_layout()

    filename = f"{model_name}_epoch{epoch:03d}_curves" if epoch is not None else f"{model_name}_curves"
    plt.savefig(out_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Also save separate loss-only file for backward compatibility
    plt.figure(figsize=pub_figsize(0.65, single_column=single_column))
    plt.plot(epochs, history["train_loss"], 'b-', label="Train Loss", linewidth=1.5)
    plt.plot(epochs, history["val_loss"], 'r-', label="Val Loss", linewidth=1.5)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.title(f"{model_name} — Loss", fontsize=11, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_name = f"{model_name}_epoch{epoch:03d}_loss" if epoch is not None else f"{model_name}_loss"
    plt.savefig(out_dir / f"{loss_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # LR curve
    if "lr" in history and len(history["lr"]) > 0:
        apply_publication_style(single_column=single_column)
        plt.figure(figsize=pub_figsize(0.65, single_column=single_column))
        plt.plot(history["lr"], label="LR", linewidth=1.5)
        plt.xlabel("Iteration", fontsize=10)
        plt.ylabel("Learning Rate", fontsize=12)
        plt.title(f"{model_name} - Learning Rate Schedule" + (f" (Epoch {epoch})" if epoch is not None else ""), fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"{model_name}_epoch{epoch:03d}_lr" if epoch is not None else f"{model_name}_lr"
        plt.savefig(out_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
        plt.close()


# -------------------------------------------------------------------
#  Multi-model comparison plots
# -------------------------------------------------------------------
def plot_model_comparison(results_dict: Dict[str, float], out_dir: Path, title="Model Accuracy Comparison"):
    """
    results_dict: {"model_name": accuracy, ...}
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    names = list(results_dict.keys())
    values = [results_dict[k] for k in names]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, values, color=plt.cm.viridis(np.linspace(0, 1, len(names))))
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_class_distribution_bars(df: pd.DataFrame, classes: List[str], out_path: Path):
    if "label_name" in df.columns:
        labels = df["label_name"].astype(str)
    else:
        labels = df["label"].astype(str)
    counts = labels.value_counts().reindex(classes, fill_value=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(10, len(classes)), 5))
    sns.barplot(x=counts.index, y=counts.values, palette="crest")
    plt.title("Class Distribution (Unified Dataset)")
    plt.ylabel("Samples")
    plt.xlabel("Class")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_dataset_similarity_heatmap(df: pd.DataFrame, classes: List[str], out_path: Path):
    if "dataset" not in df.columns:
        return
    dataset_names = sorted(df["dataset"].astype(str).unique())
    if len(dataset_names) < 2:
        return
    distributions = []
    for dataset in dataset_names:
        subset = df[df["dataset"] == dataset]
        if "label_name" in subset.columns:
            counts = subset["label_name"].astype(str).value_counts().reindex(classes, fill_value=0).values.astype(np.float32)
        else:
            counts = subset["label"].astype(str).value_counts().reindex(classes, fill_value=0).values.astype(np.float32)
        if counts.sum() == 0:
            counts = np.ones(len(classes), dtype=np.float32)
        distributions.append(counts / counts.sum())

    n = len(dataset_names)
    sims = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            sims[i, j] = 1.0 - float(_js_divergence(distributions[i], distributions[j]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(sims, annot=True, fmt=".3f", cmap="coolwarm", xticklabels=dataset_names, yticklabels=dataset_names, vmin=0, vmax=1)
    plt.title("Dataset Similarity Heatmap (1 - JS Divergence)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_augmentation_examples_grid(visuals: Dict[str, np.ndarray], out_path: Path):
    """Grid of augmentation example images. Accepts uint8 or float32 arrays."""
    if not visuals:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    items = list(visuals.items())
    cols = min(4, len(items))
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.atleast_2d(axes)
    for idx, (title, img) in enumerate(items):
        r, c = divmod(idx, cols)
        # Safely convert to float [0,1] for imshow — avoids black images from
        # accidental uint8→float truncation or out-of-range values.
        display_img = _prepare_display_image(img)
        axes[r, c].imshow(display_img, vmin=0.0, vmax=1.0)
        axes[r, c].set_title(title, fontsize=11)
        axes[r, c].axis("off")
    total_axes = rows * cols
    for idx in range(len(items), total_axes):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")
    fig.suptitle("Augmentation Examples", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_prior_work_chart(current_accuracy: float, out_path: Path):
    prior_entries = [
        ("Goodfellow et al. (2013)", 65.5),
        ("Barsoum et al. (2016)", 66.4),
        ("Pramerdorfer & Kampel (2016)", 66.8),
        ("Li et al. (2017)", 73.3),
        ("Minaee et al. (2021)", 72.1),
        ("Kollias et al. (2019)", 60.2),
    ]
    prior_entries.append(("Ours (ConvNeXt-V2)", max(0, min(1, current_accuracy)) * 100))
    labels = [entry[0] for entry in prior_entries]
    values = [entry[1] for entry in prior_entries]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    bars = plt.barh(labels, values, color=plt.cm.cividis(np.linspace(0, 1, len(labels))))
    for bar, value in zip(bars, values):
        plt.text(value + 0.5, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center")
    plt.xlabel("Accuracy (%)")
    plt.title("Prior Work Comparison")
    plt.xlim(0, max(values) + 5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# -------------------------------------------------------------------
#  Grad-CAM Heatmaps
# -------------------------------------------------------------------
def generate_gradcam(model, image_tensor, target_class, device, model_name):
    """
    Generate Grad-CAM heatmap for visualization.

    Uses both register_full_backward_hook (preferred for ConvNeXt-V2) and
    register_backward_hook as a fallback so gradients are captured reliably.
    Returns a float32 numpy array in [0, 1] (H × W) or None on failure.
    """
    model.eval()

    # Work on a fresh leaf tensor — ensures grad_fn chain is intact
    inp = image_tensor.detach().clone().to(device).unsqueeze(0)
    inp.requires_grad_(True)

    gradients: list = []
    activations: list = []

    def _forward_hook(module, inp_, out):
        activations.append(out)  # keep as tensor (on device)

    def _full_backward_hook(module, grad_in, grad_out):
        # grad_out[0]: gradient w.r.t. the *output* of this layer
        if grad_out[0] is not None:
            gradients.append(grad_out[0].detach())

    hooks = []
    target_layer = _locate_deep_feature_block(model)

    if target_layer is None:
        return None

    try:
        hooks.append(target_layer.register_forward_hook(_forward_hook))
        # Prefer full_backward_hook (works with ConvNeXt-V2 GRN blocks)
        if hasattr(target_layer, "register_full_backward_hook"):
            hooks.append(target_layer.register_full_backward_hook(_full_backward_hook))
        else:
            hooks.append(target_layer.register_backward_hook(
                lambda m, gi, go: _full_backward_hook(m, gi, go)
            ))

        # Forward
        output = model(inp)
        model.zero_grad()

        # Scalar target score — backprop from target class logit
        score = output[0, target_class]
        score.backward()

    except Exception as exc:
        print(f"  ⚠ Grad-CAM forward/backward failed: {exc}")
        for h in hooks:
            h.remove()
        return None
    finally:
        for h in hooks:
            h.remove()

    if not gradients or not activations:
        return None

    grads = gradients[0]           # (1, C, H, W)
    acts  = activations[0].detach() # (1, C, H, W)

    # Global-average-pool the gradients → channel weights
    weights = grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)

    # Weighted sum of activation maps
    cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H, W)
    cam = F.relu(cam)

    # Upsample to input spatial size
    cam = F.interpolate(
        cam, size=(inp.size(2), inp.size(3)),
        mode="bilinear", align_corners=False
    )

    cam_np = cam.squeeze().cpu().numpy().astype(np.float32)
    lo, hi = cam_np.min(), cam_np.max()
    if hi - lo < 1e-6:
        # Flat CAM → return uniform map rather than zero-divide artefact
        return np.ones_like(cam_np) * 0.5
    cam_np = (cam_np - lo) / (hi - lo)
    return cam_np


def plot_gradcam_heatmap(image, cam, out_path: Path, model_name: str, epoch: int, class_name: str, pred_class: int):
    """
    Plot side-by-side: original image  |  Grad-CAM overlay.

    Both panels receive float [0, 1] arrays — matplotlib renders these
    correctly without any uint8 cast (uint8 cast destroys sub-1 values).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_img, overlay_img = _create_gradcam_overlay(image, cam)  # float [0,1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: original image
    axes[0].imshow(base_img, vmin=0.0, vmax=1.0)   # float, NO uint8 cast
    axes[0].set_title(
        f"Original Image\nPredicted: {class_name} ({pred_class})",
        fontsize=12, fontweight="bold"
    )
    axes[0].axis("off")

    # Right: Grad-CAM heatmap overlay
    axes[1].imshow(overlay_img, vmin=0.0, vmax=1.0)  # float, NO uint8 cast
    axes[1].set_title(
        f"Grad-CAM Heatmap\nEpoch {epoch}",
        fontsize=12, fontweight="bold"
    )
    axes[1].axis("off")

    fig.suptitle(f"{model_name} - Grad-CAM Visualization", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_gradcam_gallery(entries: List[Dict], out_path: Path):
    """
    Gallery of Grad-CAM overlays (one tile per sample).
    Renders float [0,1] arrays — do NOT cast to uint8.
    Skips entries with None cam gracefully (uses uniform heatmap fallback).
    """
    if not entries:
        return
    # Filter completely invalid entries (missing image key)
    valid_entries = [e for e in entries if e.get("image") is not None]
    if not valid_entries:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = min(4, len(valid_entries))
    rows = math.ceil(len(valid_entries) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.atleast_2d(axes)
    for idx, entry in enumerate(valid_entries):
        r, c = divmod(idx, cols)
        try:
            # None cam is handled inside _create_gradcam_overlay
            _, overlay_img = _create_gradcam_overlay(entry["image"], entry.get("cam"))
            axes[r, c].imshow(overlay_img, vmin=0.0, vmax=1.0)  # float [0,1], no cast
        except Exception as exc:
            axes[r, c].text(0.5, 0.5, f"Error:\n{exc}", ha="center", va="center",
                            fontsize=8, color="red", transform=axes[r, c].transAxes)
        axes[r, c].axis("off")
        axes[r, c].set_title(entry.get("title", f"Sample {idx+1}"), fontsize=10)
    total_axes = rows * cols
    for idx in range(len(valid_entries), total_axes):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")
    fig.suptitle("Grad-CAM Gallery", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_activation_map_heatmap(model: nn.Module, image_tensor: torch.Tensor, device: torch.device, out_path: Path, model_name: str):
    target_layer = _locate_deep_feature_block(model)
    if target_layer is None:
        return
    activations: List[torch.Tensor] = []

    def forward_hook(module, _, output):
        activations.append(output.detach().cpu())

    handle = target_layer.register_forward_hook(forward_hook)
    model.eval()
    with torch.no_grad():
        _ = model(image_tensor.to(device))
    handle.remove()

    if not activations:
        return

    fmap = activations[0]
    if fmap.ndim == 4:
        fmap = fmap[0]
    activation_map = fmap.mean(0).numpy()
    activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    sns.heatmap(activation_map, cmap="inferno", cbar=True, xticklabels=False, yticklabels=False)
    plt.title(f"{model_name} Activation Map Heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_tsne_visualization(features, labels, classes, out_path: Path, model_name: str, epoch: int, perplexity: int = 30):
    """
    Generate t-SNE visualization of learned features.
    features: (N, D) feature vectors from penultimate layer
    labels: (N,) true labels
    """
    print(f"Computing t-SNE for {len(features)} samples...")

    # Reduce dimensionality if needed
    if features.shape[1] > 50:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=50)
        features = pca.fit_transform(features)
        print(f"Reduced to 50 dims via PCA (var={pca.explained_variance_ratio_.sum():.3f})")

    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(features) - 1),
                random_state=42, n_iter=1000)
    features_2d = tsne.fit_transform(features)

    labels_arr = np.array(labels)
    unique_labels = np.unique(labels_arr)
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))

    fig, ax = plt.subplots(figsize=(12, 10))
    # ONE scatter per class — avoids duplicate overlapping plots that cause
    # blobs/black patches on top of the coloured scatter
    for i, label in enumerate(unique_labels):
        mask = labels_arr == label
        class_label = classes[int(label)] if int(label) < len(classes) else f"Class {label}"
        ax.scatter(
            features_2d[mask, 0], features_2d[mask, 1],
            label=class_label,
            color=palette[i],
            alpha=0.6, s=20,
        )
    ax.set_title(f"{model_name} — t-SNE (Epoch {epoch})", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_umap_visualization(features, labels, classes, out_path: Path, model_name: str, epoch: int, n_neighbors: int = 15):
    """
    Generate UMAP visualization of learned features.
    """
    if not HAVE_UMAP:
        print("Warning: UMAP not available. Install with: pip install umap-learn")
        return
    
    print(f"Computing UMAP for {len(features)} samples...")
    
    # Reduce dimensionality if needed
    if features.shape[1] > 50:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=50)
        features = pca.fit_transform(features)
    
    # Compute UMAP
    reducer = umap.UMAP(n_neighbors=min(n_neighbors, len(features)-1), random_state=42, n_components=2)
    features_2d = reducer.fit_transform(features)
    
    # Plot
    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   label=classes[int(label)] if int(label) < len(classes) else f"Class {label}",
                   alpha=0.6, s=20, c=[colors[i]])
    
    plt.title(f"{model_name} - UMAP Visualization (Epoch {epoch})", fontsize=14, fontweight='bold')
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


# -------------------------------------------------------------------
#  Ablation Study Charts
# -------------------------------------------------------------------
def plot_ablation_study(results: Dict[str, Dict[str, float]], out_path: Path, model_name: str, single_column: bool = True):
    """
    Plot ablation study results as bar charts (single-column stacked layout for papers).
    """
    configs = [k for k in results.keys() if not k.startswith("_") and "error" not in results.get(k, {})]
    metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    apply_publication_style(single_column=single_column)

    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=pub_figsize(0.55 * n, single_column=single_column))
    if n == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        values = [results[config].get(metric, 0.0) for config in configs]
        stds = [results[config].get(f"{metric}_std", 0.0) for config in configs]
        bars = axes[idx].bar(
            range(len(configs)), values,
            yerr=stds if any(s > 0 for s in stds) else None,
            capsize=3,
            color=plt.cm.viridis(np.linspace(0, 1, len(configs))),
        )
        axes[idx].set_xticks(range(len(configs)))
        axes[idx].set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
        axes[idx].set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
        axes[idx].set_title(f"{metric.replace('_', ' ').title()}", fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width() / 2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle(f"{model_name} - Ablation Study", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


# -------------------------------------------------------------------
#  Architecture Diagram
# -------------------------------------------------------------------
def plot_architecture_diagram(model, out_path: Path, model_name: str):
    """
    Generate a simple architecture diagram showing model structure.
    """
    try:
        from torchsummary import summary
        import io
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        try:
            summary(model, (3, 256, 256), device='cpu')
            summary_str = buffer.getvalue()
        except:
            summary_str = f"{model_name} Architecture\n\nModel structure saved to diagram."
        finally:
            sys.stdout = old_stdout
        
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.text(0.1, 0.5, summary_str, fontsize=10, family='monospace', verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.title(f"{model_name} - Architecture Overview", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        # Fallback: Create a simple text-based diagram
        fig, ax = plt.subplots(figsize=(12, 8))
        layers = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                layers.append(f"{name}: {type(module).__name__}")
        
        text = f"{model_name} Architecture\n\n" + "\n".join(layers[:20])  # Limit to 20 layers
        ax.text(0.05, 0.95, text, fontsize=9, family='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.title(f"{model_name} - Architecture Overview", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()


# -------------------------------------------------------------------
#  Cross-Dataset Evaluation Charts
# -------------------------------------------------------------------
def plot_cross_dataset_evaluation(results: Dict[str, Dict[str, float]], out_path: Path, model_name: str, single_column: bool = True):
    """Plot cross-dataset evaluation heatmap (single-column journal format)."""
    train_datasets = list(results.keys())

    test_datasets = set()
    for train_data in results.values():
        if isinstance(train_data, dict) and "test_datasets" in train_data:
            test_datasets.update(train_data["test_datasets"].keys())
        else:
            test_datasets.update(train_data.keys())
    test_datasets = sorted(list(test_datasets))

    acc_matrix = np.zeros((len(train_datasets), len(test_datasets)))
    for i, train_key in enumerate(train_datasets):
        train_data = results[train_key]
        td = train_data.get("test_datasets", train_data) if isinstance(train_data, dict) else train_data
        for j, test_data in enumerate(test_datasets):
            if test_data in td:
                entry = td[test_data]
                acc_matrix[i, j] = entry.get('accuracy', 0.0) if isinstance(entry, dict) else float(entry)

    apply_publication_style(single_column=single_column)
    plt.figure(figsize=pub_figsize(0.85, single_column=single_column))
    sns.heatmap(
        acc_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
        xticklabels=test_datasets, yticklabels=train_datasets,
        cbar_kws={'label': 'Accuracy'}, annot_kws={"size": 8},
    )
    plt.title(f"{model_name}\nCross-Dataset (Train → Test)", fontsize=12, fontweight='bold')
    plt.xlabel("Test Dataset", fontsize=11)
    plt.ylabel("Train Dataset", fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


# -------------------------------------------------------------------
#  Facial Landmark Attention Heatmaps
# -------------------------------------------------------------------
def detect_facial_landmarks(image: np.ndarray):
    """
    Detect facial landmarks using MediaPipe.
    Returns list of landmark coordinates (x, y) or None if no face detected.
    """
    try:
        import mediapipe as mp
        import cv2
        
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.dtype == np.uint8 else image
        else:
            rgb_image = image
        
        # Ensure uint8
        if rgb_image.dtype != np.uint8:
            rgb_image = (rgb_image * 255).astype(np.uint8) if rgb_image.max() <= 1.0 else rgb_image.astype(np.uint8)
        
        results = face_mesh.process(rgb_image)
        face_mesh.close()
        
        if results.multi_face_landmarks:
            landmarks = []
            h, w = rgb_image.shape[:2]
            for landmark in results.multi_face_landmarks[0].landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append((x, y))
            return landmarks
        return None
    except ImportError:
        # Fallback: use simple face detection without landmarks
        try:
            import cv2
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image.astype(np.uint8)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                # Return approximate face region center
                x, y, w, h = faces[0]
                return [(x + w//2, y + h//2)]
            return None
        except:
            return None
    except Exception:
        return None


def create_landmark_attention_heatmap(image: np.ndarray, landmarks: List[Tuple[int, int]], 
                                      sigma: float = 20.0) -> np.ndarray:
    """
    Create attention heatmap from facial landmarks using Gaussian kernels.
    
    Args:
        image: Input image (H, W, 3) or (H, W)
        landmarks: List of (x, y) landmark coordinates
        sigma: Gaussian kernel standard deviation
    
    Returns:
        Heatmap array (H, W) with values in [0, 1]
    """
    if landmarks is None or len(landmarks) == 0:
        # Return uniform heatmap if no landmarks
        h, w = image.shape[:2]
        return np.ones((h, w), dtype=np.float32) * 0.5
    
    h, w = image.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    for x, y in landmarks:
        # Create Gaussian kernel for each landmark
        y_coords, x_coords = np.ogrid[:h, :w]
        gaussian = np.exp(-((x_coords - x)**2 + (y_coords - y)**2) / (2 * sigma**2))
        heatmap = np.maximum(heatmap, gaussian)
    
    # Normalize to [0, 1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap


def plot_facial_landmark_attention_heatmap(image, landmarks: List[Tuple[int, int]],
                                           class_name: str, out_path: Path,
                                           model_name: str = "ConvNeXt-V2"):
    """
    Plot facial landmark attention heatmap for a single image.

    Args:
        image: PIL Image, torch.Tensor, or np.ndarray (any dtype/range accepted)
        landmarks: List of (x, y) landmark coordinates
        class_name: Emotion class name
        out_path: Output path for PNG file
        model_name: Model name for title
    """
    # ── Normalise input to float32 [0,1] HWC first ──────────────────────────
    # _prepare_display_image handles PIL, Tensor, uint8 array, and float array
    # correctly WITHOUT zeroing low values the way `.astype(np.uint8)` would.
    display_float = _prepare_display_image(image)   # float32 [0,1], HWC

    # Keep a uint8 copy only for landmark coordinate scaling (shape reference)
    h, w = display_float.shape[:2]

    # Ensure 3-channel
    if display_float.ndim == 2:
        display_float = np.stack([display_float] * 3, axis=-1)
    if display_float.shape[2] == 4:
        display_float = display_float[:, :, :3]

    # Create heatmap
    heatmap = create_landmark_attention_heatmap(display_float, landmarks)

    # Coloured heatmap overlay
    try:
        cmap_jet = plt.colormaps["jet"]
    except AttributeError:
        cmap_jet = cm.get_cmap("jet")
    heatmap_colored = cmap_jet(heatmap)[..., :3].astype(np.float32)
    overlay = np.clip(0.6 * heatmap_colored + 0.4 * display_float, 0.0, 1.0)

    # ── Create figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Original image — float [0,1] passed directly, NO uint8 cast
    axes[0].imshow(display_float, vmin=0.0, vmax=1.0)
    axes[0].set_title(f"Original Image\nClass: {class_name}",
                      fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Landmarks overlay on original
    axes[1].imshow(display_float, vmin=0.0, vmax=1.0)
    if landmarks:
        lm_x = [lm[0] for lm in landmarks]
        lm_y = [lm[1] for lm in landmarks]
        axes[1].scatter(lm_x, lm_y, c="red", s=10, alpha=0.6, marker="o")
    axes[1].set_title(
        f"Facial Landmarks\n{len(landmarks) if landmarks else 0} points",
        fontsize=12, fontweight="bold",
    )
    axes[1].axis("off")

    # Panel 3: Attention heatmap overlay
    axes[2].imshow(overlay, vmin=0.0, vmax=1.0)
    axes[2].set_title(f"Attention Heatmap\nClass: {class_name}",
                      fontsize=12, fontweight="bold")
    axes[2].axis("off")

    plt.suptitle(f"{model_name} — Facial Landmark Attention Heatmap",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_class_landmark_heatmaps(model: nn.Module, df_val: pd.DataFrame, images_root: str,
                                     classes: List[str], device: torch.device,
                                     results_dir: Path, model_name: str,
                                     samples_per_class: int = 3):
    """
    Generate facial landmark attention heatmaps for each emotion class.

    Args:
        model: Trained model
        df_val: Validation dataframe
        images_root: Root directory for images
        classes: List of class names
        device: Device for model inference
        results_dir: Directory to save heatmaps
        model_name: Model name
        samples_per_class: Number of samples per class to visualize
    """
    # NOTE: cv2 is imported lazily inside detect_facial_landmarks — do NOT
    # import it here unconditionally; it is an optional dependency and will
    # crash on Kaggle/Colab environments that lack it.
    from pathlib import Path  # noqa: F811
    from PIL import Image as _PIL  # noqa: F811

    model.eval()
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n>>> Generating Facial Landmark Attention Heatmaps for {len(classes)} classes...")
    
    # Group validation data by class
    if "label" in df_val.columns:
        for class_idx, class_name in enumerate(classes):
            class_samples = df_val[df_val["label"] == class_idx].head(samples_per_class)
            
            if len(class_samples) == 0:
                print(f"  ⚠ No samples found for class {class_name}, skipping...")
                continue
            
            print(f"  Processing {class_name} ({len(class_samples)} samples)...")
            
            for idx, (_, row) in enumerate(class_samples.iterrows()):
                try:
                    img_path = Path(images_root) / str(row["image_path"])
                    if not img_path.exists():
                        continue
                    
                    # Load image
                    img = Image.open(img_path).convert("RGB")
                    img_array = np.array(img)
                    
                    # Detect landmarks
                    landmarks = detect_facial_landmarks(img_array)
                    
                    # Create heatmap visualization
                    out_path = figures_dir / f"{model_name}_landmark_heatmap_{class_name.lower()}_sample{idx+1}.png"
                    plot_facial_landmark_attention_heatmap(
                        img_array, landmarks, class_name, out_path, model_name
                    )
                    
                except Exception as e:
                    print(f"    ⚠ Error processing sample {idx+1} for {class_name}: {e}")
                    continue
            
            print(f"  ✓ Generated heatmaps for {class_name}")
    
    print(f"  ✓ All facial landmark attention heatmaps saved to: {figures_dir}")
