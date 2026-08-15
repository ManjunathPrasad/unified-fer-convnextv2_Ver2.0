# evaluate.py
"""
Q1 Emotion Recognition Framework – Standalone Evaluation Script

Loads a trained checkpoint and evaluates it on a given dataset CSV,
producing the full suite of publication-ready metrics and figures:

  - Overall Accuracy, F1-macro, F1-weighted, Precision, Recall
  - Per-class metrics table (CSV)
  - Classification report (CSV)
  - Confusion matrix (raw + normalised PNG)
  - ROC curves (per-class + macro-average PNG)
  - PR  curves (per-class + macro-average PNG)
  - Per-dataset breakdown (if `dataset` column present in CSV)

Usage
-----
  # Evaluate a weights-only checkpoint (.pth):
  python evaluate.py \\
      --checkpoint Results_Q1/exp_xxx/convnext_v2/convnext_v2_best.pth \\
      --csv path/to/val.csv \\
      --images path/to/images \\
      --num_classes 7

  # Use a full checkpoint (saves optimizer state too):
  python evaluate.py \\
      --checkpoint Results_Q1/exp_xxx/convnext_v2/convnext_v2_best_full.pth \\
      --full_ckpt \\
      --csv path/to/unified_dataset.csv \\
      --images path/to/images \\
      --output Results_Q1/eval_output

  # Load config from YAML:
  python evaluate.py \\
      --checkpoint convnext_v2_best.pth \\
      --csv val.csv --images images/ \\
      --config configs/kaggle_2gpu.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)

# ── Framework imports ────────────────────────────────────────────────────────
from model_factory import get_model, normalize_model_name, SUPPORTED_MODEL
from train_engine import EmotionDataset, evaluate_model
from utils import (
    ensure_dir, load_yaml, load_full_ckpt, resolve_config_auto,
    set_seed, save_json, timestamp,
)
from metrics_and_plots import (
    save_confusion_matrix as plot_confusion_matrix,
    compute_roc_pr_curves, plot_roc_curves, plot_pr_curves,
    save_classification_report,
)

# ── Emotion class names (7-class FER mapping) ─────────────────────────────────
EMOTION_NAMES = {
    0: "Angry", 1: "Disgust", 2: "Fear",
    3: "Happy", 4: "Sad",    5: "Surprise", 6: "Neutral",
}


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation function
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    checkpoint_path: str,
    csv_path: str,
    images_root: str,
    num_classes: int = 7,
    output_dir: str = "eval_output",
    model_name: str = "convnext_v2",
    input_size: int = 256,
    batch_size: int = 32,
    device_str: str = "auto",
    full_ckpt: bool = False,
    seed: int = 42,
    class_names: list = None,
) -> dict:
    """
    Run full evaluation pipeline and return results dict.

    Parameters
    ----------
    checkpoint_path : str
        Path to .pth checkpoint file (weights-only or full).
    csv_path : str
        Path to dataset CSV with columns [image_path, label, (optional) dataset].
    images_root : str
        Root directory for images.
    num_classes : int
        Number of emotion classes.
    output_dir : str
        Directory for output figures and CSV files.
    model_name : str
        Model architecture (only convnext_v2 is supported).
    input_size : int
        Input image resolution (default 256).
    batch_size : int
        Batch size for inference.
    device_str : str
        "auto", "cuda", or "cpu".
    full_ckpt : bool
        If True, load checkpoint saved by save_full_ckpt() (contains meta).
    seed : int
        Random seed (for deterministic transforms).
    class_names : list
        Optional list of class name strings. Defaults to EMOTION_NAMES 0-6.

    Returns
    -------
    dict with keys: accuracy, f1_macro, f1_weighted, precision_macro,
                    recall_macro, per_class, per_dataset, cm, report_path
    """
    set_seed(seed, deterministic=True)

    # ── Device ────────────────────────────────────────────────────────────────
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"\n{'='*65}")
    print(f"  Q1 Evaluation Script  |  {timestamp()}")
    print(f"{'='*65}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  CSV        : {csv_path}")
    print(f"  Images     : {images_root}")
    print(f"  Device     : {device}")
    print(f"  Num classes: {num_classes}")
    print(f"{'='*65}\n")

    # ── Output dir ────────────────────────────────────────────────────────────
    out = Path(output_dir)
    ensure_dir(str(out))

    # ── Load model ────────────────────────────────────────────────────────────
    model = get_model(
        name=model_name,
        num_classes=num_classes,
        pretrained=False,
        input_size=input_size,
        device=device,
    )

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if full_ckpt:
        ckpt = load_full_ckpt(str(ckpt_path), map_location=str(device))
        model.load_state_dict(ckpt["model_state"])
        ckpt_meta = {k: v for k, v in ckpt.items() if k != "model_state"}
        print(f"  Loaded full checkpoint — epoch {ckpt.get('epoch', '?')}, "
              f"best_val_acc={ckpt.get('best_val_acc', '?'):.4f}")
        save_json(ckpt_meta, str(out / "checkpoint_meta.json"))
    else:
        state = torch.load(str(ckpt_path), map_location=str(device))
        # Handle both raw state_dict and wrapped {"model_state": ...}
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state)
        print(f"  Loaded weights-only checkpoint.")

    model.eval()

    # ── Class names ────────────────────────────────────────────────────────────
    if class_names is None:
        class_names = [EMOTION_NAMES.get(i, str(i)) for i in range(num_classes)]

    # ── Dataset ───────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)

    # Filter unknown labels and map to int
    if "label" in df.columns:
        df = df[df["label"] != "unknown"].copy()
        try:
            df["label"] = df["label"].apply(lambda x: int(float(x)))
            df = df[df["label"].isin(range(num_classes))].copy()
        except Exception:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df["label"] = le.fit_transform(df["label"].astype(str))
            class_names = [str(c) for c in le.classes_]

    # Filter missing images
    from utils import filter_missing_images
    df = filter_missing_images(df, images_root)

    print(f"  Evaluating on {len(df)} samples ({len(df['label'].unique())} classes)\n")

    eval_ds = EmotionDataset(df, images_root, input_size=input_size, train=False)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for xb, yb in eval_loader:
            xb = xb.to(device)
            out = model(xb)
            probs = F.softmax(out, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.extend(probs.argmax(1).tolist())
            all_labels.extend(yb.tolist())

    all_probs  = np.vstack(all_probs)
    labels     = np.array(all_labels)
    preds      = np.array(all_preds)

    # ── Overall metrics ───────────────────────────────────────────────────────
    acc              = accuracy_score(labels, preds)
    f1_macro         = f1_score(labels, preds, average="macro",     zero_division=0)
    f1_weighted      = f1_score(labels, preds, average="weighted",  zero_division=0)
    precision_macro  = precision_score(labels, preds, average="macro",    zero_division=0)
    recall_macro     = recall_score(labels, preds, average="macro",       zero_division=0)
    cm               = confusion_matrix(labels, preds)

    print(f"{'─'*45}")
    print(f"  Accuracy         : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1-macro         : {f1_macro:.4f}")
    print(f"  F1-weighted      : {f1_weighted:.4f}")
    print(f"  Precision-macro  : {precision_macro:.4f}")
    print(f"  Recall-macro     : {recall_macro:.4f}")
    print(f"{'─'*45}\n")

    # ── Per-class metrics ─────────────────────────────────────────────────────
    per_class = {}
    for cls in range(num_classes):
        mask = labels == cls
        if mask.sum() == 0:
            continue
        per_class[class_names[cls]] = {
            "accuracy":  float(accuracy_score(labels[mask], preds[mask])),
            "precision": float(precision_score(labels == cls, preds == cls, zero_division=0)),
            "recall":    float(recall_score(labels == cls, preds == cls, zero_division=0)),
            "f1":        float(f1_score(labels == cls, preds == cls, zero_division=0)),
            "support":   int(mask.sum()),
        }

    print("  Per-class metrics:")
    print(f"  {'Class':<12} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>6}")
    print(f"  {'─'*45}")
    for cls_name, m in per_class.items():
        print(f"  {cls_name:<12} {m['accuracy']:>6.3f} {m['precision']:>6.3f} "
              f"{m['recall']:>6.3f} {m['f1']:>6.3f} {m['support']:>6d}")
    print()

    # ── Per-dataset breakdown ─────────────────────────────────────────────────
    per_dataset = {}
    if "dataset" in df.columns:
        dataset_col = df["dataset"].reset_index(drop=True)
        for ds_name in dataset_col.unique():
            mask = (dataset_col == ds_name).values
            if mask.sum() == 0:
                continue
            per_dataset[str(ds_name)] = {
                "accuracy": float(accuracy_score(labels[mask], preds[mask])),
                "f1_macro": float(f1_score(labels[mask], preds[mask], average="macro", zero_division=0)),
                "support":  int(mask.sum()),
            }
        if per_dataset:
            print("  Per-dataset breakdown:")
            for ds_name, m in per_dataset.items():
                print(f"    {ds_name}: Acc={m['accuracy']:.4f}, F1={m['f1_macro']:.4f}, N={m['support']}")
            print()

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    results = {
        "checkpoint":       str(checkpoint_path),
        "csv":              str(csv_path),
        "num_samples":      int(len(labels)),
        "num_classes":      num_classes,
        "class_names":      class_names,
        "timestamp":        timestamp(),
        "accuracy":         float(acc),
        "f1_macro":         float(f1_macro),
        "f1_weighted":      float(f1_weighted),
        "precision_macro":  float(precision_macro),
        "recall_macro":     float(recall_macro),
        "per_class":        per_class,
        "per_dataset":      per_dataset,
    }
    save_json(results, str(out / "eval_metrics.json"))
    print(f"  ✓ Metrics saved → eval_metrics.json")

    # ── Classification report CSV ─────────────────────────────────────────────
    report_path = out / "classification_report.csv"
    save_classification_report(labels, preds, class_names, report_path)
    print(f"  ✓ Classification report → classification_report.csv")

    # ── Confusion matrices ────────────────────────────────────────────────────
    try:
        plot_confusion_matrix(
            labels, preds, class_names, out,
            filename="confusion_matrix_raw",
            normalize=False,
            epoch=0, model_name=model_name,
        )
        plot_confusion_matrix(
            labels, preds, class_names, out,
            filename="confusion_matrix_norm",
            normalize=True,
            epoch=0, model_name=model_name,
        )
        print(f"  ✓ Confusion matrices (raw + normalised) saved")
    except Exception as e:
        print(f"  ⚠ Confusion matrix error: {e}")

    # ── ROC / PR curves ────────────────────────────────────────────────────────
    try:
        roc_data, pr_data = compute_roc_pr_curves(labels, all_probs, class_names)
        plot_roc_curves(roc_data, class_names, out, filename="roc_curves")
        plot_pr_curves(pr_data,  class_names, out, filename="pr_curves")
        print(f"  ✓ ROC and PR curves saved")
    except Exception as e:
        print(f"  ⚠ ROC/PR curves error: {e}")

    print(f"\n  All outputs written to: {out.resolve()}\n")
    results["report_path"] = str(report_path)
    results["cm"] = cm.tolist()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Q1 Emotion Recognition – Standalone Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pth checkpoint (weights-only or full).")
    parser.add_argument("--csv",        type=str, required=True,
                        help="Path to evaluation dataset CSV.")
    parser.add_argument("--images",     type=str, required=True,
                        help="Root directory for images.")
    parser.add_argument("--num_classes", type=int, default=7,
                        help="Number of emotion classes (default: 7).")
    parser.add_argument("--output",     type=str, default="eval_output",
                        help="Output directory for figures and CSVs.")
    parser.add_argument("--model",      type=str, default="convnext_v2",
                        help="Model architecture (default: convnext_v2).")
    parser.add_argument("--input_size", type=int, default=256,
                        help="Input image size (default: 256).")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference (default: 32).")
    parser.add_argument("--device",     type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Compute device (default: auto).")
    parser.add_argument("--full_ckpt", action="store_true", default=False,
                        help="Load a full checkpoint (contains optimizer/scheduler state).")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--config",     type=str, default=None,
                        help="Optional YAML config to override defaults. "
                             "CLI flags take priority over YAML.")
    parser.add_argument("--class_names", type=str, nargs="*", default=None,
                        help="Custom class names (space-separated). "
                             "Example: --class_names Angry Disgust Fear Happy Sad Surprise Neutral")
    return parser.parse_args()


def main():
    args = parse_args()

    # Build kwargs from args, optionally merged with YAML
    kwargs = {
        "checkpoint_path": args.checkpoint,
        "csv_path":        args.csv,
        "images_root":     args.images,
        "num_classes":     args.num_classes,
        "output_dir":      args.output,
        "model_name":      args.model,
        "input_size":      args.input_size,
        "batch_size":      args.batch_size,
        "device_str":      args.device,
        "full_ckpt":       args.full_ckpt,
        "seed":            args.seed,
        "class_names":     args.class_names,
    }

    # Merge YAML config if provided (CLI args take priority)
    if args.config:
        yaml_cfg = load_yaml(args.config)
        yaml_cfg = resolve_config_auto(yaml_cfg)
        # Only apply YAML values where CLI kept the default
        if args.num_classes == 7  and "num_classes" in yaml_cfg:
            kwargs["num_classes"]  = yaml_cfg["num_classes"]
        if args.input_size  == 256 and "input_size_ft" in yaml_cfg:
            kwargs["input_size"]   = yaml_cfg["input_size_ft"]
        if args.batch_size  == 32  and "batch_size"    in yaml_cfg:
            kwargs["batch_size"]   = yaml_cfg["batch_size"]
        if args.device      == "auto" and "device"     in yaml_cfg:
            kwargs["device_str"]   = yaml_cfg["device"]
        if args.seed        == 42  and "seed"          in yaml_cfg:
            kwargs["seed"]         = yaml_cfg["seed"]
        print(f"  Merged config from: {args.config}")

    results = run_evaluation(**kwargs)
    print(f"{'='*65}")
    print(f"  EVALUATION COMPLETE")
    print(f"  Accuracy : {results['accuracy']:.4f}  ({results['accuracy']*100:.2f}%)")
    print(f"  F1-macro : {results['f1_macro']:.4f}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
