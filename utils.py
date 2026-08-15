# utils.py
"""
Utility functions used across the Q1-ready emotion classification framework.

Includes:
 - set_seed(seed)
 - ensure_dir(path)
 - format_time(seconds)
 - load_yaml(path)
 - load_json(path)
 - save_json(obj, path)
 - log_gpu_status()
 - filter_missing_images(df, images_root)
 - timestamp()
 - model_checkpointing helpers (save_ckpt, load_ckpt)
 - EpochLogger  – per-epoch CSV logger
 - save_split_indices / load_split_indices – reproducible split persistence
 - resolve_config_auto – expand "auto" values based on device availability
"""

import os
import json
import yaml
import time
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch

# --------------------------------------------------
# General-purpose utilities
# --------------------------------------------------

def ensure_dir(path: str):
    """Create directory if missing."""
    Path(path).mkdir(parents=True, exist_ok=True)

def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set seed for reproducibility (Python, NumPy, PyTorch CPU/GPU).
    
    Args:
        seed: Random seed value
        deterministic: If True, use deterministic CUDA operations (slower but reproducible)
                      If False, use benchmark mode (faster but less reproducible)
    """
    import torch
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        # Fully deterministic mode for reproducibility (Q1 research standards)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
    else:
        # Benchmark mode for speed (non-deterministic)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def timestamp():
    """Human-readable timestamp string."""
    return time.strftime("%Y-%m-%d_%H-%M-%S")

def format_time(sec: float) -> str:
    """Convert seconds to H:M:S."""
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

# --------------------------------------------------
# YAML / JSON I/O
# --------------------------------------------------

def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML config safely."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)

def save_json(obj: Dict[str, Any], path: str):
    """Save object as JSON file."""
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)

# --------------------------------------------------
# GPU Utilities
# --------------------------------------------------

def log_gpu_status():
    """Prints GPU memory info for debugging."""
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return
    print("CUDA version:", torch.version.cuda)
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    total = props.total_memory / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    print(f"GPU: {props.name}")
    print(f"Total: {total:.2f} GB | Reserved: {reserved:.2f} GB | Allocated: {allocated:.2f} GB")

# --------------------------------------------------
# CSV / Dataset helpers
# --------------------------------------------------

def filter_missing_images(df: pd.DataFrame, images_root: str) -> pd.DataFrame:
    """
    Remove rows whose image_path does not exist under images_root.
    Returns filtered DataFrame.
    """
    images_root = Path(images_root)
    exists = df["image_path"].apply(lambda p: (images_root / str(p)).exists())
    missing_count = (~exists).sum()

    if missing_count > 0:
        print(f"[Warning] Missing {missing_count} images … filtering them out.")

    return df[exists].reset_index(drop=True)

# --------------------------------------------------
# Model checkpoint helpers
# --------------------------------------------------

def save_ckpt(path: str, model_state: Dict[str, Any], meta: Dict[str, Any] = None):
    """Save checkpoint (model weights + optional metadata)."""
    ensure_dir(Path(path).parent)
    ckpt = {
        "model_state": model_state,
        "meta": meta if meta else {}
    }
    torch.save(ckpt, path)

def save_full_ckpt(
    path: str,
    model_state: Dict[str, Any],
    optimizer_state: Optional[Dict[str, Any]] = None,
    scheduler_state: Optional[Dict[str, Any]] = None,
    scaler_state: Optional[Dict[str, Any]] = None,
    epoch: int = 0,
    best_val_acc: float = 0.0,
    seed: int = 42,
    config: Optional[Dict[str, Any]] = None,
):
    """
    Save a full training checkpoint including optimizer, scheduler, and scaler states.
    This allows resuming training from an exact point and ensures full reproducibility.
    
    Args:
        path: Output file path (.pth)
        model_state: model.state_dict()
        optimizer_state: optimizer.state_dict()
        scheduler_state: scheduler.state_dict()
        scaler_state: AMP GradScaler.state_dict() (None on CPU)
        epoch: Epoch number at checkpoint
        best_val_acc: Best validation accuracy seen so far
        seed: Random seed used for this experiment
        config: Training configuration dict
    """
    ensure_dir(Path(path).parent)
    ckpt = {
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "scaler_state": scaler_state,
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "seed": seed,
        "config": config or {},
        "timestamp": timestamp(),
    }
    torch.save(ckpt, path)

def load_ckpt(path: str, map_location: str = "cpu"):
    """Load checkpoint, return (model_state, meta_dict)."""
    ckpt = torch.load(path, map_location=map_location)
    return ckpt["model_state"], ckpt.get("meta", {})

def load_full_ckpt(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """
    Load a full checkpoint saved by save_full_ckpt().
    Returns the entire checkpoint dict with keys:
        model_state, optimizer_state, scheduler_state, scaler_state,
        epoch, best_val_acc, seed, config, timestamp
    """
    ckpt = torch.load(path, map_location=map_location)
    return ckpt

# --------------------------------------------------
# CSV row logging helper
# --------------------------------------------------

def append_to_csv(path: str, row_dict: Dict[str, Any]):
    """Append a new row to CSV (auto-create header)."""
    ensure_dir(Path(path).parent)
    df_new = pd.DataFrame([row_dict])
    if not Path(path).exists():
        df_new.to_csv(path, index=False)
    else:
        df_new.to_csv(path, index=False, mode="a", header=False)

# --------------------------------------------------
# ETA Helper
# --------------------------------------------------

def estimate_remaining_time(start_time: float, progress: float) -> str:
    """
    progress: number between 0 and 1.
    """
    elapsed = time.time() - start_time
    if progress <= 0:
        return "N/A"
    est_total = elapsed / progress
    remaining = est_total - elapsed
    return format_time(remaining)

# --------------------------------------------------
# Experiment tracking
# --------------------------------------------------

def generate_experiment_id(prefix: str = "exp") -> str:
    """Generate unique experiment ID with timestamp."""
    ts = timestamp().replace(":", "-")
    return f"{prefix}_{ts}"

def get_git_commit_hash() -> str:
    """Get current git commit hash for reproducibility."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]  # Short hash
    except Exception:
        pass
    return "unknown"

# --------------------------------------------------
# EpochLogger – structured per-epoch CSV logging
# --------------------------------------------------

class EpochLogger:
    """
    Writes one row per epoch to a CSV file for structured experiment tracking.

    Columns written:
        experiment_id, model, phase, epoch, train_loss, val_loss,
        train_acc, val_acc, lr, epoch_time_s, timestamp

    Usage::

        logger = EpochLogger(results_dir="Results_Q1/exp_xxx/convnext_v2",
                             experiment_id="exp_xxx", model_name="convnext_v2")
        logger.log(epoch=1, phase="warm", train_loss=0.9, val_loss=0.85,
                   train_acc=0.6, val_acc=0.62, lr=3e-5, epoch_time_s=45.2)
    """

    COLUMNS = [
        "experiment_id", "model", "phase", "epoch",
        "train_loss", "val_loss", "train_acc", "val_acc",
        "lr", "epoch_time_s", "timestamp"
    ]

    def __init__(self, results_dir: str, experiment_id: str = "", model_name: str = ""):
        self.results_dir = Path(results_dir)
        self.experiment_id = experiment_id
        self.model_name = model_name
        self.log_path = self.results_dir / "training_log.csv"
        ensure_dir(str(self.results_dir))

        # Write header if file does not exist yet
        if not self.log_path.exists():
            pd.DataFrame(columns=self.COLUMNS).to_csv(self.log_path, index=False)

    def log(
        self,
        epoch: int,
        phase: str,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        lr: float,
        epoch_time_s: float,
    ):
        """Append one row to the training log CSV."""
        row = {
            "experiment_id": self.experiment_id,
            "model": self.model_name,
            "phase": phase,
            "epoch": epoch,
            "train_loss": round(float(train_loss), 6),
            "val_loss": round(float(val_loss), 6),
            "train_acc": round(float(train_acc), 6),
            "val_acc": round(float(val_acc), 6),
            "lr": f"{lr:.2e}",
            "epoch_time_s": round(float(epoch_time_s), 2),
            "timestamp": timestamp(),
        }
        append_to_csv(str(self.log_path), row)

    def summary(self) -> pd.DataFrame:
        """Return the full training log as a DataFrame."""
        if self.log_path.exists():
            return pd.read_csv(self.log_path)
        return pd.DataFrame(columns=self.COLUMNS)


# --------------------------------------------------
# Reproducible split index persistence
# --------------------------------------------------

def save_split_indices(
    train_indices: List[int],
    val_indices: List[int],
    out_dir: str,
    filename: str = "split_indices.json",
    test_indices: Optional[List[int]] = None,
):
    """
    Persist the exact integer row indices of the train/val[/test] split to disk.
    """
    payload = {
        "train_indices": [int(i) for i in train_indices],
        "val_indices":   [int(i) for i in val_indices],
        "num_train":     len(train_indices),
        "num_val":       len(val_indices),
        "saved_at":      timestamp(),
    }
    if test_indices is not None:
        payload["test_indices"] = [int(i) for i in test_indices]
        payload["num_test"] = len(test_indices)
    save_json(payload, str(Path(out_dir) / filename))


def load_split_indices(
    path: str,
) -> tuple:
    """
    Load previously saved train/val split indices.

    Returns:
        (train_indices: List[int], val_indices: List[int])
    """
    data = load_json(path)
    return data["train_indices"], data["val_indices"]


# --------------------------------------------------
# YAML config auto-resolver
# --------------------------------------------------

def resolve_config_auto(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expand "auto" string values in a YAML-loaded config dict
    based on runtime device availability.

    Handles keys: device, num_workers, use_amp, channels_last,
                  pin_memory, prefetch_factor
    """
    is_cuda = torch.cuda.is_available()
    cfg = cfg.copy()

    if cfg.get("device") == "auto":
        cfg["device"] = "cuda" if is_cuda else "cpu"

    if cfg.get("num_workers") == "auto":
        if not is_cuda:
            cfg["num_workers"] = 0
        else:
            cfg["num_workers"] = min(8, os.cpu_count() or 4)

    if cfg.get("use_amp") == "auto":
        cfg["use_amp"] = is_cuda

    if cfg.get("channels_last") == "auto":
        cfg["channels_last"] = is_cuda

    if cfg.get("pin_memory") == "auto":
        cfg["pin_memory"] = is_cuda

    if cfg.get("prefetch_factor") == "auto":
        cfg["prefetch_factor"] = 4 if is_cuda else 1

    return cfg


# All keys accepted from YAML / CLI into the training pipeline
TRAIN_CONFIG_KEYS = frozenset({
    "seed", "deterministic", "val_split", "test_split", "save_split_indices",
    "model", "models", "num_classes", "input_size", "input_size_warm", "input_size_ft",
    "pretrained", "epochs_warm", "epochs_ft", "batch_size", "grad_accum_steps",
    "max_grad_norm", "lr", "weight_decay", "label_smoothing", "head_dropout",
    "early_stop", "use_mixup", "mixup_alpha", "mixup_prob", "use_cutmix",
    "cutmix_alpha", "cutmix_prob", "randaugment", "randaugment_num_ops",
    "randaugment_magnitude", "use_novel_aug", "fourier_aug_p", "contrastive_noise_p",
    "augmix_lite_p", "random_erasing_p", "balance_dataset", "balance_method",
    "min_samples_per_class", "max_samples_per_class", "use_weighted_sampler",
    "device", "num_workers", "use_amp", "channels_last", "use_torch_compile",
    "pin_memory", "prefetch_factor", "use_ema", "ema_decay", "use_focal_loss",
    "focal_gamma", "use_tta", "finetune_clean_epochs", "backbone_lr_ratio", "use_swa", "swa_epochs",
    "save_full_checkpoint", "log_csv", "results_dir", "prepare_datasets",
    "use_amp_lr_adapter", "warmup_steps", "warmup_start_factor",
    "run_benchmark_eval", "run_baselines", "baseline_models",
    "run_cross_dataset", "run_ablation", "ablation_seeds", "export_repro_bundle",
})


def merge_train_config(base: Dict[str, Any], yaml_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge a resolved YAML config into TRAIN_CONFIG, applying every known training key.
    Maps ``model`` → ``models`` list when ``models`` is absent.
    """
    merged = base.copy()
    for key, value in yaml_cfg.items():
        if key in TRAIN_CONFIG_KEYS:
            merged[key] = value
    if "model" in yaml_cfg and "models" not in merged:
        merged["models"] = [yaml_cfg["model"]]
    # Resolve single input_size into warm/ft when explicit sizes are omitted
    if "input_size" in merged:
        merged.setdefault("input_size_warm", merged["input_size"])
        merged.setdefault("input_size_ft", merged["input_size"])
    merged.setdefault("input_size_warm", merged.get("input_size", 224))
    merged.setdefault("input_size_ft", merged.get("input_size", 256))
    return merged
