# train_engine.py
"""
Training engine for Q1-ready Emotion Recognition framework.

Handles:
 - Unified training for FER2013 / RAF-DB / AffectNet
 - Warmup + Finetune strategy
 - Ablation toggles (MixUp / CutMix / RandAug / Combined / None)
 - Novel AMP scheduler (precision-aware LR adaptation)
 - Cross-dataset experiments
 - Logging, checkpoints, learning curves, ROC/PR plots
"""

import os
import sys
import time
import math
import json
import random
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
torch.backends.cudnn.benchmark = True
if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
    torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except AttributeError:
    pass

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)
from sklearn.preprocessing import label_binarize

from augmentations import (
    get_train_transforms, get_valid_transforms, get_mild_train_transforms,
    mixup_batch, cutmix_batch
)
from metrics_and_plots import (
    plot_learning_curves, save_confusion_matrix as plot_confusion_matrix,
    compute_roc_pr_curves, plot_roc_curves, plot_pr_curves,
    plot_gradcam_heatmap, generate_gradcam, plot_tsne_visualization,
    plot_umap_visualization, plot_ablation_study, plot_architecture_diagram,
    plot_cross_dataset_evaluation, plot_confusion_heatmap_unified,
    plot_cooccurrence_heatmap, plot_pr_roc_heatmaps, plot_gradcam_gallery,
    plot_activation_map_heatmap, plot_dataset_similarity_heatmap,
    plot_class_distribution_bars, plot_augmentation_examples_grid,
    plot_prior_work_chart, generate_class_landmark_heatmaps
)
from model_factory import (
    get_model, freeze_backbone, unfreeze_model, get_param_groups,
    normalize_model_name, SUPPORTED_MODEL
)
from utils import (
    ensure_dir, set_seed, format_time, generate_experiment_id,
    get_git_commit_hash, save_json, timestamp,
    save_full_ckpt, EpochLogger, save_split_indices,
)

# ---------------------------------------------------------
# Dataset Wrapper
# ---------------------------------------------------------
class EmotionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: str,
                 input_size=224, train=True,
                 aug_config=None):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.input_size = input_size
        self.train = train
        self.aug_config = aug_config or {}

        if train:
            self.transform = get_train_transforms(
                input_size=input_size,
                use_randaugment=self.aug_config.get("randaugment", True),
                use_novel_aug=self.aug_config.get("use_novel_aug", True),
                random_erasing_p=self.aug_config.get("random_erasing_p", 0.2),
                fourier_aug_p=self.aug_config.get("fourier_aug_p", 0.4),
                contrastive_noise_p=self.aug_config.get("contrastive_noise_p", 0.3),
                augmix_lite_p=self.aug_config.get("augmix_lite_p", 0.4),
                randaugment_num_ops=self.aug_config.get("randaugment_num_ops", 2),
                randaugment_magnitude=self.aug_config.get("randaugment_magnitude", 9)
            )
        else:
            self.transform = get_valid_transforms(input_size=input_size)

    def set_mild_augmentation(self):
        """Switch to light aug for final clean fine-tune epochs."""
        self.transform = get_mild_train_transforms(self.input_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root / row["image_path"]
        
        # Handle label conversion (should be int after encoding, but be safe)
        label_val = row["label"]
        if isinstance(label_val, str):
            # Try to convert string to int, skip if "unknown"
            if label_val == "unknown":
                raise ValueError("Found 'unknown' label in dataset - should have been filtered")
            try:
                label = int(label_val)
            except ValueError:
                raise ValueError(f"Invalid label value: {label_val}")
        else:
            label = int(label_val)

        try:
            # Optimized loading: open and convert in one step
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # Replace missing / corrupted files with black image
            img = Image.fromarray(np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8))

        return self.transform(img), label


# ---------------------------------------------------------
# Compute class weights / sampler
# ---------------------------------------------------------
def get_weighted_sampler(df: pd.DataFrame):
    class_counts = df["label"].value_counts().to_dict()
    weights = df["label"].map(lambda x: 1.0 / class_counts[x]).values
    sampler = WeightedRandomSampler(weights=weights,
                                    num_samples=len(weights),
                                    replacement=True)
    return sampler


def unwrap_compiled_model(model: nn.Module) -> nn.Module:
    """Return the underlying module when torch.compile wrapped the model."""
    return getattr(model, "_orig_mod", model)


class FocalLoss(nn.Module):
    """Focal loss for hard-example emphasis (helps in-the-wild AffectNet samples)."""

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            inputs, targets, reduction="none", label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def build_criterion(config: Dict):
    label_smoothing = config.get("label_smoothing", 0.1)
    if config.get("use_focal_loss", False):
        return FocalLoss(gamma=config.get("focal_gamma", 2.0), label_smoothing=label_smoothing)
    if label_smoothing > 0:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    return nn.CrossEntropyLoss()


def build_optimizer(model: nn.Module, base_lr: float, weight_decay: float, config: Dict):
    ratio = config.get("backbone_lr_ratio", 0.1)
    groups = get_param_groups(model, base_lr, backbone_lr_ratio=ratio)
    return torch.optim.AdamW(groups, weight_decay=weight_decay)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    if tensor.ndim == 4:
        tensor = tensor[0]
    img = tensor.detach().cpu().clone()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def _load_image_from_row(row: pd.Series, images_root: str, size: int) -> Optional[Image.Image]:
    img_path = Path(images_root) / str(row["image_path"])
    try:
        img = Image.open(img_path).convert("RGB")
        if size:
            img = img.resize((size, size))
        return img
    except Exception:
        return None


def generate_augmentation_examples(df_source: pd.DataFrame, images_root: str, input_size: int, aug_config: Dict, max_samples: int = 2) -> Dict[str, np.ndarray]:
    """
    Create visual samples demonstrating key augmentations.
    Returns dict {title: np.ndarray(H,W,3)}.
    """
    visuals: Dict[str, np.ndarray] = {}
    if df_source.empty:
        return visuals

    sample_count = min(max_samples, len(df_source))
    sample_indices = np.random.choice(len(df_source), size=sample_count, replace=False)
    rows = [df_source.iloc[int(idx)] for idx in sample_indices]

    primary_img = _load_image_from_row(rows[0], images_root, input_size)
    if primary_img is None:
        return visuals

    visuals["Original"] = np.array(primary_img)

    train_tf = get_train_transforms(
        input_size=input_size,
        use_randaugment=aug_config.get("randaugment", True),
        use_novel_aug=aug_config.get("use_novel_aug", True),
        random_erasing_p=0.0,  # keep visualization clean
        fourier_aug_p=aug_config.get("fourier_aug_p", 0.4),
        contrastive_noise_p=aug_config.get("contrastive_noise_p", 0.3),
        augmix_lite_p=aug_config.get("augmix_lite_p", 0.4),
        randaugment_num_ops=aug_config.get("randaugment_num_ops", 2),
        randaugment_magnitude=aug_config.get("randaugment_magnitude", 9)
    )

    rand_tensor = train_tf(primary_img.copy())
    visuals["RandAug + Novel"] = _tensor_to_uint8_image(rand_tensor)

    if len(rows) > 1:
        secondary_img = _load_image_from_row(rows[1], images_root, input_size)
        if secondary_img is None:
            secondary_img = primary_img.copy()
        second_tensor = train_tf(secondary_img.copy())

        if aug_config.get("use_mixup", True):
            mix_batch = torch.stack([rand_tensor.clone(), second_tensor.clone()], dim=0)
            labels = torch.tensor([rows[0]["label"], rows[1]["label"]], dtype=torch.long)
            mixed, _, _, _ = mixup_batch(mix_batch, labels, alpha=aug_config.get("mixup_alpha", 0.4))
            visuals["MixUp"] = _tensor_to_uint8_image(mixed[0])

        if aug_config.get("use_cutmix", False):
            cut_batch = torch.stack([rand_tensor.clone(), second_tensor.clone()], dim=0)
            labels = torch.tensor([rows[0]["label"], rows[1]["label"]], dtype=torch.long)
            cutmixed, _, _, _ = cutmix_batch(cut_batch, labels, alpha=aug_config.get("cutmix_alpha", 1.0))
            visuals["CutMix"] = _tensor_to_uint8_image(cutmixed[0])

    return visuals


# ---------------------------------------------------------
# Linear Warmup + Cosine Decay Scheduler
# ---------------------------------------------------------
class LinearWarmupCosineScheduler:
    """
    Two-phase LR scheduler:
      Phase 1 – Linear warmup: LR ramps from ``base_lr * start_factor``
                               to ``base_lr`` over ``warmup_steps`` optimizer steps.
      Phase 2 – Cosine decay:  LR decays from ``base_lr`` following a half-cosine
                               curve until training ends.

    Compatible with the existing step-per-batch update pattern in train_single_model.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
        start_factor: float = 0.01,
        eta_min_ratio: float = 0.01,
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(warmup_steps + 1, total_steps)
        self.start_factor = start_factor
        self.eta_min_ratio = eta_min_ratio
        self._step = 0

    def step(self):
        self._step += 1
        lr = self._get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _get_lr(self) -> float:
        s = self._step
        if s <= self.warmup_steps:
            # Linear ramp: start_factor * base_lr  →  base_lr
            alpha = self.start_factor + (1.0 - self.start_factor) * (s / self.warmup_steps)
            return self.base_lr * alpha
        # Cosine decay after warmup
        progress = (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        eta_min = self.base_lr * self.eta_min_ratio
        return eta_min + (self.base_lr - eta_min) * cosine_decay

    def get_last_lr(self) -> float:
        return self._get_lr()

    def state_dict(self) -> dict:
        return {"_step": self._step, "base_lr": self.base_lr,
                "warmup_steps": self.warmup_steps, "total_steps": self.total_steps,
                "start_factor": self.start_factor, "eta_min_ratio": self.eta_min_ratio}

    def load_state_dict(self, state: dict):
        self.__dict__.update(state)


class ExponentialMovingAverage:
    """Simple EMA tracker for model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9997):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        if not self.backup:
            return
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def maybe_compile_model(model: nn.Module, enable: bool):
    if enable and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as exc:
            print(f"Warning: torch.compile failed ({exc}). Continuing without compilation.")
    return model


def _unwrap(model: nn.Module) -> nn.Module:
    """Return the inner module, unwrapping nn.DataParallel if present.

    Use this whenever you need to access architecture-specific attributes
    (freeze/unfreeze, EMA shadow weights, Grad-CAM hooks, state_dict for
    checkpointing) because DataParallel proxies most but not all attribute
    access to model.module.
    """
    if isinstance(model, nn.DataParallel):
        return model.module
    return model




def build_loader(
    dataset: Dataset,
    batch_size: int,
    config: Dict,
    sampler: Optional[WeightedRandomSampler] = None,
    shuffle: bool = False
):
    # Optimize num_workers based on system and device
    is_cuda = torch.cuda.is_available()
    num_workers_default = 0 if sys.platform == "win32" else (min(8, os.cpu_count() or 4) if is_cuda else 2)
    num_workers = int(config.get("num_workers", num_workers_default))
    if sys.platform == "win32":
        num_workers = 0  # stay safe with Windows pickling
    # Disable pin_memory on CPU (not beneficial and uses extra memory)
    pin_memory = bool(config.get("pin_memory", is_cuda)) and is_cuda
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False
    }
    if sampler is not None:
        loader_kwargs["sampler"] = sampler
        loader_kwargs["shuffle"] = False
    else:
        loader_kwargs["shuffle"] = shuffle
    if num_workers > 0:
        # Reduce prefetch_factor on CPU to save memory
        default_prefetch = 1 if not is_cuda else 4
        loader_kwargs["prefetch_factor"] = int(config.get("prefetch_factor", default_prefetch))
        # Disable persistent_workers on CPU to save memory
        loader_kwargs["persistent_workers"] = is_cuda
    return DataLoader(dataset, **loader_kwargs)


# ---------------------------------------------------------
# Novel Mixed Precision Scheduler
# ---------------------------------------------------------
def adaptive_amp_lr(optimizer, base_lr, scaler, dynamic_scale=True):
    """
    Q1-style experimental feature:
    Adjust learning rate slightly depending on AMP stability / overflow.
    """
    # Only apply if scaler is available (CUDA training)
    if scaler is None:
        return
    
    scale = scaler.get_scale()

    # Overflow indicates unstable FP16 -> reduce LR slightly
    if dynamic_scale and scale < 1024:
        lr = base_lr * 0.75
    else:
        lr = base_lr

    for g in optimizer.param_groups:
        g["lr"] = lr


# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
def train_single_model(
    config: Dict,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    model_name: str,
    num_classes: int,
    device: torch.device
):
    """
    Training:
        warmup -> unfreeze -> finetune -> evaluate
    """

    # Set seed with deterministic mode for reproducibility (Q1 standards)
    deterministic_mode = config.get("deterministic", True)
    set_seed(config["seed"], deterministic=deterministic_mode)

    # -------------------------------------------------
    # Datasets and Loaders
    # -------------------------------------------------
    use_class_sampler = config.get("use_class_sampler", False)
    sampler = get_weighted_sampler(df_train) if use_class_sampler else None

    train_ds = EmotionDataset(
        df_train,
        config["images_root"],
        config["input_size_warm"],
        train=True,
        aug_config=config["augment"]
    )
    val_ds = EmotionDataset(
        df_val,
        config["images_root"],
        config["input_size_warm"],
        train=False
    )

    train_loader = build_loader(
        train_ds,
        batch_size=config["batch_size"],
        config=config,
        sampler=sampler,
        shuffle=sampler is None,
    )
    val_loader = build_loader(
        val_ds,
        batch_size=config["batch_size"],
        config=config,
        shuffle=False
    )

    # -------------------------------------------------
    # Model
    # -------------------------------------------------
    model = get_model(
        name=model_name,
        num_classes=num_classes,
        pretrained=True,
        input_size=config["input_size_warm"],
        device=device
    )

    # Add classifier head dropout if configured
    head_dropout = config.get("head_dropout", 0.0)
    if head_dropout > 0.0:
        if hasattr(model, "head") and hasattr(model.head, "fc"):
            in_features = model.head.fc.in_features
            model.head.fc = nn.Sequential(
                nn.Dropout(p=head_dropout),
                nn.Linear(in_features, num_classes)
            )
            print(f"Added head dropout layer with probability={head_dropout}")
            model.head.fc.to(device)

    freeze_backbone(model)

    channels_last_enabled = bool(config.get("channels_last", False) and torch.cuda.is_available() and device.type == "cuda")
    if channels_last_enabled:
        model.to(memory_format=torch.channels_last)

    model = maybe_compile_model(model, config.get("use_torch_compile", False))
    # ── Multi-GPU: DataParallel ─────────────────────────────────────────────
    # Auto-wraps with nn.DataParallel when ≥2 GPUs are available (e.g. Kaggle 2×T4).
    # Access the raw module via  (model.module if isinstance(model, nn.DataParallel) else model)
    # for EMA / Grad-CAM / checkpoint saving — all such sites are guarded below.
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_multi_gpu = (
        num_gpus > 1
        and config.get("multi_gpu", True)   # set multi_gpu: false in YAML to disable
        and device.type == "cuda"
    )
    if use_multi_gpu:
        model = nn.DataParallel(model)
        print(f"  \u2713 DataParallel enabled across {num_gpus} GPUs "
              f"({', '.join(f'cuda:{i}' for i in range(num_gpus))})")
        effective_bs = config["batch_size"] * num_gpus * config.get("grad_accum_steps", 1)
        print(f"  \u2713 Per-GPU batch size: {config['batch_size']}  "
              f"\u2192 effective total batch: {effective_bs}")
    # Disable EMA during warmup phase to prevent initial random head weights from dragging down early performance
    ema = None


    base_lr = config["lr_map"].get(model_name, config.get("lr", 3e-5))
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=base_lr,
        weight_decay=config["weight_decay"]
    )
    # Use CPU-appropriate scaler (disabled on CPU)
    if config["use_amp"] and torch.cuda.is_available():
        # Use new API if available, fallback to old API
        try:
            scaler = torch.amp.GradScaler('cuda', enabled=True)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None
    grad_accum = config.get("grad_accum_steps", 1)
    max_grad_norm = config.get("max_grad_norm", None)

    # Linear warmup steps: default = one full epoch of optimizer steps
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    _warmup_cfg = config.get("warmup_steps", 0)
    warmup_steps = int(_warmup_cfg) if _warmup_cfg and int(_warmup_cfg) > 0 else steps_per_epoch
    total_warm_steps = steps_per_epoch * config["epochs_warm"]
    warm_scheduler = LinearWarmupCosineScheduler(
        optimizer, base_lr=base_lr,
        warmup_steps=warmup_steps,
        total_steps=total_warm_steps,
        start_factor=config.get("warmup_start_factor", 0.01),
    )

    # Epoch logger (structured CSV)
    experiment_id = config.get("experiment_id", "")
    results_dir_for_log = Path(config.get("results_dir", "Results_Q1")) / experiment_id / model_name
    epoch_logger = EpochLogger(
        results_dir=str(results_dir_for_log),
        experiment_id=experiment_id,
        model_name=model_name,
    ) if config.get("log_csv", True) else None

    # -------------------------------------------------
    # Criterion
    # -------------------------------------------------
    criterion = build_criterion(config)

    # -------------------------------------------------
    # Training History
    # -------------------------------------------------
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": []
    }

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    # -------------------------------------------------
    # Warmup Phase
    # -------------------------------------------------
    for epoch in range(1, config["epochs_warm"] + 1):
        t0 = time.time()
        model.train()

        running_loss = 0.0
        preds, targets = [], []
        optimizer.zero_grad(set_to_none=True)
        step_count = 0

        for step, (xb, yb) in enumerate(tqdm(train_loader, desc=f"{model_name} warm E{epoch}", leave=False), 1):
            step_count = step
            xb = xb.to(device, non_blocking=True)
            if channels_last_enabled:
                xb = xb.to(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=True)

            # -------------------------------------------------
            # Ablation: MixUp / CutMix (with standard 50% probability check)
            # -------------------------------------------------
            mixup_prob = config["augment"].get("mixup_prob", 0.5)
            cutmix_prob = config["augment"].get("cutmix_prob", 0.5)
            
            if config["augment"]["use_mixup"] and random.random() < mixup_prob:
                xb, y_a, y_b, lam = mixup_batch(xb, yb, alpha=config["augment"]["mixup_alpha"])
            elif config["augment"]["use_cutmix"] and random.random() < cutmix_prob:
                xb, y_a, y_b, lam = cutmix_batch(xb, yb, alpha=config["augment"]["cutmix_alpha"])
            else:
                y_a = y_b = None
                lam = None
            
            # Identify dominant target class for accurate training accuracy logs
            if lam is not None:
                effective_targets = y_a if lam >= 0.5 else y_b
            else:
                effective_targets = yb

            # Use CPU-appropriate autocast (disabled on CPU)
            if config["use_amp"] and torch.cuda.is_available():
                # Use new API if available, fallback to old API
                try:
                    autocast_context = torch.amp.autocast('cuda', enabled=True)
                except (AttributeError, TypeError):
                    autocast_context = torch.cuda.amp.autocast(enabled=True)
                with autocast_context:
                    out = model(xb)
                    if lam is not None:
                        base_loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
                    else:
                        base_loss = criterion(out, yb)
            else:
                out = model(xb)
                if lam is not None:
                    base_loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
                else:
                    base_loss = criterion(out, yb)

            loss = base_loss / grad_accum
            
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % grad_accum == 0:
                if scaler is not None:
                    if max_grad_norm:
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if max_grad_norm:
                        clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema:
                    ema.update(model)
                if warm_scheduler:
                    warm_scheduler.step()
                if config.get("use_amp_lr_adapter", False):
                    adaptive_amp_lr(optimizer, base_lr, scaler)
                history["lr"].append(optimizer.param_groups[0]["lr"])

            running_loss += base_loss.item() * xb.size(0)
            # Move predictions to CPU immediately to free memory
            preds.extend(out.argmax(1).detach().cpu().numpy())
            targets.extend(effective_targets.detach().cpu().numpy() if effective_targets.requires_grad else effective_targets.cpu().numpy())
            
            # Periodic memory cleanup (runs on both CPU and GPU to prevent Kaggle RAM OOM)
            if step % 100 == 0:
                gc.collect()

        if step_count % grad_accum != 0:
            if scaler is not None:
                if max_grad_norm:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if max_grad_norm:
                    clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema:
                ema.update(model)
            if warm_scheduler:
                warm_scheduler.step()
            if config.get("use_amp_lr_adapter", False):
                adaptive_amp_lr(optimizer, base_lr, scaler)
            history["lr"].append(optimizer.param_groups[0]["lr"])

        train_loss = running_loss / len(train_ds)
        train_acc = accuracy_score(targets, preds)
        
        # Clear prediction lists to free memory
        del preds, targets
        gc.collect()

        if ema:
            # Apply EMA shadow weights for validation evaluation
            ema.apply_shadow(model)
            val_acc, val_loss = evaluate_model(model, val_loader, criterion, device, channels_last=channels_last_enabled)
            ema.restore(model)
        else:
            val_acc, val_loss = evaluate_model(model, val_loader, criterion, device, channels_last=channels_last_enabled)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Warm Epoch {epoch} — TrainAcc={train_acc:.3f}, ValAcc={val_acc:.3f}, "
              f"LR={current_lr:.2e}, Time={format_time(epoch_time)}")
        if epoch_logger:
            epoch_logger.log(
                epoch=epoch, phase="warm",
                train_loss=train_loss, val_loss=val_loss,
                train_acc=train_acc, val_acc=val_acc,
                lr=current_lr, epoch_time_s=epoch_time,
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if ema:
                best_state = {}
                state_dict = _unwrap(model).state_dict()
                for k, v in state_dict.items():
                    if k in ema.shadow:
                        best_state[k] = ema.shadow[k].cpu().clone()
                    else:
                        best_state[k] = v.cpu().clone()
            else:
                best_state = {k: v.cpu() for k, v in _unwrap(model).state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["early_stop"]:
                print("Early stopping warm phase.")
                break

    # Load best warm state (skip if warmup was disabled)
    if best_state:
        _unwrap(model).load_state_dict(best_state)
    elif config["epochs_warm"] == 0:
        print("  Skipped warmup phase (epochs_warm=0); proceeding to fine-tune from init weights.")

    # -------------------------------------------------
    # Finetune Phase (Higher Resolution) — unfreeze after warmup
    # -------------------------------------------------
    unfreeze_model(_unwrap(model))
    ft_input = config["input_size_ft"]
    train_ds = EmotionDataset(
        df_train, config["images_root"], ft_input,
        train=True, aug_config=config["augment"]
    )
    val_ds = EmotionDataset(
        df_val, config["images_root"], ft_input,
        train=False
    )
    sampler = get_weighted_sampler(df_train) if use_class_sampler else None
    train_loader = build_loader(
        train_ds,
        batch_size=config["batch_size"],
        config=config,
        sampler=sampler,
        shuffle=sampler is None,
    )
    val_loader = build_loader(
        val_ds,
        batch_size=config["batch_size"],
        config=config,
        shuffle=False
    )

    ft_lr_ratio = config.get("ft_lr_ratio", 0.3)
    ft_base_lr = base_lr * ft_lr_ratio
    backbone_lr_ratio = config.get("backbone_lr_ratio", 0.1)
    optimizer = build_optimizer(
        _unwrap(model),
        ft_base_lr,
        config["weight_decay"],
        config,
    )
    print(
        f"Fine-tune phase: backbone LR={ft_base_lr * backbone_lr_ratio:.2e}, "
        f"head LR={ft_base_lr:.2e}"
    )

    # Use CPU-appropriate scaler (disabled on CPU)
    if config["use_amp"] and torch.cuda.is_available():
        # Use new API if available, fallback to old API
        try:
            scaler = torch.amp.GradScaler('cuda', enabled=True)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    # Initialize Exponential Moving Average (EMA) for fine-tuning phase
    if config.get("use_ema", True):
        decay = config.get("ema_decay", 0.9997)
        ema = ExponentialMovingAverage(_unwrap(model), decay=decay)
        print(f"Initialized Exponential Moving Average (EMA) with decay={decay}")
    else:
        ema = None

    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_ft_steps = steps_per_epoch * config["epochs_ft"]
    ft_scheduler = LinearWarmupCosineScheduler(
        optimizer, base_lr=ft_base_lr,
        warmup_steps=1,            # No ramp for FT (model already warm)
        total_steps=total_ft_steps,
        start_factor=1.0,
    )

    best_ft_acc = best_val_acc
    best_ft_state = best_state
    patience_counter = 0
    finetune_clean_epochs = int(config.get("finetune_clean_epochs", 5))
    mild_aug_active = False

    for epoch in range(1, config["epochs_ft"] + 1):
        t0 = time.time()
        model.train()

        in_clean_phase = epoch > (config["epochs_ft"] - finetune_clean_epochs)
        if in_clean_phase and not mild_aug_active:
            train_ds.set_mild_augmentation()
            mild_aug_active = True
            print(f"  → Clean fine-tune phase: mild aug only (epochs {epoch}–{config['epochs_ft']})")

        # Turn off MixUp / CutMix in the clean phase
        if in_clean_phase:
            active_use_mixup = False
            active_use_cutmix = False
        else:
            active_use_mixup = config["augment"]["use_mixup"]
            active_use_cutmix = config["augment"]["use_cutmix"]

        running_loss, preds, targets = 0.0, [], []
        optimizer.zero_grad(set_to_none=True)
        step_count = 0

        for step, (xb, yb) in enumerate(tqdm(train_loader, desc=f"{model_name} FT E{epoch}", leave=False), 1):
            step_count = step
            xb = xb.to(device, non_blocking=True)
            if channels_last_enabled:
                xb = xb.to(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=True)

            # -------------------------------------------------
            # Ablation: MixUp / CutMix (with standard 50% probability check)
            # -------------------------------------------------
            mixup_prob = config["augment"].get("mixup_prob", 0.5)
            cutmix_prob = config["augment"].get("cutmix_prob", 0.5)
            
            if active_use_mixup and random.random() < mixup_prob:
                xb, y_a, y_b, lam = mixup_batch(xb, yb, alpha=config["augment"]["mixup_alpha"])
            elif active_use_cutmix and random.random() < cutmix_prob:
                xb, y_a, y_b, lam = cutmix_batch(xb, yb, alpha=config["augment"]["cutmix_alpha"])
            else:
                y_a = y_b = None
                lam = None
            
            # Identify dominant target class for accurate training accuracy logs
            if lam is not None:
                effective_targets = y_a if lam >= 0.5 else y_b
            else:
                effective_targets = yb

            # Use CPU-appropriate autocast (disabled on CPU)
            if config["use_amp"] and torch.cuda.is_available():
                # Use new API if available, fallback to old API
                try:
                    autocast_context = torch.amp.autocast('cuda', enabled=True)
                except (AttributeError, TypeError):
                    autocast_context = torch.cuda.amp.autocast(enabled=True)
                with autocast_context:
                    out = model(xb)
                    if lam is not None:
                        base_loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
                    else:
                        base_loss = criterion(out, yb)
            else:
                out = model(xb)
                if lam is not None:
                    base_loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
                else:
                    base_loss = criterion(out, yb)

            loss = base_loss / grad_accum
            
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % grad_accum == 0:
                if scaler is not None:
                    if max_grad_norm:
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if max_grad_norm:
                        clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema:
                    ema.update(model)
                if ft_scheduler:
                    ft_scheduler.step()
                if config.get("use_amp_lr_adapter", False):
                    adaptive_amp_lr(optimizer, base_lr, scaler)
                history["lr"].append(optimizer.param_groups[0]["lr"])

            running_loss += base_loss.item() * xb.size(0)
            # Move predictions to CPU immediately to free memory
            preds.extend(out.argmax(1).detach().cpu().numpy())
            targets.extend(effective_targets.detach().cpu().numpy() if effective_targets.requires_grad else effective_targets.cpu().numpy())
            
            # Periodic memory cleanup (runs on both CPU and GPU to prevent Kaggle RAM OOM)
            if step % 100 == 0:
                gc.collect()

        if step_count % grad_accum != 0:
            if scaler is not None:
                if max_grad_norm:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if max_grad_norm:
                    clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema:
                ema.update(model)
            if ft_scheduler:
                ft_scheduler.step()
            if config.get("use_amp_lr_adapter", False):
                adaptive_amp_lr(optimizer, base_lr, scaler)
            history["lr"].append(optimizer.param_groups[0]["lr"])

        train_loss = running_loss / len(train_ds)
        train_acc = accuracy_score(targets, preds)
        
        # Clear prediction lists to free memory
        del preds, targets
        gc.collect()

        if ema:
            # Apply EMA shadow weights for validation evaluation
            ema.apply_shadow(model)
            val_acc, val_loss = evaluate_model(model, val_loader, criterion, device, channels_last=channels_last_enabled)
            ema.restore(model)
        else:
            val_acc, val_loss = evaluate_model(model, val_loader, criterion, device, channels_last=channels_last_enabled)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        ft_epoch_time = time.time() - t0
        ft_current_lr = optimizer.param_groups[0]["lr"]
        print(f"FT Epoch {epoch} — TrainAcc={train_acc:.3f}, ValAcc={val_acc:.3f}, "
              f"LR={ft_current_lr:.2e}, Time={format_time(ft_epoch_time)}")
        if epoch_logger:
            epoch_logger.log(
                epoch=config["epochs_warm"] + epoch, phase="finetune",
                train_loss=train_loss, val_loss=val_loss,
                train_acc=train_acc, val_acc=val_acc,
                lr=ft_current_lr, epoch_time_s=ft_epoch_time,
            )

        if val_acc > best_ft_acc:
            best_ft_acc = val_acc
            if ema:
                best_ft_state = {}
                state_dict = model.state_dict()
                for k, v in state_dict.items():
                    if k in ema.shadow:
                        best_ft_state[k] = ema.shadow[k].cpu().clone()
                    else:
                        best_ft_state[k] = v.cpu().clone()
            else:
                best_ft_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["early_stop"]:
                print("Early stopping finetune phase.")
                break

    if best_ft_state:
        model.load_state_dict(best_ft_state)

    use_tta = config.get("use_tta", True)

    # Final Evaluation with per-dataset metrics
    results = evaluate_final(
        model, val_loader, df_val, device, model_name,
        channels_last=channels_last_enabled, use_tta=use_tta,
    )
    results["history"] = history
    results["best_val_acc"] = best_ft_acc

    return model, results


# ---------------------------------------------------------
# Evaluation Utilities
# ---------------------------------------------------------
def tta_forward(model, xb: torch.Tensor, channels_last: bool = False) -> torch.Tensor:
    """Multi-view TTA: original + horizontal flip (+ optional mild brightness)."""
    views = [xb, torch.flip(xb, dims=[3])]
    logits = []
    for view in views:
        if channels_last:
            view = view.to(memory_format=torch.channels_last)
        logits.append(model(view))
    return torch.stack(logits, dim=0).mean(dim=0)


def evaluate_model(model, loader, criterion, device, channels_last: bool = False, use_tta: bool = False):
    model.eval()
    preds = []
    targets = []
    losses = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            if channels_last:
                xb = xb.to(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=True)
            if use_tta:
                out = tta_forward(model, xb, channels_last=channels_last)
            else:
                out = model(xb)
            loss = criterion(out, yb)
            losses.append(loss.item() * xb.size(0))
            preds.extend(out.argmax(1).cpu().numpy())
            targets.extend(yb.cpu().numpy())

    acc = accuracy_score(targets, preds)
    loss = np.sum(losses) / len(preds)
    return acc, loss


def evaluate_final(model, loader, df_val, device, model_name, channels_last: bool = False, use_tta: bool = False):
    """Comprehensive evaluation with per-class and per-dataset metrics."""
    model.eval()

    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            if channels_last:
                xb = xb.to(memory_format=torch.channels_last)
            if use_tta:
                out = tta_forward(model, xb, channels_last=channels_last)
            else:
                out = model(xb)
            probs = F.softmax(out, dim=1).cpu().numpy()

            all_probs.append(probs)
            all_preds.extend(probs.argmax(1).tolist())
            all_labels.extend(yb.tolist())

    all_probs = np.vstack(all_probs)
    labels = np.array(all_labels)
    preds = np.array(all_preds)

    # Overall metrics
    final_acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weight = f1_score(labels, preds, average="weighted")
    precision_macro = precision_score(labels, preds, average="macro", zero_division=0)
    recall_macro = recall_score(labels, preds, average="macro", zero_division=0)

    # Per-class metrics
    num_classes = len(np.unique(labels))
    per_class_metrics = {}
    for cls in range(num_classes):
        cls_mask = labels == cls
        if cls_mask.sum() > 0:
            cls_acc = accuracy_score(labels[cls_mask], preds[cls_mask])
            cls_precision = precision_score(labels == cls, preds == cls, zero_division=0)
            cls_recall = recall_score(labels == cls, preds == cls, zero_division=0)
            cls_f1 = f1_score(labels == cls, preds == cls, zero_division=0)
            per_class_metrics[int(cls)] = {
                "accuracy": float(cls_acc),
                "precision": float(cls_precision),
                "recall": float(cls_recall),
                "f1": float(cls_f1),
                "support": int(cls_mask.sum())
            }

    cm = confusion_matrix(labels, preds)

    # Per-dataset evaluation if dataset column exists
    per_dataset_metrics = None
    if "dataset" in df_val.columns:
        dataset_series = df_val["dataset"].reset_index(drop=True)
        if len(dataset_series) == len(labels):
            per_dataset_metrics = {}
            for dataset in dataset_series.unique():
                mask = dataset_series == dataset
                dataset_labels = labels[mask.values]
                dataset_preds = preds[mask.values]
                if len(dataset_labels) == 0:
                    continue
                dataset_acc = accuracy_score(dataset_labels, dataset_preds)
                dataset_f1 = f1_score(dataset_labels, dataset_preds, average="macro", zero_division=0)
                per_dataset_metrics[str(dataset)] = {
                    "accuracy": float(dataset_acc),
                    "f1_macro": float(dataset_f1),
                    "support": int(mask.sum())
                }

    return {
        "labels": labels,
        "preds": preds,
        "probs": all_probs,
        "acc": final_acc,
        "f1_macro": f1_macro,
        "f1_weight": f1_weight,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "per_class_metrics": per_class_metrics,
        "per_dataset_metrics": per_dataset_metrics,
        "cm": cm,
        "model_name": model_name
    }


# ---------------------------------------------------------
# Training Pipeline Wrapper
# ---------------------------------------------------------
def run_training_pipeline(
    model_name: str,
    csv_path: str,
    images_root: str,
    num_classes: int,
    config: Dict
):
    """
    Complete training pipeline:
    1. Load and split dataset
    2. Configure training parameters
    3. Train model
    4. Save results and plots
    """
    from sklearn.model_selection import train_test_split
    from pathlib import Path
    import json
    model_name = normalize_model_name(model_name)
    
    # Import dataset balancer
    try:
        from dataset_balancer import analyze_and_balance_dataset, EMOTION_MAP
        use_balancer = config.get("balance_dataset", True)
    except ImportError:
        use_balancer = False
        print("Warning: dataset_balancer not available, using basic label encoding")
    
    # Use dataset balancer if available and enabled
    if use_balancer:
        print("\n=== Balancing and mapping dataset labels ===")
        # Create temporary balanced CSV
        temp_csv = str(Path(csv_path).parent / "temp_balanced.csv")
        
        df_balanced = analyze_and_balance_dataset(
            csv_path=csv_path,
            output_path=temp_csv,
            balance_method=config.get("balance_method", "oversample"),
            min_samples=config.get("min_samples_per_class", 500),
            max_samples=config.get("max_samples_per_class", 5000),
            filter_unknown=True
        )
        # Update csv_path to use balanced dataset
        csv_path = temp_csv
        class_names = [EMOTION_MAP[i] for i in range(7)]
        print(f"Using balanced dataset with {len(df_balanced)} samples and 7 emotion classes")
        
        # Load the balanced dataset
        df = pd.read_csv(csv_path)
    else:
        # Load dataset
        df = pd.read_csv(csv_path)
    
    # Filter missing images
    from utils import filter_missing_images
    df = filter_missing_images(df, images_root)
    
    # Define min_samples_per_class for stratified split
    min_samples_per_class = 2  # Required for stratified split
    
    if not use_balancer:
        # Filter out "unknown" labels (can't be used for classification)
        unknown_count = len(df[df["label"] == "unknown"])
        if unknown_count > 0:
            print(f"Filtering out {unknown_count} samples with 'unknown' labels")
            df = df[df["label"] != "unknown"].copy()
        
        # Map labels to standard 7 emotion classes if not already done
        # Check if labels are already 0-6
        unique_labels = sorted(df["label"].unique())
        try:
            label_ints = [int(float(l)) for l in unique_labels]
            if max(label_ints) > 6 or min(label_ints) < 0:
                print("Warning: Labels are not in standard 0-6 range, attempting to map...")
                from dataset_balancer import map_labels_to_emotions
                for dataset_name in df["dataset"].unique():
                    df = map_labels_to_emotions(df, dataset_name)
                # Filter unknown again after mapping
                df = df[df["label"] != "unknown"].copy()
                df["label"] = df["label"].apply(lambda x: int(float(x)) if x != "unknown" else None)
                df = df[df["label"].notna()].copy()
                df["label"] = df["label"].astype(int)
                df = df[df["label"].isin(range(7))].copy()
        except (ValueError, TypeError):
            pass
        
        # Check class distribution and filter classes with < 2 samples (before encoding)
        label_counts = df["label"].value_counts()
        
        # Find classes with insufficient samples
        insufficient_classes = label_counts[label_counts < min_samples_per_class].index.tolist()
        
        if insufficient_classes:
            print(f"Warning: Found {len(insufficient_classes)} class(es) with < {min_samples_per_class} samples: {insufficient_classes}")
            print(f"  Removing {len(df[df['label'].isin(insufficient_classes)])} samples from these classes")
            df = df[~df["label"].isin(insufficient_classes)].copy()
            print(f"  Remaining samples: {len(df)}")
        
        if len(df) == 0:
            raise ValueError("No samples remaining after filtering classes with insufficient samples!")
        
        # Encode labels to integers using LabelEncoder (after all filtering)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df["label"] = le.fit_transform(df["label"].astype(str))
        class_names = [str(c) for c in le.classes_.tolist()]
        df["label_name"] = le.inverse_transform(df["label"])
        
        # If we have more than 7 classes, try to map to standard emotions
        if len(class_names) > 7:
            print(f"Warning: Found {len(class_names)} classes, expected 7. Attempting to consolidate...")
            # Try to map numeric labels to 0-6
            try:
                df["label"] = df["label"].apply(lambda x: int(float(x)) if str(x).isdigit() else x)
                df = df[df["label"].isin(range(7))].copy()
                if len(df) > 0:
                    le = LabelEncoder()
                    df["label"] = le.fit_transform(df["label"].astype(str))
                    class_names = [str(c) for c in le.classes_.tolist()]
                    df["label_name"] = le.inverse_transform(df["label"])
                    print(f"Consolidated to {len(class_names)} classes")
            except:
                pass
    else:
        # When using balancer, labels should already be 0-6 integers
        # Just ensure they're integers
        df["label"] = df["label"].astype(int)
        # Create label_name column for consistency
        if "label_name" not in df.columns:
            df["label_name"] = df["label"].apply(lambda x: EMOTION_MAP.get(x, f"class_{x}"))
    
    # Check if we can use stratified split (all remaining classes should have >= 2 samples)
    label_counts_after = df["label"].value_counts()
    can_stratify = (label_counts_after >= min_samples_per_class).all()

    # Use seed from config for reproducible splits across multi-seed experiments
    split_seed = config.get("seed", 42)
    val_split  = config.get("val_split", 0.2)
    test_split = config.get("test_split", 0.0)

    # Split train/val/test
    if test_split > 0.0:
        # First split into train_val and test
        if can_stratify:
            df_train_val, df_test = train_test_split(
                df, test_size=test_split, random_state=split_seed, stratify=df["label"]
            )
            # Re-evaluate stratify check for sub-split
            label_counts_train_val = df_train_val["label"].value_counts()
            can_stratify_sub = (label_counts_train_val >= min_samples_per_class).all()
            
            val_sub_split = val_split / (1.0 - test_split)
            # Ensure within valid range
            val_sub_split = min(max(val_sub_split, 0.01), 0.99)
            if can_stratify_sub:
                df_train, df_val = train_test_split(
                    df_train_val, test_size=val_sub_split, random_state=split_seed, stratify=df_train_val["label"]
                )
            else:
                df_train, df_val = train_test_split(
                    df_train_val, test_size=val_sub_split, random_state=split_seed
                )
        else:
            df_train_val, df_test = train_test_split(
                df, test_size=test_split, random_state=split_seed
            )
            val_sub_split = val_split / (1.0 - test_split)
            val_sub_split = min(max(val_sub_split, 0.01), 0.99)
            df_train, df_val = train_test_split(
                df_train_val, test_size=val_sub_split, random_state=split_seed
            )
        df_test = df_test.reset_index(drop=True)
        print(f"3-way split: Train samples: {len(df_train)}, Val samples: {len(df_val)}, Test samples: {len(df_test)}")
    else:
        df_test = None
        # Split train/val
        if can_stratify:
            df_train, df_val = train_test_split(
                df, test_size=val_split, random_state=split_seed, stratify=df["label"]
            )
        else:
            print("Warning: Cannot use stratified split, using random split instead")
            df_train, df_val = train_test_split(
                df, test_size=val_split, random_state=split_seed
            )
        print(f"2-way split: Train samples: {len(df_train)}, Val samples: {len(df_val)}")
    
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    
    # Update num_classes based on actual unique labels
    actual_num_classes = len(df["label"].unique())
    if actual_num_classes != num_classes:
        print(f"Warning: Expected {num_classes} classes, but found {actual_num_classes} after filtering")
        num_classes = actual_num_classes
    
    print(f"Train samples: {len(df_train)}, Val samples: {len(df_val)}")
    print(f"Number of classes: {num_classes}")
    print(f"Class IDs in training set: {sorted(df_train['label'].unique())}")
    
    # Prepare full config
    # Resolve device: respect the config, but always fall back to CPU if CUDA
    # is requested but not actually available (e.g. CPU-only Colab runtime).
    _requested_device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(_requested_device).startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARNING] Config requested device='{_requested_device}' but CUDA is not available. Falling back to CPU.")
        _requested_device = "cpu"
    device = torch.device(_requested_device)
    
    # Learning rate map - optimized for 93%+ accuracy
    # Use the already-normalized model_name as key so it matches the lookup in train_single_model
    _base_lr = config.get("lr", 3e-5)
    lr_map = {
        model_name: _base_lr,          # normalized name, e.g. "convnext_v2_base"
        SUPPORTED_MODEL: _base_lr,     # alias fallback, e.g. "convnext_v2"
    }
    
    full_config = {
        "batch_size": config.get("batch_size", 128),
        "num_workers": config.get("num_workers", min(8, os.cpu_count() or 4)),
        "input_size_warm": config.get("input_size_warm", config.get("input_size", 224)),
        "input_size_ft": config.get("input_size_ft", config.get("input_size", 256)),
        "epochs_warm": config.get("epochs_warm", 3),
        "epochs_ft": config.get("epochs_ft", 20),
        "images_root": images_root,
        "seed": config.get("seed", 42),
        "lr_map": lr_map,
        "lr": _base_lr,
        "weight_decay": config.get("weight_decay", 1e-4),
        "label_smoothing": config.get("label_smoothing", 0.1),
        "head_dropout": config.get("head_dropout", 0.0),
        "use_focal_loss": config.get("use_focal_loss", False),
        "focal_gamma": config.get("focal_gamma", 2.0),
        "backbone_lr_ratio": config.get("backbone_lr_ratio", 0.1),
        "ft_lr_ratio": config.get("ft_lr_ratio", 0.3),
        "use_class_sampler": config.get("use_class_sampler", False),
        "use_tta": config.get("use_tta", True),
        "finetune_clean_epochs": config.get("finetune_clean_epochs", 5),
        "use_amp": config.get("use_amp", True),
        "early_stop": config.get("early_stop", 5),
        "augment": {
            "use_mixup": config.get("use_mixup", True),
            "mixup_alpha": config.get("mixup_alpha", 0.4),
            "mixup_prob": config.get("mixup_prob", 0.5),
            "use_cutmix": config.get("use_cutmix", False),
            "cutmix_alpha": config.get("cutmix_alpha", 1.0),
            "cutmix_prob": config.get("cutmix_prob", 0.5),
            "randaugment": config.get("randaugment", True),
            "randaugment_num_ops": config.get("randaugment_num_ops", 2),
            "randaugment_magnitude": config.get("randaugment_magnitude", 9),
            "use_novel_aug": config.get("use_novel_aug", True),
            "fourier_aug_p": config.get("fourier_aug_p", 0.4),
            "contrastive_noise_p": config.get("contrastive_noise_p", 0.3),
            "augmix_lite_p": config.get("augmix_lite_p", 0.4),
            "random_erasing_p": config.get("random_erasing_p", 0.2),
        },
        "deterministic": config.get("deterministic", False),
        "channels_last": config.get("channels_last", True),
        "use_torch_compile": config.get("use_torch_compile", False),
        "use_ema": config.get("use_ema", True),
        "ema_decay": config.get("ema_decay", 0.9997),
        "grad_accum_steps": max(1, int(config.get("grad_accum_steps", 1))),
        "max_grad_norm": config.get("max_grad_norm", 5.0),
        "prefetch_factor": config.get("prefetch_factor", 2 if not torch.cuda.is_available() else 4),
        "pin_memory": config.get("pin_memory", torch.cuda.is_available()),
        "use_amp_lr_adapter": config.get("use_amp_lr_adapter", False),
        "warmup_steps": config.get("warmup_steps", 0),
        "warmup_start_factor": config.get("warmup_start_factor", 0.01),
        "log_csv": config.get("log_csv", True),
        "save_full_checkpoint": config.get("save_full_checkpoint", True),
        "save_split_indices": config.get("save_split_indices", True),
        "experiment_id": config.get("experiment_id", generate_experiment_id()),
        "results_dir": config.get("results_dir", "Results_Q1"),
    }
    channels_last_enabled = bool(
        full_config.get("channels_last", False)
        and torch.cuda.is_available()
        and device.type == "cuda"
    )
    
    # Train model
    model, results = train_single_model(
        config=full_config,
        df_train=df_train,
        df_val=df_val,
        model_name=model_name,
        num_classes=num_classes,
        device=device
    )
    
    # Save results with experiment ID
    experiment_id = config.get("experiment_id", generate_experiment_id())
    results_dir = Path(config.get("results_dir", "Results_Q1")) / experiment_id / model_name
    ensure_dir(str(results_dir))
    figures_dir = results_dir / "figures"
    ensure_dir(str(figures_dir))
    
    # Save experiment configuration for reproducibility
    exp_config = {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "git_commit": get_git_commit_hash(),
        "timestamp": timestamp(),
        "config": {
            **config,
            "csv_path": csv_path,
            "images_root": images_root,
            "num_classes": num_classes,
            "seed": full_config["seed"],
            "deterministic": full_config.get("deterministic", True)
        }
    }
    save_json(exp_config, str(results_dir / "config.json"))
    
    # Save comprehensive metrics
    metrics = {
        "accuracy": float(results["acc"]),
        "f1_macro": float(results["f1_macro"]),
        "f1_weighted": float(results["f1_weight"]),
        "precision_macro": float(results.get("precision_macro", 0.0)),
        "recall_macro": float(results.get("recall_macro", 0.0)),
        "best_val_acc": float(results["best_val_acc"]),
        "per_class_metrics": results.get("per_class_metrics", {})
    }
    save_json(metrics, str(results_dir / "metrics.json"))
    
    # Save per-dataset metrics if available
    if results.get("per_dataset_metrics"):
        save_json(results["per_dataset_metrics"], str(results_dir / "metrics_per_dataset.json"))
    
    # Get class names
    classes = class_names
    
    # Get final epoch number
    final_epoch = len(results["history"]["train_loss"])
    
    # ================================================================
    # Q1 CONFERENCE FIGURES - Generate all requested visualizations
    # ================================================================
    
    print(f"\n>>> Generating Q1 Conference Figures (Epoch {final_epoch})...")
    
    # 1. Confusion Matrix (Enhanced)
    plot_confusion_matrix(
        results["labels"], results["preds"], classes, results_dir, 
        filename=f"{model_name}_confusion", normalize=False,
        epoch=final_epoch, model_name=model_name
    )
    plot_confusion_matrix(
        results["labels"], results["preds"], classes, results_dir, 
        filename=f"{model_name}_confusion_norm", normalize=True,
        epoch=final_epoch, model_name=model_name
    )
    if "cm" in results:
        plot_confusion_heatmap_unified(
            results["cm"], classes,
            figures_dir / f"{model_name}_confusion_heatmap_unified.png",
            title="Confusion Matrix Heatmap (Unified Dataset)"
        )
        plot_cooccurrence_heatmap(
            results["cm"], classes,
            figures_dir / f"{model_name}_cooccurrence_heatmap.png"
        )
    
    # 2. Accuracy & Loss Curves (Enhanced with epoch naming)
    plot_learning_curves(results["history"], results_dir, model_name, epoch=final_epoch)
    
    # 3. ROC/PR curves
    roc_data, pr_data = compute_roc_pr_curves(
        results["labels"], results["probs"], classes
    )
    plot_roc_curves(roc_data, classes, results_dir, filename=f"{model_name}_epoch{final_epoch:03d}_roc")
    plot_pr_curves(pr_data, classes, results_dir, filename=f"{model_name}_epoch{final_epoch:03d}_pr")
    plot_pr_roc_heatmaps(roc_data, pr_data, classes, figures_dir)
    
    plot_class_distribution_bars(
        df, classes,
        figures_dir / f"{model_name}_class_distribution.png"
    )
    plot_dataset_similarity_heatmap(
        df,
        classes,
        figures_dir / f"{model_name}_dataset_similarity.png"
    )
    
    aug_visuals = generate_augmentation_examples(
        df_train.reset_index(drop=True),
        images_root,
        full_config["input_size_warm"],
        full_config["augment"]
    )
    if aug_visuals:
        plot_augmentation_examples_grid(
            aug_visuals,
            figures_dir / f"{model_name}_augmentation_examples.png"
        )
    
    # 4. Architecture Diagram
    try:
        plot_architecture_diagram(model, results_dir / f"{model_name}_architecture.png", model_name)
        print(f"  ✓ Architecture diagram saved")
    except Exception as e:
        print(f"  ⚠ Could not generate architecture diagram: {e}")
    
    # 5. Grad-CAM Heatmaps (sample images)
    try:
        # Generate Grad-CAM for a few sample validation images
        val_dataset = EmotionDataset(df_val, full_config["images_root"], full_config["input_size_ft"], train=False)
        if len(val_dataset) > 0:
            max_gallery = min(len(val_dataset), 8)
            target_count = max_gallery if len(val_dataset) >= 6 else len(val_dataset)
            sample_indices = np.random.choice(len(val_dataset), size=target_count, replace=False)
        else:
            sample_indices = []
        gradcam_entries = []
        activation_tensor = None
        
        for idx, sample_idx in enumerate(sample_indices):
            sample_image, sample_label = val_dataset[sample_idx]
            sample_image_tensor = sample_image.unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(sample_image_tensor)
                pred_class = output.argmax(1).item()
            
            cam = generate_gradcam(model, sample_image, pred_class, device, model_name)
            if cam is not None:
                class_name = classes[pred_class] if pred_class < len(classes) else str(pred_class)
                plot_gradcam_heatmap(
                    sample_image, cam,
                    results_dir / f"{model_name}_epoch{final_epoch:03d}_gradcam_sample{idx+1}.png",
                    model_name, final_epoch, class_name, pred_class
                )
                gradcam_entries.append({
                    "image": sample_image,
                    "cam": cam,
                    "title": f"{class_name} (pred {pred_class})"
                })
                if activation_tensor is None and isinstance(sample_image, torch.Tensor):
                    activation_tensor = sample_image.unsqueeze(0).to(device)
        
        if gradcam_entries:
            plot_gradcam_gallery(
                gradcam_entries,
                figures_dir / f"{model_name}_gradcam_gallery.png"
            )
        if activation_tensor is not None:
            try:
                plot_activation_map_heatmap(
                    model,
                    activation_tensor,
                    device,
                    figures_dir / f"{model_name}_activation_map.png",
                    model_name
                )
            except Exception as e:
                print(f"  ⚠ Could not generate activation map: {e}")
        
        if gradcam_entries:
            print(f"  ✓ Grad-CAM heatmaps saved")
    except Exception as e:
        print(f"  ⚠ Could not generate Grad-CAM: {e}")
    
    # 6. t-SNE / UMAP Visualizations (feature space)
    try:
        # Extract features from penultimate layer
        model.eval()
        features_list = []
        labels_list = []
        
        # Create a simple feature extractor (hook on penultimate layer)
        def get_features_hook(module, input, output):
            features_list.append(output.cpu().numpy())
        
        # Register hook based on model architecture
        hook = None
        if hasattr(model, 'classifier'):
            hook = model.classifier.register_forward_hook(get_features_hook)
        elif hasattr(model, 'stages'):
            # timm ConvNeXt-V2 exposes stages; hook the last block
            try:
                last_stage = model.stages[-1]
                last_block = last_stage[-1]
                hook = last_block.register_forward_hook(get_features_hook)
            except Exception:
                hook = None
        
        if hook is not None:
            # Extract features from validation set (sample for efficiency)
            val_loader_feat = DataLoader(
                EmotionDataset(df_val.head(500), full_config["images_root"], full_config["input_size_ft"], train=False),
                batch_size=32, shuffle=False, num_workers=0
            )
            
            with torch.no_grad():
                for xb, yb in val_loader_feat:
                    xb = xb.to(device)
                    _ = model(xb)
                    labels_list.extend(yb.numpy())
            
            if features_list:
                features = np.vstack(features_list)
                
                # t-SNE
                plot_tsne_visualization(
                    features, np.array(labels_list), classes,
                    results_dir / f"{model_name}_epoch{final_epoch:03d}_tsne.png",
                    model_name, final_epoch
                )
                print(f"  ✓ t-SNE visualization saved")
                
                # UMAP (if available)
                try:
                    plot_umap_visualization(
                        features, np.array(labels_list), classes,
                        results_dir / f"{model_name}_epoch{final_epoch:03d}_umap.png",
                        model_name, final_epoch
                    )
                    print(f"  ✓ UMAP visualization saved")
                except:
                    pass
            
            hook.remove()
    except Exception as e:
        print(f"  ⚠ Could not generate t-SNE/UMAP: {e}")
    
    # Save classification report
    from metrics_and_plots import save_classification_report
    save_classification_report(
        results["labels"], results["preds"], classes,
        results_dir / f"{model_name}_epoch{final_epoch:03d}_classification_report.csv"
    )
    
    plot_prior_work_chart(
        results["acc"],
        figures_dir / f"{model_name}_prior_work_comparison.png"
    )
    
    # 7. Facial Landmark Attention Heatmaps for each class
    try:
        generate_class_landmark_heatmaps(
            model, df_val, images_root, classes, device,
            results_dir, model_name, samples_per_class=3
        )
        print(f"  ✓ Facial landmark attention heatmaps generated")
    except Exception as e:
        print(f"  ⚠ Could not generate landmark heatmaps: {e}")
    
    # 8. Evaluate on test split (3-way split only)
    if df_test is not None:
        print("\n=== Evaluating on independent test split ===")
        try:
            test_loader = DataLoader(
                EmotionDataset(df_test, full_config["images_root"], full_config["input_size_ft"], train=False),
                batch_size=32, shuffle=False, num_workers=0
            )
            test_criterion = build_criterion(full_config)
            use_tta_eval = full_config.get("use_tta", True)
            test_acc, test_loss = evaluate_model(
                model, test_loader, test_criterion, device,
                channels_last=channels_last_enabled, use_tta=use_tta_eval,
            )

            # Detailed test stats
            test_preds, test_labels = [], []
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb = xb.to(device)
                    if channels_last_enabled:
                        xb = xb.to(memory_format=torch.channels_last)
                    if use_tta_eval:
                        out = tta_forward(model, xb, channels_last=channels_last_enabled)
                    else:
                        out = model(xb)
                    test_preds.extend(out.argmax(1).cpu().numpy())
                    test_labels.extend(yb.numpy())
            
            from sklearn.metrics import f1_score
            test_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
            results["test_acc"] = float(test_acc)
            results["test_f1_macro"] = float(test_f1)
            results["test_loss"] = float(test_loss)
            
            print(f"Independent Test split evaluation:")
            print(f"  Test Loss: {test_loss:.4f}")
            print(f"  Test Acc : {test_acc:.4f} ({test_acc*100:.2f}%)")
            print(f"  Test F1  : {test_f1:.4f}")
        except Exception as e:
            print(f"  ⚠ Could not evaluate on test split: {e}")

    print(f"  ✓ All Q1 conference figures generated!\n")

    # ── Checkpoint saving ──────────────────────────────────────────────────────
    # 1) Bare weights file (backward-compat with cross_dataset_eval.py)
    weights_path = results_dir / f"{model_name}_best.pth"
    torch.save(model.state_dict(), weights_path)
    print(f"  ✓ Weights-only checkpoint: {weights_path.name}")

    # 2) Full checkpoint (weights + optimizer + scheduler + metadata)
    if full_config.get("save_full_checkpoint", True):
        full_ckpt_path = results_dir / f"{model_name}_best_full.pth"
        save_full_ckpt(
            path=str(full_ckpt_path),
            model_state={k: v.cpu() for k, v in model.state_dict().items()},
            epoch=len(results["history"]["train_loss"]),
            best_val_acc=float(results["best_val_acc"]),
            seed=full_config.get("seed", 42),
            config={
                **full_config,
                "csv_path": csv_path,
                "num_classes": num_classes,
            },
        )
        print(f"  ✓ Full checkpoint: {full_ckpt_path.name}")

    # 3) Persist split indices for exact reproducibility
    if full_config.get("save_split_indices", True):
        save_split_indices(
            train_indices=df_train.index.tolist(),
            val_indices=df_val.index.tolist(),
            out_dir=str(results_dir),
            test_indices=df_test.index.tolist() if df_test is not None else None,
        )
        print(f"  ✓ Split indices saved (split_indices.json)")
    
    print(f"\n>>> Results saved to: {results_dir}")
    print(f">>> Accuracy: {results['acc']:.4f}, F1-macro: {results['f1_macro']:.4f}")
    if results.get("per_dataset_metrics"):
        print(f">>> Per-dataset metrics saved to: metrics_per_dataset.json")
    
    return model, results
