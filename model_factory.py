# model_factory.py
"""
Model factory and ensemble utilities focused exclusively on ConvNeXt-V2.

Provides:
 - get_model(name, num_classes, pretrained=True, input_size=256, device='cuda')
 - freeze_backbone(model, head_keywords=("fc","classifier","head","heads"))
 - unfreeze_model(model)
 - count_parameters(model)
 - LightweightEnsemble(models, learnable=False, device='cuda')
 - save_model_states(models_dict, out_dir), load_model_states(models_dict, paths)
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import timm
    HAVE_TIMM = True
except Exception:
    timm = None
    HAVE_TIMM = False

SUPPORTED_MODEL = "convnext_v2"
SUPPORTED_MODELS = {
    "convnext_v2_tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "convnext_v2_base": "convnextv2_base.fcmae_ft_in22k_in1k",
    "convnext_v2_large": "convnextv2_large.fcmae_ft_in22k_in1k",
    "convnext_v2_huge": "convnextv2_huge.fcmae_ft_in22k_in1k",
}
# Reproducible baselines trained under identical protocol (Reviewer 2 Comment 2)
BASELINE_MODELS = {
    "resnet18": "resnet18.a1_in1k",
    "resnet50": "resnet50.a1_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "vit_small": "vit_small_patch16_224.augreg_in21k_ft_in1k",
}
ALL_MODELS = {**SUPPORTED_MODELS, **BASELINE_MODELS}
MODEL_ALIASES = {
    "convnext_v2": "convnext_v2_base",
    "convnext-v2": "convnext_v2_base",
    "convnextv2": "convnext_v2_base",
    "convnext v2": "convnext_v2_base",
    
    "convnext_v2_tiny": "convnext_v2_tiny",
    "convnext-v2-tiny": "convnext_v2_tiny",
    "convnextv2_tiny": "convnext_v2_tiny",
    "convnext v2 tiny": "convnext_v2_tiny",
    
    "convnext_v2_base": "convnext_v2_base",
    "convnext-v2-base": "convnext_v2_base",
    "convnextv2_base": "convnext_v2_base",
    "convnext v2 base": "convnext_v2_base",
    
    "convnext_v2_large": "convnext_v2_large",
    "convnext-v2-large": "convnext_v2_large",
    "convnextv2_large": "convnext_v2_large",
    "convnext v2 large": "convnext_v2_large",
    
    "convnext_v2_huge": "convnext_v2_huge",
    "convnext-v2-huge": "convnext_v2_huge",
    "convnextv2_huge": "convnext_v2_huge",
    "convnext v2 huge": "convnext_v2_huge",

    # Baseline models (same-protocol reproduction)
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet-b0": "efficientnet_b0",
    "vit_small": "vit_small",
    "vit-small": "vit_small",
}


def normalize_model_name(name: str) -> str:
    if name is None:
        raise ValueError("Model name cannot be None.")
    key = name.strip().lower()
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    if key in ALL_MODELS:
        return key
    raise ValueError(
        f"Unsupported model '{name}'. "
        f"Supported: ConvNeXt-V2 variants + baselines {list(BASELINE_MODELS.keys())}"
    )


def is_baseline_model(name: str) -> bool:
    return normalize_model_name(name) in BASELINE_MODELS

# -------------------------
# Helpers
# -------------------------
def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

# -------------------------
# Model factory
# -------------------------
def get_model(name: str, num_classes: int, pretrained: bool = True, input_size: int = 256, device: Optional[torch.device] = None):
    """
    Returns a ready-to-train model with classification head adjusted.
    """
    lname = normalize_model_name(name)
    timm_model = ALL_MODELS[lname]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not HAVE_TIMM:
        raise ImportError("timm is required for ConvNeXt-V2. Please install timm>=0.6.0.")

    try:
        m = timm.create_model(
            timm_model,
            pretrained=pretrained,
            num_classes=num_classes,
        )
        m.to(device)
        return m
    except Exception as e:
        raise RuntimeError(f"Failed to create ConvNeXt-V2 model ({timm_model}) via timm: {e}")

# -------------------------
# Freeze / unfreeze helpers
# -------------------------
def freeze_backbone(model: nn.Module, head_keywords: Tuple[str]=("fc","classifier","head","heads")):
    """
    Freeze all params except those in modules whose name contains any of head_keywords.
    """
    for n, p in model.named_parameters():
        if any(k in n.lower() for k in head_keywords):
            p.requires_grad = True
        else:
            p.requires_grad = False

def unfreeze_model(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = True


def get_param_groups(model: nn.Module, base_lr: float, backbone_lr_ratio: float = 0.1):
    """
    Build AdamW param groups with lower LR for backbone, higher for classification head.
    """
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name.lower() for k in ("head", "fc", "classifier", "heads")):
            head_params.append(param)
        else:
            backbone_params.append(param)
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": base_lr * backbone_lr_ratio})
    if head_params:
        groups.append({"params": head_params, "lr": base_lr})
    return groups if groups else [{"params": [p for p in model.parameters() if p.requires_grad], "lr": base_lr}]

# -------------------------
# Lightweight ensemble
# -------------------------
class LightweightEnsemble(nn.Module):
    """
    Simple ensemble wrapper:
     - If learnable=False: averages softmax probabilities across models.
     - If learnable=True: learnable scalar weights (logits space) per model + optional temperature.
    Useful for small-weight ensembles (Q1-style experiments).
    """
    def __init__(self, models_list: List[nn.Module], num_classes: int, learnable: bool = False, device: Optional[torch.device] = None):
        super().__init__()
        self.models = nn.ModuleList(models_list)
        self.num_models = len(self.models)
        self.num_classes = num_classes
        self.learnable = learnable
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        if learnable:
            # weight per model (initialized equally)
            w = torch.ones(self.num_models, dtype=torch.float32) / float(self.num_models)
            self.logits_w = nn.Parameter(w.to(self.device))
            # optional temperature
            self.log_temp = nn.Parameter(torch.tensor(0.0).to(self.device))
        else:
            self.register_buffer("fixed_w", torch.ones(self.num_models, dtype=torch.float32) / float(self.num_models))

        # models should be in eval mode by default for ensembling inference
        for m in self.models:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor):
        probs = []
        for m in self.models:
            out = m(x)
            probs.append(F.softmax(out, dim=1))
        stacked = torch.stack(probs, dim=0)  # (M, B, C)
        if self.learnable:
            w = F.softmax(self.logits_w, dim=0).view(self.num_models, 1, 1)  # normalized
            avg = torch.sum(stacked * w, dim=0)
            # optional temperature scaling (applied to logits before softmax would be better,
            # but we apply to final averaged probs for lightweight simplicity).
            temp = torch.exp(self.log_temp)
            eps = 1e-8
            avg = torch.clamp(avg, eps, 1.0 - eps)
            logits = torch.log(avg)
            logits = logits / temp
            return F.softmax(logits, dim=1)
        else:
            avg = torch.mean(stacked, dim=0)
            return avg

# -------------------------
# Checkpoint helpers
# -------------------------
def save_model_states(models_dict: Dict[str, nn.Module], out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, m in models_dict.items():
        path = out_dir / f"{name}_state.pth"
        torch.save(m.state_dict(), path)

def load_model_states(models_dict: Dict[str, nn.Module], paths: Dict[str, str], map_location: Optional[str] = None):
    for name, m in models_dict.items():
        p = paths.get(name)
        if p is None:
            continue
        state = torch.load(p, map_location=map_location)
        try:
            m.load_state_dict(state)
        except RuntimeError:
            # try stripping 'module.' prefixes
            new_state = {k.replace("module.", ""): v for k, v in state.items()}
            m.load_state_dict(new_state)
    return models_dict

# -------------------------
# Small CLI test
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=SUPPORTED_MODEL, help="Only convnext_v2 is supported")
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--input_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = get_model(args.model, args.num_classes, pretrained=True, input_size=args.input_size, device=device)
    print("Model created:", args.model)
    print("Total params:", count_parameters(m, trainable_only=False))
    print("Trainable params:", count_parameters(m, trainable_only=True))
