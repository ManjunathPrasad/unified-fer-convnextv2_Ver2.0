# augmentations.py
"""
Advanced augmentation suite for emotion recognition (Q1-grade).

Provides:
 - Standard pipelines:
    * get_train_transforms(input_size)
    * get_valid_transforms(input_size)
 - MixUp
 - CutMix
 - RandAugment (torchvision)
 - Random Erasing

 - Novel augmentation module (for Q1 novelty):
    * FourierAugment  (lightweight frequency-based augmentation)
    * ContrastiveNoiseAug
    * AugMixLite (multi-path stochastic blending)
    * ConsistencyColorJitter (stochastic intensity alignment)

These are all implemented so they can be toggled on/off via config YAML.
"""

import random
import math
from typing import Tuple, Optional, Dict

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms

# ----------------------------
# Helper: Standard normalize
# ----------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def normalize_tf():
    return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

# ----------------------------
# Fourier-based augmentation
# ----------------------------
class FourierAugment:
    """
    Lightweight frequency perturbation:
    - Convert image to frequency domain (FFT)
    - Randomly scale magnitude in low and mid frequencies
    - Inverse FFT to reconstruct
    - Useful for improving robustness to illumination and sensor variations

    Input: PIL Image
    Output: PIL Image
    """
    def __init__(self, alpha_low=0.8, alpha_mid=1.2, p=0.5):
        self.alpha_low = alpha_low
        self.alpha_mid = alpha_mid
        self.p = p

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img

        arr = np.asarray(img).astype(np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        h, w, c = arr.shape

        out = []
        for i in range(c):
            ch = arr[:, :, i]
            fft = np.fft.fft2(ch)
            fft_shift = np.fft.fftshift(fft)
            mag = np.abs(fft_shift)
            pha = np.angle(fft_shift)

            # Build masks
            cy, cx = h // 2, w // 2
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((Y - cy)**2 + (X - cx)**2)

            # low frequency region (large scale color/illumination)
            low_mask = dist < min(h, w) * 0.1
            # mid frequencies (texture)
            mid_mask = (dist >= min(h, w) * 0.1) & (dist < min(h, w) * 0.3)

            mag2 = mag.copy()
            mag2[low_mask] *= self.alpha_low
            mag2[mid_mask] *= self.alpha_mid

            fft_new = mag2 * np.exp(1j * pha)
            fft_ishift = np.fft.ifftshift(fft_new)
            ch_new = np.fft.ifft2(fft_ishift)
            ch_new = np.real(ch_new)
            out.append(ch_new)

        out = np.stack(out, axis=-1)
        out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out)

# ----------------------------
# Contrastive noise augmentation
# ----------------------------
class ContrastiveNoiseAug:
    """
    Adds small Gaussian noise with random variance per channel.
    Encourages local neighborhood invariance (beneficial for ViT & ConvNeXt).
    """
    def __init__(self, std_range=(2, 8), p=0.5):
        self.std_range = std_range
        self.p = p

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img
        arr = np.asarray(img).astype(np.float32)
        std = random.uniform(*self.std_range)
        noise = np.random.normal(0, std, arr.shape).astype(np.float32)
        arr2 = arr + noise
        arr2 = np.clip(arr2, 0, 255).astype(np.uint8)
        return Image.fromarray(arr2)

# ----------------------------
# Consistency color jitter
# ----------------------------
class ConsistencyColorJitter:
    """
    Smoothly adjusts brightness/contrast in a small band.
    Provides consistency regularization for subtle facial emotions.
    """
    def __init__(self, brightness=0.1, contrast=0.1, p=0.5):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img
        arr = np.asarray(img).astype(np.float32)

        # smooth brightness shift
        b_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
        c_factor = 1.0 + random.uniform(-self.contrast, self.contrast)

        arr = (arr * c_factor) + (b_factor * 10)  # small shift
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

# ----------------------------
# AugMix-Lite (novel)
# ----------------------------
class AugMixLite:
    """
    A simplified AugMix-like augmentation chain:
    - Creates 2 stochastic augmentation paths
    - Blends them with the original
    - Only uses computationally light ops (rotation, contrast, color, sharpness)

    Helps improve robustness without heavy compute.
    """
    def __init__(self, p=0.5, severity=1):
        self.p = p
        self.severity = severity
        # Use callable classes instead of lambdas for pickling compatibility
        self.ops = [
            self._op_rotate,
            self._op_brightness,
            self._op_contrast,
            self._op_saturation,
            self._op_hue,
        ]
        # Pre-create ColorJitter instances to avoid creating them each time
        self._jitter_brightness = transforms.ColorJitter(brightness=0.2)
        self._jitter_contrast = transforms.ColorJitter(contrast=0.2)
        self._jitter_saturation = transforms.ColorJitter(saturation=0.2)
        self._jitter_hue = transforms.ColorJitter(hue=0.02)

    def _op_rotate(self, img):
        """Rotation operation."""
        return img.rotate(random.uniform(-10, 10))

    def _op_brightness(self, img):
        """Brightness jitter operation."""
        return self._jitter_brightness(img)

    def _op_contrast(self, img):
        """Contrast jitter operation."""
        return self._jitter_contrast(img)

    def _op_saturation(self, img):
        """Saturation jitter operation."""
        return self._jitter_saturation(img)

    def _op_hue(self, img):
        """Hue jitter operation."""
        return self._jitter_hue(img)

    def _apply_random_op(self, x):
        """Apply a random operation from ops list."""
        op = random.choice(self.ops)
        return op(x)

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img

        m1 = self._apply_random_op(img)
        m2 = self._apply_random_op(img)
        blended = Image.blend(img, Image.blend(m1, m2, 0.5), 0.5)
        return blended

# ----------------------------
# MixUp
# ----------------------------
def mixup_data(x, y, alpha=0.4, device="cuda"):
    if alpha <= 0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# Alias for compatibility
def mixup_batch(x, y, alpha=0.4):
    """Alias for mixup_data for compatibility."""
    device = x.device if isinstance(x, torch.Tensor) else "cuda"
    return mixup_data(x, y, alpha=alpha, device=device)

# ----------------------------
# CutMix
# ----------------------------
def cutmix_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = x.size()
    index = torch.randperm(B).to(x.device)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    w = int(W * math.sqrt(1 - lam))
    h = int(H * math.sqrt(1 - lam))

    x1 = np.clip(cx - w // 2, 0, W)
    x2 = np.clip(cx + w // 2, 0, W)
    y1 = np.clip(cy - h // 2, 0, H)
    y2 = np.clip(cy + h // 2, 0, H)

    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam_eff = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam_eff

# Alias for compatibility
def cutmix_batch(x, y, alpha=1.0):
    """Alias for cutmix_data for compatibility."""
    return cutmix_data(x, y, alpha=alpha)

# ----------------------------
# Standard Train / Valid Transforms
# ----------------------------
def get_train_transforms(
    input_size: int,
    use_randaugment: bool = True,
    use_novel_aug: bool = True,
    random_erasing_p: float = 0.2,
    fourier_aug_p: float = 0.4,
    contrastive_noise_p: float = 0.3,
    augmix_lite_p: float = 0.4,
    randaugment_num_ops: int = 2,
    randaugment_magnitude: int = 9
):
    """
    Get training transforms with optional augmentations.
    
    Args:
        input_size: Target image size
        use_randaugment: Whether to use RandAugment
        use_novel_aug: Whether to use novel augmentations (FourierAugment, ContrastiveNoiseAug, AugMixLite)
        random_erasing_p: Probability for RandomErasing (0 to disable)
        fourier_aug_p: FourierAugment frequency perturbation probability
        contrastive_noise_p: Contrastive noise injection probability
        augmix_lite_p: AugMixLite multi-path blending probability
        randaugment_num_ops: Number of random ops applied in RandAugment
        randaugment_magnitude: Strength of ops in RandAugment
    """
    aug = []

    # base augmentations (always applied)
    aug.append(transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)))
    aug.append(transforms.RandomHorizontalFlip())
    aug.append(transforms.RandomRotation(12))
    
    # Color jitter (standard)
    aug.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))

    # RandAugment (if enabled)
    if use_randaugment and hasattr(transforms, "RandAugment"):
        try:
            aug.append(transforms.RandAugment(num_ops=randaugment_num_ops, magnitude=randaugment_magnitude))
        except:
            pass  # Fallback if RandAugment not available

    # Novel augmentations (if enabled)
    if use_novel_aug:
        aug.append(ConsistencyColorJitter())
        aug.append(FourierAugment(p=fourier_aug_p))
        aug.append(ContrastiveNoiseAug(p=contrastive_noise_p))
        aug.append(AugMixLite(p=augmix_lite_p))

    # Convert to tensor and normalize
    aug.append(transforms.ToTensor())
    aug.append(normalize_tf())

    # Random Erasing post-normalization (if enabled)
    if random_erasing_p > 0:
        aug.append(transforms.RandomErasing(p=random_erasing_p))

    return transforms.Compose(aug)

def get_mild_train_transforms(input_size: int):
    """
    Light augmentation for the final fine-tune epochs (clean convergence phase).
    Disables RandAugment / novel aug so the model can fit real val/test distributions.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        normalize_tf(),
    ])

def get_valid_transforms(input_size: int):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize_tf()
    ])
