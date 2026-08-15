#  Emotion Recognition Framework

> **ConvNeXt-V2 Base** · Unified FER2013 + RAF-DB + AffectNet · **89.66% validation accuracy** (27 epochs)  
> Full  research-standard pipeline with reproducible experiments, YAML configs, and comprehensive evaluation.

---

## 🎯 Features

| Category | Feature |
|----------|---------|
| **Accuracy** | 89.66% validation accuracy on unified 3-dataset benchmark (ConvNeXt-V2 Base, 27 epochs) |
| **Reproducibility** | Seed-controlled splits · YAML configs · split index persistence |
| **Training** | Linear warmup + cosine decay LR · EMA · gradient accumulation (effective batch 64) · early stopping |
| **Checkpoints** | Full checkpoint (weights + optimizer + epoch + config) + weights-only |
| **Logging** | Per-epoch CSV log (`training_log.csv`) + JSON metrics + structured results |
| **Evaluation** | Standalone `evaluate.py` CLI · confusion matrix · ROC/PR · t-SNE/UMAP · Grad-CAM |
| **Multi-Seed** | Statistical aggregation (mean ± std, 95% CI) · auto summary table |
| **Cross-Dataset** | Train-on-one, test-on-others evaluation · confusion matrices · domain-shift diagnosis |
| **Augmentation** | MixUp · CutMix · RandAugment · Fourier-based novel augmentations |
| **Datasets** | FER2013 · RAF-DB · AffectNet · multiprocessing preparation |
| **Ablation** | Multi-seed (3×) ablation with Welch t-test significance tables |
| **Notation** | Full symbol/variable reference in [NOTATION.md]
---

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# Full pipeline (prep → train → benchmark → cross-dataset → release bundle)
python master_experiment.py --config configs/kaggle_2gpu.yaml

# Optional flags
python master_experiment.py --config configs/kaggle_2gpu.yaml --run-ablation
python master_experiment.py --config configs/kaggle_2gpu.yaml --skip-cross-dataset

# Export reproducibility bundle after a run
python export_reproducibility_bundle.py \
  --experiment_dir Results_Q1/exp_main_xxx/convnext_v2 \
  --output release_bundle
```

---

## 📁 Project Structure

```
paper2/
│
├── configs/
│   └── kaggle_2gpu.yaml              #   Full config (Kaggle 2×T4, 2×GPU DataParallel)
│
├── master_experiment.py              # Main pipeline: prep → train → evaluate
├── train_engine.py                   # Training engine (warmup + finetune + logging)
├── evaluate.py                       # ✨ NEW — Standalone evaluation CLI
├── experiment_runner.py              # Multi-seed experiments + summary table
├── cross_dataset_eval.py             # Cross-dataset generalisation evaluation
├── ablation_study.py                 # Ablation study runner
│
├── dataset_preparation.py            # FER2013 / RAF-DB / AffectNet preparation
├── dataset_balancer.py               # Class-balanced sampling
├── augmentations.py                  # MixUp / CutMix / RandAugment / Novel
├── model_factory.py                  # ConvNeXt-V2 factory + freeze/unfreeze
├── metrics_and_plots.py              # All figures: confusion / ROC / PR / Grad-CAM
├── utils.py                          # ✨ UPDATED — EpochLogger, split helpers, YAML
│
├── benchmark_eval.py                 # Per-dataset standard-split evaluation
├── baseline_runner.py                # Reproduce ResNet-18 etc. under identical protocol
├── export_reproducibility_bundle.py  # Package scripts + splits + weights for GitHub
├── EXPERIMENTAL_PROTOCOL.md          # Full hyperparameter protocol (R2-C5)
├── PRIOR_WORKS_COMPARISON.md         # SOTA table with protocol labels (R1-C5, R2-C1)
├── requirements.txt
└── emotion_q1_framework/
    └── Dataset/
        ├── images/                   # Processed images (auto-created)
        └── prepared/                 # CSV files (auto-created)
```

---

## ⚙️ Configuration — YAML Files

All hyperparameters live in `configs/kaggle_2gpu.yaml`. No more editing Python source to change settings.

### `configs/kaggle_2gpu.yaml` (key fields)

```yaml
# Reproducibility
seed: 42
deterministic: true
val_split: 0.2
save_split_indices: true      # persists exact train/val rows to disk

# Model
model: convnext_v2
num_classes: 7
input_size_warm: 224
input_size_ft: 256

# Training
epochs_warm: 2
epochs_ft: 25
batch_size: 32
grad_accum_steps: 2           # effective_batch_size = batch_size × grad_accum_steps = 64
lr: 3.0e-5
warmup_steps: 0               # 0 = auto (one epoch), or set exact step count
warmup_start_factor: 0.01     # LR starts at 1% of base_lr, ramps to 100%
early_stop: 10

# Checkpoint & Logging
save_full_checkpoint: true    # saves optimizer + scheduler state too
log_csv: true                 # writes training_log.csv per epoch

# Hardware (use "auto" to let the script decide)
device: auto
use_amp: auto
channels_last: auto
```

### Config File

| File | Device | Batch | Epochs | Purpose |
|------|--------|-------|--------|---------|
| `kaggle_2gpu.yaml` | cuda | 32 | 2+25 | Kaggle 2×T4 / any CUDA GPU |

---

## 📈 Training Process

### Phase 1 — Warmup (2 epochs, 224×224)
- Backbone **frozen** — only the classification head is trained
- **Linear LR warmup**: ramps from `lr × 0.01` → `lr` over the first epoch's steps
- After warmup: cosine decay

### Phase 2 — Fine-tune (25 epochs, 256×256)
- **All layers unfrozen**, LR reset to `lr × 0.3` with cosine decay
- Exponential Moving Average (EMA, decay = 0.9997)
- Early stopping with `patience = 10`
- Gradient clipping (`max_norm = 5.0`)
- Gradient accumulation (`grad_accum_steps` configurable)

### Actual Epoch Progression (Experiment `main_2026-07-08_15-18-19`)

```
Epoch  1 [warm]    : ValAcc = 0.6767  LR = 3.00e-05  (~22 min)
Epoch  2 [warm]    : ValAcc = 0.7749  LR = 3.00e-07  (~22 min)
Epoch  3 [finetune]: ValAcc = 0.7771  LR = 8.96e-06
Epoch  6 [finetune]: ValAcc = 0.8019  LR = 8.45e-06
Epoch 10 [finetune]: ValAcc = 0.8389  LR = 6.93e-06
Epoch 15 [finetune]: ValAcc = 0.8722  LR = 4.27e-06
Epoch 20 [finetune]: ValAcc = 0.8892  LR = 1.71e-06
Epoch 27 [finetune]: ValAcc = 0.8966  LR = 9.00e-08  (best model saved)
```

---

## 📊 Output Files

Every run creates a timestamped experiment directory:

```
Results_Q1/exp_main_2026-xx-xx/convnext_v2/
│
├── training_log.csv              # ✨ Per-epoch: loss, acc, LR, time
├── config.json                   # Full experiment configuration
├── metrics.json                  # Final: accuracy, F1, precision, recall
├── metrics_per_dataset.json      # FER2013 / RAF-DB / AffectNet breakdown
├── split_indices.json            # ✨ Exact train/val row indices
│
├── convnext_v2_best.pth          # Weights-only checkpoint
├── convnext_v2_best_full.pth     # ✨ Full checkpoint (+ optimizer + config)
│
├── classification_report.csv     # Per-class precision / recall / F1
├── convnext_v2_confusion.png     # Confusion matrix (raw counts)
├── convnext_v2_confusion_norm.png# Confusion matrix (normalised)
├── convnext_v2_xxx_roc.png       # ROC curves (per-class + macro)
├── convnext_v2_xxx_pr.png        # PR curves
├── convnext_v2_xxx_tsne.png      # t-SNE feature visualisation
├── convnext_v2_xxx_umap.png      # UMAP feature visualisation
│
└── figures/
    ├── convnext_v2_gradcam_gallery.png
    ├── convnext_v2_confusion_heatmap_unified.png
    ├── convnext_v2_class_distribution.png
    ├── convnext_v2_landmark_heatmap_*.png
    └── ...
```

### Multi-Seed Run additionally produces:
```
├── individual_runs.json          # Per-seed metrics
├── aggregated_metrics.json       # Mean ± std + 95% CI
└── summary_table.csv             # ✨ Auto-generated publication table
```

---

## 🧪 Additional Scripts

### Multi-Seed Experiments (Statistical Rigour)

```bash
python experiment_runner.py \
    --model convnext_v2 \
    --csv emotion_q1_framework/Dataset/prepared/unified_dataset.csv \
    --images emotion_q1_framework/Dataset/images \
    --seeds 42 123 456 \
    --config configs/default.yaml
```

Outputs: `aggregated_metrics.json` (mean ± std, 95% CI) + **`summary_table.csv`** (auto).

### Standalone Evaluation (any checkpoint)

```bash
# Weights-only checkpoint
python evaluate.py \
    --checkpoint Results_Q1/exp_xxx/convnext_v2/convnext_v2_best.pth \
    --csv emotion_q1_framework/Dataset/prepared/unified_dataset.csv \
    --images emotion_q1_framework/Dataset/images \
    --output Results_Q1/eval_output

# Full checkpoint (shows epoch + training metadata)
python evaluate.py \
    --checkpoint Results_Q1/exp_xxx/convnext_v2/convnext_v2_best_full.pth \
    --full_ckpt \
    --config configs/default.yaml
```

Outputs: `eval_metrics.json`, `classification_report.csv`, confusion matrices, ROC/PR curves.

### Ablation Study

```bash
python ablation_study.py \
    --model convnext_v2 \
    --csv emotion_q1_framework/Dataset/prepared/unified_dataset.csv \
    --images emotion_q1_framework/Dataset/images
```

Tests: baseline → MixUp only → CutMix only → RandAugment only → full pipeline.

### Cross-Dataset Evaluation

```bash
python cross_dataset_eval.py \
    --csv emotion_q1_framework/Dataset/prepared/unified_dataset.csv \
    --images emotion_q1_framework/Dataset/images \
    --num_classes 7
```

Trains on each dataset individually, evaluates on the others.

---

## 🔬 Technical Details

### Model Architecture

| Component | Detail |
|-----------|--------|
| **Backbone** | ConvNeXt-V2 Base (`convnextv2_base.fcmae_ft_in22k_in1k` via timm) |
| **Pretrained** | ImageNet-22k → ImageNet-1k fine-tuned |
| **Input Size** | 224×224 (warmup) → 256×256 (fine-tune) |
| **Parameters** | ~89M total |

### Training Strategy

| Component | Detail |
|-----------|--------|
| **Optimizer** | AdamW (`lr=3e-5`, `weight_decay=1e-4`) |
| **LR Scheduler** | Linear Warmup (1%→100%) + Cosine Decay |
| **Loss** | CrossEntropyLoss + label smoothing (0.1) |
| **Regularisation** | Gradient clipping (`max_norm=5.0`), EMA (decay=0.9997) |
| **Mixed Precision** | AMP (CUDA only) |
| **Grad Accumulation** | Configurable via `grad_accum_steps` |

### Reproducibility

| Feature | Detail |
|---------|--------|
| **Seed** | Controls Python/NumPy/PyTorch/CUDA RNG + dataset split |
| **Split persistence** | `split_indices.json` — exact row indices saved per run |
| **Config snapshot** | Embedded in `config.json` + inside full `.pth` checkpoint |
| **Deterministic mode** | `deterministic: true` in YAML (slower but bit-exact) |
| **3-way split** | Train (65%) / Validation (20%) / Test (15%) via `test_split: 0.15` in YAML |
| **Checkpoint criterion** | Best **validation accuracy** model is saved (`convnext_v2_best.pth`) |
| **Effective batch size** | `batch_size=32` × `grad_accum_steps=2` = **64** |

---

## 📂 Reproducibility & Public Materials

All scripts needed to reproduce the reported results are included in this repository.

### Available Scripts

| Script | Purpose |
|--------|---------|
| `master_experiment.py` | End-to-end pipeline: prep → train → benchmark eval |
| `experiment_runner.py` | Multi-seed (3×) runs with mean ± std aggregation |
| `ablation_study.py` | Ablation with significance tests (`--seeds 42 123 456`) |
| `cross_dataset_eval.py` | Cross-dataset eval with confusion matrices + domain-shift report |
| `benchmark_eval.py` | Per-dataset standard-split accuracy vs. SOTA methods |
| `evaluate.py` | Standalone evaluation on any checkpoint |
| `dataset_balancer.py` | `print_dataset_report()` for verified dataset count tables |

### Checkpoint Selection Criterion

The model with the **highest validation accuracy** during fine-tuning is saved as  
`{model_name}_best.pth` / `{model_name}_best_full.pth`. Early stopping uses `patience=10` epochs.

### Reproducing the Exact Split

Each run saves `split_indices.json` containing the exact train/val/test row indices:

```python
from utils import load_split_indices
indices = load_split_indices("Results_Q1/exp_xxx/convnext_v2/")
# indices["train_indices"], indices["val_indices"], indices["test_indices"]
```

### Dataset Statistics Report

To generate publication-ready tables of raw vs. balanced sample counts:

```python
from dataset_balancer import print_dataset_report
print_dataset_report(df_before_balance, df_after_balance, out_md="dataset_stats_report.md")
```

### Symbol Reference

All mathematical symbols and variables are defined in [NOTATION.md](NOTATION.md).

---

## 🐛 Troubleshooting

### Out of Memory (OOM)
```yaml
# In your YAML config:
batch_size: 16          # reduce from 32
grad_accum_steps: 2     # keep effective batch = 32
input_size_ft: 224      # reduce from 256
```

### Low Accuracy (< 88%)
```yaml
epochs_ft: 25           # increase from 20
min_samples_per_class: 3000   # more training data
early_stop: 15          # more patience
```

### Training Not Reproducible
```yaml
seed: 42
deterministic: true     # ensures bit-exact results (slightly slower)
save_split_indices: true
```

### Resume Training from Checkpoint
```python
from utils import load_full_ckpt
ckpt = load_full_ckpt("Results_Q1/exp_xxx/convnext_v2/convnext_v2_best_full.pth")
print(f"Epoch: {ckpt['epoch']}, Best Val Acc: {ckpt['best_val_acc']:.4f}")
model.load_state_dict(ckpt["model_state"])
```

### Monitor Training Live
```powershell
# Watch training_log.csv update in real time (PowerShell)
Get-Content Results_Q1/exp_xxx/convnext_v2/training_log.csv -Wait
```

### Missing MediaPipe (Landmark Heatmaps)
```bash
pip install mediapipe   # optional; falls back to OpenCV if missing
```

---

## 🎓 Emotion Classes

| ID | Emotion | Label |
|----|---------|-------|
| 0 | Angry | anger |
| 1 | Disgust | disgust |
| 2 | Fear | fear |
| 3 | Happy | happiness |
| 4 | Sad | sadness |
| 5 | Surprise | surprise |
| 6 | Neutral | neutral |

---

## 📚 References

- **ConvNeXt-V2**: Woo et al., [arXiv:2301.00808](https://arxiv.org/abs/2301.00808)
- **FER2013**: [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- **RAF-DB**: [whdeng.cn](http://www.whdeng.cn/raf/model1.html)
- **AffectNet**: [mohammadmahoor.com](http://mohammadmahoor.com/affectnet/)
- **timm**: [github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)

---

## 📝 License

Research use only. Please cite the original dataset papers and ConvNeXt-V2 when publishing results.

---

**Last Updated**: 2026-07-09 (Experiment `main_2026-07-08_15-18-19`)
**Framework**: PyTorch + timm
**Model**: ConvNeXt-V2 Base (ImageNet-22k pretrained)
**Achieved Accuracy**: 89.4% validation accuracy on unified FER2013 + RAF-DB + AffectNet (27 epochs)
**Total Training Time**: ~11.4 hours (27 × ~25 min/epoch on GPU)
