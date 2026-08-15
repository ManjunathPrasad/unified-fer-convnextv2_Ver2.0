# Notation and Symbol Reference

> This document defines all mathematical symbols and variables used in the paper.  
> Addresses **Reviewer 1 Comment 2** (missing notation list).

---

## 1. Dataset Notation

| Symbol | Definition |
|--------|-----------|
| $\mathcal{D}$ | Full unified dataset (FER2013 ∪ RAF-DB ∪ AffectNet) |
| $\mathcal{D}_{\text{train}}$ | Training split |
| $\mathcal{D}_{\text{val}}$ | Validation split (used for model selection) |
| $\mathcal{D}_{\text{test}}$ | Independent test split (used for final reporting only) |
| $N$ | Total number of training samples |
| $C$ | Number of emotion classes (C = 7) |
| $n_c$ | Number of samples in class $c$ |
| $\hat{n}_c$ | Target number of samples per class after balancing |

---

## 2. Model and Architecture

| Symbol | Definition |
|--------|-----------|
| $f_\theta$ | The full ConvNeXt-V2 model with parameters $\theta$ |
| $\phi_\theta$ | The backbone (feature extractor) sub-network |
| $g_\theta$ | The classification head (linear layer) |
| $\mathbf{x}$ | Input image tensor, shape $(3 \times H \times W)$ |
| $\mathbf{z}$ | Feature embedding from backbone, $\mathbf{z} = \phi_\theta(\mathbf{x})$ |
| $\mathbf{o}$ | Logit output, $\mathbf{o} = g_\theta(\mathbf{z}) \in \mathbb{R}^C$ |
| $\hat{y}$ | Predicted class, $\hat{y} = \arg\max_c \mathbf{o}_c$ |
| $y$ | Ground-truth class label, $y \in \{0, 1, \ldots, C-1\}$ |

---

## 3. Training Objectives

| Symbol | Definition |
|--------|-----------|
| $\mathcal{L}$ | Total training loss |
| $\mathcal{L}_{\text{CE}}$ | Standard cross-entropy loss |
| $\mathcal{L}_{\text{LS}}$ | Label-smoothed cross-entropy loss |
| $\varepsilon$ | Label smoothing factor ($\varepsilon = 0.1$) |
| $\mathcal{L}_{\text{FL}}$ | Focal loss (optional) |
| $\gamma$ | Focal loss focusing parameter ($\gamma = 2.0$) |

### Label-Smoothed Cross-Entropy
$$\mathcal{L}_{\text{LS}} = -\sum_{c=0}^{C-1} \tilde{y}_c \log \hat{p}_c, \quad \tilde{y}_c = (1-\varepsilon)\mathbf{1}[c=y] + \frac{\varepsilon}{C}$$

### Focal Loss
$$\mathcal{L}_{\text{FL}} = -(1 - \hat{p}_y)^\gamma \log \hat{p}_y$$

---

## 4. Augmentation

| Symbol | Definition |
|--------|-----------|
| $\lambda$ | MixUp / CutMix blending coefficient, $\lambda \sim \text{Beta}(\alpha, \alpha)$ |
| $\alpha_{\text{mix}}$ | MixUp Beta distribution parameter ($\alpha_{\text{mix}} = 0.4$) |
| $\alpha_{\text{cut}}$ | CutMix Beta distribution parameter ($\alpha_{\text{cut}} = 1.0$) |
| $p_{\text{mix}}$ | Probability of applying MixUp to a batch ($p_{\text{mix}} = 0.5$) |
| $p_{\text{cut}}$ | Probability of applying CutMix to a batch ($p_{\text{cut}} = 0.5$) |
| $p_{\text{erase}}$ | Random Erasing probability ($p_{\text{erase}} = 0.2$) |
| $p_{\text{fourier}}$ | FourierAugment frequency-perturbation probability ($= 0.4$) |
| $p_{\text{noise}}$ | ContrastiveNoiseAug probability ($= 0.3$) |
| $p_{\text{augmix}}$ | AugMixLite blending probability ($= 0.4$) |

### MixUp
$$\tilde{\mathbf{x}} = \lambda \mathbf{x}_i + (1-\lambda)\mathbf{x}_j, \quad \tilde{y} = \lambda y_i + (1-\lambda) y_j$$

### CutMix
A rectangular region $M$ is cut from $\mathbf{x}_j$ and pasted into $\mathbf{x}_i$.  
$$\tilde{\mathbf{x}} = \mathbf{x}_i \odot (1 - M) + \mathbf{x}_j \odot M, \quad \lambda = 1 - \frac{|M|}{HW}$$

---

## 5. Optimisation

| Symbol | Definition |
|--------|-----------|
| $\eta$ | Base learning rate ($\eta = 3 \times 10^{-5}$) |
| $\eta_{\text{head}}$ | Head learning rate during fine-tuning ($= \eta \times r_{\text{ft}}$) |
| $\eta_{\text{backbone}}$ | Backbone learning rate during fine-tuning ($= \eta_{\text{head}} \times r_{\text{bb}}$) |
| $r_{\text{ft}}$ | Fine-tune LR ratio ($r_{\text{ft}} = 0.3$) |
| $r_{\text{bb}}$ | Backbone-to-head LR ratio ($r_{\text{bb}} = 0.1$) |
| $\lambda_{\text{wd}}$ | Weight decay ($\lambda_{\text{wd}} = 0.05$) |
| $T_{\text{warm}}$ | Number of warmup optimiser steps |
| $T_{\text{total}}$ | Total number of optimiser steps |
| $B$ | Mini-batch size ($B = 32$) |
| $G$ | Gradient accumulation steps ($G = 2$) |
| $B_{\text{eff}}$ | Effective batch size ($B_{\text{eff}} = B \times G = 64$) |

### Linear Warmup + Cosine Decay Schedule
$$\eta(t) = \begin{cases}
\eta \cdot \left(0.01 + 0.99 \cdot \frac{t}{T_{\text{warm}}}\right) & t \leq T_{\text{warm}} \\[6pt]
\eta_{\min} + (\eta - \eta_{\min}) \cdot \frac{1 + \cos\!\left(\pi \cdot \frac{t - T_{\text{warm}}}{T_{\text{total}} - T_{\text{warm}}}\right)}{2} & t > T_{\text{warm}}
\end{cases}$$

where $\eta_{\min} = 0.01 \cdot \eta$.

### Gradient Accumulation
Gradients are accumulated over $G$ mini-batches before each optimiser step:
$$\theta \leftarrow \theta - \eta \cdot \nabla_\theta \left( \frac{1}{G} \sum_{g=1}^{G} \mathcal{L}(\mathbf{x}^{(g)}, y^{(g)}) \right)$$

---

## 6. Exponential Moving Average (EMA)

| Symbol | Definition |
|--------|-----------|
| $\beta$ | EMA decay factor ($\beta = 0.9997$) |
| $\tilde{\theta}_t$ | EMA shadow weights at step $t$ |

$$\tilde{\theta}_t = \beta \cdot \tilde{\theta}_{t-1} + (1 - \beta) \cdot \theta_t$$

---

## 7. Explainability (Grad-CAM)

| Symbol | Definition |
|--------|-----------|
| $A^k$ | Activation map of the $k$-th feature channel in the target layer |
| $\alpha^k_c$ | Global-average-pooled gradient weight for class $c$, channel $k$ |
| $\text{CAM}_c$ | Class activation map for class $c$ |

$$\alpha^k_c = \frac{1}{Z} \sum_{i,j} \frac{\partial \mathbf{o}_c}{\partial A^k_{ij}}, \qquad \text{CAM}_c = \text{ReLU}\!\left(\sum_k \alpha^k_c A^k\right)$$

> **Note**: Grad-CAM is a **post-hoc visualisation** tool. It does NOT contribute any gradient signal or loss term during training. It is used exclusively for result interpretation.

---

## 8. Evaluation Metrics

| Symbol | Definition |
|--------|-----------|
| $\text{Acc}$ | Overall top-1 accuracy |
| $\text{F1}_{\text{macro}}$ | Macro-averaged F1 score (unweighted mean over classes) |
| $\text{F1}_{\text{weighted}}$ | Weighted F1 score (weighted by class support) |
| $\text{P}_{\text{macro}}$ | Macro-averaged precision |
| $\text{R}_{\text{macro}}$ | Macro-averaged recall |
| $\text{AUC}$ | Area under the ROC curve |

---

*Last updated: 2026-07-09 (Experiment `main_2026-07-08_15-18-19`). Corresponds to training code in `train_engine.py`, `augmentations.py`, `metrics_and_plots.py`.*
