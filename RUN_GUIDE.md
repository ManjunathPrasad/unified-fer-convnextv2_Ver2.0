# Reviewer-Response Reproduction Guide

This repository backs the point-by-point response letter for **Paper ID 20263887**
(*A Unified Multi-Dataset FER Framework Using Progressive Fine-Tuning of ConvNeXt-V2*).
Every reviewer comment that requires an experiment maps to a concrete command
below and to the file it produces.

---

## 0. Setup

```bash
# Python 3.10+ recommended
pip install -r requirements.txt
```

Expected dataset layout (place the raw datasets here):

```
emotion_q1_framework/Dataset/
├── fer2013.csv            # Kaggle FER2013 CSV
├── af-db.zip              # RAF-DB (zip with images + label .txt)
└── AffectNet/             # AffectNet Train/ and labels.csv
```

`images/` and `prepared/` are created automatically by the preparation step.

---

## 1. Prove the cross-dataset fix (no GPU, no images needed)

**Reviewer comments:** R1-Q6, R2-Q4, R3-Q1 (the 10.69% FER2013 collapse).

```bash
python verify_label_fix.py
```

Reproduces the original double-remap bug on synthetic labels and confirms the
idempotent fix removes it. Exit code 0 = fixed. Read `CROSS_DATASET_FIX.md` for
the full root-cause write-up you can paste into the response.

---

## 2. Prepare the unified dataset (once)

**Reviewer comments:** R3-Typo1 (split order), R1-Q3 (count reconciliation).

```bash
python dataset_preparation.py
```

Writes `emotion_q1_framework/Dataset/prepared/`:
`fer2013_prepared.csv`, `rafdb_prepared.csv`, `affectnet_prepared.csv`,
`unified_dataset.csv` (labels canonicalised **once**, with the `label_mapped`
sentinel so nothing double-remaps them).

---

## 3. Generate the no-GPU tables (accounting + comparison + proof)

**Reviewer comments:** R3-Typo5, R1-Q3, R3-Typo1 (accounting); R2-Q1/Q2, R3-Q2, R1-Q5 (comparison).

```bash
python generate_all_tables.py \
    --prepared_dir emotion_q1_framework/Dataset/prepared \
    --out_dir Results_Q1/tables
```

Produces in `Results_Q1/tables/`:
- `sample_accounting.md` / `.csv` — auditable original→excluded→retained per dataset
- `comparison_table.md` / `.csv` — SOTA table with **literature vs reproduced** and
  **standard vs unified** columns, including ResEmoteNet, POSTER V2, SSFER, MSAFNet

You can also run these two individually:

```bash
python sample_accounting.py --prepared_dir emotion_q1_framework/Dataset/prepared --out_dir Results_Q1/tables
python comparison_table.py  --out_dir Results_Q1/tables
```

---

## 4. Train the main model (GPU)

**Reviewer comments:** R2-Q5 (full protocol), R3-Typo2 (val vs test separation).

```bash
python master_experiment.py --config configs/kaggle_2gpu.yaml
```

Writes a timestamped `Results_Q1/exp_main_*/convnext_v2/` folder containing
`convnext_v2_best.pth`, `metrics.json`, `split_indices.json`,
`training_log.csv`, confusion matrices, ROC/PR, and Grad-CAM figures.
All hyperparameters live in the YAML (see `EXPERIMENTAL_PROTOCOL.md`).

> Set `CKPT=Results_Q1/exp_main_XXXX/convnext_v2/convnext_v2_best.pth` below.

---

## 5. Standard per-dataset benchmark evaluation (GPU)

**Reviewer comments:** R2-Q3, R3-Q2 (standard protocols, not the unified number).

```bash
python benchmark_eval.py \
    --checkpoint "$CKPT" \
    --images  emotion_q1_framework/Dataset/images \
    --fer_csv emotion_q1_framework/Dataset/fer2013.csv \
    --output  Results_Q1/tables/benchmark
```

Produces per-dataset accuracy on the FER2013 test partition, official RAF-DB
split, and AffectNet validation protocol → `benchmark_results.json`.
Feed that JSON back into the comparison table:

```bash
python comparison_table.py --out_dir Results_Q1/tables \
    --benchmark_json Results_Q1/tables/benchmark/benchmark_results.json \
    --unified_json  Results_Q1/exp_main_XXXX/convnext_v2/metrics.json
```

---

## 6. Cross-dataset evaluation — both directions (GPU)

**Reviewer comments:** R1-Q6, R2-Q4, R3-Q1 (multi-direction + confusion + domain-shift).

```bash
python cross_dataset_eval.py \
    --csv    emotion_q1_framework/Dataset/prepared/unified_dataset.csv \
    --images emotion_q1_framework/Dataset/images
```

Produces, for every train→test direction, accuracy / macro-P / macro-R /
macro-F1, a confusion matrix PNG, and a domain-shift diagnosis
(`cross_dataset_results.json` + figures). With the label fix in place the
FER2013 direction should no longer collapse to ~10%.

---

## 7. Multi-seed ablation with significance (GPU)

**Reviewer comments:** R2-Q6, R3-Q3 (5 seeds, mean ± std, Welch t-test).

```bash
python ablation_study.py --model convnext_v2 \
    --csv    emotion_q1_framework/Dataset/prepared/unified_dataset.csv \
    --images emotion_q1_framework/Dataset/images \
    --seeds  42 123 456 789 1000
```

Produces per-configuration mean ± std for accuracy, macro-P, macro-R, macro-F1
and pairwise Welch t-test p-values vs the full model, plus per-seed logs.

Multi-seed **main** result (for the headline number's variance):

```bash
python experiment_runner.py --model convnext_v2 \
    --csv    emotion_q1_framework/Dataset/prepared/unified_dataset.csv \
    --images emotion_q1_framework/Dataset/images \
    --seeds  42 123 456 789 1000 --config configs/default.yaml
```

---

## 8. Reproduced baselines under our protocol (GPU, optional but recommended)

**Reviewer comment:** R2-Q2 (reproduce baselines or label as literature-reported).

```bash
python baseline_runner.py \
    --csv     emotion_q1_framework/Dataset/prepared/unified_dataset.csv \
    --images  emotion_q1_framework/Dataset/images \
    --fer_csv emotion_q1_framework/Dataset/fer2013.csv \
    --config  configs/kaggle_2gpu.yaml
```

---

## 9. One command for everything (after a checkpoint exists)

```bash
python generate_all_tables.py \
    --prepared_dir emotion_q1_framework/Dataset/prepared \
    --out_dir      Results_Q1/tables \
    --images       emotion_q1_framework/Dataset/images \
    --checkpoint   "$CKPT" \
    --unified_csv  emotion_q1_framework/Dataset/prepared/unified_dataset.csv \
    --fer_csv      emotion_q1_framework/Dataset/fer2013.csv \
    --split_json   Results_Q1/exp_main_XXXX/convnext_v2/split_indices.json \
    --run-heavy
```

---

## Comment → command → output map

| Reviewer comment | Command (section) | Output file |
|---|---|---|
| R3-Q1 / R1-Q6 / R2-Q4 cross-dataset collapse | §1 verify, §6 cross-dataset | `verify_label_fix` output; `cross_dataset_results.json` + confusion PNGs |
| R3-Q2 / R2-Q3 / R1-Q5 standard vs unified | §5 benchmark, §3 comparison | `benchmark_results.json`; `comparison_table.md` |
| R2-Q1 recent methods in table | §3 comparison | `comparison_table.md` (ResEmoteNet/POSTER V2/SSFER/MSAFNet rows) |
| R2-Q2 reproduce baselines | §8 baselines, §3 comparison | reproduced rows in `comparison_table.md` |
| R3-Q3 / R2-Q6 ablation variance + significance | §7 ablation | ablation table (mean ± std + p-values) + per-seed logs |
| R2-Q5 full protocol | §4 train | `EXPERIMENTAL_PROTOCOL.md` + `config.json` |
| R3-Typo1 / R1-Q3 counts & split order | §3 accounting | `sample_accounting.md` |
| R3-Typo5 sample accounting | §3 accounting | `sample_accounting.md` |
| R3-Typo2 val vs test | §4 train | `metrics.json` (separate val/test) |
| R1-Q1 figures, R1-Q2 notation, R1-Q4 batch, formatting E-4 | manuscript edits | (see manuscript) |
| R1-Q7 / E-3 code release | this repo | GitHub URL in manuscript |
| E-2 CoI / Author Contributions | — | `MANUSCRIPT_SECTIONS.md` |

---

## Notes

- The three source datasets are **not** redistributed here; download them from
  their original providers and place them as in §0.
- `[RUN]` cells in the generated tables mark values that require a completed
  training/eval run or a literature lookup; fill them, then paste the tables
  into the manuscript and the response letter's red-italic placeholders.
- The label-mapping fix (`label_semantics.py`) is the load-bearing change; run
  §1 first — it needs nothing but Python and pandas.
