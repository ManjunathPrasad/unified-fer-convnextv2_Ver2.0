# Cross-Dataset Generalization Fix — Root Cause & Resolution

**Paper ID 20263887** — addresses Reviewer 1 (Q6), Reviewer 2 (Q4), Reviewer 3 (Q1):
the FER2013 cross-dataset collapse (macro-F1 **10.69%**, near-random for 7 classes)
while RAF-DB stayed at ~90% macro-F1.

## Root cause: a double label remap (not real domain failure)

The 7-class emotion IDs use one canonical ordering everywhere downstream:

| id | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|----|---|---|---|---|---|---|---|
| name | anger | disgust | fear | happy | sad | surprise | neutral |

RAF-DB's raw basic-emotion label file uses a **different** 1-indexed order
(1=surprise, 2=fear, 3=disgust, 4=happy, 5=sad, 6=anger, 7=neutral), so it must be
remapped to the canonical scheme exactly once.

The bug was that the remap ran **twice**:

1. `dataset_preparation.unify()` mapped RAF-DB correctly and wrote an
   already-canonical unified CSV.
2. `cross_dataset_eval.py` (and, latently, the training loader) then called the
   same remap **again** on already-canonical labels. Because the old RAF-DB remap
   tested the raw 1–7 table *first*, a correct label of `1` (disgust) was
   re-permuted to `5` (surprise), `3`→`1`, `6`→`0`, and so on.

Applying the RAF-DB permutation twice scrambles the labels. That self-consistent
scramble leaves in-distribution RAF-DB numbers healthy but makes any comparison
against a correctly-labelled dataset (FER2013) collapse to near-random — exactly
the reported 10.69% macro-F1.

`verify_label_fix.py` reproduces this: 25/35 synthetic RAF-DB labels are scrambled
after a second remap.

## The fix

A single source of truth, `label_semantics.py`, with an **idempotent** remap:

- One canonical `id ↔ name` map; RAF-DB raw→canonical map defined once.
- `map_labels_to_emotions()` is now a no-op on already-mapped tables. It detects
  the mapped state via a `label_mapped` sentinel column and/or a canonical
  `label_name` column (both written by `unify()`).
- RAF-DB raw-vs-canonical is decided **per column** (using the whole label
  column's value range), not per value — so a canonical `0–6` column is never
  re-permuted just because values `1–6` also exist in the raw table.
- `dataset_balancer.map_labels_to_emotions` / `EMOTION_MAP` / `RAFDB_MAP` are kept
  as re-exports, so no other import breaks.
- `unify()` stamps `label_mapped=True` and a canonical `label_name` column.

## What to re-run (now unblocked)

```bash
# 1. Prove the fix (no GPU/images needed)
python verify_label_fix.py

# 2. Re-prepare the unified CSV (writes the mapped sentinel)
python dataset_preparation.py            # or via master_experiment.py prep stage

# 3. Cross-dataset, BOTH directions + domain-shift diagnosis + confusion matrices
python cross_dataset_eval.py \
    --csv  emotion_q1_framework/Dataset/prepared/unified_dataset.csv \
    --images emotion_q1_framework/Dataset/images

# 4. Standard per-dataset benchmark test splits (RAF-DB / FER2013 / AffectNet)
python benchmark_eval.py

# 5. Multi-seed ablation with mean ± std + Welch t-test significance
python ablation_study.py --seeds 42 123 456 \
    --csv emotion_q1_framework/Dataset/prepared/unified_dataset.csv \
    --images emotion_q1_framework/Dataset/images
```

## Expected effect on the numbers

After the fix, RAF-DB↔FER2013 and the AffectNet directions are evaluated against
**correctly-aligned** labels, so the FER2013 direction should rise from ~10.69%
macro-F1 to a value consistent with genuine (still non-trivial) domain shift rather
than a labelling artifact. Any residual gap is then a real domain-shift result to
report and discuss, not a bug — which is exactly what Reviewer 3 asked for. Report
the recomputed four-metric table (Acc / macro-P / macro-R / macro-F1) in both
directions plus the cross-dataset confusion matrices the script now emits.

**Note:** the generalization claim in the manuscript should still be stated as
"improved cross-dataset consistency with a documented FER2013 domain gap," not
"robust generalization," unless the recomputed numbers support the stronger wording.
