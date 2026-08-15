"""
label_semantics.py
==================================================================
SINGLE SOURCE OF TRUTH for emotion-label semantics across the three
source datasets (FER2013, RAF-DB, AffectNet) and the unified 7-class
scheme.

WHY THIS FILE EXISTS
--------------------
Reviewer 1 (Q6), Reviewer 2 (Q4) and Reviewer 3 (Q1) all flagged the
cross-dataset collapse:  RAF-DB -> ~90% macro-F1  but  FER2013 -> 10.69%
macro-F1 (near-random for 7 classes).

Root cause was a *double label remap*:

  1. `dataset_preparation.unify()` already calls
     `map_labels_to_emotions()` and writes an *already-mapped* unified CSV
     (RAF-DB re-indexed correctly to the canonical 0-6 scheme).
  2. `cross_dataset_eval.py` then read that unified CSV and called
     `map_labels_to_emotions()` on the RAF-DB slice a SECOND time.

Because the old RAF-DB remap checked the 1-7 table first, an
already-correct label of `1` (disgust) was re-mapped to `5` (surprise),
`3`->`1`, `6`->`0`, etc. The permutation applied twice scrambles the
RAF-DB labels, which is exactly the systematic corruption that produces
near-random cross-dataset accuracy while the (single-mapped) unified
number stays healthy.

THE FIX
-------
- One canonical scheme, defined once, here.
- The remap is now IDEMPOTENT: a DataFrame that has already been mapped
  (detected via the `label_mapped` flag column, or the presence of a
  valid `label_name` column) is returned unchanged.
- RAF-DB raw vs. canonical is disambiguated per-column (using the whole
  label column's value range) instead of per-value, so 1-6 are never
  silently treated as raw RAF-DB codes when the column is already 0-6.
==================================================================
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Canonical unified 7-class scheme (used everywhere downstream).
# This ordering matches FER2013's native ordering and AffectNet's
# name->id mapping used throughout the repo.
# ------------------------------------------------------------------
CANONICAL_ID_TO_NAME: Dict[int, str] = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}
CANONICAL_NAME_TO_ID: Dict[str, int] = {v: k for k, v in CANONICAL_ID_TO_NAME.items()}
# Common spelling variants -> canonical id
NAME_ALIASES: Dict[str, int] = {
    "anger": 0, "angry": 0,
    "disgust": 1, "disgusted": 1,
    "fear": 2, "afraid": 2, "fearful": 2,
    "happy": 3, "happiness": 3, "joy": 3,
    "sad": 4, "sadness": 4,
    "surprise": 5, "surprised": 5, "surprized": 5,
    "neutral": 6, "none": 6,
}

# ------------------------------------------------------------------
# RAF-DB raw label convention (1-7) -> canonical 0-6.
# RAF-DB basic-emotion label file uses:
#   1=Surprise 2=Fear 3=Disgust 4=Happiness 5=Sadness 6=Anger 7=Neutral
# ------------------------------------------------------------------
RAFDB_RAW_TO_CANONICAL: Dict[int, int] = {
    1: 5,  # surprise
    2: 2,  # fear
    3: 1,  # disgust
    4: 3,  # happiness
    5: 4,  # sadness
    6: 0,  # anger
    7: 6,  # neutral
}

# Marker column written once a DataFrame's labels are in canonical space.
MAPPED_FLAG = "label_mapped"


# ==================================================================
# Idempotency detection
# ==================================================================
def is_already_mapped(df: pd.DataFrame) -> bool:
    """
    True if this DataFrame's `label` column is already in canonical 0-6
    space and must NOT be remapped again.

    Detection order (any one is sufficient):
      1. Explicit MAPPED_FLAG column set to True.
      2. A `label_name` column whose values are all canonical names
         (this column is only written *after* mapping, by unify()).
    """
    if MAPPED_FLAG in df.columns:
        try:
            flag = df[MAPPED_FLAG]
            # NaN must NOT count as True. When another dataset's rows were
            # mapped first, concat leaves this column NaN for these rows, and
            # bool(nan) is True in numpy -- which previously caused an
            # unmapped dataset to be skipped and its out-of-range labels
            # (e.g. RAF-DB's raw 7) to be silently dropped.
            if flag.notna().all() and bool(flag.fillna(False).astype(bool).all()):
                return True
        except Exception:
            pass

    if "label_name" in df.columns:
        col = df["label_name"]
        # Only trust this signal when EVERY row carries a canonical name.
        if col.notna().all():
            names = col.astype(str).str.strip().str.lower().unique()
            if len(names) > 0 and all(n in CANONICAL_NAME_TO_ID for n in names):
                return True

    return False


def _looks_like_raw_rafdb(label_series: pd.Series) -> bool:
    """
    Decide, for the WHOLE RAF-DB column, whether it is raw 1-7 codes or
    already-canonical 0-6.

    Per-column decision (not per-value) is what makes this safe: if any
    label is 7, the column is unambiguously raw RAF-DB; if the column is
    entirely within 0-6, we treat it as already canonical and do NOT
    re-permute it.
    """
    vals = pd.to_numeric(label_series, errors="coerce").dropna().astype(int)
    if len(vals) == 0:
        return False
    vmin, vmax = int(vals.min()), int(vals.max())
    # Raw RAF-DB uses 1..7 and (in practice) includes a 7 (neutral).
    if vmax == 7:
        return True
    # If the whole column sits in 0..6, it is already canonical.
    if vmin >= 0 and vmax <= 6:
        return False
    # Anything else (e.g. 1..6 with no 0 and no 7) is ambiguous; default
    # to treating it as raw only when there is no 0 present, because
    # canonical FER labels routinely include 0 (anger).
    return vmin >= 1 and vmax <= 7


# ==================================================================
# The idempotent remap
# ==================================================================
def map_labels_to_emotions(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Map dataset-specific labels of ``dataset_name`` rows to the canonical
    0-6 scheme. IDEMPOTENT: calling it twice is a no-op on the second call.

    Args:
        df:           DataFrame with 'label' and 'dataset' columns.
        dataset_name: 'FER2013' | 'RAFDB' | 'AffectNet'.

    Returns:
        New DataFrame with canonical integer labels for that dataset's
        rows, invalid/unknown rows dropped, and MAPPED_FLAG set True.
    """
    df = df.copy()

    if "label" not in df.columns or "dataset" not in df.columns:
        raise ValueError("map_labels_to_emotions requires 'label' and 'dataset' columns")

    dataset_mask = df["dataset"] == dataset_name
    if not dataset_mask.any():
        return df

    sub = df[dataset_mask].copy()

    # ---- Idempotency guard -------------------------------------------------
    if is_already_mapped(sub):
        logger.info(
            "[label_semantics] %s already mapped to canonical scheme; skipping remap.",
            dataset_name,
        )
        # still guarantee int dtype + flag
        sub["label"] = pd.to_numeric(sub["label"], errors="coerce")
        sub = sub[sub["label"].notna() & sub["label"].isin(range(7))].copy()
        sub["label"] = sub["label"].astype(int)
        sub[MAPPED_FLAG] = True
        out = pd.concat([df[~dataset_mask], sub], ignore_index=True)
        return out

    name_upper = str(dataset_name).upper().replace("-", "")

    if name_upper == "FER2013":
        sub["label"] = pd.to_numeric(sub["label"], errors="coerce")
        before = len(sub)
        sub = sub[sub["label"].notna() & sub["label"].isin(range(7))].copy()
        if len(sub) < before:
            logger.info("FER2013: dropped %d rows with out-of-range labels", before - len(sub))
        sub["label"] = sub["label"].astype(int)

    elif name_upper in {"RAFDB", "RAF"}:
        raw = _looks_like_raw_rafdb(sub["label"])
        num = pd.to_numeric(sub["label"], errors="coerce")
        if raw:
            logger.info("RAF-DB: raw 1-7 codes detected -> remapping to canonical 0-6")
            sub["label"] = num.map(RAFDB_RAW_TO_CANONICAL)
        else:
            logger.info("RAF-DB: labels already canonical 0-6 -> no remap applied")
            sub["label"] = num
        sub = sub[sub["label"].notna() & sub["label"].isin(range(7))].copy()
        sub["label"] = sub["label"].astype(int)

    elif name_upper == "AFFECTNET":
        def _map_aff(v):
            s = str(v).strip().lower()
            if s in NAME_ALIASES:
                return NAME_ALIASES[s]
            try:
                iv = int(float(v))
                return iv if 0 <= iv <= 6 else None
            except (ValueError, TypeError):
                return None
        sub["label"] = sub["label"].apply(_map_aff)
        sub = sub[sub["label"].notna()].copy()
        sub["label"] = sub["label"].astype(int)

    else:
        # Unknown dataset: only accept already-canonical ints.
        sub["label"] = pd.to_numeric(sub["label"], errors="coerce")
        sub = sub[sub["label"].notna() & sub["label"].isin(range(7))].copy()
        sub["label"] = sub["label"].astype(int)

    sub[MAPPED_FLAG] = True

    out = pd.concat([df[~dataset_mask], sub], ignore_index=True)
    return out


def attach_label_names(df: pd.DataFrame) -> pd.DataFrame:
    """Add/refresh a canonical `label_name` column from integer labels."""
    df = df.copy()
    df["label_name"] = df["label"].map(CANONICAL_ID_TO_NAME)
    return df


# Backwards-compatible aliases so existing imports keep working.
EMOTION_MAP = CANONICAL_ID_TO_NAME
EMOTION_MAP_REVERSE = CANONICAL_NAME_TO_ID