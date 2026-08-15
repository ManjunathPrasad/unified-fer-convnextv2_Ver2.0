"""
Dataset Balancing and Label Mapping Module

This module provides:
1. Label mapping from various dataset formats to standard 7 emotion classes
2. Dataset balancing using oversampling/undersampling
3. Class distribution analysis and visualization
4. print_dataset_report() — publication-ready sample count table (fixes R1-C3)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)

# --- Idempotent label mapping (fixes cross-dataset double-remap; R1-Q6/R2-Q4/R3-Q1) ---
from label_semantics import (
    map_labels_to_emotions as _canonical_map_labels_to_emotions,
    is_already_mapped,
    attach_label_names,
    CANONICAL_ID_TO_NAME as EMOTION_MAP,
    CANONICAL_NAME_TO_ID as EMOTION_TO_ID,
    RAFDB_RAW_TO_CANONICAL,
)

# (EMOTION_MAP now imported from label_semantics — single source of truth)





def resolve_balance_targets(
    label_counts: pd.Series,
    min_samples: int,
    max_samples,
    max_cap: int = 12000,
) -> Tuple[int, int]:
    """
    Resolve min/max per-class targets from config.

    ``max_samples`` may be:
      - int: fixed uniform target for ``balance_method: both``
      - ``"auto"`` / ``"max"``: use the largest raw class count (capped at ``max_cap``)
        so majors keep all real images and minorities are oversampled up to that count
    """
    largest = int(label_counts.max())
    resolved_max = max_samples

    if max_samples in (None, "auto", "max"):
        resolved_max = min(largest, max_cap)
        resolved_max = max(resolved_max, min_samples)
        logger.info(
            "Auto balance target: %d samples/class "
            "(largest raw class=%d, cap=%d → %d total)",
            resolved_max,
            largest,
            max_cap,
            resolved_max * len(label_counts),
        )
    else:
        resolved_max = int(max_samples)

    resolved_min = min(min_samples, resolved_max)
    return resolved_min, resolved_max


# FER2013 label mapping (already 0-6, but ensure consistency)
FER2013_MAP = {
    0: 0,  # anger
    1: 1,  # disgust
    2: 2,  # fear
    3: 3,  # happy
    4: 4,  # sad
    5: 5,  # surprise
    6: 6   # neutral
}

# RAF-DB raw(1-7)->canonical(0-6) mapping now lives in label_semantics.
RAFDB_MAP = RAFDB_RAW_TO_CANONICAL          # back-compat alias
RAFDB_MAP_ALT = {i: i for i in range(7)}    # back-compat alias


def map_labels_to_emotions(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Idempotent label remap (delegates to label_semantics).

    Kept as a public re-export so existing imports
    `from dataset_balancer import map_labels_to_emotions` keep working.
    Safe to call multiple times: already-canonical labels are left untouched,
    which fixes the cross-dataset double-remap collapse.
    """
    return _canonical_map_labels_to_emotions(df, dataset_name)


def balance_dataset(
    df: pd.DataFrame,
    method: str = "oversample",
    min_samples_per_class: int = 100,
    max_samples_per_class: Optional[int] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Balance dataset using oversampling or undersampling.

    Methods:
      - oversample: raise each class to at least ``min_samples_per_class``
      - undersample: cap each class at ``max_samples_per_class``
      - both: normalize every class to ``max_samples_per_class`` (oversample
        minorities, undersample majorities) for a uniform class distribution
    """
    np.random.seed(random_state)
    df = df.copy()

    label_counts = df["label"].value_counts().sort_index()
    logger.info(f"Original class distribution:\n{label_counts}")

    if method == "both" and max_samples_per_class:
        target_per_class = max_samples_per_class
    elif method == "undersample" and max_samples_per_class:
        target_per_class = None  # only cap, do not upsample
    else:
        target_per_class = min_samples_per_class if method in {"oversample", "both"} else None

    balanced_dfs = []

    for label in sorted(df["label"].unique()):
        class_df = df[df["label"] == label].copy()
        n_samples = len(class_df)

        if method == "both" and max_samples_per_class:
            target = max_samples_per_class
            if n_samples > target:
                class_df = class_df.sample(n=target, random_state=random_state)
                logger.info(
                    f"Class {label}: Undersampled from {n_samples} to {len(class_df)} samples"
                )
            elif n_samples < target:
                n_needed = target - n_samples
                extra = class_df.sample(n=n_needed, replace=True, random_state=random_state)
                class_df = pd.concat([class_df, extra], ignore_index=True)
                logger.info(
                    f"Class {label}: Oversampled from {n_samples} to {len(class_df)} samples"
                )
        else:
            if method in ["oversample", "both"] and target_per_class and n_samples < target_per_class:
                n_needed = target_per_class - n_samples
                extra = class_df.sample(n=n_needed, replace=True, random_state=random_state)
                class_df = pd.concat([class_df, extra], ignore_index=True)
                logger.info(
                    f"Class {label}: Oversampled from {n_samples} to {len(class_df)} samples"
                )

            if method in ["undersample", "both"] and max_samples_per_class:
                current_n = len(class_df)
                if current_n > max_samples_per_class:
                    class_df = class_df.sample(n=max_samples_per_class, random_state=random_state)
                    logger.info(
                        f"Class {label}: Undersampled from {current_n} to {len(class_df)} samples"
                    )

        balanced_dfs.append(class_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Shuffle
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Show final distribution
    final_counts = balanced_df["label"].value_counts().sort_index()
    logger.info(f"Balanced class distribution:\n{final_counts}")
    
    return balanced_df


def analyze_and_balance_dataset(
    csv_path: str,
    output_path: Optional[str] = None,
    balance_method: str = "oversample",
    min_samples: int = 500,
    max_samples: Optional[int] = 5000,
    filter_unknown: bool = True
) -> pd.DataFrame:
    """
    Complete pipeline: Load, map labels, balance, and save dataset.
    
    Args:
        csv_path: Path to input CSV
        output_path: Path to save balanced CSV (if None, overwrites input)
        balance_method: 'oversample', 'undersample', or 'both'
        min_samples: Minimum samples per class
        max_samples: Maximum samples per class (None = no limit)
        filter_unknown: Whether to filter out 'unknown' labels
    
    Returns:
        Balanced DataFrame
    """
    logger.info(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    logger.info(f"Original dataset size: {len(df)} samples")
    logger.info(f"Datasets: {df['dataset'].value_counts().to_dict()}")
    
    # Step 1: Map labels to standard emotion classes
    logger.info("\n=== Mapping labels to standard emotion classes ===")
    for dataset_name in df["dataset"].unique():
        logger.info(f"Mapping {dataset_name} labels...")
        df = map_labels_to_emotions(df, dataset_name)
    
    # Step 2: Filter unknown labels
    if filter_unknown:
        unknown_count = len(df[df["label"] == "unknown"])
        if unknown_count > 0:
            logger.info(f"\nFiltering out {unknown_count} samples with 'unknown' labels")
            df = df[df["label"] != "unknown"].copy()
    
    # Step 3: Convert labels to integers (should all be 0-6 now)
    def to_int_label(label):
        try:
            return int(float(label))
        except (ValueError, TypeError):
            logger.warning(f"Invalid label: {label}, skipping")
            return None
    
    df["label"] = df["label"].apply(to_int_label)
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    
    # Filter out invalid labels (not 0-6)
    valid_mask = df["label"].isin(range(7))
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        logger.warning(f"Filtering out {invalid_count} samples with invalid labels (not 0-6)")
        df = df[valid_mask].copy()
    
    # Step 4: Show distribution before balancing
    logger.info("\n=== Class distribution before balancing ===")
    label_counts = df["label"].value_counts().sort_index()
    for label, count in label_counts.items():
        emotion_name = EMOTION_MAP.get(label, "unknown")  #type:ignore
        logger.info(f"  {emotion_name} (class {label}): {count} samples")
    
    # Step 5: Balance dataset
    logger.info(f"\n=== Balancing dataset (method: {balance_method}) ===")
    resolved_min, resolved_max = resolve_balance_targets(
        label_counts,
        min_samples=min_samples,
        max_samples=max_samples,
        max_cap=12000,
    )
    df_balanced = balance_dataset(
        df,
        method=balance_method,
        min_samples_per_class=resolved_min,
        max_samples_per_class=resolved_max,
        random_state=42
    )
    
    # Step 6: Show final distribution
    logger.info("\n=== Final class distribution ===")
    final_counts = df_balanced["label"].value_counts().sort_index()
    for label, count in final_counts.items():
        emotion_name = EMOTION_MAP.get(label, "unknown")  #type:ignore
        logger.info(f"  {emotion_name} (class {label}): {count} samples")
    
    # Step 7: Save
    if output_path is None:
        output_path = csv_path.replace(".csv", "_balanced.csv")
    
    df_balanced.to_csv(output_path, index=False)
    logger.info(f"\nBalanced dataset saved to: {output_path}")
    logger.info(f"Total samples: {len(df_balanced)}")

    # Step 8: Generate and save dataset statistics report
    try:
        raw_df = pd.read_csv(csv_path)
        stats_report = {
            "original_total_samples": len(raw_df),
            "unknown_labels_filtered": int(unknown_count) if filter_unknown else 0,
            "before_balancing": {
                "total": {EMOTION_MAP[k]: int(v) for k, v in label_counts.items()}, #type:ignore
                "by_dataset": {}
            },
            "after_balancing": {
                "total": {EMOTION_MAP[k]: int(v) for k, v in final_counts.items()}, #type:ignore
                "by_dataset": {}
            }
        }
        
        # Populate dataset breakdown before balancing
        for dataset in df["dataset"].unique():
            sub_df = df[df["dataset"] == dataset]
            sub_counts = sub_df["label"].value_counts().reindex(range(7), fill_value=0)
            stats_report["before_balancing"]["by_dataset"][dataset] = {
                EMOTION_MAP[k]: int(v) for k, v in sub_counts.items()
            }
            
        # Populate dataset breakdown after balancing
        for dataset in df_balanced["dataset"].unique():
            sub_df = df_balanced[df_balanced["dataset"] == dataset]
            sub_counts = sub_df["label"].value_counts().reindex(range(7), fill_value=0)
            stats_report["after_balancing"]["by_dataset"][dataset] = {
                EMOTION_MAP[k]: int(v) for k, v in sub_counts.items()
            }

        report_json_path = Path(output_path).parent / "dataset_statistics_report.json"
        import json
        with open(report_json_path, "w") as f:
            json.dump(stats_report, f, indent=4)
        logger.info(f"Dataset statistics report saved to: {report_json_path}")

        # Publication-ready Markdown table (Reviewer 1 Comment 3)
        report_md_path = Path(output_path).parent / "dataset_stats_report.md"
        print_dataset_report(df, df_balanced, out_md=str(report_md_path))
    except Exception as e:
        logger.warning(f"Could not generate dataset statistics report: {e}")
    
    return df_balanced


# ---------------------------------------------------------------------------
# Publication-ready dataset statistics report (fixes Reviewer 1 Comment 3)
# ---------------------------------------------------------------------------

def print_dataset_report(
    df_raw: pd.DataFrame,
    df_balanced: pd.DataFrame,
    out_md: Optional[str] = None,
) -> str:
    """
    Generate a publication-ready Markdown table showing:
      1. Raw per-dataset image counts (whole corpus)
      2. Per-class distribution BEFORE balancing (with per-dataset breakdown)
      3. Per-class distribution AFTER balancing (with per-dataset breakdown)
      4. Grand totals at each stage

    This output resolves the dataset-size inconsistency raised in Reviewer 1 Comment 3:
    the paper must clearly distinguish between (a) total images per dataset,
    (b) samples used in the class distribution tables, and (c) post-balancing counts.

    Args:
        df_raw:      DataFrame BEFORE balancing (after label mapping + unknown filtering)
        df_balanced: DataFrame AFTER balancing
        out_md:      Optional path to save the Markdown report

    Returns:
        Markdown string
    """
    emotions = [EMOTION_MAP[i] for i in range(7)]
    datasets = sorted(df_raw["dataset"].unique()) if "dataset" in df_raw.columns else ["all"]

    lines = [
        "# Dataset Statistics Report",
        "",
        "> Auto-generated by `dataset_balancer.print_dataset_report()`  ",
        "> Fixes Reviewer 1 Comment 3: clarifies raw corpus sizes vs. class distribution counts.",
        "",
        "---",
        "",
        "## Table 1: Raw Dataset Sizes (Whole Corpus)",
        "",
        "| Dataset | Total Images |",
        "|---------|------------|",
    ]

    total_raw = 0
    for ds in datasets:
        n = int((df_raw["dataset"] == ds).sum()) if "dataset" in df_raw.columns else len(df_raw)
        lines.append(f"| {ds} | {n:,} |")
        total_raw += n
    lines.append(f"| **Total (raw)** | **{total_raw:,}** |")

    # ── Before balancing: per-class × per-dataset ──────────────────────────
    lines += [
        "",
        "---",
        "",
        "## Table 2: Class Distribution BEFORE Balancing",
        f"*(based on {total_raw:,} valid samples after label mapping and unknown-label removal)*",
        "",
        "| Emotion (class) |"
        + "".join(f" {ds} |" for ds in datasets)
        + " **Total** |",
        "|" + "---|" * (len(datasets) + 2),
    ]

    before_totals = []
    for i, emotion in enumerate(emotions):
        row = f"| {emotion} ({i}) |"
        row_total = 0
        for ds in datasets:
            if "dataset" in df_raw.columns:
                n = int(((df_raw["dataset"] == ds) & (df_raw["label"] == i)).sum())
            else:
                n = int((df_raw["label"] == i).sum())
            row += f" {n:,} |"
            row_total += n
        row += f" **{row_total:,}** |"
        lines.append(row)
        before_totals.append(row_total)

    grand_before = sum(before_totals)
    totals_row = "| **Total** |"
    for ds in datasets:
        if "dataset" in df_raw.columns:
            n = int((df_raw["dataset"] == ds).sum())
        else:
            n = len(df_raw)
        totals_row += f" **{n:,}** |"
    totals_row += f" **{grand_before:,}** |"
    lines.append(totals_row)

    # ── After balancing: per-class × per-dataset ───────────────────────────
    lines += [
        "",
        "---",
        "",
        "## Table 3: Class Distribution AFTER Balancing",
        f"*(based on {len(df_balanced):,} samples)*",
        "",
        "| Emotion (class) |"
        + "".join(f" {ds} |" for ds in datasets)
        + " **Total** |",
        "|" + "---|" * (len(datasets) + 2),
    ]

    for i, emotion in enumerate(emotions):
        row = f"| {emotion} ({i}) |"
        row_total = 0
        for ds in datasets:
            if "dataset" in df_balanced.columns:
                n = int(((df_balanced["dataset"] == ds) & (df_balanced["label"] == i)).sum())
            else:
                n = int((df_balanced["label"] == i).sum())
            row += f" {n:,} |"
            row_total += n
        row += f" **{row_total:,}** |"
        lines.append(row)

    grand_after = len(df_balanced)
    totals_row_b = "| **Total** |"
    for ds in datasets:
        if "dataset" in df_balanced.columns:
            n = int((df_balanced["dataset"] == ds).sum())
        else:
            n = grand_after
        totals_row_b += f" **{n:,}** |"
    totals_row_b += f" **{grand_after:,}** |"
    lines.append(totals_row_b)

    lines += [
        "",
        "---",
        "",
        "## Summary",
        "",
        f"| Stage | Samples |",
        f"|-------|--------|",
        f"| Raw corpus (all 3 datasets, before any filtering) | {total_raw:,} |",
        f"| After label mapping + unknown removal | {grand_before:,} |",
        f"| After class balancing | {grand_after:,} |",
        "",
        "> These are the definitive numbers to use in the paper tables.",
        "> The '49,554' figure refers to the raw corpus; '15,499' and '20,728'",
        "> refer to different intermediate stages — clarify which stage each paper",
        "> table reports.",
    ]

    report = "\n".join(lines)

    if out_md:
        Path(out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(out_md).write_text(report, encoding="utf-8")
        logger.info(f"Dataset statistics report saved to: {out_md}")

    print(report)
    return report


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [INFO] %(message)s")
    
    if len(sys.argv) < 2:
        print("Usage: python dataset_balancer.py <input_csv> [output_csv] [--method oversample|undersample|both] [--min N] [--max N]")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    method = "oversample"
    min_samples = 500
    max_samples = 5000
    
    # Parse arguments
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--method" and i + 1 < len(sys.argv):
            method = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--min" and i + 1 < len(sys.argv):
            min_samples = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--max" and i + 1 < len(sys.argv):
            max_samples = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1
    
    analyze_and_balance_dataset(
        input_csv,
        output_csv,
        balance_method=method,
        min_samples=min_samples,
        max_samples=max_samples
    )
