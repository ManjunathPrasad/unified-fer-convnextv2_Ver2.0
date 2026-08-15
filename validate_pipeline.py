#!/usr/bin/env python3
"""Quick data-pipeline validation (no GPU / training required)."""

from pathlib import Path

import pandas as pd

from dataset_balancer import EMOTION_MAP, analyze_and_balance_dataset, print_dataset_report


def _resolve_dataset_dir() -> Path:
    """Auto-detect nested Dataset/Dataset layout."""
    root = Path(__file__).parent
    candidates = [
        root / "emotion_q1_framework" / "Dataset" / "Dataset",
        root / "emotion_q1_framework" / "Dataset",
        root / "Dataset" / "Dataset",
        root / "Dataset",
    ]
    for c in candidates:
        if (c / "fer2013.csv").exists() or (c / "images").exists():
            return c
    return candidates[0]


def main() -> int:
    base = _resolve_dataset_dir()
    unified = base / "prepared" / "unified_dataset.csv"
    images = base / "images"

    if not unified.exists():
        print(f"FAIL: missing {unified}")
        print("  Run: python master_experiment.py --config configs/kaggle_2gpu.yaml")
        print("  Or rebuild CSVs from existing images via dataset_preparation.py")
        return 1

    df = pd.read_csv(unified)
    labels = pd.to_numeric(df["label"], errors="coerce")
    invalid = labels.isna() | ~labels.isin(range(7))
    if invalid.any():
        print(f"FAIL: {invalid.sum()} rows have invalid labels in unified CSV")
        return 1

    fer = df[df["dataset"] == "FER2013"]
    if len(fer) == 0:
        print("WARN: no FER2013 rows in unified dataset")
    elif not fer["label"].isin(range(7)).all():
        print("FAIL: FER2013 labels still corrupted (expected 0-6)")
        return 1

    missing = (~df["image_path"].apply(lambda p: (images / str(p)).exists())).sum()
    print(f"Unified samples: {len(df)} | missing images: {missing}")
    print(f"Datasets: {df['dataset'].value_counts().to_dict()}")
    print(f"FER2013 label range: {sorted(fer['label'].unique()) if len(fer) else 'n/a'}")

    stats_md = base / "prepared" / "dataset_stats_report.md"
    if stats_md.exists():
        print(f"Dataset stats report: {stats_md} (exists)")
    else:
        print("WARN: dataset_stats_report.md not yet generated")

    temp = base / "prepared" / "validation_balanced.csv"
    balanced = analyze_and_balance_dataset(
        csv_path=str(unified),
        output_path=str(temp),
        balance_method="both",
        min_samples=500,
        max_samples=1000,
        filter_unknown=True,
    )
    counts = balanced["label"].value_counts().sort_index()
    print("Balanced smoke counts:", {EMOTION_MAP[k]: int(v) for k, v in counts.items()})
    print("PASS: data pipeline labels are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
