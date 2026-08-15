"""
sample_accounting.py
==================================================================
Produces the auditable per-dataset sample-accounting summary required by:
  - Reviewer 3 Typo 5 : original pool -> excluded (corrupt/dup/unsupported label)
                        -> retained, per dataset, traceable.
  - Reviewer 1 Q3     : reconcile inconsistent dataset-size numbers.
  - Reviewer 3 Typo 1 : split-before-vs-after balancing, exact per-split counts.

It derives every number deterministically from the prepared CSVs and (if
present) the preparation manifest, so each retained total is reproducible.

Inputs (auto-discovered under --prepared_dir):
  fer2013_prepared.csv | rafdb_prepared.csv | affectnet_prepared.csv | unified_dataset.csv
Optional:
  --split_json  path to split_indices.json (from a training run) for the
                exact train/val/test counts per split.

Outputs (to --out_dir):
  sample_accounting.md
  sample_accounting.csv

Run:
  python sample_accounting.py \
      --prepared_dir emotion_q1_framework/Dataset/prepared \
      --out_dir Results_Q1/tables \
      [--split_json Results_Q1/exp_main_xxx/convnext_v2/split_indices.json]
==================================================================
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

RUN = "[RUN]"

DATASETS = [
    ("FER2013", ["fer2013_prepared.csv", "fer_prepared.csv"]),
    ("RAFDB", ["rafdb_prepared.csv", "raf_prepared.csv", "rafdb.csv"]),
    ("AffectNet", ["affectnet_prepared.csv", "affnet_prepared.csv"]),
]


def _find(prep_dir: Path, names: List[str]) -> Optional[Path]:
    for n in names:
        p = prep_dir / n
        if p.exists():
            return p
    return None


def _manifest(prep_dir: Path) -> Dict:
    """Optional preparation manifest with exclusion counts."""
    for name in ["preparation_manifest.json", "prepare_stats.json"]:
        p = prep_dir / name
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
    return {}


def account_dataset(name: str, csv_path: Optional[Path], manifest: Dict) -> Dict:
    """Build the accounting row for one dataset."""
    entry = manifest.get(name, {}) if isinstance(manifest, dict) else {}
    retained = RUN
    partitions = ""
    if csv_path and csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            retained = len(df)
            # FER2013 partition breakdown if a Usage column exists
            if name == "FER2013" and "Usage" in df.columns:
                counts = df["Usage"].astype(str).str.strip().value_counts().to_dict()
                partitions = "; ".join(f"{k}={v}" for k, v in counts.items())
        except Exception as e:
            print(f"  (could not read {csv_path}: {e})")

    original = entry.get("original_pool", RUN)
    excluded = entry.get("excluded", RUN)
    # If we know original and retained, derive excluded
    if excluded == RUN and isinstance(original, int) and isinstance(retained, int):
        excluded = original - retained

    return {
        "dataset": name,
        "original_pool": original,
        "excluded": excluded,
        "retained": retained,
        "partitions": partitions or entry.get("partitions", RUN),
        "notes": entry.get("notes", {
            "FER2013": "Grayscale; state train/public-test/private-test origin",
            "RAFDB": "7 basic emotions; official split",
            "AffectNet": "Manually-annotated subset; explain reduction from full pool",
        }.get(name, "")),
    }


def build(prepared_dir: str, split_json: Optional[str]) -> Dict:
    prep = Path(prepared_dir)
    manifest = _manifest(prep)

    rows = []
    total_retained = 0
    for name, names in DATASETS:
        csv_path = _find(prep, names)
        row = account_dataset(name, csv_path, manifest)
        rows.append(row)
        if isinstance(row["retained"], int):
            total_retained += row["retained"]

    # Unified total (prefer the actual unified CSV)
    unified_csv = _find(prep, ["unified_dataset.csv"])
    unified_total = RUN
    if unified_csv and unified_csv.exists():
        try:
            unified_total = len(pd.read_csv(unified_csv))
        except Exception:
            pass
    if unified_total == RUN and total_retained:
        unified_total = total_retained

    # Split accounting
    split_rows = []
    if split_json and Path(split_json).exists():
        try:
            sj = json.loads(Path(split_json).read_text())
            for k in ["train", "val", "test"]:
                idx = sj.get(k) or sj.get(f"{k}_indices")
                if idx is not None:
                    split_rows.append({"split": k, "count": len(idx)})
        except Exception as e:
            print(f"  (could not parse split_json: {e})")

    return {"rows": rows, "unified_total": unified_total, "splits": split_rows}


def to_markdown(data: Dict) -> str:
    out = [
        "# Auditable Sample-Accounting Summary",
        "",
        "> Addresses Reviewer 3 (Typo 5), Reviewer 1 (Q3), Reviewer 3 (Typo 1).",
        "> Every retained total is traced from the original pool through each exclusion step.",
        "",
        "## Per-dataset accounting",
        "",
        "| Dataset | Original pool | Excluded (corrupt/dup/unsupported label) | Retained | Source partitions | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for r in data["rows"]:
        out.append(
            f"| {r['dataset']} | {r['original_pool']} | {r['excluded']} | "
            f"{r['retained']} | {r['partitions']} | {r['notes']} |"
        )
    out.append(f"| **Unified total** | — | — | **{data['unified_total']}** | — | Before split & balancing |")

    out += [
        "",
        "## Split accounting (split BEFORE balancing; balancing/augmentation on TRAIN only)",
        "",
        "| Split | Count | Balanced/augmented? |",
        "|---|---|---|",
    ]
    if data["splits"]:
        for s in data["splits"]:
            aug = "yes (balanced + augmented)" if s["split"] == "train" else "no (natural distribution)"
            out.append(f"| {s['split']} | {s['count']} | {aug} |")
    else:
        out += [
            f"| train | {RUN} | yes (balanced + augmented) |",
            f"| val | {RUN} | no (natural distribution) |",
            f"| test | {RUN} | no (natural distribution) |",
        ]
    out += [
        "",
        "**Pipeline order (explicit):** (1) split the unified images once at 70:15:15 with a "
        "fixed seed; (2) apply balancing + augmentation to the TRAIN split only; (3) leave "
        "val and test at their natural distribution.",
        "",
    ]
    return "\n".join(out)


def to_csv(data: Dict, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "original_pool", "excluded", "retained", "partitions", "notes"])
        for r in data["rows"]:
            w.writerow([r["dataset"], r["original_pool"], r["excluded"], r["retained"],
                        r["partitions"], r["notes"]])
        w.writerow(["Unified total", "", "", data["unified_total"], "", "Before split & balancing"])


def main():
    ap = argparse.ArgumentParser(description="Generate the auditable sample-accounting table.")
    ap.add_argument("--prepared_dir", default="emotion_q1_framework/Dataset/prepared")
    ap.add_argument("--out_dir", default="Results_Q1/tables")
    ap.add_argument("--split_json", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = build(args.prepared_dir, args.split_json)

    md_path = Path(args.out_dir) / "sample_accounting.md"
    csv_path = Path(args.out_dir) / "sample_accounting.csv"
    md_path.write_text(to_markdown(data))
    to_csv(data, str(csv_path))
    print(f"✓ Wrote {md_path}")
    print(f"✓ Wrote {csv_path}")


if __name__ == "__main__":
    main()
