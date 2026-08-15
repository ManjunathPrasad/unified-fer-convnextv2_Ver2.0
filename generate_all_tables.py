"""
generate_all_tables.py
==================================================================
One command that regenerates EVERY table/figure the response letter
promises, so "produced by the released code" is literally true.

It orchestrates the individual generators and evaluators. Each step is
independent and skips gracefully if its inputs (a trained checkpoint,
prepared CSVs) are not present yet — printing exactly what to run first.

Steps and the reviewer comments they satisfy:
  1. verify_label_fix.py        -> proves the cross-dataset fix   (R1-Q6, R2-Q4, R3-Q1)
  2. sample_accounting.py       -> auditable counts               (R3-Typo5, R1-Q3, R3-Typo1)
  3. comparison_table.py        -> SOTA table (protocol-labelled) (R2-Q1/Q2, R3-Q2, R1-Q5)
  4. benchmark_eval.py          -> standard per-dataset results   (R2-Q3, R3-Q2)   [needs checkpoint]
  5. cross_dataset_eval.py      -> both-direction cross-dataset   (R1-Q6, R2-Q4, R3-Q1) [needs checkpoint]
  6. ablation_study.py          -> 5-seed ablation + t-test       (R2-Q6, R3-Q3)   [trains models]
  7. experiment_runner.py       -> multi-seed main result         (R3-Q3)          [trains models]

Run (tables that need no GPU/checkpoint):
  python generate_all_tables.py --prepared_dir emotion_q1_framework/Dataset/prepared \
      --out_dir Results_Q1/tables

Run everything (after you have a trained checkpoint):
  python generate_all_tables.py --prepared_dir emotion_q1_framework/Dataset/prepared \
      --out_dir Results_Q1/tables --images emotion_q1_framework/Dataset/images \
      --checkpoint Results_Q1/exp_main_xxx/convnext_v2/convnext_v2_best.pth \
      --unified_csv emotion_q1_framework/Dataset/prepared/unified_dataset.csv \
      --fer_csv emotion_q1_framework/Dataset/fer2013.csv --run-heavy
==================================================================
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PY = sys.executable


def _run(cmd: list, label: str) -> bool:
    print("\n" + "=" * 68)
    print(f"STEP: {label}")
    print("  $ " + " ".join(str(c) for c in cmd))
    print("=" * 68)
    try:
        subprocess.run(cmd, check=True)
        print(f"  ✓ {label} done")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ {label} failed (exit {e.returncode}) — see message above")
        return False
    except FileNotFoundError as e:
        print(f"  ✗ {label} could not start: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description="Regenerate all reviewer-response tables/figures.")
    ap.add_argument("--prepared_dir", default="emotion_q1_framework/Dataset/prepared")
    ap.add_argument("--out_dir", default="Results_Q1/tables")
    ap.add_argument("--images", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--unified_csv", default=None)
    ap.add_argument("--fer_csv", default=None)
    ap.add_argument("--split_json", default=None)
    ap.add_argument("--benchmark_json", default=None)
    ap.add_argument("--unified_json", default=None)
    ap.add_argument("--seeds", nargs="+", default=["42", "123", "456", "789", "1000"])
    ap.add_argument("--run-heavy", action="store_true",
                    help="Also run benchmark/cross-dataset/ablation (needs checkpoint + GPU).")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    here = Path(__file__).parent
    ok, skip = [], []

    # 1. Label-fix proof (no GPU/images) -------------------------------------
    if _run([PY, str(here / "verify_label_fix.py")], "1. Verify label-mapping fix"):
        ok.append("verify_label_fix")

    # 2. Sample accounting (no GPU) ------------------------------------------
    cmd = [PY, str(here / "sample_accounting.py"),
           "--prepared_dir", args.prepared_dir, "--out_dir", args.out_dir]
    if args.split_json:
        cmd += ["--split_json", args.split_json]
    if _run(cmd, "2. Sample-accounting table"):
        ok.append("sample_accounting")

    # 3. Comparison table (no GPU) -------------------------------------------
    cmd = [PY, str(here / "comparison_table.py"), "--out_dir", args.out_dir]
    if args.benchmark_json:
        cmd += ["--benchmark_json", args.benchmark_json]
    if args.unified_json:
        cmd += ["--unified_json", args.unified_json]
    if _run(cmd, "3. SOTA comparison table"):
        ok.append("comparison_table")

    # ---- heavy steps (need a trained checkpoint) ---------------------------
    if args.run_heavy:
        if args.checkpoint and args.images and args.fer_csv:
            cmd = [PY, str(here / "benchmark_eval.py"),
                   "--checkpoint", args.checkpoint, "--images", args.images,
                   "--fer_csv", args.fer_csv, "--output", str(Path(args.out_dir) / "benchmark")]
            if _run(cmd, "4. Standard per-dataset benchmark"):
                ok.append("benchmark_eval")
        else:
            skip.append("4. benchmark_eval (needs --checkpoint --images --fer_csv)")

        if args.unified_csv and args.images:
            cmd = [PY, str(here / "cross_dataset_eval.py"),
                   "--csv", args.unified_csv, "--images", args.images]
            if _run(cmd, "5. Cross-dataset (both directions)"):
                ok.append("cross_dataset_eval")
        else:
            skip.append("5. cross_dataset_eval (needs --unified_csv --images)")

        if args.unified_csv and args.images:
            cmd = [PY, str(here / "ablation_study.py"), "--model", "convnext_v2",
                   "--csv", args.unified_csv, "--images", args.images, "--seeds", *args.seeds]
            if _run(cmd, "6. Multi-seed ablation + significance"):
                ok.append("ablation_study")
        else:
            skip.append("6. ablation_study (needs --unified_csv --images)")
    else:
        skip += [
            "4. benchmark_eval  (pass --run-heavy + checkpoint)",
            "5. cross_dataset_eval (pass --run-heavy + checkpoint)",
            "6. ablation_study  (pass --run-heavy + checkpoint)",
        ]

    # ---- summary -----------------------------------------------------------
    print("\n" + "#" * 68)
    print("SUMMARY")
    print("#" * 68)
    print("Completed : " + (", ".join(ok) if ok else "(none)"))
    if skip:
        print("Skipped   :")
        for s in skip:
            print("   - " + s)
    print(f"\nTables written to: {args.out_dir}")
    print("Fill any remaining [RUN] cells from the heavy-step outputs, then paste")
    print("the tables into the manuscript and the response letter.")


if __name__ == "__main__":
    main()
