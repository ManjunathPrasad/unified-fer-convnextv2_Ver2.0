# baseline_runner.py
"""
Train lightweight baseline models under the IDENTICAL protocol as the proposed
ConvNeXt-V2 framework, then evaluate on official standard test splits.

Addresses Reviewer 1 Comment 5 and Reviewer 2 Comments 1–2:
  - Literature SOTA numbers are NOT directly comparable.
  - These baselines ARE trained/evaluated with our exact YAML config, splits,
    augmentation policy, and benchmark_eval protocol.

Baselines: ResNet-18, ResNet-50, EfficientNet-B0, ViT-Small (all via timm).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

from train_engine import run_training_pipeline
from benchmark_eval import run_benchmark_eval, generate_prior_works_comparison
from utils import ensure_dir, generate_experiment_id, load_yaml, resolve_config_auto, save_json

DEFAULT_BASELINES = ["resnet18", "resnet50", "efficientnet_b0"]


def run_baselines_under_protocol(
    csv_path: str,
    images_root: str,
    fer2013_orig_csv: str,
    num_classes: int = 7,
    base_config: Optional[Dict] = None,
    baseline_models: Optional[List[str]] = None,
    experiment_id: Optional[str] = None,
) -> Dict:
    """
    Train each baseline with the same config as the main experiment, then
    evaluate on FER2013 / RAF-DB / AffectNet official test splits.
    """
    if base_config is None:
        base_config = {}
    if baseline_models is None:
        baseline_models = DEFAULT_BASELINES
    if experiment_id is None:
        experiment_id = generate_experiment_id("baselines")

    results_dir = Path(base_config.get("results_dir", "Results_Q1")) / experiment_id
    ensure_dir(str(results_dir))

    all_results = {}

    print(f"\n{'='*70}")
    print("BASELINE REPRODUCTION (identical protocol)")
    print(f"Experiment ID: {experiment_id}")
    print(f"Models: {baseline_models}")
    print(f"{'='*70}\n")

    for model_name in baseline_models:
        print(f"\n--- Training baseline: {model_name} ---")
        config = base_config.copy()
        config["experiment_id"] = f"{experiment_id}_{model_name}"
        config["deterministic"] = config.get("deterministic", True)

        try:
            _, train_results = run_training_pipeline(
                model_name=model_name,
                csv_path=csv_path,
                images_root=images_root,
                num_classes=num_classes,
                config=config,
            )

            model_dir = results_dir / model_name
            ckpt = model_dir / f"{model_name}_best_full.pth"
            if not ckpt.exists():
                ckpt = model_dir / f"{model_name}_best.pth"

            bench_out = model_dir / "benchmark_eval"
            bench_metrics = {}
            if ckpt.exists():
                bench_metrics = run_benchmark_eval(
                    checkpoint_path=str(ckpt),
                    images_root=images_root,
                    fer2013_orig_csv=fer2013_orig_csv,
                    num_classes=num_classes,
                    output_dir=str(bench_out),
                    model_name=model_name,
                    input_size=config.get("input_size_ft", config.get("input_size", 256)),
                    batch_size=config.get("batch_size", 32),
                    full_ckpt=ckpt.name.endswith("_full.pth"),
                )

            all_results[model_name] = {
                "unified_test_accuracy": float(train_results.get("acc", 0.0)),
                "unified_test_f1_macro": float(train_results.get("f1_macro", 0.0)),
                "benchmark_splits": bench_metrics,
                "protocol": "Reproduced (identical YAML config + standard test splits)",
            }
            print(f"  ✓ {model_name}: unified acc={train_results.get('acc', 0):.4f}")

        except Exception as e:
            print(f"  ✗ {model_name} failed: {e}")
            all_results[model_name] = {"error": str(e)}

    save_json(all_results, str(results_dir / "baseline_results.json"))

    # Merge into PRIOR_WORKS_COMPARISON.md
    ours_ckpt = None
    for m in base_config.get("models", ["convnext_v2"]):
        main_dir = Path(base_config.get("results_dir", "Results_Q1"))
        for exp in sorted(main_dir.glob("exp_main_*"), reverse=True):
            ck = exp / m / f"{m}_best_full.pth"
            if ck.exists():
                ours_ckpt = ck
                break
        if ours_ckpt:
            break

    generate_prior_works_comparison(
        our_benchmark_dir=str(results_dir),
        baseline_results=all_results,
        output_path=str(results_dir / "PRIOR_WORKS_COMPARISON.md"),
    )

    print(f"\n  ✓ Baseline results: {results_dir / 'baseline_results.json'}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baselines under identical protocol")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--fer_csv", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/kaggle_2gpu.yaml")
    parser.add_argument("--models", type=str, nargs="+", default=DEFAULT_BASELINES)
    parser.add_argument("--num_classes", type=int, default=7)
    args = parser.parse_args()

    cfg = resolve_config_auto(load_yaml(args.config))
    run_baselines_under_protocol(
        csv_path=args.csv,
        images_root=args.images,
        fer2013_orig_csv=args.fer_csv,
        num_classes=args.num_classes,
        base_config=cfg,
        baseline_models=args.models,
    )
