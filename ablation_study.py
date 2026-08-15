# ablation_study.py
"""
Q1 Research Standards: Systematic ablation study runner.

Runs controlled experiments to evaluate the contribution of each component:
- Baseline (no augmentations)
- MixUp only
- CutMix only
- RandAugment only
- Novel augmentations only
- Full pipeline (all augmentations)

Reviewer fixes:
- Default seeds changed to [42, 123, 456] for valid variance statistics
- Pairwise statistical significance tests added (Welch t-test)
- "Explainability-aware training" clarified as post-hoc visualisation,
  NOT a training objective (no loss term involved)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import argparse
from scipy import stats as scipy_stats

from train_engine import run_training_pipeline
from utils import ensure_dir, generate_experiment_id, save_json
import torch


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------
ABLATION_CONFIGS = {
    "baseline": {
        "use_mixup": False,
        "use_cutmix": False,
        "randaugment": False,
        "use_novel_aug": False,
        "balance_dataset": True,
        "epochs_warm": 3,
        "description": "Baseline: no advanced augmentations (unified + balanced + progressive FT)",
    },
    "no_balancing": {
        "use_mixup": True,
        "use_cutmix": False,
        "randaugment": True,
        "use_novel_aug": True,
        "balance_dataset": False,
        "epochs_warm": 3,
        "description": "Ablation: unified dataset WITHOUT class balancing",
    },
    "no_progressive_ft": {
        "use_mixup": True,
        "use_cutmix": False,
        "randaugment": True,
        "use_novel_aug": True,
        "balance_dataset": True,
        "epochs_warm": 0,
        "description": "Ablation: no progressive fine-tuning (epochs_warm=0, single-phase FT)",
    },
    "fer2013_only": {
        "use_mixup": True,
        "use_cutmix": False,
        "randaugment": True,
        "use_novel_aug": True,
        "balance_dataset": True,
        "epochs_warm": 3,
        "csv_override": "fer2013_prepared.csv",
        "description": "Ablation: FER2013 only (no multi-dataset fusion)",
    },
    "mixup_only": {
        "use_mixup": True,
        "use_cutmix": False,
        "randaugment": False,
        "use_novel_aug": False,
        "balance_dataset": True,
        "epochs_warm": 3,
        "description": "MixUp only",
    },
    "cutmix_only": {
        "use_mixup": False,
        "use_cutmix": True,
        "randaugment": False,
        "use_novel_aug": False,
        "balance_dataset": True,
        "epochs_warm": 3,
        "description": "CutMix only",
    },
    "randaugment_only": {
        "use_mixup": False,
        "use_cutmix": False,
        "randaugment": True,
        "use_novel_aug": False,
        "balance_dataset": True,
        "epochs_warm": 3,
        "description": "RandAugment only",
    },
    "novel_aug_only": {
        "use_mixup": False,
        "use_cutmix": False,
        "randaugment": False,
        "use_novel_aug": True,
        "balance_dataset": True,
        "epochs_warm": 3,
        "description": (
            "Novel augmentations only "
            "(FourierAugment, ContrastiveNoiseAug, AugMixLite)"
        ),
    },
    "full_pipeline": {
        "use_mixup": True,
        "use_cutmix": False,
        "randaugment": True,
        "use_novel_aug": True,
        "balance_dataset": True,
        "epochs_warm": 3,
        "description": "Full pipeline: fusion + balancing + progressive FT + all augmentations",
    },
    # NOTE: "explainability_aware_training" is NOT an ablation variant.
    # Grad-CAM / landmark heatmaps are POST-HOC VISUALISATION ONLY and do
    # not contribute any loss term or gradient signal during training.
    # They are listed as an evaluation feature, not a training component.
}


# ---------------------------------------------------------------------------
# Statistical significance testing
# ---------------------------------------------------------------------------

def _pairwise_significance(
    per_seed_results: Dict[str, List[float]],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Welch two-sample t-test between every pair of ablation configurations.

    Returns a DataFrame with columns:
      config_a, config_b, mean_a, mean_b, delta, t_stat, p_value, significant
    """
    configs = list(per_seed_results.keys())
    rows = []
    for i in range(len(configs)):
        for j in range(i + 1, len(configs)):
            a, b = configs[i], configs[j]
            vals_a = per_seed_results[a]
            vals_b = per_seed_results[b]
            if len(vals_a) < 2 or len(vals_b) < 2:
                continue
            t_stat, p_val = scipy_stats.ttest_ind(vals_a, vals_b, equal_var=False)
            rows.append({
                "config_a": a,
                "config_b": b,
                "mean_a": float(np.mean(vals_a)),
                "mean_b": float(np.mean(vals_b)),
                "delta": float(np.mean(vals_a) - np.mean(vals_b)),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "significant": bool(p_val < alpha),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

def run_ablation_study(
    model_name: str,
    csv_path: str,
    images_root: str,
    num_classes: int,
    base_config: Dict,
    ablation_configs: Dict = None,
    seeds: List[int] = None,
    experiment_id: str = None,
) -> Dict:
    """
    Run systematic ablation study over multiple random seeds for variance reporting.

    Default seeds are [42, 123, 456] (3 runs) so that std > 0 and statistical
    significance tests are meaningful.  Pass seeds=[42] only for a quick smoke test.
    """
    if ablation_configs is None:
        ablation_configs = ABLATION_CONFIGS

    # --- DEFAULT: 3 seeds for publication-quality variance ---
    if seeds is None:
        seeds = [42, 123, 456]

    if experiment_id is None:
        experiment_id = generate_experiment_id("ablation")

    results: Dict[str, Dict] = {}
    # Track per-seed accuracy lists for significance testing
    per_seed_accuracy: Dict[str, List[float]] = {}

    print(f"\n{'='*70}")
    print(f"ABLATION STUDY: {model_name}")
    print(f"Experiment ID: {experiment_id}")
    print(f"Seeds: {seeds}  (n={len(seeds)} runs per config)")
    print(f"{'='*70}\n")

    for config_name, ablation_cfg in ablation_configs.items():
        print(f"\n{'='*70}")
        print(f"Running: {config_name}")
        print(f"Description: {ablation_cfg['description']}")
        print(f"{'='*70}\n")

        runs_acc:        List[float] = []
        runs_f1_macro:   List[float] = []
        runs_f1_weighted: List[float] = []
        runs_precision:  List[float] = []
        runs_recall:     List[float] = []

        for seed in seeds:
            print(f"--- Running seed {seed} ---")
            config = base_config.copy()
            config.update({k: v for k, v in ablation_cfg.items() if k not in ("description", "csv_override")})
            config["seed"] = seed
            config["experiment_id"] = f"{experiment_id}_{config_name}_seed{seed}"
            config["results_dir"] = base_config.get("results_dir", "Results_Q1")

            run_csv = csv_path
            csv_override = ablation_cfg.get("csv_override")
            if csv_override:
                override_path = Path(csv_path).parent / csv_override
                if override_path.exists():
                    run_csv = str(override_path)
                    print(f"  Using dataset override: {override_path.name}")
                else:
                    print(f"  ⚠ csv_override not found: {override_path}, using unified CSV")

            try:
                model, train_results = run_training_pipeline(
                    model_name=model_name,
                    csv_path=run_csv,
                    images_root=images_root,
                    num_classes=num_classes,
                    config=config,
                )

                runs_acc.append(float(train_results["acc"]))
                runs_f1_macro.append(float(train_results["f1_macro"]))
                runs_f1_weighted.append(float(train_results["f1_weight"]))
                runs_precision.append(float(train_results.get("precision_macro", 0.0)))
                runs_recall.append(float(train_results.get("recall_macro", 0.0)))

            except Exception as e:
                print(f"✗ {config_name} seed {seed} failed: {e}")
                continue

        if not runs_acc:
            print(f"✗ All seeds failed for {config_name}")
            results[config_name] = {"error": "All seeds failed"}
            continue

        per_seed_accuracy[config_name] = runs_acc

        # Compute mean and standard deviations
        n = len(runs_acc)
        results[config_name] = {
            "description": ablation_cfg["description"],
            "num_completed_seeds": n,
            "seeds_used": seeds[:n],
            "accuracy": float(np.mean(runs_acc)),
            "accuracy_std": float(np.std(runs_acc, ddof=1)) if n > 1 else 0.0,
            "f1_macro": float(np.mean(runs_f1_macro)),
            "f1_macro_std": float(np.std(runs_f1_macro, ddof=1)) if n > 1 else 0.0,
            "f1_weighted": float(np.mean(runs_f1_weighted)),
            "f1_weighted_std": float(np.std(runs_f1_weighted, ddof=1)) if n > 1 else 0.0,
            "precision_macro": float(np.mean(runs_precision)),
            "precision_macro_std": float(np.std(runs_precision, ddof=1)) if n > 1 else 0.0,
            "recall_macro": float(np.mean(runs_recall)),
            "recall_macro_std": float(np.std(runs_recall, ddof=1)) if n > 1 else 0.0,
        }

        print(
            f"✓ {config_name} ({n} seeds): "
            f"Acc={results[config_name]['accuracy']:.4f} "
            f"± {results[config_name]['accuracy_std']:.4f}, "
            f"F1={results[config_name]['f1_macro']:.4f} "
            f"± {results[config_name]['f1_macro_std']:.4f}"
        )

    # --- Pairwise statistical significance ---
    significance_df = _pairwise_significance(per_seed_accuracy)

    # Save ablation results
    results_dir = Path(base_config.get("results_dir", "Results_Q1")) / experiment_id / model_name
    ensure_dir(str(results_dir))

    save_json(results, str(results_dir / "ablation_results.json"))

    if not significance_df.empty:
        sig_path = results_dir / "ablation_significance_tests.csv"
        significance_df.to_csv(sig_path, index=False)
        print(f"  ✓ Statistical significance table saved: {sig_path.name}")
        # Also embed in results JSON
        results["_significance_tests"] = significance_df.to_dict(orient="records")
        save_json(results, str(results_dir / "ablation_results.json"))

    # Generate ablation study chart
    try:
        from metrics_and_plots import plot_ablation_study
        plot_ablation_study(
            results,
            results_dir / f"{model_name}_ablation_study.png",
            model_name,
        )
        print(f"  ✓ Ablation study chart saved")
    except Exception as e:
        print(f"  ⚠ Could not generate ablation chart: {e}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"ABLATION STUDY SUMMARY: {model_name}")
    print(f"{'='*70}")
    for config_name, result in results.items():
        if config_name.startswith("_") or "error" in result:
            continue
        print(
            f"{config_name:20s}: "
            f"Acc={result['accuracy']:.4f} ± {result['accuracy_std']:.4f}  "
            f"F1={result['f1_macro']:.4f} ± {result['f1_macro_std']:.4f}"
        )

    if not significance_df.empty:
        print(f"\n{'='*70}")
        print("PAIRWISE SIGNIFICANCE (Welch t-test, α=0.05)")
        print(f"{'='*70}")
        for _, row in significance_df.iterrows():
            sig_marker = "✓ sig." if row["significant"] else "  n.s."
            print(
                f"  {row['config_a']:20s} vs {row['config_b']:20s}: "
                f"Δ={row['delta']:+.4f}  p={row['p_value']:.4f}  {sig_marker}"
            )

    print(f"{'='*70}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study runner")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--csv", type=str, required=True, help="Path to unified dataset CSV")
    parser.add_argument("--images", type=str, required=True, help="Path to images root")
    parser.add_argument("--num_classes", type=int, default=7, help="Number of classes")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 456],
        help="Random seeds (≥2 required for meaningful variance and significance tests)",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs_warm", type=int, default=2)
    parser.add_argument("--epochs_ft", type=int, default=8)

    args = parser.parse_args()

    base_config = {
        "batch_size": args.batch_size,
        "num_workers": 4,
        "input_size": 256,
        "epochs_warm": args.epochs_warm,
        "epochs_ft": args.epochs_ft,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "results_dir": "Results_Q1",
        "deterministic": True,
        "use_amp": True,
        "label_smoothing": 0.1,
        "weight_decay": 1e-4,
    }

    results = run_ablation_study(
        model_name=args.model,
        csv_path=args.csv,
        images_root=args.images,
        num_classes=args.num_classes,
        base_config=base_config,
        seeds=args.seeds,
    )
    print("Ablation study completed!")
