# experiment_runner.py
"""
Q1 Research Standards: Multi-run experiment runner with statistical aggregation.

This script:
1. Runs experiments with multiple random seeds for robustness
2. Aggregates results statistically (mean ± std)
3. Performs significance testing between models
4. Generates publication-ready summary tables
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from scipy import stats

from train_engine import run_training_pipeline
from utils import ensure_dir, generate_experiment_id, save_json, load_yaml, resolve_config_auto
import torch


def run_multiple_seeds(
    model_name: str,
    csv_path: str,
    images_root: str,
    num_classes: int,
    base_config: Dict,
    seeds: List[int] = [42, 123, 456],
    experiment_id: str = None
) -> Dict:
    """
    Run training with multiple random seeds and aggregate results.
    
    Returns aggregated metrics with mean, std, and confidence intervals.
    """
    if experiment_id is None:
        experiment_id = generate_experiment_id()
    
    all_results = []
    all_metrics = []
    
    print(f"\n{'='*60}")
    print(f"Running {model_name} with {len(seeds)} different seeds")
    print(f"Experiment ID: {experiment_id}")
    print(f"{'='*60}\n")
    
    for seed_idx, seed in enumerate(seeds, 1):
        print(f"\n--- Run {seed_idx}/{len(seeds)}: Seed={seed} ---")
        
        config = base_config.copy()
        config["seed"] = seed
        config["experiment_id"] = f"{experiment_id}_seed{seed}"
        
        try:
            model, results = run_training_pipeline(
                model_name=model_name,
                csv_path=csv_path,
                images_root=images_root,
                num_classes=num_classes,
                config=config
            )
            
            metrics = {
                "seed": seed,
                "accuracy": float(results["acc"]),
                "f1_macro": float(results["f1_macro"]),
                "f1_weighted": float(results["f1_weight"]),
                "precision_macro": float(results.get("precision_macro", 0.0)),
                "recall_macro": float(results.get("recall_macro", 0.0)),
                "best_val_acc": float(results.get("best_val_acc", 0.0))
            }
            
            all_results.append(results)
            all_metrics.append(metrics)
            
            print(f"✓ Seed {seed}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")
            
        except Exception as e:
            print(f"✗ Seed {seed} failed: {e}")
            continue
    
    if not all_metrics:
        raise RuntimeError("All runs failed!")
    
    # Statistical aggregation
    df_metrics = pd.DataFrame(all_metrics)
    
    aggregated = {
        "model_name": model_name,
        "experiment_id": experiment_id,
        "num_runs": len(all_metrics),
        "seeds": seeds[:len(all_metrics)],
        "mean": {},
        "std": {},
        "min": {},
        "max": {},
        "ci_95": {}  # 95% confidence intervals
    }
    
    metric_keys = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro", "best_val_acc"]
    
    for key in metric_keys:
        if key in df_metrics.columns:
            values = df_metrics[key].values
            aggregated["mean"][key] = float(np.mean(values))
            aggregated["std"][key] = float(np.std(values))
            aggregated["min"][key] = float(np.min(values))
            aggregated["max"][key] = float(np.max(values))
            
            # 95% confidence interval (t-distribution)
            if len(values) > 1:
                sem = stats.sem(values)  # Standard error of the mean
                ci = stats.t.interval(0.95, len(values)-1, loc=np.mean(values), scale=sem)
                aggregated["ci_95"][key] = [float(ci[0]), float(ci[1])]
            else:
                aggregated["ci_95"][key] = [float(np.mean(values)), float(np.mean(values))]
    
    # Save individual runs
    results_dir = Path(base_config.get("results_dir", "Results_Q1")) / experiment_id / model_name
    ensure_dir(str(results_dir))
    
    save_json(all_metrics, str(results_dir / "individual_runs.json"))
    save_json(aggregated, str(results_dir / "aggregated_metrics.json"))
    
    print(f"\n{'='*60}")
    print(f"Aggregated Results for {model_name}:")
    print(f"{'='*60}")
    print(f"Runs: {len(all_metrics)}")
    print(f"Accuracy: {aggregated['mean']['accuracy']:.4f} ± {aggregated['std']['accuracy']:.4f}")
    print(f"F1-macro: {aggregated['mean']['f1_macro']:.4f} ± {aggregated['std']['f1_macro']:.4f}")
    print(f"95% CI (Accuracy): [{aggregated['ci_95']['accuracy'][0]:.4f}, {aggregated['ci_95']['accuracy'][1]:.4f}]")
    print(f"{'='*60}\n")

    # Automatically generate summary table
    summary_table = generate_summary_table(
        [aggregated],
        out_path=results_dir / "summary_table.csv",
    )

    return aggregated, all_metrics


def compare_models_statistically(
    aggregated_results: List[Dict],
    metric: str = "accuracy",
    significance_level: float = 0.05
) -> pd.DataFrame:
    """
    Perform statistical comparison between models using t-test.
    """
    n_models = len(aggregated_results)
    comparison_matrix = np.zeros((n_models, n_models))
    p_values = np.ones((n_models, n_models))
    
    model_names = [r["model_name"] for r in aggregated_results]
    
    # Load individual runs for each model
    all_runs = {}
    for agg_result in aggregated_results:
        model_name = agg_result["model_name"]
        experiment_id = agg_result["experiment_id"]
        results_dir = Path(agg_result.get("results_dir", "Results_Q1")) / experiment_id / model_name
        runs_file = results_dir / "individual_runs.json"
        
        if runs_file.exists():
            with open(runs_file, "r") as f:
                all_runs[model_name] = json.load(f)
    
    # Pairwise t-tests
    comparisons = []
    for i in range(n_models):
        for j in range(i+1, n_models):
            model_i = model_names[i]
            model_j = model_names[j]
            
            if model_i in all_runs and model_j in all_runs:
                runs_i = [r[metric] for r in all_runs[model_i]]
                runs_j = [r[metric] for r in all_runs[model_j]]
                
                if len(runs_i) > 1 and len(runs_j) > 1:
                    t_stat, p_val = stats.ttest_ind(runs_i, runs_j)
                    significant = p_val < significance_level
                    
                    mean_diff = np.mean(runs_i) - np.mean(runs_j)
                    
                    comparisons.append({
                        "model_1": model_i,
                        "model_2": model_j,
                        "mean_diff": float(mean_diff),
                        "p_value": float(p_val),
                        "significant": significant
                    })
                    
                    comparison_matrix[i, j] = mean_diff
                    p_values[i, j] = p_val
    
    if comparisons:
        return pd.DataFrame(comparisons)
    return pd.DataFrame()


def generate_summary_table(aggregated_results: List[Dict], out_path: Path):
    """Generate publication-ready summary table."""
    rows = []
    for result in aggregated_results:
        row = {
            "Model": result["model_name"],
            "Accuracy (mean±std)": f"{result['mean']['accuracy']:.4f}±{result['std']['accuracy']:.4f}",
            "F1-macro (mean±std)": f"{result['mean']['f1_macro']:.4f}±{result['std']['f1_macro']:.4f}",
            "Precision-macro": f"{result['mean']['precision_macro']:.4f}",
            "Recall-macro": f"{result['mean']['recall_macro']:.4f}",
            "Num Runs": result["num_runs"]
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nSummary table saved to: {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-run experiment runner")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--csv", type=str, required=True, help="Path to unified dataset CSV")
    parser.add_argument("--images", type=str, required=True, help="Path to images root")
    parser.add_argument("--num_classes", type=int, default=7, help="Number of classes")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456], help="Random seeds")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (e.g. configs/default.yaml). "
                             "CLI flags override YAML values.")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs_warm", type=int, default=None)
    parser.add_argument("--epochs_ft", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true", default=None,
                        help="Deterministic training (overrides YAML)")

    args = parser.parse_args()

    # ── Base config from YAML (if provided) ──────────────────────────────────
    if args.config:
        config = load_yaml(args.config)
        config = resolve_config_auto(config)  # expand "auto" values
        print(f"Loaded config from: {args.config}")
    else:
        config = {
            "batch_size": 32,
            "num_workers": 4,
            "input_size": 256,
            "epochs_warm": 2,
            "epochs_ft": 8,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "results_dir": "Results_Q1",
            "deterministic": True,
            "use_mixup": True,
            "mixup_alpha": 0.4,
            "randaugment": True,
            "use_amp": True,
            "log_csv": True,
            "save_full_checkpoint": True,
            "save_split_indices": True,
        }

    # ── CLI overrides (None means "not supplied by user") ────────────────────
    if args.batch_size  is not None: config["batch_size"]   = args.batch_size
    if args.epochs_warm is not None: config["epochs_warm"]  = args.epochs_warm
    if args.epochs_ft   is not None: config["epochs_ft"]    = args.epochs_ft
    if args.deterministic is not None: config["deterministic"] = args.deterministic

    aggregated, individual = run_multiple_seeds(
        model_name=args.model,
        csv_path=args.csv,
        images_root=args.images,
        num_classes=args.num_classes,
        base_config=config,
        seeds=args.seeds
    )

    print("Experiment completed successfully!")

