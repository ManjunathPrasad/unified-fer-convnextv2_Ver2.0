# master_experiment.py
"""
MASTER EXPERIMENT RUNNER – Q1 READY

This script:
 1. Prepares all datasets:
      - FER2013  (local CSV provided)
      - RAF-DB   (local ZIP)
      - AffectNet (local directory)
 2. Creates unified_dataset.csv
 3. Trains the ConvNeXt-V2 backbone end-to-end
 4. Runs ablation experiments
 5. Runs cross-dataset evaluation
"""

import os
from pathlib import Path
import pandas as pd
import torch

# Optimize PyTorch for CPU training
if not torch.cuda.is_available():
    # Set optimal number of threads for CPU
    # Use all available cores for computation, but leave some for data loading
    cpu_count = os.cpu_count() or 4
    torch.set_num_threads(max(1, cpu_count - 2))  # Leave 2 cores for data loading
    # Disable MKL threading to avoid oversubscription (PyTorch handles it)
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', str(max(1, cpu_count - 2)))

# IMPORT YOUR FRAMEWORK MODULES
from dataset_preparation import (
    prepare_fer2013,
    prepare_rafdb,
    prepare_affectnet,
    unify,
    rebuild_prepared_csv_from_images,
)

from train_engine import run_training_pipeline
from utils import ensure_dir, generate_experiment_id, load_yaml, resolve_config_auto, merge_train_config


# =========================================================================== #
# =============================== USER PATHS ================================= #
# =========================================================================== #

# Check if running in a Kaggle environment
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or "/kaggle/" in str(Path(__file__).resolve())

# Base directory for finding input datasets (read-only)
BASE_DIR = Path(__file__).parent / "emotion_q1_framework" if Path(__file__).parent.name != "emotion_q1_framework" else Path(__file__).parent
# Fallback to hardcoded path if structure is different
if not BASE_DIR.exists():
    BASE_DIR = Path.cwd() / "emotion_q1_framework"
if not BASE_DIR.exists():
    BASE_DIR = Path(__file__).parent

DATASET_DIR = BASE_DIR / "Dataset"
# Auto-detect nested structure if dataset files are present there
if (DATASET_DIR / "Dataset" / "fer2013.csv").exists() or (DATASET_DIR / "Dataset" / "af-db.zip").exists() or (DATASET_DIR / "Dataset" / "af-db").exists():
    DATASET_DIR = DATASET_DIR / "Dataset"

# Input dataset locations (read-only)
FER_CSV_IN = DATASET_DIR / "fer2013.csv"
RAF_ZIP_IN = DATASET_DIR / "af-db.zip"
AFFECTNET_DIR = DATASET_DIR / "AffectNet"

# Output base directory (must be writable)
if IS_KAGGLE:
    OUTPUT_BASE_DIR = Path("/kaggle/working/emotion_q1_framework")
else:
    OUTPUT_BASE_DIR = BASE_DIR

# Output folders
IMAGES_ROOT = OUTPUT_BASE_DIR / "Dataset" / "Dataset" / "images" if IS_KAGGLE else DATASET_DIR / "images"
PREPARED_DIR = OUTPUT_BASE_DIR / "Dataset" / "Dataset" / "prepared" if IS_KAGGLE else DATASET_DIR / "prepared"

ensure_dir(IMAGES_ROOT) #type:ignore
ensure_dir(PREPARED_DIR) #type:ignore


# =========================================================================== #
# ======================= 1. DATASET PREPARATION ============================= #
# =========================================================================== #

def _try_rebuild_csv(dataset_key: str, out_csv: Path, fer_orig: str = None) -> bool: #type:ignore
    """Rebuild prepared CSV from existing image folders when CSV is missing."""
    folder_map = {"fer2013": "FER2013", "rafdb": "RAFDB", "affectnet": "AffectNet"}
    ds_name = folder_map.get(dataset_key, dataset_key.upper())
    img_sub = IMAGES_ROOT / ds_name
    if not img_sub.exists() or not any(img_sub.rglob("*.jpg")):
        return False
    try:
        rebuild_prepared_csv_from_images(
            str(IMAGES_ROOT), ds_name, str(out_csv),
            fer_orig_csv=fer_orig if dataset_key == "fer2013" else None, #type:ignore
            raf_zip_path=str(RAF_ZIP_IN) if dataset_key == "rafdb" else None, #type:ignore
            affectnet_dir=str(AFFECTNET_DIR) if dataset_key == "affectnet" else None, #type:ignore
        )
        return out_csv.exists() and len(pd.read_csv(out_csv)) > 0
    except Exception as e:
        print(f"  [WARN] Rebuild CSV failed for {ds_name}: {e}")
        return False


def prepare_all_datasets(datasets: list = None): #type:ignore
    """Prepare specified datasets, continuing with available ones if some fail."""
    prepared_csvs = []
    
    # Normalize datasets list to lower-case if provided
    if datasets is not None:
        datasets = [d.lower() for d in datasets]
        print(f"Dataset preparation filter enabled: preparing only {datasets}")

    # 1. FER2013
    if datasets is None or "fer2013" in datasets:
        print("\n================ Preparing FER2013 ================")
        fer_prepared_csv = PREPARED_DIR / "fer2013_prepared.csv"
        if fer_prepared_csv.exists() and len(pd.read_csv(fer_prepared_csv)) > 0:
            prepared_csvs.append(str(fer_prepared_csv))
            print(f"[OK] FER2013 CSV already exists: {len(pd.read_csv(fer_prepared_csv))} rows")
        elif _try_rebuild_csv("fer2013", fer_prepared_csv, str(FER_CSV_IN)):
            prepared_csvs.append(str(fer_prepared_csv))
            print(f"[OK] FER2013 rebuilt from images")
        else:
            try:
                prepare_fer2013(
                    input_path=str(FER_CSV_IN),
                    images_root=str(IMAGES_ROOT),
                    out_csv=str(fer_prepared_csv)
                )
                if fer_prepared_csv.exists():
                    df_check = pd.read_csv(fer_prepared_csv)
                    if len(df_check) > 0:
                        prepared_csvs.append(str(fer_prepared_csv))
                        print(f"[OK] FER2013 prepared successfully: {len(df_check)} images")
                    else:
                        print("[WARN] FER2013 CSV is empty, skipping...")
                else:
                    print("[WARN] FER2013 preparation failed, skipping...")
            except Exception as e:
                print(f"[ERR] FER2013 preparation failed: {e}")
                print("  Continuing with other datasets...")

    # 2. RAF-DB
    if datasets is None or "rafdb" in datasets or "raf-db" in datasets:
        print("\n================ Preparing RAF-DB ================")
        raf_prepared_csv = PREPARED_DIR / "rafdb_prepared.csv"
        if raf_prepared_csv.exists() and len(pd.read_csv(raf_prepared_csv)) > 0:
            prepared_csvs.append(str(raf_prepared_csv))
            print(f"[OK] RAF-DB CSV already exists")
        elif _try_rebuild_csv("rafdb", raf_prepared_csv):
            prepared_csvs.append(str(raf_prepared_csv))
        else:
            try:
                prepare_rafdb(
                    input_path=str(RAF_ZIP_IN),
                    images_root=str(IMAGES_ROOT),
                    out_csv=str(raf_prepared_csv)
                )
                if raf_prepared_csv.exists():
                    df_check = pd.read_csv(raf_prepared_csv)
                    if len(df_check) > 0:
                        prepared_csvs.append(str(raf_prepared_csv))
                        print(f"[OK] RAF-DB prepared successfully: {len(df_check)} images")
                    else:
                        print("[WARN] RAF-DB CSV is empty, skipping...")
                else:
                    print("[WARN] RAF-DB preparation failed, skipping...")
            except Exception as e:
                print(f"[ERR] RAF-DB preparation failed: {e}")
                print("  Continuing with other datasets...")

    # 3. AffectNet
    if datasets is None or "affectnet" in datasets:
        print("\n================ Preparing AffectNet ================")
        affectnet_prepared_csv = PREPARED_DIR / "affectnet_prepared.csv"
        if affectnet_prepared_csv.exists() and len(pd.read_csv(affectnet_prepared_csv)) > 0:
            prepared_csvs.append(str(affectnet_prepared_csv))
            print(f"[OK] AffectNet CSV already exists")
        elif _try_rebuild_csv("affectnet", affectnet_prepared_csv):
            prepared_csvs.append(str(affectnet_prepared_csv))
        else:
            try:
                prepare_affectnet(
                    input_path=str(AFFECTNET_DIR),
                    images_root=str(IMAGES_ROOT),
                    out_csv=str(affectnet_prepared_csv)
                )
                if affectnet_prepared_csv.exists():
                    df_check = pd.read_csv(affectnet_prepared_csv)
                    if len(df_check) > 0:
                        prepared_csvs.append(str(affectnet_prepared_csv))
                        print(f"[OK] AffectNet prepared successfully: {len(df_check)} images")
                    else:
                        print("[WARN] AffectNet CSV is empty, skipping...")
                else:
                    print("[WARN] AffectNet preparation failed, skipping...")
            except Exception as e:
                print(f"[ERR] AffectNet preparation failed: {e}")
                print("  Continuing with other datasets...")

    if not prepared_csvs:
        raise RuntimeError("No datasets were successfully prepared! Check your dataset files and paths.")

    print("\n================ Unifying All Datasets ================")
    unified_csv = PREPARED_DIR / "unified_dataset.csv"

    fer_csv_path = None
    raf_csv_path = None
    aff_csv_path = None

    for csv_path in prepared_csvs:
        csv_name = Path(csv_path).name.lower()
        if "fer2013" in csv_name or "fer" in csv_name:
            fer_csv_path = csv_path
        elif "raf" in csv_name or "rafdb" in csv_name:
            raf_csv_path = csv_path
        elif "affectnet" in csv_name or "affect" in csv_name:
            aff_csv_path = csv_path

    unify(
        fer_csv=fer_csv_path,
        raf_csv=raf_csv_path,
        aff_csv=aff_csv_path,
        images_root=str(IMAGES_ROOT),
        unified_csv_out=str(unified_csv),
    )

    print("\n>>> Unified dataset written at:", unified_csv)
    df_final = pd.read_csv(unified_csv)
    print(f">>> Total images in unified dataset: {len(df_final)}")
    print(f">>> Per-dataset counts: {df_final['dataset'].value_counts().to_dict()}")

    # Auto-generate dataset statistics report (Reviewer 1 Comment 3)
    try:
        from dataset_balancer import map_labels_to_emotions, print_dataset_report, balance_dataset
        df_raw = df_final.copy()
        for ds in df_raw["dataset"].unique():
            df_raw = map_labels_to_emotions(df_raw, ds)
        df_raw = df_raw[df_raw["label"] != "unknown"].copy()
        df_raw["label"] = pd.to_numeric(df_raw["label"], errors="coerce")
        df_raw = df_raw[df_raw["label"].notna() & df_raw["label"].isin(range(7))].copy()
        df_raw["label"] = df_raw["label"].astype(int)
        df_bal_preview = balance_dataset(
            df_raw, method="both",
            min_samples_per_class=TRAIN_CONFIG.get("min_samples_per_class", 8000),
            max_samples_per_class=TRAIN_CONFIG.get("max_samples_per_class", 8000),
            random_state=TRAIN_CONFIG.get("seed", 42),
        )
        stats_md = PREPARED_DIR / "dataset_stats_report.md"
        print_dataset_report(df_raw, df_bal_preview, out_md=str(stats_md))
        print(f">>> Dataset statistics report: {stats_md}")
    except Exception as e:
        print(f"  ⚠ Could not generate dataset stats report: {e}")

    return unified_csv


# =========================================================================== #
# ===================== 2. TRAINING CONFIGURATION =========================== #
# =========================================================================== #

# GPU-aware configuration for 10X speedup while maintaining accuracy
IS_CUDA = torch.cuda.is_available()
GPU_PROFILE = None
LOW_MEM_GPU = False
if IS_CUDA:
    try:
        props = torch.cuda.get_device_properties(0)
        GPU_PROFILE = {
            "name": props.name,
            "total_gb": round(props.total_memory / (1024 ** 3), 1)
        }
        gpu_name_lower = GPU_PROFILE["name"].lower()
        LOW_MEM_GPU = ("t4" in gpu_name_lower) or GPU_PROFILE["total_gb"] <= 16
    except Exception:
        GPU_PROFILE = None
        LOW_MEM_GPU = False

if IS_CUDA:
    if LOW_MEM_GPU:
        # Google Colab T4 (15 GB) safe defaults
        BATCH_SIZE = 32
        NUM_WORKERS = 2
        PREFETCH_FACTOR = 2
        GRAD_ACCUM_STEPS = 2  # Effective batch size 64
    else:
        # High-memory GPUs can be pushed harder
        BATCH_SIZE = 128
        NUM_WORKERS = min(16, os.cpu_count() or 8)
        PREFETCH_FACTOR = 8
        GRAD_ACCUM_STEPS = 1
    USE_AMP = True  # 2X speedup with mixed precision
    CHANNELS_LAST = True  # 10-20% speedup on modern GPUs
    USE_TORCH_COMPILE = True  # 20-30% speedup with PyTorch 2.0+
    PIN_MEMORY = True  # Faster GPU transfer
else:
    # CPU optimizations - aggressively reduced for memory safety and speed
    cpu_count = os.cpu_count() or 4
    BATCH_SIZE = 8  # Further reduced to prevent OOM and speed up iterations on CPU
    NUM_WORKERS = 1  # Reduced to 1 to prevent memory issues and context switching overhead
    USE_AMP = False  # Not beneficial on CPU
    CHANNELS_LAST = False  # Not beneficial on CPU
    USE_TORCH_COMPILE = False  # Disabled on CPU due to backward pass bugs in compiled code
    PIN_MEMORY = False  # Not beneficial on CPU
    PREFETCH_FACTOR = 1  # Reduced to minimum to save memory
    GRAD_ACCUM_STEPS = 4  # Increased to maintain effective batch size (8 * 4 = 32)

TRAIN_CONFIG = {
    "batch_size": BATCH_SIZE,  # Optimized for maximum speedup
    "num_workers": NUM_WORKERS,  # Optimized for faster data loading
    "input_size": 160 if not IS_CUDA else 256,  # Increased to 256 on GPU for 93%+ accuracy
    "epochs_warm": 1 if not IS_CUDA else 3,  # Increased warm epochs for better initialization
    "epochs_ft": 8 if not IS_CUDA else 20,  # Increased to 20 epochs for 93%+ accuracy
    "device": "cuda" if IS_CUDA else "cpu",
    "models": ["convnext_v2"],
    
    # Q1 Research Standards — reproducible by default (Reviewer 2 Comment 5)
    "seed": 42,
    "deterministic": True,
    "save_split_indices": True,
    "use_amp": USE_AMP,
    "use_mixup": True,  # ENABLED - MixUp improves generalization and accuracy
    "mixup_alpha": 0.4,
    "randaugment": True,  # ENABLED - RandAugment is crucial for high accuracy
    "use_novel_aug": True,  # ENABLED - Novel augmentations help generalization
    "label_smoothing": 0.1,  # ENABLED - Label smoothing improves generalization
    "weight_decay": 1e-4,
    "early_stop": 5 if not IS_CUDA else 10,  # Increased patience for better convergence
    "channels_last": CHANNELS_LAST,  # Enabled on GPU for 10-20% speedup
    "use_torch_compile": USE_TORCH_COMPILE,  # Enabled on GPU for 20-30% speedup
    "pin_memory": PIN_MEMORY,  # Enabled on GPU for faster transfer
    "prefetch_factor": PREFETCH_FACTOR,  # Tuned per device profile
    "grad_accum_steps": GRAD_ACCUM_STEPS,  # Maintains effective batch size per device
    "use_ema": IS_CUDA,  # Enabled on GPU for accuracy improvement

    # Dataset Balancing - OPTIMIZED FOR HIGH ACCURACY (93%+)
    "balance_dataset": True,  # BALANCED DATASET IS BETTER for 93%+ accuracy
    "balance_method": "both",  # Use both oversample and undersample for best balance
    "min_samples_per_class": 8000,
    "max_samples_per_class": 8000,
    "use_class_sampler": False,
    "use_tta": True,
    "backbone_lr_ratio": 0.1,
    "ft_lr_ratio": 0.3,
    "val_split": 0.15,
    "test_split": 0.15,
    "results_dir": str(OUTPUT_BASE_DIR / "Results_Q1"),

    # Post-training evaluation pipeline (Reviewer fixes)
    "run_benchmark_eval": True,
    "run_baselines": False,       # Set True to train ResNet-18 etc. under identical protocol
    "baseline_models": ["resnet18", "efficientnet_b0"],
    "run_cross_dataset": True,
    "run_ablation": False,        # Set True for full ablation (3 seeds × 8 configs — slow)
    "ablation_seeds": [42, 123, 456],
    "export_repro_bundle": True,
}


# =========================================================================== #
# ============================ 3. TRAINING RUN =============================== #
# =========================================================================== #

def run_post_training_eval(unified_csv_path, experiment_id, model_name, train_results=None):
    """Run benchmark, baselines, cross-dataset, ablation, and export bundle."""
    results_base = Path(TRAIN_CONFIG["results_dir"]) / experiment_id
    model_dir = results_base / model_name
    ckpt_path = model_dir / f"{model_name}_best_full.pth"
    if not ckpt_path.exists():
        ckpt_path = model_dir / f"{model_name}_best.pth"

    num_classes = 7

    # ── Benchmark eval (standard splits) ─────────────────────────────────────
    if TRAIN_CONFIG.get("run_benchmark_eval", True) and ckpt_path.exists():
        print(f"\n================ BENCHMARK EVALUATION: {model_name} ================")
        try:
            from benchmark_eval import run_benchmark_eval, generate_prior_works_comparison
            bench_out = model_dir / "benchmark_eval"
            bench_metrics = run_benchmark_eval(
                checkpoint_path=str(ckpt_path),
                images_root=str(IMAGES_ROOT),
                fer2013_orig_csv=str(FER_CSV_IN),
                num_classes=num_classes,
                output_dir=str(bench_out),
                model_name=model_name,
                input_size=TRAIN_CONFIG.get("input_size_ft", TRAIN_CONFIG.get("input_size", 256)),
                batch_size=BATCH_SIZE,
                full_ckpt=ckpt_path.name.endswith("_full.pth"),
            )
            generate_prior_works_comparison(
                our_metrics=bench_metrics,
                our_unified_metrics=train_results, #type:ignore
                output_path=str(results_base / "PRIOR_WORKS_COMPARISON.md"),
            )
            print(f"Benchmark complete: {bench_out / 'comparative_analysis_report.md'}")
        except Exception as e:
            print(f"Benchmark evaluation failed: {e}")

    # ── Baseline reproduction (identical protocol) ───────────────────────────
    if TRAIN_CONFIG.get("run_baselines", False):
        print(f"\n================ BASELINE REPRODUCTION ================")
        try:
            from baseline_runner import run_baselines_under_protocol
            run_baselines_under_protocol(
                csv_path=str(unified_csv_path),
                images_root=str(IMAGES_ROOT),
                fer2013_orig_csv=str(FER_CSV_IN),
                num_classes=num_classes,
                base_config=TRAIN_CONFIG.copy(),
                baseline_models=TRAIN_CONFIG.get("baseline_models", ["resnet18"]),
                experiment_id=f"{experiment_id}_baselines",
            )
        except Exception as e:
            print(f"Baseline reproduction failed: {e}")

    # ── Cross-dataset evaluation + domain-shift diagnosis ───────────────────
    if TRAIN_CONFIG.get("run_cross_dataset", True):
        print(f"\n================ CROSS-DATASET EVALUATION ================")
        try:
            from cross_dataset_eval import run_cross_dataset_experiments
            cross_cfg = TRAIN_CONFIG.copy()
            cross_cfg["experiment_id"] = f"{experiment_id}_cross"
            cross_cfg["epochs_warm"] = min(2, cross_cfg.get("epochs_warm", 3))
            cross_cfg["epochs_ft"] = min(8, cross_cfg.get("epochs_ft", 20))
            run_cross_dataset_experiments(
                csv_path=str(unified_csv_path),
                images_root=str(IMAGES_ROOT),
                num_classes=num_classes,
                base_config=cross_cfg,
                models=[model_name],
            )
        except Exception as e:
            print(f"Cross-dataset evaluation failed: {e}")

    # ── Ablation study (multi-seed + significance) ───────────────────────────
    if TRAIN_CONFIG.get("run_ablation", False):
        print(f"\n================ ABLATION STUDY ================")
        try:
            from ablation_study import run_ablation_study
            ablation_cfg = TRAIN_CONFIG.copy()
            ablation_cfg["epochs_warm"] = min(2, ablation_cfg.get("epochs_warm", 3))
            ablation_cfg["epochs_ft"] = min(10, ablation_cfg.get("epochs_ft", 20))
            run_ablation_study(
                model_name=model_name,
                csv_path=str(unified_csv_path),
                images_root=str(IMAGES_ROOT),
                num_classes=num_classes,
                base_config=ablation_cfg,
                seeds=TRAIN_CONFIG.get("ablation_seeds", [42, 123, 456]),
                experiment_id=f"{experiment_id}_ablation",
            )
        except Exception as e:
            print(f"Ablation study failed: {e}")

    # ── Generalization Audit (R1-C5/C6, R2-C3/C4) ───────────────────────────
    # Runs: label-distribution comparison, label-mapping audit, bidirectional
    # cross-dataset matrix, FER2013 collapse explanation, benchmark completeness.
    if TRAIN_CONFIG.get("run_cross_dataset", True) or TRAIN_CONFIG.get("run_benchmark_eval", True):
        print(f"\n================ GENERALIZATION AUDIT ================")
        try:
            from generalization_audit import run_full_generalization_audit
            unified_csv_for_audit = Path(PREPARED_DIR) / "unified_dataset.csv"
            cross_results_json = results_base / "cross_dataset_results.json"
            benchmark_metrics_json = model_dir / "benchmark_eval" / "benchmark_metrics.json"
            if unified_csv_for_audit.exists():
                run_full_generalization_audit(
                    unified_csv=str(unified_csv_for_audit),
                    cross_results_json=str(cross_results_json),
                    benchmark_metrics_json=str(benchmark_metrics_json),
                    output_dir=str(results_base),
                    unified_train_metrics=train_results,
                )
                print(f"  ✓ Generalization audit saved to: {results_base / 'generalization_audit'}")
            else:
                print(f"  ⚠ unified_dataset.csv not found for audit; skipping.")
        except Exception as e:
            print(f"Generalization audit failed: {e}")

    # ── Export reproducibility bundle ─────────────────────────────────────────
    if TRAIN_CONFIG.get("export_repro_bundle", True) and model_dir.exists():
        try:
            from export_reproducibility_bundle import export_reproducibility_bundle  #type:ignore
            bundle_out = results_base / "release_bundle"
            export_reproducibility_bundle(
                experiment_dir=str(model_dir),
                output_dir=str(bundle_out),
                include_weights=True,
            )
        except Exception as e:
            print(f"  ⚠ Reproducibility bundle export failed: {e}")


def run_all_training(unified_csv_path, experiment_id=None):
    """Train all models with shared experiment ID for comparison."""
    if experiment_id is None:
        experiment_id = generate_experiment_id("main")
    
    df = pd.read_csv(unified_csv_path)
    num_classes = len(sorted(df["label"].unique()))
    
    print(f"\n{'='*70}")
    print(f"TRAINING ConvNeXt-V2")
    print(f"Experiment ID: {experiment_id}")
    print(f"{'='*70}")
    
    # Display optimization settings
    device_type = "GPU" if IS_CUDA else "CPU"
    input_size = TRAIN_CONFIG["input_size"]
    epochs_warm = TRAIN_CONFIG["epochs_warm"]
    epochs_ft = TRAIN_CONFIG["epochs_ft"]
    gpu_info = ""
    if IS_CUDA and GPU_PROFILE:
        gpu_info = f"{GPU_PROFILE['name']} ({GPU_PROFILE['total_gb']} GB)"
    mode_note = "T4/Colab-safe settings" if LOW_MEM_GPU else "High-memory GPU settings"
    print(f"\n[TRAINING CONFIGURATION - Optimized for {device_type}]")
    print(f"  Device: {device_type}")
    if gpu_info:
        print(f"  GPU: {gpu_info}")
        print(f"  GPU Mode: {mode_note}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Num Workers: {NUM_WORKERS}")
    print(f"  Input Size: {input_size}")
    print(f"  Epochs (Warm/FT): {epochs_warm}/{epochs_ft}")
    print(f"  Mixed Precision (AMP): {USE_AMP}")
    print(f"  Channels Last: {CHANNELS_LAST}")
    print(f"  Torch Compile: {USE_TORCH_COMPILE}")
    print(f"  Prefetch Factor: {PREFETCH_FACTOR}")
    print(f"  Effective Batch Size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    if not IS_CUDA:
        print(f"  [CPU MODE: Reduced settings for memory safety and faster iteration]")
    print(f"{'='*70}\n")

    for model_name in TRAIN_CONFIG["models"]:
        print(f"\n================ TRAINING MODEL: {model_name} ================")
        config = TRAIN_CONFIG.copy()
        config["experiment_id"] = experiment_id
        model, results = run_training_pipeline(
            model_name=model_name,
            csv_path=str(unified_csv_path),
            images_root=str(IMAGES_ROOT),
            num_classes=num_classes,
            config=config
        )

        run_post_training_eval(unified_csv_path, experiment_id, model_name, train_results=results)

    # Auto-generate summary table for all trained models
    try:
        from experiment_runner import generate_summary_table
        results_base = Path(TRAIN_CONFIG["results_dir"]) / experiment_id
        # Collect aggregated JSON files from each model subdirectory
        agg_results = []
        for model_name in TRAIN_CONFIG["models"]:
            agg_path = results_base / model_name / "aggregated_metrics.json"
            if agg_path.exists():
                import json
                with open(agg_path) as f:
                    agg_results.append(json.load(f))
        if agg_results:
            generate_summary_table(
                agg_results,
                out_path=results_base / "summary_table.csv",
            )
            print(f"\n  ✓ Summary table saved to: {results_base / 'summary_table.csv'}")
    except Exception as e:
        print(f"Could not generate summary table: {e}")


# =========================================================================== #
# ============================= MASTER PIPELINE ============================== #
# =========================================================================== #

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Q1 Master Experiment Pipeline")
    parser.add_argument("--config", type=str, default="configs/kaggle_2gpu.yaml",
                        help="YAML config (default: configs/kaggle_2gpu.yaml)")
    parser.add_argument("--skip-cross-dataset", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--run-baselines", action="store_true")
    args = parser.parse_args()

    # Load and merge YAML config
    if args.config and Path(args.config).exists():
        yaml_cfg = resolve_config_auto(load_yaml(args.config))
        print(f"\n  Loaded config from: {args.config}")
        globals()["TRAIN_CONFIG"] = merge_train_config(TRAIN_CONFIG, yaml_cfg)
    elif args.config:
        print(f"Config not found: {args.config}, using built-in defaults")

    if args.skip_cross_dataset:
        TRAIN_CONFIG["run_cross_dataset"] = False
    if args.skip_benchmark:
        TRAIN_CONFIG["run_benchmark_eval"] = False
    if args.run_ablation:
        TRAIN_CONFIG["run_ablation"] = True
    if args.run_baselines:
        TRAIN_CONFIG["run_baselines"] = True

    # Post-config path normalization for Kaggle / Read-only environments
    if IS_KAGGLE:
        # Redirect results_dir to writable directory under /kaggle/working/
        res_dir = Path(TRAIN_CONFIG.get("results_dir", "Results_Q1"))
        if not res_dir.is_absolute() or "/kaggle/input" in str(res_dir):
            TRAIN_CONFIG["results_dir"] = str(Path("/kaggle/working") / res_dir.name)
        print(f"  [Kaggle Environment Detected] Redirected results_dir to writable path: {TRAIN_CONFIG['results_dir']}")

    print("\n" + "="*70)
    print("Q1 Emotion Recognition Project – MASTER PIPELINE")
    print("Research Standards: Reproducible, Comprehensive Evaluation")
    print("="*70)

    prepare_list = TRAIN_CONFIG.get("prepare_datasets", None)
    unified_csv = prepare_all_datasets(datasets=prepare_list) #type:ignore

    print("\n================ Starting Training ================")
    run_all_training(unified_csv)

    print("\n" + "="*70)
    print("FINISHED ALL EXPERIMENTS")
    print("="*70)
    print("\nOutput files per model:")
    print("  - convnext_v2_best_full.pth        : Full training checkpoint")
    print("  - benchmark_eval/                  : Per-dataset standard-split results")
    print("      comparative_analysis_report.md : SOTA comparison table")
    print("      benchmark_metrics.json         : FER2013 / RAF-DB / AffectNet accuracy")
    print("\nFor statistical analysis with multiple runs:")
    print("  python experiment_runner.py --model convnext_v2 --csv <path> --images <path> --config configs/kaggle_2gpu.yaml")
    print("\nFor ablation studies (3 seeds, significance tests):")
    print("  python ablation_study.py --model convnext_v2 --csv <path> --images <path> --seeds 42 123 456")
    print("\nFor cross-dataset evaluation (with domain-shift diagnosis):")
    print("  python cross_dataset_eval.py --csv <path> --images <path>")
    print("\nFor standalone evaluation:")
    print("  python evaluate.py --checkpoint <path>.pth --csv <path> --images <path>")
    print("="*70)


if __name__ == "__main__":
    main()
