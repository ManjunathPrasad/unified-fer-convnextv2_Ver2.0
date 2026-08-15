# dataset_preparation.py
"""
LOCAL-ONLY dataset preparation for Q1 emotion framework.
Uses ONLY your provided dataset paths:

FER2013 CSV   -> C:\\paper2Q1\\emotion_q1_framework\\Dataset\\fer2013.csv
RAF-DB ZIP    -> C:\\paper2Q1\\emotion_q1_framework\\Dataset\\af-db.zip
AffectNet     -> C:\\paper2Q1\\emotion_q1_framework\\Dataset\\AffectNet

Outputs:
 - images_root/FER2013/*.jpg
 - images_root/RAFDB/*.jpg
 - images_root/AffectNet/*.jpg
 - CSV files inside out_dir:

   fer2013_prepared.csv
   rafdb_prepared.csv
   affectnet_prepared.csv
   unified_dataset.csv
"""

from pathlib import Path
import shutil
import zipfile
import pandas as pd
import numpy as np
from PIL import Image
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [INFO] %(message)s")
logger = logging.getLogger("dataset_prep")

# -------------------------------------------------------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def pil_save_rgb(img: Image.Image, out_path: Path):
    ensure_dir(out_path.parent)
    img.convert("RGB").save(out_path, format="JPEG", quality=95, optimize=True)

def _detect_fer2013_format(input_path: Path):
    """
    Detect FER2013 CSV layout before pandas assigns pixel values as column names.

    Returns:
        (dataframe, pixels_format, pixel_cols, has_usage)
    """
    with open(input_path, "r", encoding="utf-8", errors="replace") as handle:
        first_line = handle.readline().strip()
    if not first_line:
        return pd.DataFrame(), "empty", None, False

    first_fields = first_line.split(",")
    num_cols = len(first_fields)

    # Standard headered Kaggle export: emotion,pixels,Usage
    if first_fields[0].lower() in {"emotion", "label"} and "pixels" in first_line.lower():
        df = pd.read_csv(input_path)
        if "Usage" not in df.columns and len(df.columns) >= 3:
            df = df.rename(columns={df.columns[2]: "Usage"})
        return df, "string", None, "Usage" in df.columns

    # Headerless wide matrix: 2304 pixels + 7 one-hot labels [+ optional Usage]
    if num_cols >= 2311 and first_fields[0].isdigit():
        df = pd.read_csv(input_path, header=None)
        df["pixels"] = "onehot_columns"
        has_usage = num_cols >= 2312
        if has_usage:
            df["Usage"] = df.iloc[:, -1].astype(str)
        logger.info(
            "Detected headerless FER2013 matrix: %d columns (%s)",
            num_cols,
            "with Usage" if has_usage else "one-hot labels only",
        )
        return df, "onehot_columns", None, has_usage

    # Headerless: emotion + 2304 pixel columns [+ Usage]
    if num_cols >= 2305 and first_fields[0].isdigit():
        df = pd.read_csv(input_path, header=None)
        df = df.rename(columns={0: "emotion"})
        df["pixels"] = "columns"
        has_usage = num_cols >= 2306
        if has_usage:
            df = df.rename(columns={num_cols - 1: "Usage"})
        logger.info("Detected headerless FER2013 pixel-column format (%d columns)", num_cols)
        pixel_cols = [str(c) for c in range(1, min(2305, num_cols))]
        return df, "columns", pixel_cols[:2304], has_usage

    # Fallback: try default headered read
    df = pd.read_csv(input_path)
    if "emotion" in df.columns and "pixels" in df.columns:
        return df, "string", None, "Usage" in df.columns

    # Last resort: first row was mis-read as header (e.g. columns named '254')
    if str(df.columns[0]).replace(".", "", 1).isdigit() and len(df.columns) >= 2311:
        df = pd.read_csv(input_path, header=None)
        df["pixels"] = "onehot_columns"
        logger.info("Recovered headerless one-hot FER2013 after mis-parse")
        return df, "onehot_columns", None, False

    raise ValueError(
        f"FER2013 CSV format not recognized ({num_cols} columns in first row). "
        "Expected Kaggle (emotion,pixels,Usage) or 2311-column pixel matrix."
    )


def _process_fer2013_row(args):
    """Process a single FER2013 row - optimized for multiprocessing."""
    i, row_dict, images_root, pixels_format, pixel_cols = args
    try:
        # Check if one-hot format is marked
        if pixels_format == "onehot_columns" or row_dict.get("pixels") == "onehot_columns":
            # The label is the argmax of columns 2304 to 2310
            one_hot_vals = []
            for col in range(2304, 2311):
                val = row_dict.get(col)
                if val is None:
                    val = row_dict.get(str(col), 0)
                one_hot_vals.append(int(float(val)))
            label = int(np.argmax(one_hot_vals))
            
            # The pixels are in columns 0 to 2303
            pixel_values = []
            for col in range(2304):
                val = row_dict.get(col)
                if val is None:
                    val = row_dict.get(str(col), 0)
                if val is None or val == '' or (isinstance(val, float) and np.isnan(val)):
                    pixel_values.append(0)
                else:
                    pixel_values.append(int(float(val)))
        else:
            # Fallback to standard parsing
            # Convert dict back to Series-like access
            # Handle both "emotion" key and numeric key (0) for emotion column
            if "emotion" in row_dict:
                label = int(row_dict["emotion"])
            elif 0 in row_dict:
                label = int(row_dict[0])
            else:
                # Try to find emotion in first column
                first_key = min([k for k in row_dict.keys() if isinstance(k, (int, str)) and str(k).isdigit()], default=None)
                if first_key is not None:
                    label = int(row_dict[first_key])
                else:
                    return None
            
            if pixels_format == "string":
                pixels_str = str(row_dict["pixels"]).strip()
                if not pixels_str or pixels_str == 'nan':
                    return None
                pixel_values = [int(x) for x in pixels_str.split()[:2304]]
            else:
                pixel_values = []
                if pixel_cols is None:
                    return None
                for col in pixel_cols:
                    try:
                        val = row_dict.get(col)
                        if val is None:
                            val = row_dict.get(str(col), 0)
                        # Check for NaN or empty values
                        if val is None or val == '' or (isinstance(val, float) and np.isnan(val)):
                            pixel_values.append(0)
                        else:
                            pixel_values.append(int(float(val)))
                    except (ValueError, TypeError):
                        pixel_values.append(0)
                
                # Ensure we have exactly 2304 pixels
                if len(pixel_values) < 2304:
                    # Pad with zeros if needed
                    pixel_values.extend([0] * (2304 - len(pixel_values)))
                elif len(pixel_values) > 2304:
                    # Truncate if we have more
                    pixel_values = pixel_values[:2304]
        
        if len(pixel_values) != 2304:
            return None
            
        pixels = np.array(pixel_values, dtype=np.uint8).reshape(48, 48)
        im = Image.fromarray(pixels, mode='L').resize((224, 224), Image.Resampling.LANCZOS)
        
        fname = f"fer2013_{i:06d}.jpg"
        out_path = Path(images_root) / fname
        pil_save_rgb(im, out_path)
        
        usage = row_dict.get("Usage")
        if usage is None or (isinstance(usage, float) and np.isnan(usage)):
            usage = ""
        return (f"FER2013/{fname}", label, "FER2013", str(usage).strip())
    except Exception as e:
        # Log first few errors for debugging
        if i < 5:
            logger.debug(f"Error processing row {i}: {e}")
        return None

def _process_rafdb_image(args):
    """Process a single RAF-DB image - optimized for multiprocessing."""
    src_path_str, dest_path_str, img_name, label = args
    try:
        src_path = Path(src_path_str)
        dest_path = Path(dest_path_str)
        if not src_path.exists():
            return None
        pil_save_rgb(Image.open(src_path), dest_path)
        return (f"RAFDB/{img_name}", label, "RAFDB")
    except Exception:
        return None

def _process_affectnet_image(args):
    """Process a single AffectNet image - optimized for multiprocessing."""
    src_path_str, dest_path_str, img_name, label = args
    try:
        src_path = Path(src_path_str)
        dest_path = Path(dest_path_str)
        if not src_path.exists():
            return None
        pil_save_rgb(Image.open(src_path), dest_path)
        return (f"AffectNet/{img_name}", label, "AffectNet")
    except Exception:
        return None

# -------------------------------------------------------------------------
# FER2013 preparation (LOCAL CSV ONLY)
# -------------------------------------------------------------------------
def prepare_fer2013(input_path: str, images_root: str, out_csv: str) -> Path:
    images_root = Path(images_root) / "FER2013" #type:ignore
    out_csv = Path(out_csv) #type:ignore
    input_path = Path(input_path) #type:ignore
    ensure_dir(images_root); ensure_dir(out_csv.parent) #type:ignore

    if not input_path.exists(): #type:ignore
        raise FileNotFoundError(f"FER2013 CSV not found at:\n{input_path}")

    logger.info("Reading FER2013 from local CSV…")
    df, pixels_format, pixel_cols, has_usage = _detect_fer2013_format(input_path) #type:ignore

    if len(df) == 0 or pixels_format == "empty":
        logger.warning(f"FER2013 CSV is empty! Check the file at: {input_path}")
        out_df = pd.DataFrame(columns=["image_path", "label", "dataset", "Usage"])
        out_df.to_csv(out_csv, index=False)
        return out_csv #type:ignore

    if pixels_format == "string":
        logger.info("Detected FER2013 format: pixels as space-separated string")
    elif pixels_format == "onehot_columns":
        logger.info("Detected FER2013 format: 2304 pixels + 7 one-hot label columns")
    elif pixels_format == "columns":
        logger.info(
            "Detected FER2013 format: pixels spread across %d columns",
            len(pixel_cols or []),
        )
    
    # Use multiprocessing for faster processing
    num_workers = min(cpu_count(), 8)  # Limit to 8 workers to avoid overhead
    logger.info(f"Processing FER2013 with {num_workers} workers...")
    
    # Prepare arguments for multiprocessing - convert rows to dicts for pickling
    # Convert images_root to string for pickling
    images_root_str = str(images_root)
    process_args = [(i, df.iloc[i].to_dict(), images_root_str, pixels_format, pixel_cols) for i in range(len(df))]
    
    # Process in parallel
    rows = []
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(_process_fer2013_row, process_args),
            total=len(df),
            desc="Processing FER2013 images"
        ))
        rows = [r for r in results if r is not None]
    
    error_count = len(df) - len(rows)
    if error_count > 0:
        logger.warning(f"Skipped {error_count} rows due to errors")

    out_df = pd.DataFrame(rows, columns=["image_path", "label", "dataset", "Usage"])
    if not has_usage:
        out_df["Usage"] = ""
    # Sanity check: labels must be emotion ids 0-6, not pixel intensities
    out_df["label"] = pd.to_numeric(out_df["label"], errors="coerce")
    invalid = out_df["label"].isna() | ~out_df["label"].isin(range(7))
    if invalid.any():
        logger.warning(
            "Dropping %d FER2013 rows with invalid labels (expected 0-6)",
            int(invalid.sum()),
        )
        out_df = out_df[~invalid].copy()
    out_df["label"] = out_df["label"].astype(int)
    out_df.to_csv(out_csv, index=False)
    logger.info(f"FER2013 prepared: {len(out_df)} images (from {len(df)} total rows)")
    return out_csv #type:ignore

# -------------------------------------------------------------------------
# RAF-DB (LOCAL ZIP ONLY)
# -------------------------------------------------------------------------
def prepare_rafdb(input_path: str, images_root: str, out_csv: str) -> Path:
    images_root = Path(images_root) / "RAFDB" #type:ignore
    out_csv = Path(out_csv) #type:ignore
    input_path = Path(input_path) #type:ignore
    ensure_dir(images_root); ensure_dir(out_csv.parent) #type:ignore

    # Check if we have an unzipped folder instead (common on Kaggle since it auto-extracts zip files)
    unzipped_dir = input_path.parent / input_path.stem  # e.g., DATASET_DIR / "af-db" #type:ignore
    
    if not input_path.exists() and unzipped_dir.exists() and unzipped_dir.is_dir(): #type:ignore
        logger.info(f"RAF-DB ZIP not found, but found unzipped directory: {unzipped_dir}. Bypassing extraction.")
        temp_dir = unzipped_dir
    else:
        # Check direct fallback directory named "af-db" in parent
        fallback_dir = input_path.parent / "af-db" #type:ignore
        if not input_path.exists() and fallback_dir.exists() and fallback_dir.is_dir(): #type:ignore
            logger.info(f"RAF-DB ZIP not found, but found fallback directory: {fallback_dir}. Bypassing extraction.")
            temp_dir = fallback_dir
        else:
            if not input_path.exists(): #type:ignore
                raise FileNotFoundError(f"RAF-DB ZIP not found at:\n{input_path} and no unzipped directory found.")
            
            temp_dir = out_csv.parent / "rafdb_raw" #type:ignore
            if temp_dir.exists(): shutil.rmtree(temp_dir)
            ensure_dir(temp_dir)

            logger.info("Extracting RAF-DB ZIP…")
            with zipfile.ZipFile(input_path, "r") as z:
                z.extractall(temp_dir)

    rows = []
    # Try to find CSV label files first (newer format)
    csv_label_files = list(temp_dir.rglob("*label*.csv"))
    
    # Build image lookup map once for efficiency
    image_map = {}
    for img_file in temp_dir.rglob("*"):
        if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            image_map[img_file.name] = img_file
            # Also map without extension
            name_no_ext = img_file.stem
            if name_no_ext not in image_map:
                image_map[name_no_ext] = img_file
    
    if csv_label_files:
        # Process CSV label files
        logger.info(f"Found {len(csv_label_files)} CSV label file(s)")
        for csv_file in csv_label_files:
            logger.info(f"Processing: {csv_file.name}")
            try:
                df_labels = pd.read_csv(csv_file)
                # Check common CSV formats
                # Format 1: image_name, label
                # Format 2: filename, label or similar
                if len(df_labels.columns) >= 2:
                    img_col = df_labels.columns[0]
                    label_col = df_labels.columns[1]
                    
                    # Prepare arguments for multiprocessing
                    process_args = []
                    for _, row in df_labels.iterrows():
                        img_name = str(row[img_col]).strip()
                        label = str(row[label_col]).strip()
                        
                        if not img_name or img_name == 'nan':
                            continue
                        
                        # Find image file using map
                        src = image_map.get(img_name) or image_map.get(Path(img_name).stem)
                        if not src:
                            # Try with different extensions
                            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                                name_with_ext = Path(img_name).stem + ext
                                src = image_map.get(name_with_ext)
                                if src:
                                    break
                        
                        if not src:
                            continue
                        
                        dest = images_root / Path(img_name).name #type:ignore
                        process_args.append((str(src), str(dest), Path(img_name).name, label))
                    
                    # Process in parallel
                    num_workers = min(cpu_count(), 8)
                    logger.info(f"Processing {len(process_args)} RAF-DB images with {num_workers} workers...")
                    with Pool(processes=num_workers) as pool:
                        results = list(tqdm(
                            pool.imap(_process_rafdb_image, process_args),
                            total=len(process_args),
                            desc=f"Processing {csv_file.name}"
                        ))
                        rows.extend([r for r in results if r is not None])
                            
            except Exception as e:
                logger.warning(f"Error processing CSV file {csv_file}: {e}")
                continue
    
    # Fallback to TXT files if no CSV found
    if not rows:
        label_patterns = [
            "*list*label*.txt",
            "*label*.txt",
            "*train*.txt",
            "*test*.txt",
            "*.txt"
        ]
        
        label_files = []
        for pattern in label_patterns:
            label_files = list(temp_dir.rglob(pattern))
            if label_files:
                # Filter out very large files (likely not label files)
                label_files = [f for f in label_files if f.stat().st_size < 10 * 1024 * 1024]  # < 10MB
                if label_files:
                    break
        
        if not label_files:
            # List directory structure for debugging
            logger.error("RAF-DB label file not found (neither CSV nor TXT)")
            logger.info("Top-level directories/files in archive:")
            for item in list(temp_dir.iterdir())[:20]:
                logger.info(f"  - {item.name} ({'DIR' if item.is_dir() else 'FILE'})")
            raise RuntimeError("RAF-DB label file not found. Check the ZIP structure.")

        label_file = label_files[0]
        logger.info(f"Using TXT label file: {label_file} ({label_file.stat().st_size} bytes)")

        # Prepare arguments for multiprocessing
        process_args = []
        for line in label_file.read_text().splitlines():
            if not line.strip(): continue
            parts = line.split()
            if len(parts) < 2:
                continue
            img_name, label = parts[0], parts[1]

            src = image_map.get(img_name) or image_map.get(Path(img_name).stem)
            if not src:
                # Try with different extensions
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    name_with_ext = Path(img_name).stem + ext
                    src = image_map.get(name_with_ext)
                    if src:
                        break
            
            if not src:
                continue

            dest = images_root / img_name
            process_args.append((str(src), str(dest), img_name, label))
        
        # Process in parallel
        num_workers = min(cpu_count(), 8)
        logger.info(f"Processing {len(process_args)} RAF-DB images with {num_workers} workers...")
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(_process_rafdb_image, process_args),
                total=len(process_args),
                desc="Processing RAF-DB images"
            ))
            rows.extend([r for r in results if r is not None])

    pd.DataFrame(rows, columns=["image_path","label","dataset"]).to_csv(out_csv, index=False)
    logger.info(f"RAF-DB prepared: {len(rows)} images")
    return out_csv  #type:ignore

# -------------------------------------------------------------------------
# AffectNet (LOCAL FOLDER ONLY)
# -------------------------------------------------------------------------
def prepare_affectnet(input_path: str, images_root: str, out_csv: str) -> Path:
    images_root = Path(images_root) / "AffectNet" #type:ignore
    out_csv = Path(out_csv) #type:ignore
    input_path = Path(input_path) #type:ignore
    ensure_dir(images_root); ensure_dir(out_csv.parent) #type:ignore

    if not input_path.exists(): #type:ignore
        raise FileNotFoundError(f"AffectNet folder not found at:\n{input_path}")

    # Load labels.csv if present
    label_map = {}
    labels_csv = input_path / "labels.csv" #type:ignore
    if labels_csv.exists():
        logger.info(f"Loading AffectNet labels from {labels_csv}...")
        try:
            df_labels = pd.read_csv(labels_csv)
            # EMOTION_NAME_TO_INT mapping
            emotion_name_to_int = {
                "anger": 0, "angry": 0,
                "disgust": 1,
                "fear": 2,
                "happy": 3, "happiness": 3,
                "sad": 4, "sadness": 4,
                "surprise": 5,
                "neutral": 6
            }
            for _, row in df_labels.iterrows():
                pth = str(row.get("pth", "")).strip()
                lbl = str(row.get("label", "")).strip().lower()
                val = emotion_name_to_int.get(lbl, "unknown")
                if pth:
                    # Key by filename
                    label_map[Path(pth).name] = val
                    # Key by relative path structure
                    label_map[pth.replace("\\", "/")] = val
        except Exception as e:
            logger.warning(f"Failed to parse AffectNet labels.csv: {e}")

    rows = []
    logger.info("Processing AffectNet (Train/ + Test/ only, valid labels required)…")

    # Official AffectNet layout: only Train/ and Test/ subfolders (fixes count inflation)
    scan_roots = []
    for sub in ("Train", "Test"):
        sub_dir = input_path / sub #type:ignore
        if sub_dir.exists():
            scan_roots.append(sub_dir)
    if not scan_roots:
        logger.warning("AffectNet Train/Test folders not found; scanning entire directory")
        scan_roots = [input_path]

    img_paths = []
    for root in scan_roots:
        img_paths.extend(
            p for p in root.rglob("*") #type:ignore
            if p.suffix.lower() in [".jpg", ".png", ".jpeg"]
        )

    # Prepare arguments for multiprocessing
    process_args = []
    skipped_unknown = 0
    for p in img_paths:
        rel_path = p.relative_to(input_path).as_posix()
        label = label_map.get(rel_path)
        if label is None:
            label = label_map.get(p.name)
        if label is None:
            # Fallback to directory name mapping
            label = "unknown"
            for parent in [p.parent, p.parent.parent]:
                name = parent.name.lower()
                if name in ["anger", "angry", "0"]: label = 0; break
                if name in ["disgust", "1"]: label = 1; break
                if name in ["fear", "2"]: label = 2; break
                if name in ["happy", "happiness", "3"]: label = 3; break
                if name in ["sad", "sadness", "4"]: label = 4; break
                if name in ["surprise", "5"]: label = 5; break
                if name in ["neutral", "6"]: label = 6; break

        # Skip unlabelled / non-emotion images (fixes 49k vs 30k mismatch)
        if label == "unknown":
            skipped_unknown += 1
            continue

        # Preserve Train/Test subfolder in output path
        split_subdir = "Test" if "test" in p.as_posix().lower() else "Train"
        dest_path = images_root / split_subdir / p.name #type:ignore
        img_name_with_subdir = f"{split_subdir}/{p.name}"

        process_args.append((str(p), str(dest_path), img_name_with_subdir, label))

    if skipped_unknown:
        logger.info("Skipped %d AffectNet images with unknown/missing labels", skipped_unknown)
    
    # Process in parallel
    num_workers = min(cpu_count(), 8)
    logger.info(f"Processing {len(process_args)} AffectNet images with {num_workers} workers...")
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(_process_affectnet_image, process_args),
            total=len(process_args),
            desc="Processing AffectNet images"
        ))
        rows = [r for r in results if r is not None]

    out_df = pd.DataFrame(rows, columns=["image_path", "label", "dataset"])
    out_df = out_df[out_df["label"].astype(str) != "unknown"].copy()
    out_df["label"] = pd.to_numeric(out_df["label"], errors="coerce")
    out_df = out_df[out_df["label"].notna() & out_df["label"].isin(range(7))].copy()
    out_df["label"] = out_df["label"].astype(int)
    out_df.to_csv(out_csv, index=False)
    logger.info(f"AffectNet prepared: {len(out_df)} images (valid 7-class labels only)")
    return out_csv #type:ignore


# -------------------------------------------------------------------------
# Rebuild prepared CSV from existing image folders (skip re-extraction)
# -------------------------------------------------------------------------
def _load_rafdb_label_map(raf_zip_path: Path) -> dict:
    """Load RAF-DB filename → label mapping from the local ZIP."""
    label_map = {}
    if not raf_zip_path or not Path(raf_zip_path).exists():
        return label_map
    try:
        with zipfile.ZipFile(raf_zip_path, "r") as z:
            for name in z.namelist():
                if "label" in name.lower() and name.lower().endswith(".txt"):
                    text = z.read(name).decode("utf-8", errors="replace")
                    for line in text.splitlines():
                        parts = line.split()
                        if len(parts) >= 2:
                            fname, lbl = parts[0], parts[1]
                            label_map[fname] = lbl
                            label_map[Path(fname).name] = lbl
                    break
    except Exception as e:
        logger.warning(f"Could not read RAF-DB labels from ZIP: {e}")
    return label_map


def _load_affectnet_label_map(affectnet_dir: Path) -> dict:
    """Load AffectNet labels.csv into a lookup dict."""
    label_map = {}
    labels_csv = Path(affectnet_dir) / "labels.csv"
    if not labels_csv.exists():
        return label_map
    emotion_name_to_int = {
        "anger": 0, "angry": 0, "disgust": 1, "fear": 2,
        "happy": 3, "happiness": 3, "sad": 4, "sadness": 4,
        "surprise": 5, "neutral": 6,
    }
    try:
        df_labels = pd.read_csv(labels_csv)
        for _, row in df_labels.iterrows():
            pth = str(row.get("pth", "")).strip()
            lbl = str(row.get("label", "")).strip().lower()
            val = emotion_name_to_int.get(lbl, "unknown")
            if pth:
                label_map[pth.replace("\\", "/")] = val
                label_map[Path(pth).name] = val
    except Exception as e:
        logger.warning(f"Could not parse AffectNet labels.csv: {e}")
    return label_map


def rebuild_prepared_csv_from_images(
    images_root: str,
    dataset_name: str,
    out_csv: str,
    fer_orig_csv: str = None, #type:ignore
    raf_zip_path: str = None, #type:ignore
    affectnet_dir: str = None, #type:ignore
) -> Path:
    """
    Fast path when images already exist but prepared CSV was deleted.
    Scans images_root/{dataset_name}/ and writes a prepared CSV with correct labels.
    """
    images_root = Path(images_root) #type:ignore
    out_csv = Path(out_csv) #type:ignore
    ensure_dir(out_csv.parent) #type:ignore

    folder_map = {"FER2013": "FER2013", "RAFDB": "RAFDB", "AffectNet": "AffectNet"}
    folder = folder_map.get(dataset_name, dataset_name)
    img_dir = images_root / folder #type:ignore
    if not img_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {img_dir}")

    usage_map = {}
    if dataset_name == "FER2013" and fer_orig_csv and Path(fer_orig_csv).exists():
        try:
            df_fer, fmt, _, has_usage = _detect_fer2013_format(Path(fer_orig_csv))
            if has_usage and "Usage" in df_fer.columns:
                for i in range(len(df_fer)):
                    usage_map[i] = str(df_fer.iloc[i]["Usage"]).strip()
        except Exception as e:
            logger.warning(f"Could not load FER2013 Usage column: {e}")

    raf_labels = _load_rafdb_label_map(Path(raf_zip_path)) if dataset_name == "RAFDB" else {}
    aff_labels = _load_affectnet_label_map(Path(affectnet_dir)) if dataset_name == "AffectNet" else {}

    # Scan only relevant subfolders
    if dataset_name == "AffectNet":
        scan_dirs = [d for d in (img_dir / "Train", img_dir / "Test") if d.exists()]
        if not scan_dirs:
            scan_dirs = [img_dir]
    else:
        scan_dirs = [img_dir]

    rows = []
    for scan_dir in scan_dirs:
        for p in sorted(scan_dir.rglob("*")):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            rel = f"{folder}/{p.relative_to(img_dir).as_posix()}".replace("\\", "/")
            label = "unknown"

            if dataset_name == "FER2013":
                label = 0
                stem = p.stem
                usage = ""
                if stem.startswith("fer2013_"):
                    try:
                        idx = int(stem.split("_")[-1])
                        usage = usage_map.get(idx, "")
                    except ValueError:
                        pass
                rows.append({"image_path": rel, "label": label, "dataset": dataset_name, "Usage": usage})
            elif dataset_name == "RAFDB":
                label = raf_labels.get(p.name, raf_labels.get(p.stem, "unknown"))
                rows.append({"image_path": rel, "label": label, "dataset": dataset_name})
            elif dataset_name == "AffectNet":
                label = aff_labels.get(rel.replace(f"{folder}/", ""), aff_labels.get(p.name, "unknown"))
                if label != "unknown":
                    rows.append({"image_path": rel, "label": label, "dataset": dataset_name})

    if not rows:
        logger.warning(f"No images found to rebuild CSV for {dataset_name}")
        return out_csv #type:ignore

    df = pd.DataFrame(rows)

    if dataset_name == "FER2013" and fer_orig_csv:
        try:
            df_fer, fmt, pixel_cols, _ = _detect_fer2013_format(Path(fer_orig_csv))
            for i, row in df.iterrows():
                stem = Path(row["image_path"]).stem
                if stem.startswith("fer2013_"):
                    idx = int(stem.split("_")[-1])
                    if idx < len(df_fer):
                        if fmt == "onehot_columns":
                            oh = [
                                int(float(df_fer.iloc[idx].get(c, df_fer.iloc[idx].get(str(c), 0))))
                                for c in range(2304, 2311)
                            ]
                            df.at[i, "label"] = int(np.argmax(oh))
                        elif "emotion" in df_fer.columns:
                            df.at[i, "label"] = int(df_fer.iloc[idx]["emotion"])
                        elif 0 in df_fer.columns:
                            df.at[i, "label"] = int(df_fer.iloc[idx][0])
        except Exception as e:
            logger.warning(f"Could not re-derive FER2013 labels: {e}")

    from dataset_balancer import map_labels_to_emotions
    df["dataset"] = dataset_name
    df = map_labels_to_emotions(df, dataset_name)
    df = df[df["label"] != "unknown"].copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].notna() & df["label"].isin(range(7))].copy()
    df["label"] = df["label"].astype(int)
    df.to_csv(out_csv, index=False)
    logger.info(f"Rebuilt {dataset_name} CSV from images: {len(df)} rows → {out_csv}")
    return out_csv #type:ignore

# -------------------------------------------------------------------------
# Create unified CSV
# -------------------------------------------------------------------------
def unify(fer_csv=None, raf_csv=None, aff_csv=None, images_root=None, unified_csv_out=None):
    """Unify multiple dataset CSVs into one. All CSV parameters are optional."""
    logger.info("Combining all datasets into unified CSV...")
    dfs = []
    
    if fer_csv and Path(fer_csv).exists():
        df = pd.read_csv(fer_csv)
        if len(df) > 0:
            dfs.append(df)
            logger.info(f"  Added FER2013: {len(df)} images")
        else:
            logger.warning(f"  FER2013 CSV exists but is empty, skipping...")
    
    if raf_csv and Path(raf_csv).exists():
        df = pd.read_csv(raf_csv)
        if len(df) > 0:
            dfs.append(df)
            logger.info(f"  Added RAF-DB: {len(df)} images")
        else:
            logger.warning(f"  RAF-DB CSV exists but is empty, skipping...")
    
    if aff_csv and Path(aff_csv).exists():
        df = pd.read_csv(aff_csv)
        if len(df) > 0:
            dfs.append(df)
            logger.info(f"  Added AffectNet: {len(df)} images")
        else:
            logger.warning(f"  AffectNet CSV exists but is empty, skipping...")
    
    if not dfs:
        raise ValueError("No valid datasets to unify! Check that at least one dataset CSV has data.")
    
    all_df = pd.concat(dfs, ignore_index=True)

    # Validate + canonicalise labels ONCE, then stamp the mapped flag so no
    # downstream stage double-remaps (fixes R1-Q6/R2-Q4/R3-Q1 cross-dataset collapse).
    from label_semantics import (
        map_labels_to_emotions, attach_label_names, MAPPED_FLAG, CANONICAL_ID_TO_NAME,
    )

    for dataset_name in all_df["dataset"].unique():
        all_df = map_labels_to_emotions(all_df, dataset_name)
    all_df = all_df[all_df["label"] != "unknown"].copy()
    all_df["label"] = pd.to_numeric(all_df["label"], errors="coerce")
    all_df = all_df[all_df["label"].notna() & all_df["label"].isin(range(7))].copy()
    all_df["label"] = all_df["label"].astype(int)
    all_df = attach_label_names(all_df)          # canonical label_name column
    all_df[MAPPED_FLAG] = True                    # idempotency sentinel

    all_df.to_csv(unified_csv_out, index=False)
    logger.info(f"Unified dataset CSV saved: {unified_csv_out} ({len(all_df)} total images)")
    return unified_csv_out

# -------------------------------------------------------------------------
# CLI USAGE
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Default paths for standalone execution - use current directory
    BASE_DIR = Path(__file__).parent / "emotion_q1_framework" if Path(__file__).parent.name != "emotion_q1_framework" else Path(__file__).parent
    if not BASE_DIR.exists():
        BASE_DIR = Path.cwd() / "emotion_q1_framework"
    if not BASE_DIR.exists():
        BASE_DIR = Path(__file__).parent
        
    DATASET_DIR = BASE_DIR / "Dataset"
    # Auto-detect nested structure if dataset files are present there
    if (DATASET_DIR / "Dataset" / "fer2013.csv").exists() or (DATASET_DIR / "Dataset" / "af-db.zip").exists():
        DATASET_DIR = DATASET_DIR / "Dataset"
        
    fer_in = DATASET_DIR / "fer2013.csv"
    raf_in = DATASET_DIR / "af-db.zip"
    aff_in = DATASET_DIR / "AffectNet"
    
    images_root = DATASET_DIR / "images" if (DATASET_DIR / "images").parent.name == "Dataset" else BASE_DIR / "images"
    out_dir      = DATASET_DIR / "prepared" if (DATASET_DIR / "prepared").parent.name == "Dataset" else BASE_DIR / "prepared"

    ensure_dir(Path(images_root))
    ensure_dir(Path(out_dir))

    fer_csv_out = str(Path(out_dir) / "fer2013_prepared.csv")
    raf_csv_out = str(Path(out_dir) / "rafdb_prepared.csv")
    aff_csv_out = str(Path(out_dir) / "affectnet_prepared.csv")

    if fer_in.exists():
        prepare_fer2013(str(fer_in), str(images_root), fer_csv_out)
    if raf_in.exists():
        prepare_rafdb(str(raf_in), str(images_root), raf_csv_out)
    if aff_in.exists():
        prepare_affectnet(str(aff_in), str(images_root), aff_csv_out)

    unify(
        fer_csv_out if fer_in.exists() else None,
        raf_csv_out if raf_in.exists() else None,
        aff_csv_out if aff_in.exists() else None,
        str(images_root),
        Path(out_dir) / "unified_dataset.csv"
    )

    logger.info("All datasets prepared successfully.")
