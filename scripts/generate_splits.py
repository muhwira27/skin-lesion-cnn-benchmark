"""Unified split generation script for PAD-UFES-20.

Supports both single train/val/test split and K-fold cross-validation.

Usage:
    # Generate single split
    python scripts/generate_splits.py --mode single
    
    # Generate 5-fold CV splits
    python scripts/generate_splits.py --mode cv --n_folds 5
"""

import argparse
import json
import os
from collections import Counter
from typing import List, Tuple, Optional, Dict

import pandas as pd

try:
    from sklearn.model_selection import StratifiedKFold, train_test_split
except Exception as e:
    raise RuntimeError(
        "scikit-learn is required. Install with: pip install scikit-learn"
    ) from e


REQUIRED_COLS = {"img_id", "patient_id", "diagnostic"}


def patient_primary_label(df: pd.DataFrame) -> pd.Series:
    """Primary label per patient for stratification."""
    counts = (
        df.groupby(["patient_id", "diagnostic"])
          .size()
          .reset_index(name="n")
          .sort_values(["patient_id", "n"], ascending=[True, False])
    )
    primary = counts.drop_duplicates("patient_id").set_index("patient_id")["diagnostic"]
    return primary


def image_class_counts(df: pd.DataFrame, patient_ids: List[str]) -> Counter:
    """Count images per class for given patient IDs."""
    sub = df[df["patient_id"].isin(patient_ids)]
    return Counter(sub["diagnostic"].tolist())


def check_constraints(
    df: pd.DataFrame,
    train_p: List[str],
    val_p: List[str],
    test_p: List[str],
    min_mel_val: int,
    min_mel_test: int
) -> Tuple[bool, Dict]:
    """Check if a split/fold satisfies all constraints."""
    train_c = image_class_counts(df, train_p)
    val_c = image_class_counts(df, val_p)
    test_c = image_class_counts(df, test_p)
    
    classes = sorted(df["diagnostic"].unique().tolist())
    
    for cls in classes:
        if val_c.get(cls, 0) < 1:
            return False, {"reason": f"class_missing_in_val:{cls}"}
        if test_c.get(cls, 0) < 1:
            return False, {"reason": f"class_missing_in_test:{cls}"}
    
    if val_c.get("MEL", 0) < min_mel_val:
        return False, {"reason": "mel_too_few_in_val"}
    if test_c.get("MEL", 0) < min_mel_test:
        return False, {"reason": "mel_too_few_in_test"}
    
    return True, {
        "train_counts": dict(train_c),
        "val_counts": dict(val_c),
        "test_counts": dict(test_c)
    }


def normalize_patient_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize patient_id column to stable string format."""
    df = df.copy()
    pid = df["patient_id"].astype(str).str.strip()
    
    if not pid.str.fullmatch(r"\d+").all():
        extracted = pid.str.extract(r"(\d+)", expand=False)
        pid = extracted.fillna(pid)
    
    df["patient_id"] = pid
    return df


def generate_single_split(df, train_ratio, val_ratio, test_ratio, min_mel_val, min_mel_test, seed_start, trials):
    """Generate single train/val/test split."""
    patients = sorted(df["patient_id"].unique().tolist())
    primary = patient_primary_label(df)
    patient_labels = [primary.loc[p] for p in patients]
    
    for i in range(trials):
        seed = seed_start + i
        
        # Split patients: train_val vs test
        train_val_patients, test_patients = train_test_split(
            patients,
            test_size=test_ratio,
            random_state=seed,
            stratify=patient_labels,
        )
        
        # Split train_val into train vs val
        val_prop = val_ratio / (train_ratio + val_ratio)
        primary_tv = primary.loc[train_val_patients].tolist()
        train_patients, val_patients = train_test_split(
            train_val_patients,
            test_size=val_prop,
            random_state=seed + 999,
            stratify=primary_tv,
        )
        
        ok, stats = check_constraints(df, train_patients, val_patients, test_patients, min_mel_val, min_mel_test)
        
        if ok:
            return {
                "train_patients": sorted(train_patients),
                "val_patients": sorted(val_patients),
                "test_patients": sorted(test_patients),
                "seed": seed,
                "stats": stats,
            }, None
    
    return None, "Failed to find valid split after all trials"


def generate_cv_splits(df, n_folds, val_ratio, min_mel_val, min_mel_test, seed, trials):
    """Generate K-fold CV splits."""
    patients = sorted(df["patient_id"].unique().tolist())
    primary = patient_primary_label(df)
    patient_labels = [primary.loc[p] for p in patients]    
    for trial in range(trials):
        current_seed = seed + trial
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=current_seed)
        
        folds = []
        all_folds_valid = True
        
        for fold_id, (train_val_idx, test_idx) in enumerate(skf.split(patients, patient_labels)):
            train_val_patients = [patients[i] for i in train_val_idx]
            test_patients = [patients[i] for i in test_idx]
            
            # Further split train_val into train and val
            primary_tv = primary.loc[train_val_patients].tolist()
            try:
                train_patients, val_patients = train_test_split(
                    train_val_patients,
                    test_size=val_ratio,
                    random_state=current_seed + fold_id,
                    stratify=primary_tv,
                )
            except ValueError:
                all_folds_valid = False
                break
            
            ok, stats = check_constraints(df, train_patients, val_patients, test_patients, min_mel_val, min_mel_test)
            
            if not ok:
                all_folds_valid = False
                break
            
            folds.append({
                "fold_id": fold_id,
                "train_patients": sorted(train_patients),
                "val_patients": sorted(val_patients),
                "test_patients": sorted(test_patients),
                "stats": stats,
            })
        
        if all_folds_valid and len(folds) == n_folds:
            return folds, current_seed, None
    
    return None, None, "Failed to find valid CV splits after all trials"


def main():
    ap = argparse.ArgumentParser(description="Generate patient-level data splits for PAD-UFES-20")
    ap.add_argument("--mode", choices=["single", "cv"], default="cv", help="Split mode: single or cross-validation")
    ap.add_argument("--csv", default="data/metadata.csv", help="Path to metadata CSV")
    ap.add_argument("--out", default=None, help="Output path (auto-determined if not specified)")
    
    # Single split parameters
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    
    # CV parameters
    ap.add_argument("--n_folds", type=int, default=5, help="Number of folds for CV mode")
    
    # Common parameters
    ap.add_argument("--min_mel_val", type=int, default=3, help="Minimum MEL samples in val")
    ap.add_argument("--min_mel_test", type=int, default=3, help="Minimum MEL samples in test")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--trials", type=int, default=100, help="Number of trials to find valid split")
    
    args = ap.parse_args()
    
    # Auto-determine output path
    if args.out is None:
        args.out = "data/cv_splits.json" if args.mode == "cv" else "data/splits.json"
    
    # Load and validate data
    df = pd.read_csv(args.csv)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    
    df = df.dropna(subset=["img_id", "patient_id", "diagnostic"]).copy()
    df["diagnostic"] = df["diagnostic"].astype(str)
    df = normalize_patient_ids(df)
    
    print(f"[INFO] Mode: {args.mode}")
    print(f"[INFO] Total patients: {df['patient_id'].nunique()}")
    print(f"[INFO] Total images: {len(df)}")
    
    if args.mode == "single":
        # Generate single split
        print(f"[INFO] Generating single train/val/test split...")
        result, error = generate_single_split(
            df, args.train_ratio, args.val_ratio, args.test_ratio,
            args.min_mel_val, args.min_mel_test, args.seed, args.trials
        )
        
        if result is None:
            raise RuntimeError(f"Failed to generate split: {error}")
        
        output = result
        
    else:  # cv mode
        # Generate CV splits
        print(f"[INFO] Generating {args.n_folds}-fold CV splits...")
        folds, best_seed, error = generate_cv_splits(
            df, args.n_folds, args.val_ratio,
            args.min_mel_val, args.min_mel_test, args.seed, args.trials
        )
        
        if folds is None:
            raise RuntimeError(f"Failed to generate CV splits: {error}")
        
        output = {
            "n_folds": args.n_folds,
            "seed": best_seed,
            "stratified": True,
            "val_ratio": args.val_ratio,
            "folds": folds,
        }
    
    # Save to file
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n[OK] Wrote splits to: {args.out}")
    
    # Print summary
    if args.mode == "single":
        print(f"Seed: {output['seed']}")
        print(f"Train: {len(output['train_patients'])} patients")
        print(f"Val:   {len(output['val_patients'])} patients")
        print(f"Test:  {len(output['test_patients'])} patients")
    else:
        print(f"Seed: {output['seed']}")
        print(f"Folds: {len(output['folds'])}")
        for fold in output['folds']:
            print(f"  Fold {fold['fold_id']}: train={len(fold['train_patients'])}, "
                  f"val={len(fold['val_patients'])}, test={len(fold['test_patients'])}")


if __name__ == "__main__":
    main()
