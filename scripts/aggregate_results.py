"""Unified results aggregation script.

Auto-detects workflow type (basic vs CV) and aggregates results accordingly.

Usage:
    # Auto-detect and aggregate
    python scripts/aggregate_results.py --runs_dir outputs/runs --out outputs/report.csv
    
    # Force specific mode
    python scripts/aggregate_results.py --mode cv --runs_dir outputs/runs --out outputs/cv_report.csv
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None


def detect_workflow_mode(runs_dir: Path) -> str:
    """Auto-detect if runs are CV or basic based on folder names."""
    if not runs_dir.exists():
        return "unknown"
    
    runs = [d for d in runs_dir.iterdir() if d.is_dir()]
    cv_runs = [d for d in runs if "_fold" in d.name]
    
    if not runs:
        return "empty"
    elif len(cv_runs) >= len(runs) * 0.5:  # >50% are CV runs
        return "cv"
    else:
        return "basic"


def compute_bootstrap_ci(values: np.ndarray, confidence: float = 0.95, n_bootstrap: int = 10000):
    """Compute bootstrap confidence interval."""
    if len(values) < 2:
        val = float(values[0]) if len(values) == 1 else np.nan
        return val, val
    
    if scipy_stats is None:
        # Fallback: use mean ± std * 1.96 (approximate 95% CI)
        mean = np.mean(values)
        margin = 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))
        return float(mean - margin), float(mean + margin)
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
    
    return float(lower), float(upper)


def aggregate_basic(runs_dir: Path) -> pd.DataFrame:
    """Aggregate basic (non-CV) runs - simple summary."""
    rows = []
    
    for run_dir in sorted(runs_dir.glob("*")):
        if not run_dir.is_dir():
            continue
        
        # Skip CV fold runs
        if "_fold" in run_dir.name:
            continue
        
        # Prefer test metrics; fallback to val
        metrics_path = run_dir / "test_metrics.json"
        split_name = "test"
        if not metrics_path.exists():
            metrics_path = run_dir / "val_metrics.json"
            split_name = "val"
        if not metrics_path.exists():
            continue
        
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        
        row = {
            "run_dir": str(run_dir),
            "split": split_name,
            "backbone": m.get("backbone", run_dir.name),
            "img_size": m.get("img_size"),
            "seed": m.get("seed"),
            "macro_f1": m.get("macro_f1"),
            "balanced_accuracy": m.get("balanced_accuracy"),
            "mean_recall": m.get("mean_recall"),
            "params_m": m.get("params_m"),
            "latency_ms": m.get("latency_ms"),
        }
        
        # Add per-class recall
        per_class = m.get("per_class_recall", {}) or {}
        for k, v in per_class.items():
            row[f"recall_{k}"] = v
        
        rows.append(row)
    
    if not rows:
        raise RuntimeError(f"No runs found in {runs_dir}")
    
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False)


def aggregate_cv(runs_dir: Path) -> pd.DataFrame:
    """Aggregate CV runs with statistics."""
    backbone_folds = defaultdict(list)
    
    for run_path in runs_dir.iterdir():
        if not run_path.is_dir():
            continue
        
        # Only process CV fold runs
        if "_fold" not in run_path.name:
            continue
        
        test_metrics_path = run_path / "test_metrics.json"
        run_meta_path = run_path / "run_meta.json"
        
        if not test_metrics_path.exists():
            continue
        
        # Load metrics
        with open(test_metrics_path, "r") as f:
            metrics = json.load(f)
        
        # Load metadata to get fold info
        cv_fold = None
        if run_meta_path.exists():
            with open(run_meta_path, "r") as f:
                meta = json.load(f)
                cv_fold = meta.get("cv_fold")
        
        if cv_fold is not None:
            backbone = metrics["backbone"]
            metrics["cv_fold"] = cv_fold
            metrics["run_dir"] = str(run_path)
            backbone_folds[backbone].append(metrics)
    
    if not backbone_folds:
        raise RuntimeError(f"No CV fold results found in {runs_dir}")
    
    # Aggregate metrics across folds
    rows = []
    
    for backbone, fold_results in backbone_folds.items():
        n_folds = len(fold_results)
        fold_results = sorted(fold_results, key=lambda x: x["cv_fold"])
        
        # Aggregate scalar metrics
        scalar_metrics = ["macro_f1", "balanced_accuracy", "mean_recall", "params_m", "latency_ms"]
        
        for metric_name in scalar_metrics:
            values = np.array([fold[metric_name] for fold in fold_results])
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if n_folds > 1 else 0.0
            
            if n_folds > 1:
                ci_lower, ci_upper = compute_bootstrap_ci(values, confidence=0.95)
            else:
                ci_lower = ci_upper = mean
            
            rows.append({
                "backbone": backbone,
                "metric": metric_name,
                "mean": mean,
                "std": std,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n_folds": n_folds,
            })
        
        # Aggregate per-class recall
        if "per_class_recall" in fold_results[0]:
            class_names = list(fold_results[0]["per_class_recall"].keys())
            
            for class_name in class_names:
                values = np.array([
                    fold["per_class_recall"][class_name]
                    for fold in fold_results
                ])
                
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1)) if n_folds > 1 else 0.0
                
                if n_folds > 1:
                    ci_lower, ci_upper = compute_bootstrap_ci(values, confidence=0.95)
                else:
                    ci_lower = ci_upper = mean
                
                rows.append({
                    "backbone": backbone,
                    "metric": f"recall_{class_name}",
                    "mean": mean,
                    "std": std,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "n_folds": n_folds,
                })
    
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Aggregate benchmark results (auto-detects basic vs CV)")
    ap.add_argument("--runs_dir", default="outputs/runs", help="Directory containing run folders")
    ap.add_argument("--out", default="outputs/report.csv", help="Output CSV path")
    ap.add_argument("--mode", choices=["auto", "basic", "cv"], default="auto", 
                    help="Aggregation mode (auto-detect by default)")
    ap.add_argument("--verbose", action="store_true", help="Print detailed statistics")
    args = ap.parse_args()
    
    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    
    # Detect mode if auto
    if args.mode == "auto":
        mode = detect_workflow_mode(runs_dir)
        if mode == "empty":
            raise RuntimeError(f"No runs found in {runs_dir}")
        elif mode == "unknown":
            mode = "basic"  # Default fallback
        print(f"[INFO] Auto-detected mode: {mode}")
    else:
        mode = args.mode
    
    print(f"[INFO] Aggregating results from: {runs_dir}")
    print(f"[INFO] Mode: {mode}")
    
    # Aggregate based on mode
    if mode == "cv":
        df = aggregate_cv(runs_dir)
        
        if args.verbose:
            print("\n=== Top Backbones by Macro-F1 ===")
            macro_f1_df = df[df["metric"] == "macro_f1"].sort_values("mean", ascending=False)
            for _, row in macro_f1_df.iterrows():
                print(f"{row['backbone']:25s}: {row['mean']:.4f} ± {row['std']:.4f} "
                      f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] (n={row['n_folds']})")
    else:
        df = aggregate_basic(runs_dir)
        
        if args.verbose:
            print("\n=== Top Backbones by Macro-F1 ===")
            top5 = df.head(5)
            for _, row in top5.iterrows():
                print(f"{row['backbone']:25s}: {row['macro_f1']:.4f}")
    
    # Save to CSV
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format="%.6f")
    
    print(f"\n[OK] Saved aggregated results to: {output_path}")
    print(f"Total runs processed: {len(df) if mode != 'cv' else sum(df['n_folds'].unique())}")


if __name__ == "__main__":
    main()
