"""Statistical analysis and comparison tools for CV results.

This script provides statistical testing utilities to compare backbones:
1. Paired t-test: Compare two backbones across CV folds
2. Effect size (Cohen's d): Measure practical significance
3. Visualization: Comparison plots with error bars

Usage:
    # Compare resnet50 vs efficientnetv2
    python scripts/statistical_analysis.py --runs_dir outputs/runs \\
        --backbone1 resnet50 --backbone2 tf_efficientnetv2_s --metric macro_f1
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def load_backbone_fold_metrics(runs_dir: Path, backbone: str, metric: str) -> Tuple[List[float], List[int]]:
    """Load metric values for a specific backbone across folds."""
    fold_values = []
    fold_ids = []
    
    for run_path in runs_dir.iterdir():
        if not run_path.is_dir():
            continue
        
        run_meta_path = run_path / "run_meta.json"
        test_metrics_path = run_path / "test_metrics.json"
        
        if not (run_meta_path.exists() and test_metrics_path.exists()):
            continue
        
        with open(run_meta_path, "r") as f:
            meta = json.load(f)
        
        if meta.get("backbone") != backbone or meta.get("cv_fold") is None:
            continue
        
        with open(test_metrics_path, "r") as f:
            metrics = json.load(f)
        
        # Extract metric value
        if metric in metrics:
            value = metrics[metric]
        elif metric.startswith("recall_"):
            class_name = metric.replace("recall_", "")
            value = metrics.get("per_class_recall", {}).get(class_name)
        else:
            continue
        
        if value is not None:
            fold_values.append(float(value))
            fold_ids.append(int(meta["cv_fold"]))
    
    # Sort by fold_id
    sorted_pairs = sorted(zip(fold_ids, fold_values))
    fold_ids = [p[0] for p in sorted_pairs]
    fold_values = [p[1] for p in sorted_pairs]
    
    return fold_values, fold_ids


def paired_ttest(values1: List[float], values2: List[float]) -> Dict:
    """Perform paired t-test."""
    if len(values1) != len(values2):
        raise ValueError(f"Mismatched number of folds: {len(values1)} vs {len(values2)}")
    
    if len(values1) < 2:
        return {
            "t_statistic": np.nan,
            "p_value": np.nan,
            "significant": False,
            "note": "Insufficient samples for t-test (need >= 2 folds)"
        }
    
    arr1 = np.array(values1)
    arr2 = np.array(values2)
    
    t_stat, p_value = stats.ttest_rel(arr1, arr2)
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "mean_diff": float(np.mean(arr1 - arr2)),
        "std_diff": float(np.std(arr1 - arr2, ddof=1)),
    }


def cohens_d(values1: List[float], values2: List[float]) -> float:
    """Compute Cohen's d effect size for paired data."""
    arr1 = np.array(values1)
    arr2 = np.array(values2)
    diff = arr1 - arr2
    
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    
    if std_diff == 0:
        return 0.0
    
    return float(mean_diff / std_diff)


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def plot_comparison(
    backbone1: str,
    backbone2: str,
    values1: List[float],
    values2: List[float],
    metric: str,
    output_path: Path
):
    """Create comparison visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Box plot
    data = [values1, values2]
    labels = [backbone1, backbone2]
    ax1.boxplot(data, labels=labels, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax1.set_ylabel(metric.replace("_", " ").title())
    ax1.set_title(f"Distribution Across Folds")
    ax1.grid(axis='y', alpha=0.3)
    
    # Fold-by-fold comparison
    fold_ids = list(range(len(values1)))
    x = np.array(fold_ids)
    width = 0.35
    
    ax2.bar(x - width/2, values1, width, label=backbone1, alpha=0.8)
    ax2.bar(x + width/2, values2, width, label=backbone2, alpha=0.8)
    ax2.set_xlabel('Fold ID')
    ax2.set_ylabel(metric.replace("_", " ").title())
    ax2.set_title('Per-Fold Comparison')
    ax2.set_xticks(x)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved comparison plot to: {output_path}")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Statistical comparison of CV results")
    ap.add_argument("--runs_dir", default="outputs/runs", help="Directory containing run folders")
    ap.add_argument("--backbone1", required=True, help="First backbone to compare")
    ap.add_argument("--backbone2", required=True, help="Second backbone to compare")
    ap.add_argument("--metric", default="macro_f1", help="Metric to compare (e.g., macro_f1, recall_MEL)")
    ap.add_argument("--plot", action="store_true", help="Generate comparison plot")
    ap.add_argument("--plot_out", default="outputs/comparison.png", help="Plot output path")
    args = ap.parse_args()
    
    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    
    print(f"[INFO] Loading fold results for {args.backbone1}...")
    values1, folds1 = load_backbone_fold_metrics(runs_dir, args.backbone1, args.metric)
    
    print(f"[INFO] Loading fold results for {args.backbone2}...")
    values2, folds2 = load_backbone_fold_metrics(runs_dir, args.backbone2, args.metric)
    
    if not values1:
        raise ValueError(f"No CV fold results found for {args.backbone1}")
    if not values2:
        raise ValueError(f"No CV fold results found for {args.backbone2}")
    
    if len(values1) != len(values2):
        print(f"[WARN] Mismatched number of folds: {args.backbone1}={len(values1)}, {args.backbone2}={len(values2)}")
        print(f"  {args.backbone1} folds: {folds1}")
        print(f"  {args.backbone2} folds: {folds2}")
        # Try to align by common folds
        common_folds = set(folds1).intersection(set(folds2))
        if not common_folds:
            raise ValueError("No common folds found between backbones")
        
        print(f"[INFO] Using common folds: {sorted(common_folds)}")
        values1 = [v for i, v in zip(folds1, values1) if i in common_folds]
        values2 = [v for i, v in zip(folds2, values2) if i in common_folds]
    
    # Compute statistics
    print(f"\n=== Statistical Comparison: {args.backbone1} vs {args.backbone2} ===")
    print(f"Metric: {args.metric}")
    print(f"Number of folds: {len(values1)}\n")
    
    # Descriptive statistics
    print(f"{args.backbone1}:")
    print(f"  Mean: {np.mean(values1):.6f}")
    print(f"  Std:  {np.std(values1, ddof=1):.6f}")
    print(f"  Min:  {np.min(values1):.6f}")
    print(f"  Max:  {np.max(values1):.6f}")
    
    print(f"\n{args.backbone2}:")
    print(f"  Mean: {np.mean(values2):.6f}")
    print(f"  Std:  {np.std(values2, ddof=1):.6f}")
    print(f"  Min:  {np.min(values2):.6f}")
    print(f"  Max:  {np.max(values2):.6f}")
    
    # Paired t-test
    print("\n--- Paired t-test ---")
    ttest_result = paired_ttest(values1, values2)
    if "note" in ttest_result:
        print(f"Note: {ttest_result['note']}")
    else:
        print(f"t-statistic: {ttest_result['t_statistic']:.4f}")
        print(f"p-value:     {ttest_result['p_value']:.6f}")
        print(f"Significant: {'YES (p < 0.05)' if ttest_result['significant'] else 'NO (p >= 0.05)'}")
        print(f"Mean diff:   {ttest_result['mean_diff']:.6f} ± {ttest_result['std_diff']:.6f}")
    
    # Effect size
    print("\n--- Effect Size ---")
    d = cohens_d(values1, values2)
    interpretation = interpret_cohens_d(d)
    print(f"Cohen's d: {d:.4f} ({interpretation})")
    
    if d > 0:
        print(f"Interpretation: {args.backbone1} performs better on average (positive effect)")
    elif d < 0:
        print(f"Interpretation: {args.backbone2} performs better on average (negative effect)")
    else:
        print(f"Interpretation: No difference in performance")
    
    # Plot
    if args.plot:
        output_path = Path(args.plot_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plot_comparison(args.backbone1, args.backbone2, values1, values2, args.metric, output_path)
    
    # Conclusion
    print("\n=== Conclusion ===")
    if ttest_result.get("significant", False):
        winner = args.backbone1 if ttest_result["mean_diff"] > 0 else args.backbone2
        print(f"✓ {winner} is STATISTICALLY SIGNIFICANTLY better")
        print(f"  (p={ttest_result['p_value']:.6f}, Cohen's d={d:.4f} [{interpretation}])")
    else:
        print(f"✗ No statistically significant difference found")
        print(f"  (p={ttest_result.get('p_value', 'N/A')}, Cohen's d={d:.4f} [{interpretation}])")


if __name__ == "__main__":
    main()
