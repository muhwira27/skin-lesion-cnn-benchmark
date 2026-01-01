import subprocess
import sys
from pathlib import Path

def main():
    import argparse
    
    ap = argparse.ArgumentParser(description="Evaluate all CV fold runs")
    ap.add_argument("--runs_dir", default="outputs/runs", help="Directory containing run folders")
    ap.add_argument("--backbones", nargs="+", default=None,
                    help="Specific backbones to evaluate (default: all). Example: --backbones resnet50 densenet121")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip runs that already have test_metrics.json")
    args = ap.parse_args()
    
    runs_dir = Path(args.runs_dir)
    
    if not runs_dir.exists():
        print("[ERROR] No runs directory found")
        return
    
    # Find all CV fold runs
    fold_runs = [d for d in runs_dir.iterdir() if d.is_dir() and "_fold" in d.name]
    
    if not fold_runs:
        print("[ERROR] No CV fold runs found")
        return
    
    # Filter by backbones if specified
    if args.backbones:
        filtered_runs = []
        for run_dir in fold_runs:
            parts = run_dir.name.split("_")
            fold_idx = next((i for i, p in enumerate(parts) if p.startswith("fold")), None)
            if fold_idx:
                backbone = "_".join(parts[:fold_idx])
                if backbone in args.backbones:
                    filtered_runs.append(run_dir)
        fold_runs = filtered_runs
    
    if not fold_runs:
        print(f"[ERROR] No runs found for backbones: {args.backbones}")
        return
    
    print(f"Found {len(fold_runs)} CV fold runs to evaluate")
    if args.backbones:
        print(f"Filtering for backbones: {args.backbones}")
    if args.skip_existing:
        print("Skipping runs with existing test_metrics.json")
    print()
    
    evaluated = 0
    skipped = 0
    failed = 0
    
    for i, run_dir in enumerate(sorted(fold_runs), 1):
        # Extract backbone and fold from dir name
        parts = run_dir.name.split("_")
        
        # Find fold index
        fold_idx = next((i for i, p in enumerate(parts) if p.startswith("fold")), None)
        if not fold_idx:
            continue
            
        backbone = "_".join(parts[:fold_idx])
        fold_num = parts[fold_idx].replace("fold", "")
        
        # Check if already evaluated
        if args.skip_existing and (run_dir / "test_metrics.json").exists():
            print(f"[{i}/{len(fold_runs)}] {backbone} fold {fold_num} [SKIP - already evaluated]")
            skipped += 1
            continue
        
        print(f"[{i}/{len(fold_runs)}] Evaluating: {backbone} fold {fold_num}")
        
        cmd = [
            sys.executable,
            "src/eval.py",
            "--config", "configs/config.yaml",
            "--backbone", backbone,
            "--cv_fold", fold_num,
            "--run_dir", str(run_dir)
        ]
        
        ret = subprocess.call(cmd)
        
        if ret != 0:
            print(f"[WARN] Evaluation failed for {run_dir.name}")
            failed += 1
        else:
            print(f"[OK] {run_dir.name}")
            evaluated += 1
        print()
    
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total runs found: {len(fold_runs)}")
    print(f"Evaluated: {evaluated}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"\n[DONE] All evaluations completed!")

if __name__ == "__main__":
    main()
