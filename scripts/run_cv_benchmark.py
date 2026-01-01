import subprocess
import sys
from datetime import datetime
from pathlib import Path

# All available backbones
ALL_BACKBONES = [
    "resnet50",
    "densenet121", 
    "efficientnet_b0",
    "tf_efficientnetv2_s",
    "convnext_tiny",
    "seresnet50",
    "vit_small_patch16_224",
    "regnety_032",
    "mobilenet_v3_large",
    "shufflenet_v2_x1_0",
]

N_FOLDS = 5

def get_completed_runs(runs_dir):
    """Get set of completed (backbone, fold) combinations."""
    completed = set()
    runs_dir = Path(runs_dir)
    
    if not runs_dir.exists():
        return completed
    
    for run_path in runs_dir.iterdir():
        if not run_path.is_dir() or "_fold" not in run_path.name:
            continue
        
        # Check if training completed (has best.ckpt)
        if (run_path / "best.ckpt").exists():
            # Extract backbone and fold from folder name
            parts = run_path.name.split("_")
            fold_idx = next((i for i, p in enumerate(parts) if p.startswith("fold")), None)
            
            if fold_idx:
                backbone = "_".join(parts[:fold_idx])
                fold_num = int(parts[fold_idx].replace("fold", ""))
                completed.add((backbone, fold_num))
    
    return completed

def main():
    import argparse
    
    ap = argparse.ArgumentParser(description="Run CV benchmark training for selected backbones")
    ap.add_argument("--backbones", nargs="+", default=None,
                    help="Specific backbones to train (default: all). Example: --backbones resnet50 densenet121")
    ap.add_argument("--folds", nargs="+", type=int, default=None,
                    help="Specific folds to train (default: all 0-4). Example: --folds 0 1 2")
    ap.add_argument("--resume", action="store_true",
                    help="Skip already completed runs (checks for best.ckpt)")
    ap.add_argument("--runs_dir", default="outputs/runs",
                    help="Directory to check for completed runs (for --resume)")
    args = ap.parse_args()
    
    # Determine which backbones to train
    if args.backbones:
        backbones = args.backbones
        # Validate
        invalid = [b for b in backbones if b not in ALL_BACKBONES]
        if invalid:
            print(f"[ERROR] Invalid backbones: {invalid}")
            print(f"Available: {ALL_BACKBONES}")
            return
    else:
        backbones = ALL_BACKBONES
    
    # Determine which folds to train
    if args.folds:
        folds = args.folds
        if any(f < 0 or f >= N_FOLDS for f in folds):
            print(f"[ERROR] Invalid folds. Must be 0-{N_FOLDS-1}")
            return
    else:
        folds = list(range(N_FOLDS))
    
    # Get completed runs if resuming
    completed = get_completed_runs(args.runs_dir) if args.resume else set()
    
    total = len(backbones) * len(folds)
    completed_count = 0
    failed = []
    skipped = []
    
    start_time = datetime.now()
    
    print(f"[INFO] Training Configuration:")
    print(f"  Backbones: {backbones}")
    print(f"  Folds: {folds}")
    print(f"  Total runs: {total}")
    if args.resume:
        print(f"  Resume mode: ON (skipping completed runs)")
    print()
    
    for backbone in backbones:
        print(f"\n{'='*60}")
        print(f"Training: {backbone}")
        print(f"{'='*60}")
        
        for fold in folds:
            completed_count += 1
            
            # Check if already completed
            if args.resume and (backbone, fold) in completed:
                print(f"\n[{completed_count}/{total}] {backbone} - Fold {fold} [SKIP - already completed]")
                skipped.append(f"{backbone}_fold{fold}")
                continue
            
            print(f"\n[{completed_count}/{total}] {backbone} - Fold {fold}")
            
            cmd = [
                sys.executable,
                "src/train.py",
                "--config", "configs/config.yaml",
                "--backbone", backbone,
                "--cv_fold", str(fold)
            ]
            
            ret = subprocess.call(cmd)
            
            if ret != 0:
                print(f"[ERROR] Failed: {backbone} fold {fold}")
                failed.append(f"{backbone}_fold{fold}")
            else:
                print(f"[OK] Completed: {backbone} fold {fold}")
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Total runs: {total}")
    print(f"Successful: {total - len(failed) - len(skipped)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Failed: {len(failed)}")
    print(f"Duration: {duration}")
    
    if skipped:
        print(f"\nSkipped runs (already completed):")
        for s in skipped:
            print(f"  - {s}")
    
    if failed:
        print(f"\nFailed runs:")
        for f in failed:
            print(f"  - {f}")
        print(f"\nTo retry failed runs:")
        print(f"python scripts/run_cv_benchmark.py --backbones {' '.join(set(f.split('_fold')[0] for f in failed))}")

if __name__ == "__main__":
    main()
