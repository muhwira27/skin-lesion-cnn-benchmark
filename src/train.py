from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import inspect
import torch.nn as nn
from torch.cuda.amp import GradScaler as CudaGradScaler

try:
    from torch import amp as torch_amp
    _autocast = torch_amp.autocast
except Exception:  # pragma: no cover
    from torch.cuda.amp import autocast as _autocast
from tqdm import tqdm

from dataset import PADUfesConfig, make_loaders
from metrics import compute_classification_metrics
from model import build_model, freeze_backbone, unfreeze_all, unfreeze_head
from utils import load_yaml, save_json, set_seed, now_tag, get_device, count_params_m


def compute_class_weights(train_df, label2id: Dict[str, int]) -> torch.Tensor:
    counts = train_df["diagnostic"].value_counts().to_dict()
    n_classes = len(label2id)
    w = np.zeros(n_classes, dtype=np.float32)
    for label, idx in label2id.items():
        w[idx] = 1.0 / max(counts.get(label, 1), 1)
    w = w / w.sum() * n_classes
    return torch.tensor(w, dtype=torch.float32)


def make_scheduler(optimizer, total_epochs: int, warmup_epochs: int):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        t = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def run_eval(model, loader, device, id2label):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        y_true.append(y.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return compute_classification_metrics(y_true, y_pred, id2label)


def train_one_epoch(model, loader, optimizer, scaler, device, criterion, amp: bool, grad_clip_norm: float, epoch: int):
    model.train()
    losses = []
    pbar = tqdm(loader, desc=f"train e{epoch}", leave=True)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if "device_type" in inspect.signature(_autocast).parameters:
            ctx = _autocast(enabled=amp, device_type=device.type)
        else:
            ctx = _autocast(enabled=amp)
        with ctx:
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        if grad_clip_norm and grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        losses.append(float(loss.detach().cpu()))
        pbar.set_postfix(loss=float(np.mean(losses)))
    return float(np.mean(losses)) if losses else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--cv_fold", type=int, default=None, help="CV fold ID (0-indexed). If provided, uses CV splits.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    label2id = cfg["labels"]["label2id"]
    id2label = {int(k): v for k, v in cfg["labels"]["id2label"].items()}

    seed = int(cfg["seed"])
    set_seed(seed)

    device = get_device(cfg["device"])
    amp = bool(cfg.get("amp", True)) and (device.type == "cuda")
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"[INFO] device={device.type} cuda_available={cuda_available} amp={amp} device_name={device_name}")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        sig = inspect.signature(torch.amp.GradScaler)
        if "device" in sig.parameters:
            scaler = torch.amp.GradScaler(device=device.type, enabled=amp)
        elif "device_type" in sig.parameters:
            scaler = torch.amp.GradScaler(device_type=device.type, enabled=amp)
        else:
            if device.type == "cuda":
                scaler = torch.amp.GradScaler("cuda", enabled=amp)
            else:
                scaler = torch.amp.GradScaler(enabled=amp)
    else:
        scaler = CudaGradScaler(enabled=amp)

    img_size = int(cfg["train"]["img_size"])
    
    # Build run name with CV fold if applicable
    if args.cv_fold is not None:
        fold_suffix = f"_fold{args.cv_fold}"
    else:
        fold_suffix = ""
    
    run_name = args.run_name or f"{args.backbone}{fold_suffix}_img{img_size}_seed{seed}_{now_tag()}"
    run_dir = Path("outputs/runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    sampler_name = "weighted_sampler" if bool(cfg["train"].get("weighted_sampler", False)) else "shuffle"
    run_meta = {
        "backbone": args.backbone,
        "seed": seed,
        "img_size": img_size,
        "cv_fold": args.cv_fold,  # Track which fold was used (None for single-split)
        "loss": {
            "type": str(cfg.get("loss", {}).get("name", "cross_entropy")),
            "class_weights": bool(cfg.get("loss", {}).get("class_weights", True)),
        },
        "sampler": sampler_name,
    }
    save_json(str(run_dir / "run_meta.json"), run_meta)

    # Auto-select splits file based on cv_fold
    if args.cv_fold is not None:
        # Use CV splits for cross-validation
        splits_path = "data/cv_splits.json"
    else:
        # Use regular splits for basic workflow
        splits_path = cfg["data"]["splits_path"]
    
    data_cfg = PADUfesConfig(
        img_dir=cfg["data"]["img_dir"],
        csv_path=cfg["data"]["csv_path"],
        splits_path=splits_path,
        img_size=img_size,
        label2id=label2id,
    )
    train_loader, val_loader, _, _, train_df = make_loaders(
        data_cfg,
        aug_cfg=cfg["augment"],
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]),
        use_weighted_sampler=bool(cfg["train"].get("weighted_sampler", False)),
        cv_fold=args.cv_fold,  # Pass CV fold to data loader
    )

    model = build_model(args.backbone, num_classes=len(label2id)).to(device)
    params_m = count_params_m(model)

    class_w = compute_class_weights(train_df, label2id).to(device) if cfg["loss"].get("class_weights", True) else None
    if class_w is not None:
        class_weight_map = {label: float(class_w[idx].cpu().item()) for label, idx in label2id.items()}
        run_meta["loss"]["class_weights_values"] = class_weight_map
        save_json(str(run_dir / "run_meta.json"), run_meta)
    label_smoothing = float(cfg["loss"].get("label_smoothing", 0.0))
    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=label_smoothing)

    stage1_epochs = int(cfg["train"]["stage1"]["epochs"])
    stage2_epochs = int(cfg["train"]["stage2"]["epochs"])
    warmup_epochs = int(cfg["train"].get("warmup_epochs", 0))
    grad_clip = float(cfg["train"].get("grad_clip_norm", 0.0))

    best_metric = -1.0
    best_path = run_dir / "best.ckpt"
    patience = int(cfg["train"]["early_stopping"]["patience"])
    patience_left = patience
    history = []
    label_order = [k for k, _ in sorted(label2id.items(), key=lambda x: x[1])]

    def format_recalls(recalls: Dict[str, float]) -> str:
        parts = []
        for label in label_order:
            val = recalls.get(label, 0.0)
            parts.append(f"{label}:{val:.3f}")
        return " ".join(parts)

    def save_ckpt(path: Path):
        payload = {
            "backbone": args.backbone,
            "img_size": img_size,
            "seed": seed,
            "label2id": label2id,
            "state_dict": model.state_dict(),
        }
        torch.save(payload, str(path))

    # Stage 1
    freeze_backbone(model)
    unfreeze_head(model)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg["train"]["stage1"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = make_scheduler(optimizer, total_epochs=stage1_epochs, warmup_epochs=min(warmup_epochs, stage1_epochs))

    for epoch in range(stage1_epochs):
        epoch_display = epoch + 1
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, criterion, amp, grad_clip, epoch_display)
        scheduler.step()
        val_m = run_eval(model, val_loader, device, id2label)

        history.append({"epoch": epoch_display, "stage": 1, "train_loss": train_loss, **val_m})
        save_json(str(run_dir / "val_metrics.json"), {
            "backbone": args.backbone, "img_size": img_size, "seed": seed, "params_m": params_m,
            **val_m, "history": history
        })

        score = float(val_m["macro_f1"])
        lr = float(optimizer.param_groups[0]["lr"])
        print(
            f"[VAL] epoch={epoch_display} stage=1 loss={train_loss:.4f} "
            f"macro_f1={val_m['macro_f1']:.4f} bal_acc={val_m['balanced_accuracy']:.4f} "
            f"mean_recall={val_m['mean_recall']:.4f} lr={lr:.6f} "
            f"recall={format_recalls(val_m.get('per_class_recall', {}))}"
        )
        if score > best_metric:
            best_metric = score
            patience_left = patience
            save_ckpt(best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    # Stage 2
    unfreeze_all(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["stage2"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = make_scheduler(optimizer, total_epochs=stage2_epochs, warmup_epochs=min(warmup_epochs, stage2_epochs))

    start_epoch = len(history)
    for e in range(stage2_epochs):
        epoch = start_epoch + e
        epoch_display = epoch + 1
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, criterion, amp, grad_clip, epoch_display)
        scheduler.step()
        val_m = run_eval(model, val_loader, device, id2label)

        history.append({"epoch": epoch_display, "stage": 2, "train_loss": train_loss, **val_m})
        save_json(str(run_dir / "val_metrics.json"), {
            "backbone": args.backbone, "img_size": img_size, "seed": seed, "params_m": params_m,
            **val_m, "history": history
        })

        score = float(val_m["macro_f1"])
        lr = float(optimizer.param_groups[0]["lr"])
        print(
            f"[VAL] epoch={epoch_display} stage=2 loss={train_loss:.4f} "
            f"macro_f1={val_m['macro_f1']:.4f} bal_acc={val_m['balanced_accuracy']:.4f} "
            f"mean_recall={val_m['mean_recall']:.4f} lr={lr:.6f} "
            f"recall={format_recalls(val_m.get('per_class_recall', {}))}"
        )
        if score > best_metric:
            best_metric = score
            patience_left = patience
            save_ckpt(best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    print(f"[DONE] Best val macro_f1={best_metric:.4f}  checkpoint={best_path}")


if __name__ == "__main__":
    main()
