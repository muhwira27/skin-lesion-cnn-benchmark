from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import inspect
from tqdm import tqdm

from dataset import PADUfesConfig, make_loaders
from metrics import compute_classification_metrics
from model import build_model
from utils import load_yaml, save_json, get_device, latest_run_dir, count_params_m


@torch.no_grad()
def predict(model, loader, device, amp: bool):
    model.eval()
    y_true, y_pred = [], []
    n_images = 0
    t_start = time.time()

    for x, y in tqdm(loader, desc="test", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            sig = inspect.signature(torch.amp.autocast)
            if "device_type" in sig.parameters:
                ctx = torch.amp.autocast(enabled=amp, device_type=device.type)
            else:
                ctx = torch.amp.autocast(enabled=amp)
        else:
            ctx = torch.cuda.amp.autocast(enabled=amp)
        with ctx:
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize()

        n_images += x.shape[0]
        y_true.append(y.cpu().numpy())
        y_pred.append(pred.cpu().numpy())

    t_end = time.time()
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    latency_ms = (t_end - t_start) * 1000.0 / max(1, n_images)
    return y_true, y_pred, float(latency_ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--run_dir", default=None, help="If empty, uses latest run for this backbone")
    ap.add_argument("--cv_fold", type=int, default=None, help="CV fold ID (optional, for consistency with training)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = get_device(cfg["device"])
    amp = bool(cfg.get("amp", True)) and (device.type == "cuda")

    run_dir = args.run_dir or latest_run_dir("outputs/runs", args.backbone)
    if run_dir is None:
        raise RuntimeError(f"No run found for {args.backbone} in outputs/runs/")
    run_dir = Path(run_dir)

    ckpt_path = run_dir / "best.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    label2id = ckpt["label2id"]
    id2label = {v: k for k, v in label2id.items()}
    img_size = int(ckpt.get("img_size", cfg["train"]["img_size"]))

    # Auto-select splits file based on cv_fold
    if args.cv_fold is not None:
        splits_path = "data/cv_splits.json"
    else:
        splits_path = cfg["data"]["splits_path"]

    data_cfg = PADUfesConfig(
        img_dir=cfg["data"]["img_dir"],
        csv_path=cfg["data"]["csv_path"],
        splits_path=splits_path,
        img_size=img_size,
        label2id=label2id,
    )
    _, _, test_loader, _, _ = make_loaders(
        data_cfg,
        aug_cfg=cfg["augment"],
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]),
        cv_fold=args.cv_fold,  # Support CV fold if needed
    )

    model = build_model(args.backbone, num_classes=len(label2id))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)

    params_m = count_params_m(model)

    y_true, y_pred, latency_ms = predict(model, test_loader, device, amp)
    m = compute_classification_metrics(y_true, y_pred, id2label)

    out = {
        "backbone": args.backbone,
        "img_size": img_size,
        "seed": int(ckpt.get("seed", cfg["seed"])),
        "params_m": float(params_m),
        "latency_ms": float(latency_ms),
        **m,
    }
    save_json(str(run_dir / "test_metrics.json"), out)
    print(f"[OK] {run_dir / 'test_metrics.json'}")
    print(f"macro_f1={out['macro_f1']:.4f}  balanced_acc={out['balanced_accuracy']:.4f}  latency_ms={out['latency_ms']:.2f}")


if __name__ == "__main__":
    main()
