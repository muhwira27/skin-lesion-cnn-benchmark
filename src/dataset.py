from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Tuple, Set, List, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class PADUfesConfig:
    img_dir: str
    csv_path: str
    splits_path: str
    img_size: int
    label2id: Dict[str, int]


class PADUfesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform: A.Compose, label2id: Dict[str, int]):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.label2id = label2id

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_id = str(row["img_id"]).strip()
        label_str = str(row["diagnostic"]).strip()

        if label_str not in self.label2id:
            raise KeyError(
                f"Unknown label '{label_str}'. Known: {list(self.label2id.keys())}")

        label = int(self.label2id[label_str])

        path = self.img_dir / img_id
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = self.transform(image=img)
        x = out["image"]
        return x, label


def build_transforms(img_size: int, aug_cfg: Dict, is_train: bool) -> A.Compose:
    if is_train:
        scale_min = float(aug_cfg["random_resized_crop"]["scale_min"])
        scale_max = float(aug_cfg["random_resized_crop"]["scale_max"])
        rot = int(aug_cfg["rotate_deg"])
        hflip_p = float(aug_cfg["hflip_p"])
        bc = float(aug_cfg["brightness_contrast"])
        gamma = float(aug_cfg["gamma"])
        noise_p = float(aug_cfg["gaussian_noise_p"])
        blur_p = float(aug_cfg["blur_p"])

        import inspect
        rrc_kwargs = {
            "scale": (scale_min, scale_max),
            "ratio": (0.9, 1.1),
        }
        # Albumentations v1 uses height/width; v2 uses size.
        sig = inspect.signature(A.RandomResizedCrop)
        if "size" in sig.parameters:
            rrc_kwargs["size"] = (img_size, img_size)
        else:
            rrc_kwargs["height"] = img_size
            rrc_kwargs["width"] = img_size

        noise_kwargs = {"p": noise_p}
        sig_noise = inspect.signature(A.GaussNoise)
        if "var_limit" in sig_noise.parameters:
            noise_kwargs["var_limit"] = (10.0, 50.0)
        elif "std_range" in sig_noise.parameters:
            noise_kwargs["std_range"] = (
                math.sqrt(10.0) / 255.0, math.sqrt(50.0) / 255.0)

        return A.Compose(
            [
                A.RandomResizedCrop(**rrc_kwargs),
                A.Rotate(limit=rot, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
                A.HorizontalFlip(p=hflip_p),
                A.RandomBrightnessContrast(
                    brightness_limit=bc, contrast_limit=bc, p=0.7),
                A.RandomGamma(gamma_limit=(int((1 - gamma) * 100),
                              int((1 + gamma) * 100)), p=0.4),
                A.GaussNoise(**noise_kwargs),
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=3, p=1.0),
                        A.GaussianBlur(blur_limit=3, p=1.0),
                    ],
                    p=blur_p,
                ),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    else:
        # eval/val/test: deterministic
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )


def _norm_pid_series(pid: pd.Series) -> pd.Series:
    """
    Normalize patient_id column into a stable digit string.
    Accepts ints/strings like 'PAT_1516', '1516', '1516.0' and returns '1516'.
    """
    pid = pid.astype(str).str.strip()
    # Handle float-like "123.0"
    pid = pid.str.replace(r"\.0$", "", regex=True)
    # Extract digits if non-numeric
    extracted = pid.str.extract(r"(\d+)", expand=False)
    pid = extracted.fillna(pid)
    return pid


def _norm_pid_list(xs: List) -> Set[str]:
    """
    Normalize patient_id list from splits.json to the same format as _norm_pid_series.
    """
    out: List[str] = []
    import re

    for x in xs:
        s = str(x).strip()
        if s.endswith(".0") and s.replace(".0", "").isdigit():
            s = s.replace(".0", "")
        m = re.search(r"(\d+)", s)
        out.append(m.group(1) if m else s)
    return set(out)


def load_split_dfs(
    cfg: PADUfesConfig,
    cv_fold: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, str]]:
    """
    Load train/val/test dataframes from splits.
    
    Args:
        cfg: Dataset configuration
        cv_fold: If provided, load from CV splits (expects cv_splits.json format)
                 If None, load from regular splits.json
    
    Returns:
        train_df, val_df, test_df, id2label
    """
    df = pd.read_csv(cfg.csv_path)

    for c in ["img_id", "patient_id", "diagnostic"]:
        if c not in df.columns:
            raise ValueError(f"metadata.csv missing required column: {c}")

    df = df.dropna(subset=["img_id", "patient_id", "diagnostic"]).copy()
    df["img_id"] = df["img_id"].astype(str).str.strip()
    df["diagnostic"] = df["diagnostic"].astype(str).str.strip()
    df["patient_id"] = _norm_pid_series(df["patient_id"])

    import json
    with open(cfg.splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # Check if this is a CV splits file or regular splits file
    if "folds" in splits:
        # CV splits format
        if cv_fold is None:
            raise ValueError(
                f"CV splits file detected ({cfg.splits_path}), but cv_fold not specified. "
                "Please provide --cv_fold argument."
            )
        
        if cv_fold < 0 or cv_fold >= len(splits["folds"]):
            raise ValueError(
                f"cv_fold={cv_fold} out of range. Available folds: 0-{len(splits['folds'])-1}"
            )
        
        fold_data = splits["folds"][cv_fold]
        train_p = _norm_pid_list(fold_data["train_patients"])
        val_p = _norm_pid_list(fold_data["val_patients"])
        test_p = _norm_pid_list(fold_data["test_patients"])
    else:
        # Regular splits format
        if cv_fold is not None:
            raise ValueError(
                f"Regular splits file detected ({cfg.splits_path}), but cv_fold={cv_fold} was specified. "
                "Use CV splits file (--cv_splits_path) when using --cv_fold."
            )
        
        train_p = _norm_pid_list(splits["train_patients"])
        val_p = _norm_pid_list(splits["val_patients"])
        test_p = _norm_pid_list(splits["test_patients"])

    train_df = df[df["patient_id"].isin(train_p)].copy()
    val_df = df[df["patient_id"].isin(val_p)].copy()
    test_df = df[df["patient_id"].isin(test_p)].copy()

    id2label = {v: k for k, v in cfg.label2id.items()}
    return train_df, val_df, test_df, id2label


def build_weighted_sampler(train_df: pd.DataFrame) -> WeightedRandomSampler:
    """
    WeightedRandomSampler over images based on inverse class frequency.
    This oversamples minority classes without changing the loss.
    """
    counts = train_df["diagnostic"].value_counts().to_dict()

    def w(label: str) -> float:
        c = max(int(counts.get(str(label), 1)), 1)
        return 1.0 / c

    weights = train_df["diagnostic"].map(w).astype(float)
    weight_tensor = torch.as_tensor(weights.values, dtype=torch.double)

    return WeightedRandomSampler(
        weights=weight_tensor,
        num_samples=len(weight_tensor),
        replacement=True,
    )


def make_loaders(
    cfg: PADUfesConfig,
    aug_cfg: Dict,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    use_weighted_sampler: bool = False,
    cv_fold: Optional[int] = None,
):
    train_df, val_df, test_df, id2label = load_split_dfs(cfg, cv_fold=cv_fold)

    t_train = build_transforms(cfg.img_size, aug_cfg["train"], is_train=True)
    t_eval = build_transforms(
        cfg.img_size, aug_cfg["val"], is_train=False)

    train_ds = PADUfesDataset(train_df, cfg.img_dir, t_train, cfg.label2id)
    val_ds = PADUfesDataset(val_df, cfg.img_dir, t_eval, cfg.label2id)
    test_ds = PADUfesDataset(test_df, cfg.img_dir, t_eval, cfg.label2id)

    if use_weighted_sampler:
        sampler = build_weighted_sampler(train_df)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,   # ✅ must be False when sampler is used
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, id2label, train_df
