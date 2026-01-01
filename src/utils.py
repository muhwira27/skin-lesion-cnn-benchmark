import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def now_tag() -> str:
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def get_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_params_m(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def latest_run_dir(runs_root: str, backbone: str) -> Optional[str]:
    root = Path(runs_root)
    if not root.exists():
        return None
    candidates = [d for d in root.glob(f"{backbone}_*") if d.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])
