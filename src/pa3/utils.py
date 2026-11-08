from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fmt_time(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def get_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def amp_enabled(flag: str, device: torch.device) -> bool:
    """auto: liga só em GPU com tensor cores (cc >= 7, ex. T4); Pascal (1080 Ti) fica em FP32."""
    if flag == "on":
        return device.type == "cuda"
    if flag == "off" or device.type != "cuda":
        return False
    return torch.cuda.get_device_capability(device)[0] >= 7


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def save_json(obj, path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def load_json(path):
    return json.loads(Path(path).read_text())
