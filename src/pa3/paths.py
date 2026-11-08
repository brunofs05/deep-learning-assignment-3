"""Resolução de caminhos: funciona igual local e no Kaggle."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"
RESULTS_DIR = REPO_ROOT / "results"


def _looks_like_data_root(p: Path) -> bool:
    return (p / "classes.csv").is_file() and (p / "labels").is_dir() and (p / "images").is_dir()


def data_root() -> Path:
    """Ordem: $PA3_DATA → /kaggle/input/** → <repo>/data/pa3_dados."""
    env = os.environ.get("PA3_DATA")
    if env:
        p = Path(env)
        if not _looks_like_data_root(p):
            raise FileNotFoundError(f"PA3_DATA={p} não contém classes.csv/labels/images")
        return p
    kaggle = Path("/kaggle/input")
    if kaggle.is_dir():
        for cand in sorted(kaggle.glob("**/classes.csv")):
            if _looks_like_data_root(cand.parent):
                return cand.parent
    local = REPO_ROOT / "data" / "pa3_dados"
    if _looks_like_data_root(local):
        return local
    raise FileNotFoundError(
        "Dados não encontrados. Rode scripts/get_data.sh ou defina PA3_DATA."
    )
