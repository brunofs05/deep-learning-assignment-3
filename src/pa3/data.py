"""Leitura dos splits, Dataset e transforms."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .paths import data_root

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# split -> (csv relativo a labels/, pasta de imagens)
SPLITS = {
    "budget1": ("rotulos_1.csv", "train"),
    "budget5": ("rotulos_5.csv", "train"),
    "budget20": ("rotulos_20.csv", "train"),
    "val": ("val.csv", "val"),
    "test_public": ("test_public.csv", "test_public"),
}


def load_classes(root: Path | None = None) -> list[str]:
    root = root or data_root()
    df = pd.read_csv(root / "classes.csv").sort_values("idx")
    return df["label"].tolist()


def load_split(name: str, root: Path | None = None) -> pd.DataFrame:
    """DataFrame com image_id, label, y (int) e path.

    `name` em SPLITS ou "train_all" (pool inteiro, sem rótulo: label=None, y=-1).
    """
    root = root or data_root()
    classes = load_classes(root)
    to_idx = {c: i for i, c in enumerate(classes)}
    if name == "train_all":
        paths = sorted((root / "images" / "train").glob("*.jpg"))
        df = pd.DataFrame({"image_id": [p.stem for p in paths], "label": None, "y": -1})
        df["path"] = [str(p) for p in paths]
        return df
    csv, folder = SPLITS[name]
    df = pd.read_csv(root / "labels" / csv)
    df["y"] = df["label"].map(to_idx)
    if df["y"].isna().any():
        raise ValueError(f"{name}: classes desconhecidas {set(df.loc[df.y.isna(), 'label'])}")
    df["y"] = df["y"].astype(int)
    df["path"] = [str(root / "images" / folder / f"{i}.jpg") for i in df["image_id"]]
    return df


def budget_split(budget: int) -> str:
    if budget not in (1, 5, 20):
        raise ValueError("budget deve ser 1, 5 ou 20")
    return f"budget{budget}"


class ImageDataset(Dataset):
    def __init__(self, paths: list[str], labels: list[int] | None, transform):
        self.paths = list(paths)
        self.labels = list(labels) if labels is not None else [-1] * len(self.paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), torch.tensor(self.labels[i])

    @classmethod
    def from_df(cls, df: pd.DataFrame, transform) -> "ImageDataset":
        return cls(df["path"].tolist(), df["y"].tolist(), transform)


def train_transform(img_size: int = 224) -> T.Compose:
    # Escala mínima moderada: o animal costuma ser pequeno, crop agressivo o tira do quadro.
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.2, 0.02),
        T.RandomGrayscale(p=0.2),  # imita fotos noturnas em IR
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(img_size: int = 224) -> T.Compose:
    # Redimensiona o quadro inteiro (sem center crop) para não cortar animais na borda.
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
