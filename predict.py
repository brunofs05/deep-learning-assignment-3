"""Classifica uma pasta de imagens com o modelo entregue. Roda em CPU, sem internet.

Uso: python predict.py --imagens PASTA --saida preds.csv [--pesos weights/model.pt]
Saída: CSV com colunas image_id,label (nomes de classes.csv).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent / "src"))
from pa3.data import ImageDataset, eval_transform  # noqa: E402
from pa3.models import build_model, count_params  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--imagens", required=True, type=Path)
    p.add_argument("--saida", required=True, type=Path)
    p.add_argument("--pesos", type=Path, default=Path(__file__).parent / "weights" / "model.pt")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    torch.set_num_threads(max(torch.get_num_threads(), 1))
    ckpt = torch.load(args.pesos, map_location="cpu")
    classes = ckpt["classes"]
    model = build_model(ckpt["model"], len(classes), pretrained=False)  # sem download
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"modelo: {ckpt['model']} | parâmetros: {count_params(model):,}")

    paths = sorted(q for q in args.imagens.iterdir() if q.suffix.lower() in EXTS)
    dl = DataLoader(ImageDataset([str(q) for q in paths], None, eval_transform(ckpt["img_size"])),
                    batch_size=64, num_workers=args.workers)
    preds = []
    with torch.no_grad():
        for x, _ in dl:
            preds.extend(model(x).argmax(1).tolist())
    pd.DataFrame({"image_id": [q.stem for q in paths], "label": [classes[i] for i in preds]}
                 ).to_csv(args.saida, index=False)
    print(f"{len(paths)} imagens → {args.saida}")


if __name__ == "__main__":
    main()
