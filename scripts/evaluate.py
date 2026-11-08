"""Avalia o best.pt de uma run num split rotulado (padrão: test_public).

Ex.: python scripts/evaluate.py --run runs/piso_resnet18_b5_s0
Gera metrics_<split>.json, per_class_<split>.csv, preds_<split>.csv e confusion_<split>.png na run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from pa3.data import ImageDataset, eval_transform, load_split
from pa3.engine import predict
from pa3.metrics import compute_metrics, per_class_table, plot_confusion
from pa3.models import build_model
from pa3.utils import amp_enabled, get_device, load_json, save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--split", default="test_public", choices=["val", "test_public"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit-batches", type=int, default=None)
    args = p.parse_args()

    device = get_device(args.device)
    ckpt = torch.load(args.run / "best.pt", map_location="cpu")
    cfg = load_json(args.run / "config.json")
    classes = ckpt["classes"]
    model = build_model(ckpt["model"], len(classes), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    df = load_split(args.split)
    dl = DataLoader(ImageDataset.from_df(df, eval_transform(ckpt["img_size"])), batch_size=64,
                    num_workers=args.workers)
    logits, y = predict(model, dl, device, amp_enabled("auto", device), args.limit_batches)
    pred = logits.argmax(1)
    df = df.iloc[: len(y)]

    m = compute_metrics(y, pred, len(classes)) | {
        "split": args.split, "run": args.run.name, "method": cfg["method"], "budget": cfg["budget"],
        "seed": cfg["seed"], "model": ckpt["model"], "best_epoch": ckpt["epoch"], "val_f1": ckpt["val_f1"],
    }
    save_json(m, args.run / f"metrics_{args.split}.json")
    per_class_table(y, pred, classes).to_csv(args.run / f"per_class_{args.split}.csv", index=False)
    probs = torch.softmax(torch.from_numpy(logits), 1).numpy()
    pd.DataFrame({"image_id": df["image_id"], "label": df["label"],
                  "pred": [classes[i] for i in pred], "conf": probs.max(1)}
                 ).to_csv(args.run / f"preds_{args.split}.csv", index=False)
    plot_confusion(y, pred, classes, args.run / f"confusion_{args.split}.png",
                   title=f"{args.run.name} — {args.split} (F1 macro {m['f1_macro']:.3f}, acc {m['accuracy']:.3f})")
    print(f"{args.run.name} @ {args.split}: F1 macro {m['f1_macro']:.4f} | acc {m['accuracy']:.4f} "
          f"| melhor época {ckpt['epoch']}")


if __name__ == "__main__":
    main()
