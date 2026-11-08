"""Treino supervisionado com um orçamento de rótulos; seleção de época pelo F1 macro no VAL.

Ex.: python scripts/train.py --model resnet18 --budget 5 --seed 0
O test_public NÃO é lido aqui (use scripts/evaluate.py).
"""
from __future__ import annotations

import argparse
import math
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pa3.data import ImageDataset, budget_split, eval_transform, load_classes, load_split, train_transform
from pa3.engine import eval_loss, predict, train_one_epoch
from pa3.metrics import compute_metrics
from pa3.models import MAX_PARAMS, build_model, count_params, head_parameters
from pa3.paths import RUNS_DIR, data_root
from pa3.utils import amp_enabled, fmt_time, get_device, git_hash, save_json, seed_everything

DEFAULTS = {
    #           épocas  lr(cabeça) mult. backbone  weight decay
    "smallcnn": dict(epochs=300, lr=1e-3, backbone_lr_mult=1.0, weight_decay=5e-4),
    "resnet18": dict(epochs=60, lr=1e-3, backbone_lr_mult=0.1, weight_decay=5e-2),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(DEFAULTS), default="resnet18")
    p.add_argument("--budget", type=int, choices=[1, 5, 20], default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", default=None, help="nome no curva_orcamento.csv (padrão: piso_<model>)")
    p.add_argument("--epochs", type=int, help="épocas (o mínimo de passos pode aumentar)")
    p.add_argument("--min-steps", type=int, default=240,
                   help="mínimo de passos de otimização; com orçamento 1 (1 batch/época) aumenta as épocas")
    p.add_argument("--lr", type=float)
    p.add_argument("--backbone-lr-mult", type=float)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--warmup-epochs", type=float, default=0.05, help="fração do treino (<1) ou épocas (>=1)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--limit-batches", type=int, default=None, help="smoke test")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()
    for k, v in DEFAULTS[args.model].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    args.method = args.method or f"piso_{args.model}"
    args.run_name = args.run_name or f"{args.method}_b{args.budget}_s{args.seed}"
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)
    amp = amp_enabled(args.amp, device)
    classes = load_classes()
    run_dir = RUNS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(budget_split(args.budget))
    val_df = load_split("val")
    g = torch.Generator().manual_seed(args.seed)
    pw = args.workers > 0
    train_dl = DataLoader(
        ImageDataset.from_df(train_df, train_transform(args.img_size)),
        batch_size=min(args.batch_size, len(train_df)), shuffle=True, num_workers=args.workers,
        pin_memory=device.type == "cuda", generator=g, persistent_workers=pw,
    )
    # Mesmo conjunto de treino, sem aumentação: mede o quanto o modelo decorou (comparável ao val).
    train_eval_dl = DataLoader(ImageDataset.from_df(train_df, eval_transform(args.img_size)),
                               batch_size=64, num_workers=args.workers, persistent_workers=pw)
    val_dl = DataLoader(ImageDataset.from_df(val_df, eval_transform(args.img_size)),
                        batch_size=64, num_workers=args.workers, persistent_workers=pw)

    model = build_model(args.model, len(classes), pretrained=not args.no_pretrained).to(device)
    n_params = count_params(model)
    assert n_params <= MAX_PARAMS, n_params
    head = head_parameters(model)
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": args.lr * args.backbone_lr_mult}, {"params": head, "lr": args.lr}],
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = len(train_dl) if args.limit_batches is None else min(len(train_dl), args.limit_batches)
    if args.limit_batches is None:
        args.epochs = max(args.epochs, math.ceil(args.min_steps / steps_per_epoch))
    total = args.epochs * steps_per_epoch
    warm = round(args.warmup_epochs * total) if args.warmup_epochs < 1 else round(args.warmup_epochs * steps_per_epoch)
    warm = max(warm, 1)

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(enabled=amp)

    config = vars(args) | {
        "n_params": n_params, "device": str(device), "amp_used": amp, "git": git_hash(),
        "data_root": str(data_root()), "n_train": len(train_df), "classes": classes,
        "total_steps": total, "warmup_steps": warm,
    }
    save_json(config, run_dir / "config.json")
    print(f"[{args.run_name}] {args.model} | {n_params:,} params | {device} amp={amp} | img {args.img_size} | "
          f"lr {args.lr:g} (backbone ×{args.backbone_lr_mult:g}) | {len(train_df)} imagens, "
          f"{args.epochs} épocas × {steps_per_epoch} passos = {total}", flush=True)

    history, best = [], (-1.0, float("inf"))
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_dl, optimizer, scheduler, criterion, device, scaler, amp,
                             args.limit_batches)
        tl_logits, tl_y = predict(model, train_eval_dl, device, amp, args.limit_batches)
        logits, y = predict(model, val_dl, device, amp, args.limit_batches)
        m = compute_metrics(y, logits.argmax(1), len(classes))
        vl = eval_loss(logits, y)
        row = {"epoch": epoch,
               "train_loss": tr["loss"], "train_acc": tr["acc"],  # com aumentação + label smoothing
               "train_clean_loss": eval_loss(tl_logits, tl_y),    # sem aumentação, CE pura
               "train_clean_acc": float((tl_logits.argmax(1) == tl_y).mean()),
               "val_loss": vl, "val_f1": m["f1_macro"], "val_acc": m["accuracy"],
               "lr": optimizer.param_groups[-1]["lr"], "time_s": time.time() - t0}
        history.append(row)
        improved = (m["f1_macro"], -vl) > (best[0], -best[1])
        if improved:
            best = (m["f1_macro"], vl)
            torch.save({"model": args.model, "img_size": args.img_size, "classes": classes,
                        "state_dict": model.state_dict(), "epoch": epoch, "val_f1": m["f1_macro"]},
                       run_dir / "best.pt")
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        elapsed = time.time() - t_start
        eta = elapsed / epoch * (args.epochs - epoch)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs or improved:
            print(f"ep {epoch:3d}/{args.epochs} | treino limpo loss {row['train_clean_loss']:.3f} "
                  f"acc {row['train_clean_acc']:.3f} | val loss {vl:.3f} F1 {m['f1_macro']:.3f} "
                  f"acc {m['accuracy']:.3f} | {row['time_s']:.1f}s/ép, decorrido {fmt_time(elapsed)}, "
                  f"ETA {fmt_time(eta)}{' *' if improved else ''}", flush=True)

    best_row = max(history, key=lambda r: (r["val_f1"], -r["val_loss"]))
    save_json({"best_epoch": best_row["epoch"], "val_f1": best_row["val_f1"], "val_acc": best_row["val_acc"],
               "val_loss": best_row["val_loss"], "train_time_s": time.time() - t_start},
              run_dir / "best_val.json")
    print(f"melhor época {best_row['epoch']} (val F1 {best_row['val_f1']:.3f}) | "
          f"tempo total {fmt_time(time.time() - t_start)} → {run_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
