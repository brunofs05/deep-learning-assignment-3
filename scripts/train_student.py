"""Trilha C — treina o aluno (ResNet-18) com pseudo-rótulos do professor + os rótulos verdadeiros do orçamento.

Pseudo-rótulos vêm de scripts/teacher_improve.py. Tratamento do ruído (ver docs/DECISOES.md):
  * filtro por confiança: só entram as imagens sem rótulo com max-prob do professor >= tau;
  * rótulos suaves com temperatura: alvo = softmax(log p / temp) (temp > 1 suaviza);
  * peso das pseudo-amostras na perda (--pseudo-weight) e rótulos verdadeiros repetidos (--labeled-repeat).
Seleção de época, tau e temp: F1 macro no VAL (do aluno). O test_public só é lido em evaluate.py.

Ex.: python scripts/train_student.py --teacher siglip --budget 5 --tau 0.5 --temp 1.0
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.special import softmax
from torch.utils.data import DataLoader, Dataset

from pa3.data import ImageDataset, budget_split, eval_transform, load_classes, load_split, train_transform
from pa3.engine import eval_loss, predict
from pa3.metrics import compute_metrics
from pa3.models import MAX_PARAMS, build_model, count_params, head_parameters
from pa3.paths import RESULTS_DIR, RUNS_DIR, data_root
from pa3.utils import amp_enabled, fmt_time, get_device, git_hash, save_json, seed_everything


class SoftDataset(Dataset):
    def __init__(self, paths, targets, weights, transform):
        self.paths, self.targets, self.weights, self.transform = list(paths), targets, weights, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.targets[i], self.weights[i]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", default="siglip")
    p.add_argument("--budget", type=int, choices=[1, 5, 20], default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tau", type=float, default=0.5, help="confiança mínima do professor para usar a imagem")
    p.add_argument("--temp", type=float, default=1.0, help="temperatura dos rótulos suaves")
    p.add_argument("--pseudo-weight", type=float, default=1.0)
    p.add_argument("--labeled-repeat", type=int, default=5, help="cópias de cada imagem rotulada por época")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr-mult", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-2)
    p.add_argument("--warmup-epochs", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--label-smoothing", type=float, default=0.1, help="só nos rótulos verdadeiros")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--limit-batches", type=int, default=None, help="smoke test")
    p.add_argument("--method", default=None, help="nome no curva_orcamento.csv (padrão: aluno_<teacher>)")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()
    args.method = args.method or f"aluno_{args.teacher}"
    args.run_name = args.run_name or f"{args.method}_b{args.budget}_s{args.seed}_tau{args.tau:g}_T{args.temp:g}"
    return args


def build_targets(args, classes):
    """Alvos (N,C), pesos (N,) e máscara de uso sobre o pool 'train_all'."""
    t_dir = RESULTS_DIR / "teacher" / args.teacher / f"improved_b{args.budget}"
    pool = load_split("train_all")
    ids = pd.read_csv(t_dir / "pseudo_ids_train.csv")["image_id"].tolist()
    assert ids == pool["image_id"].tolist(), "ordem do professor diferente do pool"
    lp = np.load(t_dir / "log_probs_train.npy")
    conf = softmax(lp, axis=1).max(1)
    soft = softmax(lp / args.temp, axis=1)

    nc = len(classes)
    lab = load_split(budget_split(args.budget))
    pos = {i: k for k, i in enumerate(ids)}
    lab_idx = np.array([pos[i] for i in lab["image_id"]])
    is_lab = np.zeros(len(ids), bool)
    is_lab[lab_idx] = True

    targets = soft.copy()
    onehot = np.full((len(lab_idx), nc), args.label_smoothing / (nc - 1))
    onehot[np.arange(len(lab_idx)), lab["y"].to_numpy()] = 1 - args.label_smoothing
    targets[lab_idx] = onehot
    weights = np.where(is_lab, 1.0, args.pseudo_weight)
    use = is_lab | (conf >= args.tau)
    return pool, targets.astype(np.float32), weights.astype(np.float32), use, is_lab, lab


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)
    amp = amp_enabled(args.amp, device)
    classes = load_classes()
    run_dir = RUNS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    pool, targets, weights, use, is_lab, lab = build_targets(args, classes)
    idx_unl = np.where(use & ~is_lab)[0]
    idx_lab = np.where(is_lab)[0]
    order = np.concatenate([idx_unl] + [idx_lab] * args.labeled_repeat)
    ds = SoftDataset(pool["path"].to_numpy()[order], torch.from_numpy(targets[order]),
                     torch.from_numpy(weights[order]), train_transform(args.img_size))
    pw = args.workers > 0
    g = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                          pin_memory=device.type == "cuda", generator=g, persistent_workers=pw, drop_last=True)
    val_df = load_split("val")
    val_dl = DataLoader(ImageDataset.from_df(val_df, eval_transform(args.img_size)), batch_size=64,
                        num_workers=args.workers, persistent_workers=pw)
    lab_dl = DataLoader(ImageDataset.from_df(lab, eval_transform(args.img_size)), batch_size=64,
                        num_workers=args.workers, persistent_workers=pw)

    model = build_model("resnet18", len(classes), pretrained=True).to(device)
    n_params = count_params(model)
    assert n_params <= MAX_PARAMS, n_params
    head_ids = {id(p) for p in head_parameters(model)}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": args.lr * args.backbone_lr_mult},
         {"params": head_parameters(model), "lr": args.lr}], weight_decay=args.weight_decay)
    spe = len(train_dl) if args.limit_batches is None else min(len(train_dl), args.limit_batches)
    total = args.epochs * spe
    warm = max(round(args.warmup_epochs * total) if args.warmup_epochs < 1 else round(args.warmup_epochs * spe), 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    scaler = torch.amp.GradScaler(enabled=amp)

    config = vars(args) | {
        "model": "resnet18", "n_params": n_params, "device": str(device), "amp_used": amp, "git": git_hash(),
        "data_root": str(data_root()), "n_labeled": int(is_lab.sum()), "n_pseudo_used": int(len(idx_unl)),
        "n_pseudo_total": int((~is_lab).sum()), "n_epoch_samples": int(len(ds)), "total_steps": total,
        "classes": classes}
    save_json(config, run_dir / "config.json")
    print(f"[{args.run_name}] resnet18 | {n_params:,} params | {device} | img {args.img_size} | "
          f"{len(idx_lab)} rotuladas ×{args.labeled_repeat} + {len(idx_unl)}/{int((~is_lab).sum())} pseudo "
          f"(tau {args.tau:g}, T {args.temp:g}) | {args.epochs} ép × {spe} passos", flush=True)

    history, best, t_start = [], (-1.0, float("inf")), time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for b, (x, t, w) in enumerate(train_dl):
            if args.limit_batches is not None and b >= args.limit_batches:
                break
            x, t, w = x.to(device, non_blocking=True), t.to(device), w.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(x)
            loss = (-(t * F.log_softmax(logits.float(), 1)).sum(1) * w).sum() / w.sum()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            tot += loss.item() * len(x)
            n += len(x)
        ll, ly = predict(model, lab_dl, device, amp, args.limit_batches)
        logits, y = predict(model, val_dl, device, amp, args.limit_batches)
        m = compute_metrics(y, logits.argmax(1), len(classes))
        vl = eval_loss(logits, y)
        row = {"epoch": epoch, "train_loss": tot / max(n, 1), "train_labeled_acc": float((ll.argmax(1) == ly).mean()),
               "val_loss": vl, "val_f1": m["f1_macro"], "val_acc": m["accuracy"], "time_s": time.time() - t0}
        history.append(row)
        improved = (m["f1_macro"], -vl) > (best[0], -best[1])
        if improved:
            best = (m["f1_macro"], vl)
            torch.save({"model": "resnet18", "img_size": args.img_size, "classes": classes,
                        "state_dict": model.state_dict(), "epoch": epoch, "val_f1": m["f1_macro"]},
                       run_dir / "best.pt")
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        el = time.time() - t_start
        print(f"ep {epoch:2d}/{args.epochs} | loss {row['train_loss']:.3f} | rotuladas acc {row['train_labeled_acc']:.3f} "
              f"| val loss {vl:.3f} F1 {m['f1_macro']:.3f} acc {m['accuracy']:.3f} | {row['time_s']:.0f}s/ép, "
              f"decorrido {fmt_time(el)}, ETA {fmt_time(el / epoch * (args.epochs - epoch))}"
              f"{' *' if improved else ''}", flush=True)

    best_row = max(history, key=lambda r: (r["val_f1"], -r["val_loss"]))
    save_json({"best_epoch": best_row["epoch"], "val_f1": best_row["val_f1"], "val_acc": best_row["val_acc"],
               "val_loss": best_row["val_loss"], "train_time_s": time.time() - t_start}, run_dir / "best_val.json")
    print(f"melhor época {best_row['epoch']} (val F1 {best_row['val_f1']:.3f}) → {run_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
