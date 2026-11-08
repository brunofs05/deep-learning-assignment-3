"""Curva de treino e validação de uma run (history.csv).

Painéis: loss e acurácia de treino LIMPO (sem aumentação, CE pura, modo eval) × val, e F1 macro no val.
Marca: época em que o treino limpo chega a 100% (decorou), mínimo da loss de val e época escolhida.

Ex.: python scripts/plot_curves.py runs/piso_resnet18_b5_s0 --out results/parte0/curva_resnet18.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pa3.utils import load_json, save_json


def memorization_stats(h: pd.DataFrame) -> dict:
    full = h[h.train_clean_acc >= 0.999]
    return {
        "epoch_train_clean_acc_95": int(h[h.train_clean_acc >= 0.95].epoch.min()) if (h.train_clean_acc >= 0.95).any() else None,
        "epoch_train_clean_acc_100": int(full.epoch.min()) if len(full) else None,
        "epoch_min_val_loss": int(h.loc[h.val_loss.idxmin(), "epoch"]),
        "min_val_loss": float(h.val_loss.min()),
        "epoch_best_val_f1": int(h.loc[h.val_f1.idxmax(), "epoch"]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    h = pd.read_csv(args.run / "history.csv")
    s = memorization_stats(h)
    best = load_json(args.run / "best_val.json")["best_epoch"]
    save_json(s | {"best_epoch_selected": best}, args.run / "memorization.json")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ax_loss, ax_acc, ax_f1 = axes
    ax_loss.plot(h.epoch, h.train_clean_loss, label="treino (limpo)")
    ax_loss.plot(h.epoch, h.val_loss, label="val")
    ax_loss.plot(h.epoch, h.train_loss, ":", color="gray", label="treino (aumentado + label smoothing)")
    ax_loss.set(title="Loss (entropia cruzada)", xlabel="época")
    ax_acc.plot(h.epoch, h.train_clean_acc, label="treino (limpo)")
    ax_acc.plot(h.epoch, h.val_acc, label="val")
    ax_acc.set(title="Acurácia", xlabel="época", ylim=(0, 1.02))
    ax_f1.plot(h.epoch, h.val_f1, color="C1", label="F1 macro val")
    ax_f1.set(title="F1 macro no val", xlabel="época", ylim=(0, max(0.5, h.val_f1.max() + 0.05)))

    marks = [(s["epoch_train_clean_acc_100"], "C3", "treino limpo = 100%"),
             (s["epoch_min_val_loss"], "C2", "mín. loss val"),
             (best, "k", "época escolhida (val F1)")]
    for ax in axes:
        for ep, c, lab in marks:
            if ep:
                ax.axvline(ep, color=c, ls="--", lw=1, alpha=0.7, label=lab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(args.title or args.run.name)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"salvo em {args.out} | {s} | escolhida {best}")


if __name__ == "__main__":
    main()
