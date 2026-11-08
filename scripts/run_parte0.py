"""Reproduz a Parte 0 inteira e o piso da Parte 2, com progresso e ETA globais.

Etapas:
  1. Sweep de hiperparâmetros do piso escolhido SÓ no val (orçamento 5):
       resnet18: lr da cabeça × resolução, 2 seeds; smallcnn: lr, 1 seed.
  2. Runs finais com a melhor config: smallcnn b5 s0; resnet18 b5 s{0,1,2}, b1 s0, b20 s0.
  3. Avaliação no test_public (só das runs finais), curvas, matriz de confusão, curva_orcamento.csv.

Retomável: runs com best_val.json são puladas. Ex.: python scripts/run_parte0.py
"""
from __future__ import annotations

import itertools
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from pa3.paths import RESULTS_DIR, RUNS_DIR
from pa3.utils import fmt_time

PY = [sys.executable]
SCRIPTS = Path(__file__).parent
OUT = RESULTS_DIR / "parte0"
N_TRAIN = {1: 21, 5: 105, 20: 420}

R18_LRS = [3e-4, 1e-3, 3e-3]
R18_IMGS = [224, 320, 384]
R18_SEEDS = [0, 1]
CNN_LRS = [1e-3, 3e-3]


def job(name, model, budget, seed, lr, img, epochs, method, extra=()):
    steps = -(-N_TRAIN[budget] // 32)
    epochs = max(epochs, -(-240 // steps))
    cost = epochs * (2 * N_TRAIN[budget] + N_TRAIN[budget] + 202) * (img / 224) ** 2
    args = ["--model", model, "--budget", str(budget), "--seed", str(seed), "--lr", str(lr),
            "--img-size", str(img), "--method", method, "--run-name", name, *extra]
    return {"name": name, "args": args, "cost": cost}


def run_jobs(jobs, label, state):
    for j in jobs:
        state["i"] += 1
        done = (RUNS_DIR / j["name"] / "best_val.json").exists()
        el = time.time() - state["t0"]
        rate = el / state["cost_done"] if state["cost_done"] else None
        eta = fmt_time(rate * state["cost_left"]) if rate else "?"
        print(f"\n===== [{state['i']}/{state['n']}] {label}: {j['name']} | decorrido {fmt_time(el)} | "
              f"ETA global ~{eta} {'(já feito, pulando)' if done else ''}", flush=True)
        if not done:
            subprocess.run(PY + [str(SCRIPTS / "train.py"), *j["args"]], check=True)
            state["cost_done"] += j["cost"]
        state["cost_left"] -= j["cost"]


def val_f1(name):
    return json.loads((RUNS_DIR / name / "best_val.json").read_text())["val_f1"]


def main():
    sweep_r18 = [job(f"sweep_resnet18_lr{lr:g}_img{img}_s{s}", "resnet18", 5, s, lr, img, 60, "sweep_resnet18")
                 for lr, img, s in itertools.product(R18_LRS, R18_IMGS, R18_SEEDS)]
    sweep_cnn = [job(f"sweep_smallcnn_lr{lr:g}_s0", "smallcnn", 5, 0, lr, 224, 300, "sweep_smallcnn",
                     ["--no-pretrained"]) for lr in CNN_LRS]
    # custo estimado das finais (com a config mais cara) só para o ETA inicial
    finals_guess = ([job("x", "resnet18", 5, 0, 1e-3, 320, 60, "x")] * 3 +
                    [job("x", "resnet18", b, 0, 1e-3, 320, 60, "x") for b in (1, 20)] +
                    [job("x", "smallcnn", 5, 0, 1e-3, 224, 300, "x")])
    state = {"t0": time.time(), "i": 0, "n": len(sweep_r18) + len(sweep_cnn) + len(finals_guess),
             "cost_done": 0.0, "cost_left": sum(j["cost"] for j in sweep_r18 + sweep_cnn + finals_guess)}

    print(f"Parte 0: {state['n']} treinos ({len(sweep_r18) + len(sweep_cnn)} de sweep + {len(finals_guess)} finais)")
    run_jobs(sweep_r18, "sweep resnet18", state)
    run_jobs(sweep_cnn, "sweep smallcnn", state)

    # ---- seleção no VAL ----
    rows = [{"model": "resnet18", "lr": lr, "img": img, "seed": s,
             "val_f1": val_f1(f"sweep_resnet18_lr{lr:g}_img{img}_s{s}")}
            for lr, img, s in itertools.product(R18_LRS, R18_IMGS, R18_SEEDS)]
    rows += [{"model": "smallcnn", "lr": lr, "img": 224, "seed": 0, "val_f1": val_f1(f"sweep_smallcnn_lr{lr:g}_s0")}
             for lr in CNN_LRS]
    sweep = pd.DataFrame(rows)
    agg = sweep.groupby(["model", "lr", "img"]).val_f1.agg(["mean", "std", "count"]).reset_index()
    OUT.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(OUT / "sweep_val.csv", index=False)
    agg.to_csv(OUT / "sweep_val_resumo.csv", index=False)
    best = {m: agg[agg.model == m].sort_values("mean", ascending=False).iloc[0] for m in ("resnet18", "smallcnn")}
    print("\nSweep (F1 macro no val, média entre seeds):\n" + agg.round(4).to_string(index=False))
    for m, b in best.items():
        print(f"→ {m}: lr {b.lr:g}, img {int(b.img)} (val F1 {b['mean']:.3f})")
    (OUT / "config_escolhida.json").write_text(json.dumps(
        {m: {"lr": float(b.lr), "img_size": int(b.img), "val_f1_mean": float(b["mean"])} for m, b in best.items()},
        indent=2))

    # ---- finais ----
    r, c = best["resnet18"], best["smallcnn"]
    finals = [job("piso_smallcnn_b5_s0", "smallcnn", 5, 0, c.lr, int(c.img), 300, "piso_smallcnn", ["--no-pretrained"])]
    finals += [job(f"piso_resnet18_b5_s{s}", "resnet18", 5, s, r.lr, int(r.img), 60, "piso_resnet18") for s in (0, 1, 2)]
    finals += [job(f"piso_resnet18_b{b}_s0", "resnet18", b, 0, r.lr, int(r.img), 60, "piso_resnet18") for b in (1, 20)]
    state["cost_left"] = sum(j["cost"] for j in finals)
    run_jobs(finals, "final", state)

    # ---- avaliação no test_public e figuras ----
    print("\n===== avaliação no test_public (runs finais)", flush=True)
    for j in finals:
        subprocess.run(PY + [str(SCRIPTS / "evaluate.py"), "--run", str(RUNS_DIR / j["name"])], check=True)
    for name, fig in [("piso_resnet18_b5_s0", "curva_resnet18_b5.png"), ("piso_smallcnn_b5_s0", "curva_smallcnn_b5.png")]:
        subprocess.run(PY + [str(SCRIPTS / "plot_curves.py"), str(RUNS_DIR / name), "--out", str(OUT / fig)], check=True)
    for name, tag in [("piso_resnet18_b5_s0", "resnet18_b5"), ("piso_smallcnn_b5_s0", "smallcnn_b5")]:
        shutil.copy(RUNS_DIR / name / "confusion_test_public.png", OUT / f"confusion_{tag}.png")
        shutil.copy(RUNS_DIR / name / "per_class_test_public.csv", OUT / f"per_class_{tag}.csv")
        shutil.copy(RUNS_DIR / name / "memorization.json", OUT / f"memorizacao_{tag}.json") \
            if (RUNS_DIR / name / "memorization.json").exists() else None
    subprocess.run(PY + [str(SCRIPTS / "collect_results.py")], check=True)
    print(f"\nParte 0 concluída em {fmt_time(time.time() - state['t0'])}. Resultados em {OUT}")


if __name__ == "__main__":
    main()
