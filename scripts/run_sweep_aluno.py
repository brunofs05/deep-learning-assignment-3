"""Trilha C — escolhe tau e T do aluno pelo F1 macro no VAL (seed 0, orçamento 5, professor dado).

Retomável: runs com best_val.json são puladas. O test_public NÃO é lido aqui.
Ex.: python scripts/run_sweep_aluno.py --teacher siglip
Saídas: results/trilha_c/<teacher>/sweep_val.csv e config_escolhida.json
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from pa3.paths import RESULTS_DIR, RUNS_DIR
from pa3.utils import fmt_time, load_json, save_json

GRID = [(0.5, 1.0), (0.0, 1.0), (0.7, 1.0), (0.5, 2.0)]  # (tau, T); o primeiro é o padrão a priori


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", default="siglip")
    p.add_argument("--budget", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    out = RESULTS_DIR / "trilha_c" / args.teacher
    out.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).parent / "train_student.py"
    t0, rows = time.time(), []
    for i, (tau, temp) in enumerate(GRID, 1):
        name = f"aluno_{args.teacher}_b{args.budget}_s{args.seed}_tau{tau:g}_T{temp:g}"
        done = RUNS_DIR / name / "best_val.json"
        print(f"\n===== [{i}/{len(GRID)}] tau={tau:g} T={temp:g} | decorrido {fmt_time(time.time() - t0)} "
              f"{'(já feito)' if done.exists() else ''}", flush=True)
        if not done.exists():
            subprocess.run([sys.executable, str(script), "--teacher", args.teacher, "--budget", str(args.budget),
                            "--seed", str(args.seed), "--tau", str(tau), "--temp", str(temp),
                            "--workers", str(args.workers), "--run-name", name], check=True)
        b = load_json(done)
        cfg = load_json(RUNS_DIR / name / "config.json")
        rows.append({"tau": tau, "temp": temp, "n_pseudo_used": cfg["n_pseudo_used"], "best_epoch": b["best_epoch"],
                     "val_f1": b["val_f1"], "val_acc": b["val_acc"], "run": name})
        pd.DataFrame(rows).to_csv(out / "sweep_val.csv", index=False)
    df = pd.DataFrame(rows).sort_values(["val_f1", "val_acc"], ascending=False)
    best = df.iloc[0]
    save_json({"tau": float(best["tau"]), "temp": float(best["temp"]), "val_f1": float(best["val_f1"]),
               "run": best["run"]}, out / "config_escolhida.json")
    print("\n", df.round(4).to_string(index=False), "\nescolhida:", best["run"])


if __name__ == "__main__":
    main()
