"""Agrega os números da Parte 2 em results/curva_orcamento.csv (entregável).

Linhas:
  * runs/*/metrics_test_public.json do piso e do aluno (sweeps, smoke tests e runs *_tau* de sweep ficam de fora:
    foram escolhidos no val e não entram na curva);
  * professor da Trilha C (o escolhido em results/parte2/professor_escolhido.json): melhorado por orçamento e
    zero-shot (orcamento = 0: não usa rótulo nenhum).
"""
from __future__ import annotations

import json

import pandas as pd

from pa3.paths import RESULTS_DIR, RUNS_DIR

EXCLUDE_PREFIXES = ("sweep_", "smoke")


def run_rows():
    rows = []
    for f in sorted(RUNS_DIR.glob("*/metrics_test_public.json")):
        name = f.parent.name
        if name.startswith(EXCLUDE_PREFIXES) or "_tau" in name:
            continue
        m = json.loads(f.read_text())
        rows.append({"metodo": m["method"], "orcamento": m["budget"], "seed": m["seed"],
                     "f1_macro": m["f1_macro"], "acuracia": m["accuracy"]})
    return rows


def teacher_rows():
    choice = RESULTS_DIR / "parte2" / "professor_escolhido.json"
    if not choice.exists():
        return []
    t = json.loads(choice.read_text())["professor"]
    tdir = RESULTS_DIR / "teacher" / t
    rows = []
    zs = json.loads((tdir / "metrics_test_public.json").read_text())
    rows.append({"metodo": f"professor_{t}_zeroshot", "orcamento": 0, "seed": 0,
                 "f1_macro": zs["f1_macro"], "acuracia": zs["accuracy"]})
    for f in sorted(tdir.glob("improved_b*/metrics_test_public.json")):
        m = json.loads(f.read_text())
        rows.append({"metodo": f"professor_{t}_melhorado", "orcamento": m["budget"], "seed": 0,
                     "f1_macro": m["f1_macro"], "acuracia": m["accuracy"]})
    return rows


def main():
    rows = run_rows() + teacher_rows()
    if not rows:
        raise SystemExit("nenhuma run avaliada em test_public")
    df = pd.DataFrame(rows).sort_values(["metodo", "orcamento", "seed"])
    df[["f1_macro", "acuracia"]] = df[["f1_macro", "acuracia"]].round(4)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "curva_orcamento.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print("\nmédia ± desvio:")
    print(df.groupby(["metodo", "orcamento"])[["f1_macro", "acuracia"]].agg(["mean", "std", "count"]).round(4))
    print(f"\nsalvo em {out}")


if __name__ == "__main__":
    main()
