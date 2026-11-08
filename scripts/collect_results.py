"""Agrega runs/*/metrics_test_public.json em results/curva_orcamento.csv (entregável da Parte 2)."""
from __future__ import annotations

import json

import pandas as pd

from pa3.paths import RESULTS_DIR, RUNS_DIR

rows = []
for f in sorted(RUNS_DIR.glob("*/metrics_test_public.json")):
    if f.parent.name.startswith(("sweep_", "smoke")):
        continue  # sweeps são escolhidos no val e não entram na curva
    m = json.loads(f.read_text())
    rows.append({"metodo": m["method"], "orcamento": m["budget"], "seed": m["seed"],
                 "f1_macro": round(m["f1_macro"], 4), "acuracia": round(m["accuracy"], 4)})
if not rows:
    raise SystemExit("nenhuma run avaliada em test_public")
df = pd.DataFrame(rows).sort_values(["metodo", "orcamento", "seed"])
RESULTS_DIR.mkdir(exist_ok=True)
out = RESULTS_DIR / "curva_orcamento.csv"
df.to_csv(out, index=False)
print(df.to_string(index=False))
print("\nmédia ± desvio:")
print(df.groupby(["metodo", "orcamento"])[["f1_macro", "acuracia"]].agg(["mean", "std", "count"]).round(4))
print(f"\nsalvo em {out}")
