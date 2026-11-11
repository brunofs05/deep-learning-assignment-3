"""Trilha C — leitura da matriz de ruído do professor e confiança como sinal de erro/OOD (só reportar).

Ex.: python scripts/analisar_ruido.py --teacher siglip --run runs/aluno_siglip_b5_s0_tau0_T1
Saídas: results/teacher/<teacher>/improved_b5/top_confusoes.csv, results/trilha_c/<teacher>/confianca_aluno.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from pa3.paths import RESULTS_DIR
from pa3.utils import save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", default="siglip")
    p.add_argument("--run", required=True, type=Path)
    args = p.parse_args()
    d = RESULTS_DIR / "teacher" / args.teacher / "improved_b5"
    cm = pd.read_csv(d / "confusion_test_public.csv", index_col=0)
    rows = [{"verdadeiro": a, "previsto": b, "n": int(cm.loc[a, b]), "total_classe": int(cm.loc[a].sum()),
             "fracao": round(float(cm.loc[a, b] / cm.loc[a].sum()), 3)}
            for a in cm.index for b in cm.columns if a != b and cm.loc[a, b] > 0]
    top = pd.DataFrame(rows).sort_values("n", ascending=False).head(25)
    top.to_csv(d / "top_confusoes.csv", index=False)
    print(top.head(12).to_string(index=False))

    pr = pd.read_csv(args.run / "preds_test_public.csv")
    ok = (pr["label"] == pr["pred"]).to_numpy()
    res = {"conf_media_acertos": float(pr["conf"][ok].mean()), "conf_media_erros": float(pr["conf"][~ok].mean()),
           "auroc_conf_acerto": float(roc_auc_score(ok, pr["conf"])), "n": int(len(pr))}
    out = RESULTS_DIR / "trilha_c" / args.teacher / "confianca_aluno.json"
    save_json(res, out)
    print(res)


if __name__ == "__main__":
    main()
