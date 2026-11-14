"""Trilha C — professor melhorado com os 105 rótulos + pseudo-rótulos para o train.

Entrada: saídas de scripts/teacher.py (embeddings em runs/teacher/<nome>/, logits em results/teacher/<nome>/).
Professor melhorado = mistura, em log-probabilidade, do zero-shot e de uma regressão logística (probe) sobre os
embeddings treinada nos 105 rótulos do orçamento:
    log p = (1-alpha) * log_softmax(zero-shot) + alpha * log_softmax(probe)
(alpha, C) escolhidos por validação cruzada estratificada (5 dobras) NOS PRÓPRIOS 105 RÓTULOS. O val não é lido
(o professor não pode vê-lo durante o treino). O test_public só mede o professor.

Ex.: python scripts/teacher_improve.py --name siglip --budget 5
Saídas em results/teacher/<nome>/improved_b<orç>/:
  config.json, cv_grid.csv, pseudo_probs_train.npy (N,21), pseudo_ids_train.csv,
  metrics_test_public.json, per_class_test_public.csv, confusion_test_public.{png,csv},
  noite_vs_dia_test_public.csv, tau_precisao_cobertura.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import log_softmax, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from pa3.data import budget_split, load_classes, load_split
from pa3.metrics import compute_metrics, per_class_table, plot_confusion
from pa3.paths import RESULTS_DIR, RUNS_DIR
from pa3.utils import save_json, seed_everything

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
CS = [1.0, 10.0, 100.0, 1000.0]
TAUS = [0.0, 0.3, 0.5, 0.7, 0.9]


def fit_probe(x, y, C, n_classes):
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    clf.fit(x, y)
    return clf


def probe_logits(clf, x, n_classes):
    out = np.full((len(x), n_classes), -30.0)
    out[:, clf.classes_] = clf.decision_function(x)
    return out


def mix(zs_logits, probe_lg, alpha):
    return (1 - alpha) * log_softmax(zs_logits, axis=1) + alpha * log_softmax(probe_lg, axis=1)


def is_night(path: str) -> bool:
    """Foto em infravermelho/escala de cinza: canais quase iguais."""
    a = np.asarray(Image.open(path).convert("RGB").resize((64, 64)), dtype=np.float32)
    return float(np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean()) < 3.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="siglip")
    p.add_argument("--budget", type=int, default=5, choices=[1, 5, 20])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    seed_everything(args.seed)

    classes = load_classes()
    nc = len(classes)
    emb_dir = RUNS_DIR / "teacher" / args.name
    res_dir = RESULTS_DIR / "teacher" / args.name
    out = res_dir / f"improved_b{args.budget}"
    out.mkdir(parents=True, exist_ok=True)

    ids_tr = pd.read_csv(res_dir / "ids_train_all.csv")["image_id"].tolist()
    x_tr = np.load(emb_dir / "emb_train_all.npy").astype(np.float32)
    zs_tr = np.load(res_dir / "logits_train_all.npy")
    x_te = np.load(emb_dir / "emb_test_public.npy").astype(np.float32)
    zs_te = np.load(res_dir / "logits_test_public.npy")
    test = load_split("test_public")
    y_te = test["y"].to_numpy()

    lab = load_split(budget_split(args.budget))
    pos = {i: k for k, i in enumerate(ids_tr)}
    lab_idx = np.array([pos[i] for i in lab["image_id"]])
    y_lab = lab["y"].to_numpy()
    xl, zl = x_tr[lab_idx], zs_tr[lab_idx]

    # 1) (alpha, C) por validação cruzada nos rótulos do orçamento (sem val, sem test)
    n_splits = min(5, int(np.bincount(y_lab).min()))
    rows = []
    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        for C in CS:
            oof = np.zeros((len(xl), nc))
            for tr, va in skf.split(xl, y_lab):
                oof[va] = probe_logits(fit_probe(xl[tr], y_lab[tr], C, nc), xl[va], nc)
            for a in ALPHAS:
                pred = mix(zl, oof, a).argmax(1)
                rows.append({"C": C, "alpha": a, "cv_f1": f1_score(y_lab, pred, average="macro", zero_division=0)})
    else:  # orçamento 1: não há dobras; usa só o zero-shot misturado com alpha fixo
        rows = [{"C": 10.0, "alpha": 0.5, "cv_f1": float("nan")}]
    grid = pd.DataFrame(rows)
    grid.to_csv(out / "cv_grid.csv", index=False)
    best = grid.sort_values(["cv_f1", "alpha"], ascending=[False, True]).iloc[0] if n_splits >= 2 else grid.iloc[0]
    C, alpha = float(best["C"]), float(best["alpha"])
    print(f"CV nos {len(xl)} rótulos: C={C:g}, alpha={alpha:g} (F1 macro CV {best['cv_f1']:.3f})")

    # 2) ajuste final nos 105 rótulos; professor melhorado em train e test_public
    clf = fit_probe(xl, y_lab, C, nc)
    lp_tr = mix(zs_tr, probe_logits(clf, x_tr, nc), alpha)
    lp_te = mix(zs_te, probe_logits(clf, x_te, nc), alpha)
    np.save(out / "pseudo_probs_train.npy", softmax(lp_tr, axis=1).astype(np.float32))
    np.save(out / "log_probs_train.npy", lp_tr.astype(np.float32))
    pd.DataFrame({"image_id": ids_tr}).to_csv(out / "pseudo_ids_train.csv", index=False)

    # 3) medidas no test_public (só medir o professor)
    pred = lp_te.argmax(1)
    zs_pred = zs_te.argmax(1)
    m = compute_metrics(y_te, pred, nc) | {
        "teacher": args.name, "budget": args.budget, "C": C, "alpha": alpha, "cv_f1": float(best["cv_f1"]),
        "zero_shot_f1_macro": compute_metrics(y_te, zs_pred, nc)["f1_macro"],
        "zero_shot_accuracy": compute_metrics(y_te, zs_pred, nc)["accuracy"],
    }
    save_json(m, out / "metrics_test_public.json")
    per_class_table(y_te, pred, classes).to_csv(out / "per_class_test_public.csv", index=False)
    cm = plot_confusion(y_te, pred, classes, out / "confusion_test_public.png",
                        title=f"professor melhorado ({args.name}, b{args.budget}) — test_public "
                              f"(F1 macro {m['f1_macro']:.3f}, acc {m['accuracy']:.3f})")
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out / "confusion_test_public.csv")

    # noite (cinza/IR) vs dia
    night = np.array([is_night(pth) for pth in test["path"]])
    rows = []
    for nm, mask in [("noite_IR", night), ("dia_cor", ~night)]:
        if mask.sum():
            for tag, pr in [("zero_shot", zs_pred), ("melhorado", pred)]:
                rows.append({"grupo": nm, "professor": tag, "n": int(mask.sum()),
                             "acc": float((pr[mask] == y_te[mask]).mean()),
                             "f1_macro": compute_metrics(y_te[mask], pr[mask], nc)["f1_macro"]})
    pd.DataFrame(rows).to_csv(out / "noite_vs_dia_test_public.csv", index=False)

    # precisão e cobertura vs tau (diagnóstico no test_public; tau é escolhido pelo aluno no val)
    conf = softmax(lp_te, axis=1).max(1)
    rows = []
    for tau in TAUS:
        keep = conf >= tau
        for c in range(nc):
            sel = keep & (pred == c)          # pseudo-rotulados como c que passam no filtro
            true_c = y_te == c
            rows.append({"tau": tau, "classe": classes[c], "n_previstos": int(sel.sum()),
                         "precisao": float((y_te[sel] == c).mean()) if sel.sum() else float("nan"),
                         # cobertura: fração das imagens da classe c que recebem pseudo-rótulo (passam no filtro)
                         "cobertura": float((keep & true_c).sum() / max(true_c.sum(), 1)),
                         # fração da classe c que recebe o pseudo-rótulo CERTO
                         "recall_correto": float((sel & true_c).sum() / max(true_c.sum(), 1))})
        rows.append({"tau": tau, "classe": "(todas)", "n_previstos": int(keep.sum()),
                     "precisao": float((pred[keep] == y_te[keep]).mean()) if keep.sum() else float("nan"),
                     "cobertura": float(keep.mean())})
    pd.DataFrame(rows).to_csv(out / "tau_precisao_cobertura.csv", index=False)

    save_json({"name": args.name, "budget": args.budget, "C": C, "alpha": alpha, "seed": args.seed,
               "n_labeled": int(len(xl))}, out / "config.json")
    print(f"professor melhorado {args.name} b{args.budget} @ test_public: F1 macro {m['f1_macro']:.4f} | "
          f"acc {m['accuracy']:.4f}  (zero-shot: {m['zero_shot_f1_macro']:.4f} / {m['zero_shot_accuracy']:.4f})")
    print(pd.DataFrame(rows[-1:]).to_string(index=False))


if __name__ == "__main__":
    main()
