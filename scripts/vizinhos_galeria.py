"""Parte 3 — atalhos: vizinhos mais próximos no TRAIN das imagens da galeria, no espaço de features do modelo entregue.

Para cada imagem de `results/parte3/selecao.csv`, busca as k imagens do train mais parecidas segundo o próprio modelo
(saída do avgpool da ResNet-18, 512-d, similaridade de cosseno) e mostra com que rótulo cada vizinha entrou no
treino do aluno: o rótulo verdadeiro (se está em rotulos_5) ou o pseudo-rótulo do professor (argmax; o aluno
treina com o alvo suave, T = 1, mas o argmax é a classe para onde ele foi puxado). Se as vizinhas
de um erro são o MESMO CENÁRIO com o rótulo previsto, o modelo está reconhecendo o fundo/câmera, não o animal.

Só lê o train (vizinhas) e as imagens da galeria (já escolhidas por galeria.py); nada é escolhido aqui.
Ex.: python scripts/vizinhos_galeria.py --pesos weights/model.pt --teacher clip-l
Saídas: results/parte3/vizinhos.png, vizinhos.csv
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from pa3.data import ImageDataset, budget_split, eval_transform, load_split
from pa3.models import build_model
from pa3.paths import REPO_ROOT, RESULTS_DIR
from pa3.utils import get_device


@torch.no_grad()
def features(model, df, tf, device, workers):
    model.fc_backup, model.fc = model.fc, torch.nn.Identity()
    dl = DataLoader(ImageDataset(df["path"], None, tf), batch_size=64, num_workers=workers)
    out = [model(x.to(device)).float().cpu() for x, _ in dl]
    model.fc = model.fc_backup
    f = torch.cat(out)
    return torch.nn.functional.normalize(f, dim=1).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pesos", default=str(REPO_ROOT / "weights" / "model.pt"))
    ap.add_argument("--teacher", default="clip-l")
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = RESULTS_DIR / "parte3"
    device = get_device()

    ckpt = torch.load(args.pesos, map_location="cpu")
    classes = ckpt["classes"]
    model = build_model(ckpt["model"], len(classes), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    tf = eval_transform(ckpt["img_size"])

    # rótulo com que cada imagem do train entrou no treino do aluno
    train = load_split("train_all")
    imp = RESULTS_DIR / "teacher" / args.teacher / f"improved_b{args.budget}"
    pseudo = pd.Series(np.load(imp / "pseudo_probs_train.npy").argmax(1),
                       index=pd.read_csv(imp / "pseudo_ids_train.csv")["image_id"])
    lab = load_split(budget_split(args.budget)).set_index("image_id")["label"]
    train["rotulo_treino"] = [lab[i] if i in lab.index else classes[pseudo[i]] for i in train["image_id"]]
    train["origem"] = ["verdadeiro" if i in lab.index else "pseudo" for i in train["image_id"]]

    sel = pd.read_csv(out / "selecao.csv")
    test = load_split("test_public").set_index("image_id")
    q = test.loc[sel["image_id"]].reset_index()
    print(f"features do train ({len(train)} imagens) em {device}...", flush=True)
    ft = features(model, train, tf, device, args.workers)
    fq = features(model, q, tf, device, args.workers)
    sims = fq @ ft.T

    # ---- o test_public inteiro: quanto a previsão segue o rótulo de treino da vizinha mais próxima? ----
    tp = test.reset_index()
    met = pd.read_csv(out / "metricas_cam_test_public.csv").set_index("image_id").loc[tp["image_id"]]
    pred, conf = met["pred"].to_numpy(), met["conf"].to_numpy()
    s_all = features(model, tp, tf, device, args.workers) @ ft.T
    nn1 = s_all.argmax(1)
    viz = train["rotulo_treino"].to_numpy()[nn1]
    y = tp["label"].to_numpy()
    acerto, segue, viz_ok = pred == y, pred == viz, viz == y
    alta = s_all.max(1) >= 0.9  # vizinha quase idêntica: mesma câmera/cena
    resumo_global = {
        "n": int(len(tp)),
        "frac_pred_igual_rotulo_da_vizinha": float(segue.mean()),
        "frac_vizinha_cos_ge_0.9": float(alta.mean()),
        "acc_modelo": float(acerto.mean()),
        "acc_quando_vizinha_tem_rotulo_certo": float(acerto[viz_ok].mean()),
        "acc_quando_vizinha_tem_rotulo_errado": float(acerto[~viz_ok].mean()),
        "frac_vizinha_rotulo_certo": float(viz_ok.mean()),
        "erros_que_seguem_a_vizinha": float(segue[~acerto].mean()),
        # sinais de "não confie" disponíveis no modelo entregue (substitutos de OOD; ver DECISOES 6b.14/6d.4)
        "conf_media_acertos": float(conf[acerto].mean()), "conf_media_erros": float(conf[~acerto].mean()),
        "auroc_conf_acerto": float(roc_auc_score(acerto, conf)),
        "sim_vizinha_media_acertos": float(s_all.max(1)[acerto].mean()),
        "sim_vizinha_media_erros": float(s_all.max(1)[~acerto].mean()),
        "auroc_sim_vizinha_acerto": float(roc_auc_score(acerto, s_all.max(1))),
        "nota": "vizinha = imagem do train mais parecida nas features do modelo entregue; rótulo = o de treino "
                "(verdadeiro do orçamento ou pseudo-rótulo do professor). Só diagnóstico.",
    }
    pd.Series(resumo_global).to_json(out / "vizinhos_resumo_test_public.json", indent=2, force_ascii=False)
    print(pd.Series(resumo_global).drop("nota").round(3).to_string(), flush=True)

    rows = []
    fig, axs = plt.subplots(len(sel), args.k + 1, figsize=(2.6 * (args.k + 1), 2.3 * len(sel)))
    for i, r in sel.iterrows():
        nn = np.argsort(-sims[i])[:args.k]
        ax = axs[i, 0]
        ax.imshow(Image.open(q.loc[i, "path"]).convert("RGB"))
        ax.set_title(f"[{r.grupo}] {r.label}\n→ {r.pred} ({r.conf:.2f})", fontsize=7, color="C3" if r.grupo == "erro" else "C2")
        for j, n in enumerate(nn, 1):
            t = train.iloc[n]
            rows.append({"image_id": r.image_id, "grupo": r.grupo, "label": r.label, "pred": r.pred, "rank": j,
                         "vizinha": t.image_id, "sim": float(sims[i, n]), "rotulo_treino": t.rotulo_treino,
                         "origem": t.origem})
            axs[i, j].imshow(Image.open(t.path).convert("RGB"))
            cor = "C3" if t.rotulo_treino == r.pred and r.grupo == "erro" else "k"
            axs[i, j].set_title(f"{t.rotulo_treino} ({t.origem})\ncos {sims[i, n]:.2f}", fontsize=7, color=cor)
        for ax in axs[i]:
            ax.axis("off")
            ax.set_aspect("auto")
    fig.suptitle(f"Vizinhas mais próximas no train (features do modelo entregue); título = rótulo com que a vizinha "
                 f"entrou no treino", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out / "vizinhos.png", dpi=110)
    plt.close(fig)
    df = pd.DataFrame(rows)
    df.round(4).to_csv(out / "vizinhos.csv", index=False)
    resumo = df.assign(igual_pred=df.rotulo_treino == df.pred).groupby(["grupo", "image_id", "label", "pred"],
                                                                          sort=False).igual_pred.sum()
    print(f"nº de vizinhas (de {args.k}) com o rótulo PREVISTO:\n{resumo.to_string()}")
    print(f"salvo em {out / 'vizinhos.png'}")


if __name__ == "__main__":
    main()
