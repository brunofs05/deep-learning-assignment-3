"""Parte 3 — galeria de falhas e atalhos (Grad-CAM sobre o modelo entregue).

Roda o modelo no test_public (só para REPORTAR, nada é escolhido com ele), calcula o Grad-CAM na `layer4` e gera:
  * `erro_*.png`  — imagem, classe verdadeira, prevista, confiança, CAM da classe prevista e da verdadeira;
  * `acerto_*.png` — 5 acertos com o CAM da classe prevista (olhou o animal ou o fundo?);
  * `galeria_erros.png`, `galeria_acertos.png` — painéis para os slides;
  * `candidatos_erros.csv`, `selecao.csv`, `metricas_cam_test_public.csv` — lastro numérico.

Seleção (determinística, sem olhar o desfecho "bonito"):
  * erros: os de MAIOR confiança, uma classe verdadeira por erro (erro confiante = falha mais grave). Para trocar,
    use --erros ID1 ID2 ID3 depois de olhar `candidatos_erros.csv`;
  * acertos: sorteio com semente fixa, uma classe por acerto (sem escolher os mais bonitos).

Ex.: python scripts/galeria.py --pesos weights/model.pt
"""
from __future__ import annotations

import argparse
import colorsys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from pa3.data import eval_transform, load_split
from pa3.models import build_model, count_params
from pa3.paths import REPO_ROOT, RESULTS_DIR


class GradCAM:
    """Grad-CAM clássico sobre uma camada conv (aqui, `layer4` da ResNet-18)."""

    def __init__(self, model, layer):
        self.model, self.act, self.grad = model, None, None
        layer.register_forward_hook(self._fwd)

    def _fwd(self, _, __, out):
        self.act = out
        out.register_hook(lambda g: setattr(self, "grad", g))

    def __call__(self, x, target: int | None = None):
        """Devolve (logits, cam[h, w] em [0,1], classe alvo). `x` é um lote de 1."""
        self.model.zero_grad()
        logits = self.model(x)
        t = int(logits.argmax(1)) if target is None else target
        logits[0, t].backward()
        w = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.act).sum(1))[0].detach()
        cam = cam / cam.max() if cam.max() > 0 else cam
        return logits.detach(), cam.numpy(), t


def upsample(cam: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """CAM (h, w) → (H, W) bilinear; `size` = (W, H) da imagem original."""
    t = torch.from_numpy(cam)[None, None]
    out = F.interpolate(t, size=(size[1], size[0]), mode="bilinear", align_corners=False)[0, 0].numpy()
    return np.clip(out, 0, 1)


def cam_stats(cam_full: np.ndarray) -> dict:
    """Onde está a massa do CAM: área que concentra 50% dela, centro, e massa na borda (10% de cada lado)."""
    c = cam_full / (cam_full.sum() + 1e-9)
    srt = np.sort(c.ravel())[::-1]
    area50 = float((np.cumsum(srt) < 0.5).sum() / srt.size)
    H, W = c.shape
    ys, xs = np.mgrid[0:H, 0:W]
    bh, bw = int(0.1 * H), int(0.1 * W)
    border = np.ones_like(c, dtype=bool)
    border[bh:H - bh, bw:W - bw] = False
    return {"cam_area50": area50, "cam_cx": float((c * xs).sum() / W), "cam_cy": float((c * ys).sum() / H),
            "cam_massa_borda": float(c[border].sum())}


def mean_saturation(img: Image.Image) -> float:
    a = np.asarray(img.resize((64, 64))).astype(np.float32) / 255
    mx, mn = a.max(2), a.min(2)
    return float(np.mean((mx - mn) / (mx + 1e-6)))


def overlay(ax, img: Image.Image, cam_full: np.ndarray | None, title: str):
    ax.imshow(img)
    if cam_full is not None:
        ax.imshow(cam_full, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def figura_erro(path, img, row, cam_pred, cam_true):
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    overlay(axs[0], img, None, f"{row.image_id}\nverdadeira: {row.label}")
    overlay(axs[1], img, cam_pred, f"CAM da classe PREVISTA: {row.pred}\nconfiança {row.conf:.2f}")
    overlay(axs[2], img, cam_true, f"CAM da classe VERDADEIRA: {row.label}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figura_acerto(path, img, row, cam_pred):
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2))
    overlay(axs[0], img, None, f"{row.image_id}\nverdadeira = prevista: {row.label}")
    overlay(axs[1], img, cam_pred, f"CAM | confiança {row.conf:.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def painel(path, itens, titulo):
    """itens: lista de (img, cam, legenda)."""
    n = len(itens)
    fig, axs = plt.subplots(2, n, figsize=(3.6 * n, 6.6))
    axs = np.asarray(axs).reshape(2, n)
    for j, (img, cam, leg) in enumerate(itens):
        overlay(axs[0, j], img, None, leg)
        overlay(axs[1, j], img, cam, "Grad-CAM (classe prevista)")
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pesos", default=str(REPO_ROOT / "weights" / "model.pt"))
    ap.add_argument("--saida", default=str(RESULTS_DIR / "parte3"))
    ap.add_argument("--n-erros", type=int, default=3)
    ap.add_argument("--n-acertos", type=int, default=5)
    ap.add_argument("--erros", nargs="*", default=None, help="image_ids dos erros a mostrar (sobrescreve a regra)")
    ap.add_argument("--acertos", nargs="*", default=None, help="image_ids dos acertos (sobrescreve o sorteio)")
    ap.add_argument("--seed", type=int, default=0, help="semente do sorteio dos acertos")
    ap.add_argument("--limite", type=int, default=None, help="só as N primeiras imagens (teste rápido)")
    args = ap.parse_args()

    out = __import__("pathlib").Path(args.saida)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.pesos, map_location="cpu")
    classes = ckpt["classes"]
    model = build_model(ckpt["model"], len(classes), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"modelo {ckpt['model']} | {count_params(model):,} parâmetros | img {ckpt['img_size']} | {args.pesos}")
    tf = eval_transform(ckpt["img_size"])
    cam_engine = GradCAM(model, model.layer4)

    df = load_split("test_public")
    if args.limite:
        df = df.head(args.limite)
    rows, cams = [], {}
    for k, r in enumerate(df.itertuples(), 1):
        img = Image.open(r.path).convert("RGB")
        x = tf(img)[None].requires_grad_(False)
        logits, cam, t = cam_engine(x)
        prob = torch.softmax(logits, 1)[0]
        cf = upsample(cam, img.size)
        st = cam_stats(cf)
        rows.append({"image_id": r.image_id, "label": r.label, "pred": classes[t], "conf": float(prob[t]),
                     "acertou": classes[t] == r.label, "largura": img.size[0], "altura": img.size[1],
                     "saturacao": mean_saturation(img), "luminosidade": float(np.asarray(img.convert("L")).mean()),
                     **st})
        if k % 100 == 0:
            print(f"{k}/{len(df)}", flush=True)
    res = pd.DataFrame(rows)
    res["noturna"] = res["saturacao"] < 0.08
    res.to_csv(out / "metricas_cam_test_public.csv", index=False)
    print(f"test_public: acurácia {res.acertou.mean():.3f} em {len(res)} imagens")

    # ---- seleção ----
    erros = res[~res.acertou].sort_values("conf", ascending=False)
    erros_cand = erros.drop_duplicates("label").head(15)  # um por classe verdadeira, mais confiantes primeiro
    erros_cand.to_csv(out / "candidatos_erros.csv", index=False)
    sel_e = res[res.image_id.isin(args.erros)] if args.erros else erros_cand.head(args.n_erros)
    sel_e = sel_e.set_index("image_id").loc[args.erros if args.erros else sel_e.image_id].reset_index()
    ac = res[res.acertou]
    if args.acertos:
        sel_a = ac.set_index("image_id").loc[args.acertos].reset_index()
    else:
        sel_a = ac.sample(frac=1, random_state=args.seed).drop_duplicates("label").head(args.n_acertos)
    sel = pd.concat([sel_e.assign(grupo="erro"), sel_a.assign(grupo="acerto")])
    sel.to_csv(out / "selecao.csv", index=False)

    # ---- figuras ----
    pe, pa = [], []
    for i, row in enumerate(sel_e.itertuples(), 1):
        img = Image.open(df.set_index("image_id").loc[row.image_id, "path"]).convert("RGB")
        x = tf(img)[None]
        _, c_pred, _ = cam_engine(x)
        _, c_true, _ = cam_engine(x, target=classes.index(row.label))
        cp, ct = upsample(c_pred, img.size), upsample(c_true, img.size)
        figura_erro(out / f"erro_{i}_{row.image_id}.png", img, row, cp, ct)
        pe.append((img, cp, f"{row.label} → {row.pred}\nconf {row.conf:.2f}"))
    for i, row in enumerate(sel_a.itertuples(), 1):
        img = Image.open(df.set_index("image_id").loc[row.image_id, "path"]).convert("RGB")
        _, c_pred, _ = cam_engine(tf(img)[None])
        cp = upsample(c_pred, img.size)
        figura_acerto(out / f"acerto_{i}_{row.image_id}.png", img, row, cp)
        pa.append((img, cp, f"{row.label}\nconf {row.conf:.2f}"))
    painel(out / "galeria_erros.png", pe, "Três erros confiantes do modelo entregue (test_public)")
    painel(out / "galeria_acertos.png", pa, "Cinco acertos sorteados (test_public)")

    cols = ["image_id", "grupo", "label", "pred", "conf", "largura", "altura", "noturna", "saturacao",
            "cam_area50", "cam_cx", "cam_cy", "cam_massa_borda"]
    print(sel[cols].round(3).to_string(index=False))
    print(f"\nfiguras em {out}")


if __name__ == "__main__":
    main()
