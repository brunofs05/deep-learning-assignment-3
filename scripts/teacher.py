"""Trilha C — professor zero-shot (foundation model aberto, via open_clip).

Extrai embeddings de imagem e logits zero-shot (nomes das classes como prompt) para o train inteiro
(sem rótulo) e para o test_public (só para MEDIR o professor). O val NUNCA é lido aqui.
O professor só gera arquivos; ele não entra no modelo entregue.

Ex.: python scripts/teacher.py --name bioclip
Saídas:
  runs/teacher/<name>/emb_<split>.npy      embeddings normalizados (fp16)   [gitignored]
  results/teacher/<name>/logits_<split>.npy  logits zero-shot (N, 21)
  results/teacher/<name>/ids_<split>.csv     image_id na mesma ordem
  results/teacher/<name>/text_emb.npy        embeddings de texto das classes
  results/teacher/<name>/{metrics,per_class,confusion}_test_public.*
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from pa3.data import ImageDataset, load_classes, load_split
from pa3.metrics import compute_metrics, per_class_table, plot_confusion
from pa3.paths import RESULTS_DIR, RUNS_DIR
from pa3.utils import amp_enabled, fmt_time, get_device, save_json, seed_everything
# nome curto -> id open_clip (pesos abertos no Hugging Face)
MODELS = {
    "bioclip": "hf-hub:imageomics/bioclip",
    "siglip": "hf-hub:timm/ViT-B-16-SigLIP",
    "clip-l": "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
}

TEMPLATES = [
    "a photo of {}.",
    "a camera trap photo of {}.",
    "an infrared night photo of {}, taken by a trail camera.",
    "a blurry photo of a small {} in the wild.",
]


def humanize(label: str) -> str:
    return label.replace("_", " ")


def full_frame_transform(model, preprocess) -> T.Compose:
    """Redimensiona o quadro inteiro (sem center crop, que cortaria animais na borda)."""
    size = model.visual.image_size
    size = size if isinstance(size, int) else size[0]
    norm = next(t for t in preprocess.transforms if isinstance(t, T.Normalize))
    return T.Compose([T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), norm])


@torch.no_grad()
def text_embeddings(model, tokenizer, classes, device):
    embs = []
    for c in classes:
        tok = tokenizer([t.format(humanize(c)) for t in TEMPLATES]).to(device)
        e = model.encode_text(tok).float()
        e = e / e.norm(dim=-1, keepdim=True)
        e = e.mean(0)
        embs.append(e / e.norm())
    return torch.stack(embs)  # (C, D)


@torch.no_grad()
def image_embeddings(model, df, tfm, device, batch_size, workers, amp):
    dl = DataLoader(ImageDataset.from_df(df, tfm), batch_size=batch_size, num_workers=workers)
    out, t0 = [], time.time()
    for b, (x, _) in enumerate(dl):
        # FP16 só em GPU com tensor cores; na Pascal (1080 Ti) FP16 é muito mais lento que FP32
        with torch.autocast(device_type=device.type, enabled=amp):
            e = model.encode_image(x.to(device)).float()
        out.append((e / e.norm(dim=-1, keepdim=True)).cpu())
        if b % 20 == 0:
            el = time.time() - t0
            eta = el / (b + 1) * (len(dl) - b - 1)
            print(f"  {b + 1}/{len(dl)} lotes | {(b + 1) * batch_size / max(el, 1e-9):.0f} img/s | "
                  f"decorrido {fmt_time(el)}, ETA {fmt_time(eta)}", flush=True)
    print(f"  {len(df)} imagens em {fmt_time(time.time() - t0)}", flush=True)
    return torch.cat(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", choices=list(MODELS), default="bioclip")
    p.add_argument("--hf-id", default=None, help="sobrescreve o id do open_clip")
    p.add_argument("--splits", nargs="+", default=["test_public", "train_all"],
                   choices=["test_public", "train_all"], help="o val é proibido")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    import open_clip

    seed_everything(0)
    device = get_device(args.device)
    classes = load_classes()
    hf_id = args.hf_id or MODELS[args.name]
    model, _, preprocess = open_clip.create_model_and_transforms(hf_id)
    tokenizer = open_clip.get_tokenizer(hf_id)
    model = model.to(device).eval()
    tfm = full_frame_transform(model, preprocess)
    scale = model.logit_scale.exp().item()
    amp = amp_enabled("auto", device)
    print(f"[{args.name}] {hf_id} | {sum(p.numel() for p in model.parameters()):,} params | {device} amp={amp}",
          flush=True)

    emb_dir = RUNS_DIR / "teacher" / args.name
    res_dir = RESULTS_DIR / "teacher" / args.name
    emb_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    txt = text_embeddings(model, tokenizer, classes, device).cpu()
    np.save(res_dir / "text_emb.npy", txt.numpy().astype(np.float16))

    for split in args.splits:
        df = load_split(split)
        print(f"{split}: {len(df)} imagens", flush=True)
        emb = image_embeddings(model, df, tfm, device, args.batch_size, args.workers, amp)
        logits = scale * emb @ txt.T
        np.save(emb_dir / f"emb_{split}.npy", emb.numpy().astype(np.float16))
        np.save(res_dir / f"logits_{split}.npy", logits.numpy().astype(np.float32))
        df[["image_id"]].to_csv(res_dir / f"ids_{split}.csv", index=False)
        if split == "test_public":
            y, pred = df["y"].to_numpy(), logits.argmax(1).numpy()
            m = compute_metrics(y, pred, len(classes)) | {"teacher": args.name, "hf_id": hf_id, "split": split}
            save_json(m, res_dir / "metrics_test_public.json")
            per_class_table(y, pred, classes).to_csv(res_dir / "per_class_test_public.csv", index=False)
            plot_confusion(y, pred, classes, res_dir / "confusion_test_public.png",
                           title=f"professor {args.name} zero-shot — test_public "
                                 f"(F1 macro {m['f1_macro']:.3f}, acc {m['accuracy']:.3f})")
            print(f"professor {args.name} @ test_public: F1 macro {m['f1_macro']:.4f} | acc {m['accuracy']:.4f}")


if __name__ == "__main__":
    main()
