"""Parte 2 — curva de orçamento da Trilha C, ponta a ponta, com progresso, ETA e monitor de GPU.

Etapas:
  1. Professores (SigLIP e CLIP-L): embeddings + zero-shot em test_public e train (o val nunca é lido).
  2. Professor melhorado (zero-shot + probe) para cada professor × orçamento {1, 5, 20}.
  3. Escolha do professor pelo F1 de validação cruzada NOS 105 RÓTULOS do orçamento 5 (sem val, sem test_public).
  4. Alunos ResNet-18 (384 px, τ = 0, T = 1, escolhidos no val na Parte 1): b1 s0, b5 s{0,1,2}, b20 s0.
  5. Avaliação no test_public, curva_orcamento.csv, gráfico e ganho por classe; análises da Parte 1 refeitas
     com o professor escolhido (aluno × professor por classe, matriz de ruído, confiança).

Retomável: o que já existe é pulado. Ex.: python scripts/run_parte2.py
Log de GPU: runs/parte2_gpu.csv (uso %, memória, temperatura a cada 10 s).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd

from pa3.paths import RESULTS_DIR, RUNS_DIR
from pa3.utils import fmt_time, save_json

PY = [sys.executable]
S = Path(__file__).parent
OUT = RESULTS_DIR / "parte2"
TEACHERS = ["siglip", "clip-l"]
BUDGETS = [1, 5, 20]
STUDENT_JOBS = [(1, 0), (5, 0), (5, 1), (5, 2), (20, 0)]
N_UNL_PLUS_REP = {1: 6484 + 21 * 5, 5: 6400 + 105 * 5, 20: 6085 + 420 * 5}  # amostras por época do aluno


class GpuMonitor(threading.Thread):
    """Amostra nvidia-smi a cada `every` s; `window()` devolve a média desde a última marca."""

    def __init__(self, path: Path, every: float = 10.0):
        super().__init__(daemon=True)
        self.path, self.every, self.samples, self.mark = path, every, [], 0
        self.stop_ev = threading.Event()

    def run(self):
        with open(self.path, "a") as f:
            while not self.stop_ev.is_set():
                try:
                    o = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                         "--format=csv,noheader,nounits"], text=True).strip().split(", ")
                    util, mem, temp, pw = float(o[0]), float(o[1]), float(o[2]), float(o[3])
                    self.samples.append((util, mem, temp))
                    f.write(f"{time.strftime('%H:%M:%S')},{util},{mem},{temp},{pw}\n")
                    f.flush()
                except Exception:
                    pass
                self.stop_ev.wait(self.every)

    def window(self) -> str:
        w = self.samples[self.mark:]
        self.mark = len(self.samples)
        if not w:
            return "GPU: sem amostras"
        u = sum(s[0] for s in w) / len(w)
        return (f"GPU na etapa: uso médio {u:.0f}% | memória máx {max(s[1] for s in w) / 1024:.1f} GB | "
                f"temp máx {max(s[2] for s in w):.0f}°C")


def sh(cmd: list[str]):
    subprocess.run(PY + cmd, check=True)


def step(title: str, t0: float):
    print(f"\n{'=' * 100}\n{title} | decorrido {fmt_time(time.time() - t0)}\n{'=' * 100}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--teacher", choices=TEACHERS, default=None, help="força o professor (pula a escolha por CV)")
    args = ap.parse_args()

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    gpu = GpuMonitor(RUNS_DIR / "parte2_gpu.csv")
    gpu.start()

    # ---------- 1. professores ----------
    for i, t in enumerate(TEACHERS, 1):
        step(f"[1/5] professor {i}/{len(TEACHERS)}: {t} (embeddings + zero-shot)", t0)
        if (RUNS_DIR / "teacher" / t / "emb_train_all.npy").exists():
            print("já extraído, pulando")
        else:
            sh([str(S / "teacher.py"), "--name", t, "--splits", "test_public", "train_all",
                "--workers", str(args.workers)])
        print(gpu.window(), flush=True)

    # ---------- 2. professor melhorado ----------
    step("[2/5] professor melhorado (zero-shot + probe nos rótulos do orçamento)", t0)
    for t in TEACHERS:
        for b in BUDGETS:
            sh([str(S / "teacher_improve.py"), "--name", t, "--budget", str(b)])

    # ---------- 3. escolha do professor por CV nos 105 rótulos ----------
    step("[3/5] escolha do professor: F1 macro de validação cruzada nos 105 rótulos (orçamento 5)", t0)
    rows = []
    for t in TEACHERS:
        d = RESULTS_DIR / "teacher" / t
        cfg = json.loads((d / "improved_b5" / "metrics_test_public.json").read_text())
        rows.append({"professor": t, "cv_f1_b5": cfg["cv_f1"], "alpha": cfg["alpha"], "C": cfg["C"],
                     "test_public_zeroshot_f1": cfg["zero_shot_f1_macro"], "test_public_melhorado_f1": cfg["f1_macro"]})
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "comparacao_professores.csv", index=False)
    teacher = args.teacher or comp.sort_values("cv_f1_b5", ascending=False).iloc[0]["professor"]
    print(comp.round(4).to_string(index=False))
    print(f"→ professor escolhido: {teacher} (critério: maior F1 de CV nos 105 rótulos; test_public só reportado)")
    save_json({"professor": teacher, "criterio": "maior cv_f1 no orçamento 5 (CV estratificada nos 105 rótulos)",
               "forcado": args.teacher is not None, "comparacao": rows}, OUT / "professor_escolhido.json")

    # ---------- 4. alunos ----------
    method = f"aluno_{teacher}"
    jobs = [(b, s, f"{method}_b{b}_s{s}") for b, s in STUDENT_JOBS]
    done_cost, left_cost = 0.0, sum(N_UNL_PLUS_REP[b] for b, _, _ in jobs)
    t_students = time.time()
    for k, (b, s, name) in enumerate(jobs, 1):
        cost = N_UNL_PLUS_REP[b]
        rate = (time.time() - t_students) / done_cost if done_cost else 35.0 * args.epochs / 6925  # s por amostra-época
        eta = fmt_time(rate * left_cost)
        step(f"[4/5] aluno {k}/{len(jobs)}: {name} | ETA dos alunos ~{eta}"
             f"{' (estimativa inicial)' if not done_cost else ''}", t0)
        if (RUNS_DIR / name / "best_val.json").exists():
            print("já treinado, pulando")
        else:
            t1 = time.time()
            sh([str(S / "train_student.py"), "--teacher", teacher, "--budget", str(b), "--seed", str(s),
                "--tau", "0", "--temp", "1", "--img-size", str(args.img_size), "--epochs", str(args.epochs),
                "--workers", str(args.workers), "--method", method, "--run-name", name])
            done_cost += cost
            print(f"aluno {name} levou {fmt_time(time.time() - t1)} | {gpu.window()}", flush=True)
        left_cost -= cost

    # ---------- 5. avaliação, curva e análises ----------
    step("[5/5] avaliação no test_public, curva de orçamento, ganho por classe e análises", t0)
    for _, _, name in jobs:
        sh([str(S / "evaluate.py"), "--run", str(RUNS_DIR / name), "--workers", str(args.workers)])
    sh([str(S / "collect_results.py")])
    sh([str(S / "plot_parte2.py")])
    sh([str(S / "compare_student_teacher.py"), "--teacher", teacher, "--run", str(RUNS_DIR / f"{method}_b5_s0")])
    sh([str(S / "analisar_ruido.py"), "--teacher", teacher, "--run", str(RUNS_DIR / f"{method}_b5_s0")])

    gpu.stop_ev.set()
    tot = gpu.samples
    if tot:
        print(f"\nGPU no total: uso médio {sum(x[0] for x in tot) / len(tot):.0f}% em {len(tot)} amostras")
    print(f"Parte 2 concluída em {fmt_time(time.time() - t0)}. Resultados em {OUT}")


if __name__ == "__main__":
    main()
