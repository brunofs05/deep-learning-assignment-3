"""Parte 2 — gráfico da curva de orçamento e ganho por classe no orçamento 5 (só reportar, test_public).

Lê results/curva_orcamento.csv e as runs; escreve em results/parte2/:
  curva_orcamento.png              F1 macro × rótulos/classe (log), piso, aluno, professor, teto
  ganho_por_classe_b5.{csv,png}    F1 por classe (média de 3 seeds): aluno − piso, com o professor ao lado
Ex.: python scripts/plot_parte2.py
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pa3.paths import RESULTS_DIR, RUNS_DIR

TETO = 0.686
BUDGETS = [1, 5, 20]


def per_class_mean(prefix: str, seeds) -> pd.Series:
    d = [pd.read_csv(RUNS_DIR / f"{prefix}_s{s}" / "per_class_test_public.csv").set_index("label")["f1"] for s in seeds]
    return pd.concat(d, axis=1).mean(axis=1)


def main():
    out = RESULTS_DIR / "parte2"
    out.mkdir(parents=True, exist_ok=True)
    t = json.loads((out / "professor_escolhido.json").read_text())["professor"]
    df = pd.read_csv(RESULTS_DIR / "curva_orcamento.csv")
    aluno, piso = f"aluno_{t}", "piso_resnet18"

    # ---------- curva de orçamento ----------
    fig, ax = plt.subplots(figsize=(8, 5.2))
    styles = [(piso, "Piso (ResNet-18, só rótulos)", "C0", "o"),
              (aluno, f"Aluno ResNet-18 (Trilha C, professor {t})", "C1", "s"),
              (f"professor_{t}_melhorado", f"Professor {t} melhorado (zero-shot + probe)", "C2", "^")]
    for metodo, lab, cor, mk in styles:
        g = df[df.metodo == metodo].groupby("orcamento").f1_macro.agg(["mean", "std", "count"]).reindex(BUDGETS)
        if g["mean"].isna().all():
            continue
        ax.errorbar(g.index, g["mean"], yerr=g["std"].fillna(0), marker=mk, color=cor, capsize=4, lw=2, label=lab)
        for b, row in g.iterrows():
            if pd.notna(row["mean"]):
                txt = f"{row['mean']:.3f}" + (f"±{row['std']:.3f}" if row["count"] > 1 else "")
                off = {"C0": (8, -14), "C1": (8, -16), "C2": (-14, 9)}[cor]  # aluno abaixo, professor acima
                ax.annotate(txt, (b, row["mean"]), textcoords="offset points", xytext=off,
                            fontsize=7.5, color=cor, ha="left" if cor != "C2" else "right")
    zs = df[df.metodo == f"professor_{t}_zeroshot"].f1_macro
    if len(zs):
        ax.axhline(zs.iloc[0], color="C2", ls=":", lw=1.5, label=f"Professor {t} zero-shot ({zs.iloc[0]:.3f})")
    ax.axhline(TETO, color="k", ls="--", lw=1.2, label=f"Teto (todos os rótulos, {TETO})")
    ax.set_xscale("log")
    ax.set_xticks(BUDGETS, [f"{b}\n({21 * b} imagens)" for b in BUDGETS])
    ax.set_xlim(0.75, 32)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("rótulos por classe (escala log)")
    ax.set_ylabel("F1 macro no test_public")
    ax.set_title("Curva de orçamento — orçamento 5: média ± desvio de 3 seeds")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "curva_orcamento.png", dpi=150)
    plt.close(fig)

    # ---------- ganho por classe no orçamento 5 ----------
    g = pd.DataFrame({
        "f1_piso": per_class_mean(f"{piso}_b5", [0, 1, 2]),
        "f1_aluno": per_class_mean(f"{aluno}_b5", [0, 1, 2]),
        "f1_professor_melhorado": pd.read_csv(RESULTS_DIR / "teacher" / t / "improved_b5" / "per_class_test_public.csv")
        .set_index("label")["f1"],
    })
    g["support"] = pd.read_csv(RUNS_DIR / f"{piso}_b5_s0" / "per_class_test_public.csv").set_index("label")["support"]
    g["ganho_aluno_vs_piso"] = g.f1_aluno - g.f1_piso
    g["aluno_vs_professor"] = g.f1_aluno - g.f1_professor_melhorado
    g = g.sort_values("ganho_aluno_vs_piso", ascending=False)
    g.round(4).to_csv(out / "ganho_por_classe_b5.csv")

    fig, ax = plt.subplots(figsize=(9, 7))
    cores = ["C2" if v > 0.05 else ("C3" if v < -0.05 else "0.6") for v in g.ganho_aluno_vs_piso]
    ax.barh(g.index[::-1], g.ganho_aluno_vs_piso[::-1], color=cores[::-1])
    for i, (lab, row) in enumerate(g[::-1].iterrows()):
        ax.text(row.ganho_aluno_vs_piso + (0.01 if row.ganho_aluno_vs_piso >= 0 else -0.01), i,
                f"{row.f1_piso:.2f} → {row.f1_aluno:.2f}", va="center", fontsize=7,
                ha="left" if row.ganho_aluno_vs_piso >= 0 else "right")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("ganho de F1 (aluno − piso), média de 3 seeds")
    ax.set_title(f"Ganho por classe no orçamento 5 — aluno ({t}) vs piso\n"
                 f"F1 macro: piso {g.f1_piso.mean():.3f} → aluno {g.f1_aluno.mean():.3f}")
    lim = max(abs(g.ganho_aluno_vs_piso).max() + 0.15, 0.3)
    ax.set_xlim(-lim, lim)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out / "ganho_por_classe_b5.png", dpi=150)
    plt.close(fig)
    print(g.round(3).to_string())
    print(f"\nsalvo em {out}")


if __name__ == "__main__":
    main()
