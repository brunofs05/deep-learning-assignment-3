"""Trilha C — tabela por classe no test_public: piso × professor melhorado × aluno (só para REPORTAR).

Ex.: python scripts/compare_student_teacher.py --teacher siglip --run runs/aluno_siglip_b5_s0_tau0_T1
Saída: results/trilha_c/<teacher>/aluno_vs_professor_b5.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pa3.paths import RESULTS_DIR


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", default="siglip")
    p.add_argument("--run", required=True, type=Path)
    args = p.parse_args()
    a = pd.read_csv(args.run / "per_class_test_public.csv").set_index("label")
    t = pd.read_csv(RESULTS_DIR / "teacher" / args.teacher / "improved_b5" / "per_class_test_public.csv").set_index("label")
    z = pd.read_csv(RESULTS_DIR / "teacher" / args.teacher / "per_class_test_public.csv").set_index("label")
    pi = pd.read_csv(RESULTS_DIR / "parte0" / "per_class_resnet18_b5_media3seeds.csv").set_index("label")
    d = pd.DataFrame({"support": a["support"], "f1_piso_3seeds": pi["f1"], "f1_prof_zeroshot": z["f1"],
                      "f1_prof_melhorado": t["f1"], "f1_aluno": a["f1"]})
    d["aluno_menos_prof"] = d["f1_aluno"] - d["f1_prof_melhorado"]
    d["aluno_menos_piso"] = d["f1_aluno"] - d["f1_piso_3seeds"]
    d = d.sort_values("aluno_menos_prof", ascending=False)
    out = RESULTS_DIR / "trilha_c" / args.teacher / "aluno_vs_professor_b5.csv"
    d.round(4).to_csv(out)
    print(d.round(2).to_string())
    print(d[["f1_piso_3seeds", "f1_prof_zeroshot", "f1_prof_melhorado", "f1_aluno"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
