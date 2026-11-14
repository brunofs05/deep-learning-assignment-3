"""Trilha C — tabela por classe no test_public: piso × professor melhorado × aluno (só para REPORTAR).

Ex.: python scripts/compare_student_teacher.py --teacher siglip --run runs/aluno_siglip_b5_s0_tau0_T1
     python scripts/compare_student_teacher.py --teacher clip-l --run runs/aluno_clip-l_b5_s{0,1,2}
Com várias runs (seeds), f1_aluno é a média; seeds_acima_prof conta em quantas o aluno supera o professor.
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
    p.add_argument("--run", required=True, type=Path, nargs="+")
    args = p.parse_args()
    per = [pd.read_csv(r / "per_class_test_public.csv").set_index("label") for r in args.run]
    fa = pd.concat([x["f1"] for x in per], axis=1)
    t = pd.read_csv(RESULTS_DIR / "teacher" / args.teacher / "improved_b5" / "per_class_test_public.csv").set_index("label")
    z = pd.read_csv(RESULTS_DIR / "teacher" / args.teacher / "per_class_test_public.csv").set_index("label")
    pi = pd.read_csv(RESULTS_DIR / "parte0" / "per_class_resnet18_b5_media3seeds.csv").set_index("label")
    d = pd.DataFrame({"support": per[0]["support"], "f1_piso_3seeds": pi["f1"], "f1_prof_zeroshot": z["f1"],
                      "f1_prof_melhorado": t["f1"], "f1_aluno": fa.mean(axis=1)})
    if len(per) > 1:
        d["f1_aluno_min"], d["f1_aluno_max"] = fa.min(axis=1), fa.max(axis=1)
        d["seeds_acima_prof"] = fa.gt(t["f1"], axis=0).sum(axis=1)
    d["aluno_menos_prof"] = d["f1_aluno"] - d["f1_prof_melhorado"]
    d["aluno_menos_piso"] = d["f1_aluno"] - d["f1_piso_3seeds"]
    d = d.sort_values("aluno_menos_prof", ascending=False)
    out = RESULTS_DIR / "trilha_c" / args.teacher / "aluno_vs_professor_b5.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    d.round(4).to_csv(out)
    print(d.round(2).to_string())
    print(d[["f1_piso_3seeds", "f1_prof_zeroshot", "f1_prof_melhorado", "f1_aluno"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
