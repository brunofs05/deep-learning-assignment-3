# Revisão da Parte 1 (Trilha C) — branch `trilha-c-professor`

Revisão feita em 2026-11-13 sobre os commits `46612cc` e `f3a9798`, contra o enunciado (`docs/PA3.md`)
e a coerência com a Parte 0 e as Partes 2–3. Decisões que saíram daqui estão em `docs/DECISOES.md` (seção 6c).

## 1. Requisitos obrigatórios da Trilha C

| Requisito do enunciado | Status | Onde |
|---|---|---|
| F1 zero-shot do professor no test_public, nomes das classes como prompt | ✅ | `results/teacher/<nome>/metrics_test_public.json` (BioCLIP 0,304, SigLIP 0,381, CLIP-L 0,451) |
| Matriz de confusão do professor = matriz de ruído; onde erra sistematicamente (noite, pequenos, classes parecidas) | ✅ (pequenos: não medido, justificado) | `improved_b5/confusion_test_public.*`, `top_confusoes.csv`, `noite_vs_dia_test_public.csv`, DECISOES 6b.13 |
| Tratar o ruído (filtro, rótulos suaves, SELFIE ou professor melhorado com os 105 rótulos) | ✅ professor melhorado + τ + T | `teacher_improve.py`, `train_student.py`, sweep em `results/trilha_c/siglip/` |
| Em quais classes o aluno supera o professor, e por quê | ✅ (hipóteses explicitadas) | `aluno_vs_professor_b5.csv`, DECISOES 6b.12 |
| Pergunta da capivara (OOD), sem implementar | ✅ | DECISOES 6b.14 + `confianca_aluno.json` |
| Professor só com pesos abertos, local | ✅ | open_clip / Hugging Face |
| Professor nunca vê o val durante o treino | ✅ verificado no código | `teacher.py` nem aceita `--splits val`; o probe usa CV só nos rótulos do orçamento |
| Professor fora do modelo entregue | ✅ | o aluno é uma ResNet-18 pura, salva no mesmo formato do piso (`predict.py` funciona) |
| Mesmo backbone da Parte 0 (b) e orçamento 5 | ✅ backbone / ⚠ resolução | ResNet-18 ImageNet; mas aluno em 224 px e piso em 384 px (ver 2.2) |

**Conclusão geral:** a Parte 1 cumpre o enunciado. A documentação é cuidadosa e honesta (ressalvas de ruído do val,
de seed única, de heurística de noite, de escolhas a priori). Os pontos abaixo são de coerência e rigor, não de
requisito faltando.

## 2. Problemas encontrados

### 2.1 FP16 forçado em qualquer GPU no `teacher.py` — corrigido
`torch.autocast(enabled=device.type == "cuda")` liga FP16 em qualquer GPU. Na 1080 Ti (Pascal) FP16 é muito
mais lento que FP32. Trocado por `amp_enabled("auto", device)`, igual ao resto do código. Também ganhou img/s e ETA.

### 2.2 Aluno em 224 px × piso em 384 px — corrigido na Parte 2
A comparação aluno × piso misturava método e resolução (a própria DECISOES 6b.10 apontava). Na Parte 2 o aluno
roda em **384 px**, igual ao piso. τ = 0 e T = 1 (escolhidos no val em 224 px) são mantidos; a ressalva fica registrada.

### 2.3 Escolha do professor olhando o test_public — trocado o critério
A comparação BioCLIP × SigLIP × CLIP-L foi feita pelo F1 no test_public. O enunciado permite o professor olhar o
test_public **para medir o professor**, mas usar esse número para **escolher** é seleção de modelo no teste.
Novo critério: **F1 macro de validação cruzada nos 105 rótulos** (`cv_f1` do professor melhorado), que não usa
nem val nem test_public. Os números zero-shot no test_public continuam como diagnóstico.

### 2.4 CLIP-L não rodado (aviso da DECISOES 6b.7) — resolvido
Em CPU era inviável. Na 1080 Ti os embeddings do train saem em minutos; a Parte 2 roda SigLIP e CLIP-L e escolhe
pelo critério de 2.3.

### 2.5 Artefatos fora do git — regenerados e conferidos
Os embeddings (`runs/teacher/`) e as runs do aluno ficaram na máquina da Elisa (`runs/` é gitignored).
Regenerados na 1080 Ti: zero-shot do SigLIP idêntico (F1 0,3805, acc 0,4353) e logits do train com argmax
100% igual e diferença máxima de 0,002. **O pipeline é reprodutível.**

### 2.6 "Cobertura" por classe no diagnóstico de τ — corrigido
Em `tau_precisao_cobertura.csv`, a cobertura por classe era `n_previstos_c / n_verdadeiros_c` (pode passar de 1;
não é cobertura). Agora: `cobertura` = fração das imagens da classe que passa no filtro, mais `recall_correto` =
fração da classe que recebe o pseudo-rótulo certo. A linha `(todas)` já estava certa.

### 2.7 Uma seed no aluno — resolvido na Parte 2
A Parte 2 pede 3 seeds no orçamento 5; o aluno roda com seeds 0, 1, 2.

## 3. Pontos que ficam como ressalva (não corrigidos)

- **Orçamento 1:** com 1 rótulo por classe não há CV possível, então o professor melhorado usa α = 0,5 e C = 10
  fixados a priori. Aceitável, mas tem que ser dito.
- **`labeled-repeat` = 5 fixo em todos os orçamentos:** a proporção rotulado:pseudo muda (b1: 105 de 6.589 amostras;
  b20: 2.100 de 8.185). É uma escolha razoável, mas não foi ajustada.
- **Hiperparâmetros do aluno sem busca:** 15 épocas, `pseudo-weight` = 1, mesma receita do piso.
- **Noite × dia** é uma heurística de saturação de cor, não metadado.
- **Prompts:** o zero-shot usa um ensemble de 4 templates com o nome da classe. O enunciado fala em "nomes das
  classes como prompt", o que isso atende; se alguém perguntar, dá para citar que não houve ajuste de prompt em
  nenhum conjunto rotulado.
- **Notebook do Kaggle** instala `open_clip_torch` sem `transformers`/`sentencepiece` (o Kaggle já traz o
  `transformers`, então provavelmente funciona, mas não foi testado).

## 4. Coerência com as próximas partes

- **Parte 2:** precisa do professor melhorado nos orçamentos 1 e 20 e de alunos em b1, b5 × 3 seeds e b20 —
  implementado em `scripts/run_parte2.py`.
- **Parte 3:** o modelo final será o aluno do orçamento 5. Qual seed entregar deve ser decidido pelo **F1 no val**
  (não no test_public). O extra `cam` (grad-cam) já está no `pyproject.toml`.
- **Entregável `weights/model.pt`:** o `train_student.py` salva no mesmo formato do piso; `predict.py` funciona sem mudança.
