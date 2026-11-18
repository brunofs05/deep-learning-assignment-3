# Parte 2 — Curva de orçamento (Trilha C, professor CLIP-L)

Reproduzir: `uv run python scripts/run_parte2.py` (~1h10 numa GTX 1080 Ti; retomável).

## Gráfico obrigatório — `curva_orcamento.png`
F1 macro no test_public × rótulos por classe (eixo log), com piso, aluno, professor (melhorado e zero-shot) e teto.
Números em `../curva_orcamento.csv` (entregável: método, orçamento, seed, F1 macro, acurácia).

| Rótulos/classe | Piso ResNet-18 | **Aluno ResNet-18** | Professor CLIP-L melhorado | Professor zero-shot | Teto |
|---|---|---|---|---|---|
| 1 (21 imagens) | 0,214 | **0,523** | 0,513 | 0,451 | 0,686 |
| 5 (105 imagens), 3 seeds | 0,454 ± 0,027 | **0,625 ± 0,038** | 0,641 | 0,451 | 0,686 |
| 20 (420 imagens) | 0,602 | **0,723** | 0,751 | 0,451 | 0,686 |

- O aluno recupera **~74% da distância piso→teto** no orçamento 5 sem nenhum rótulo a mais.
- Com 1 rótulo/classe o aluno já supera o piso com 5 rótulos/classe.
- Com 20 rótulos/classe aluno e professor passam do teto (que não usa o CLIP-L).
- O aluno fica 0,02–0,03 abaixo do professor (empata no orçamento 1): destila quase tudo, não supera no geral.

## Ganho por classe no orçamento 5 — `ganho_por_classe_b5.{png,csv}`
F1 por classe, média de 3 seeds, aluno − piso (o CSV traz também o professor melhorado).

- **Mais ganharam:** american_crow (0,27 → 0,77), american_black_bear (0,43 → 0,89), red_fox, striped_skunk, horse, grey_fox (~+0,30).
- **Não ganharam:** white_tailed_deer, chicken, dog, eastern_chipmunk — já eram fáceis para o piso (F1 > 0,75).
- **Piorou:** domestic_cat (0,29 → 0,09). O professor acerta só 6 de 48 gatos (espalha em red_fox, woodchuck, raccoon);
  o aluno herda o ruído estruturado e cada seed erra para um lugar diferente.
- bird e bobcat continuam ruins (0,14 e 0,20): também são fracos no professor (0,27 e 0,26).

## Escolha do professor — `comparacao_professores.csv`, `professor_escolhido.json`
Critério: F1 macro de validação cruzada do professor melhorado **nos 105 rótulos** (sem val, sem test_public).
CLIP-L 0,598 × SigLIP 0,535 → CLIP-L. Os F1 no test_public estão no CSV só como relato.

## Ressalvas
- τ = 0 e T = 1 vieram do sweep da Parte 1 (SigLIP, 224 px); não foram re-escolhidos.
- No orçamento 1, o professor melhorado usa α = 0,5 e C = 10 fixos (não há CV com 1 rótulo/classe).
- Variância entre seeds do aluno é alta (0,583–0,656); a seed de melhor val (s1) teve o pior test_public.
