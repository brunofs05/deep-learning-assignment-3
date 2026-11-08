# Parte 0 — Piso

Reproduzir: `uv run python scripts/run_parte0.py` (~1 h numa GTX 1080 Ti; retomável). Log: `runs/parte0.log`.

## 1. Seleção de hiperparâmetros (só no val, orçamento 5)

Sweep em `sweep_val.csv` / `sweep_val_resumo.csv`. ResNet-18: lr da cabeça {3e-4, 1e-3, 3e-3} (backbone ×0,1)
× resolução {224, 320, 384}, 2 seeds. CNN pequena: lr {1e-3, 3e-3}, 224, 1 seed.

| Modelo | Config escolhida | F1 macro val (média das seeds) |
|---|---|---|
| ResNet-18 | lr 1e-3, **384 px** | 0,442 ± 0,023 |
| CNN pequena | lr 1e-3, 224 px | 0,341 |

Ressalva honesta: com 202 imagens no val, as diferenças entre configs (0,39–0,44) estão perto do ruído entre
seeds (desvios de 0,01–0,07). O padrão mais consistente é que **384 px ganhou nos dois melhores lr**, o que
é coerente com animais ocupando poucos pixels. Fixo para o resto do PA: ResNet-18, 384 px, lr 1e-3.

## 2. Resultados no test_public (orçamento 5)

| Modelo | Parâmetros | Seeds | Melhor época (val) | **F1 macro** | Acurácia |
|---|---|---|---|---|---|
| (a) CNN pequena do zero | 1.879.797 | 0 | 193 / 300 | **0,307** | 0,328 |
| (b) ResNet-18 ImageNet + fine-tuning | 11.187.285 | 0, 1, 2 | 40, 38, 26 / 60 | **0,454 ± 0,027** | 0,501 ± 0,019 |
| Teto (monitor, todos os rótulos, 224 px) | — | — | — | 0,686 | — |

Por seed, a ResNet-18 teve 0,424 / 0,477 / 0,461. O pré-treino em ImageNet vale **+0,15 de F1** sobre treinar do zero.

Piso nos outros orçamentos (Parte 2, seed 0): 1 rótulo/classe → **0,214**; 20 rótulos/classe → **0,602**.

Figuras: `curva_resnet18_b5.png`, `curva_smallcnn_b5.png`, `confusion_resnet18_b5.png`, `confusion_smallcnn_b5.png`.

## 3. Quando a ResNet-18 começa a decorar (seed 0, `memorizacao_resnet18_b5.json`)

Curvas medidas no treino **limpo** (sem aumentação, modo eval, CE sem label smoothing), comparáveis ao val:

- A acurácia de treino passa de 95% na **época 11** e chega a **100% na época 22**. A loss de treino limpo cai a ~0,1.
- A loss de val cai rápido até a época ~15 (≈ 2,1) e **para de cair**: dali em diante fica estável em 2,05–2,2
  (mínimo na época 45), um gap de ~20× para o treino.
- O F1 de val cresce até ~0,35 na época 15 e depois só oscila entre 0,33 e 0,43.
- **Leitura:** com 105 imagens, o modelo decora o conjunto de treino em **~10–20 épocas**. A partir daí, o treino
  não gera mais informação útil; o pouco que o val ainda melhora (até a época 40) vem do decaimento do LR e
  do weight decay, que suavizam o modelo, e não de aprender mais sobre os animais.
- A CNN do zero demora muito mais para decorar (95% na época 161, 100% na 251) e generaliza pior.

## 4. Por classe (ResNet-18, média de 3 seeds — `per_class_resnet18_b5_media3seeds.csv`)

- **Fáceis:** eastern_chipmunk (0,92), virginia_opossum (0,84), white_tailed_deer (0,82), dog (0,81), chicken (0,76).
- **Difíceis:** bird (0,07), eastern_cottontail (0,09), bobcat (0,13), american_crow (0,27), grey_fox (0,28), domestic_cat (0,29).
- **Confusões sistemáticas** (matriz da seed 0): bird → chicken/esquilo-cinza; grey_fox → coyote;
  american_crow → woodchuck; raccoon ↔ skunk ↔ red_fox; bear espalhado (cavalo, woodchuck, raccoon).
- **"Classes-lixeira":** woodchuck (recall 0,70 / precisão 0,18), eastern_gray_squirrel (0,94 / 0,51) e horse
  (recall 1,0, precisão 0,23 com só 4 exemplos) absorvem animais escuros, pequenos ou ambíguos.
  É aqui que o professor da Trilha C precisa mostrar ganho.
