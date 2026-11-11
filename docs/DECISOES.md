# Decisões do projeto (PA3)

Registro das escolhas que tomamos e por quê. Cada decisão tem data, o que foi decidido e a justificativa.
Quando uma decisão mudar, atualizar aqui (sem apagar a antiga, marcar como **substituída**).

Resumo rápido:

| Tema | Decisão |
|---|---|
| Trilha da Parte 1 | **C — Professor e aluno** (professor: SigLIP; **CLIP-L pendente**, ver 6b.7) |
| Backbone (Parte 0 b em diante) | **ResNet-18, pesos ImageNet (torchvision IMAGENET1K_V1)** |
| Resolução de entrada | **384 px** (escolhida no val) |
| Métrica de seleção | **F1 macro no val** (desempate: menor loss de val) |
| Compute | Planejado: 1080 Ti local + Kaggle. **Na prática (Parte 1): tudo em CPU nesta máquina** (sem GPU NVIDIA), ver 6b.3/6b.4 |
| Professor (Trilha C) | **SigLIP ViT-B/16**, melhorado com os 105 rótulos (probe + zero-shot); **CLIP-L pendente** (6b.7) |
| Tratamento do ruído | Professor melhorado + rótulos verdadeiros pesados; filtro τ e temperatura testados e **descartados** (τ = 0, T = 1), ver 6b.8 e 6b.11 |
| Aluno (Parte 1) | ResNet-18, **224 px**, 15 épocas, 1 seed (o piso usa 384 px), ver 6b.10 |
| Ambiente | **uv**, Python 3.11, PyTorch do índice **cu126** |

---

## 1. Escopo e estratégia

### 1.1 Trilha C (professor → aluno) — 2026-11-07
**Decisão:** na Parte 1, um foundation model aberto (BioCLIP/CLIP) rotula o train e uma ResNet-18 (aluno) aprende
com esses pseudo-rótulos mais os 105 rótulos verdadeiros.

**Por quê:**
- **Melhor custo-benefício no prazo** (entrega 16/11): o professor só faz inferência, uma vez, sobre ~7,5k imagens.
  A Trilha A exige pré-treino contrastivo longo (muita GPU) e a B (FixMatch) é instável com 5 rótulos/classe e classes desbalanceadas.
- **Maior chance de ganho real:** modelos como o BioCLIP foram treinados em dados de espécies; o zero-shot deles
  provavelmente já fica perto do teto (0,686), e o aluno destila esse conhecimento num modelo de ≤12M parâmetros que roda em CPU.
- **Análise rica para a apresentação:** a matriz de confusão do professor é a matriz de ruído dos pseudo-rótulos,
  e a pergunta "em quais classes o aluno supera o professor?" rende discussão concreta.
- **Restrições aceitas:** o professor só pode ver train e test_public (este só para medir o professor), nunca o val
  durante o treino, e não entra no modelo entregue.

### 1.2 Backbone: ResNet-18 — 2026-11-07
**Decisão:** ResNet-18 com pesos de ImageNet, fixa da Parte 0 (b) até o fim.

**Por quê:**
- **Mesmo modelo do teto do monitor** (ResNet-18 ImageNet → 0,686), o que torna a comparação piso/método/teto direta.
- 11,19M parâmetros: **cabe no limite de 12M** e roda rápido em CPU (811 imagens em ~17 s).
- **Grad-CAM simples** (última camada conv `layer4`), útil para a Parte 3.
- Alternativas: EfficientNet-B0 e MobileNetV3 têm menos parâmetros, mas são mais sensíveis a hiperparâmetros
  com poucos dados e se afastam da referência do teto.

---

## 2. Infraestrutura

### 2.1 Compute: 1080 Ti local + Kaggle — 2026-11-07
**Decisão:** o mesmo código roda nos dois lugares. O caminho dos dados é resolvido automaticamente
(`$PA3_DATA` → `/kaggle/input/**` → `data/pa3_dados`).

**Por quê:**
- **1080 Ti (11 GB) como principal:** sem cota nem limite de sessão, dados no disco; em FP32 é parecida com a P100 do Kaggle.
  Na prática, ~1 s/época (224 px) e ~2 s/época (384 px) no orçamento 5, ~10× mais rápido que a CPU.
- **Kaggle como reserva** (30 h/semana de GPU) para rodar seeds/orçamentos em paralelo se faltar tempo.

### 2.2 Ambiente: uv, Python 3.11, PyTorch cu126 — 2026-11-07
**Decisão:** `pyproject.toml` + `uv.lock`; Python fixado em 3.11; torch/torchvision vindos do índice `cu126`.

**Por quê:**
- **Reprodutibilidade:** o lock fixa versões exatas, e o enunciado exige que tudo seja reproduzível a partir do repo.
- **Python 3.11** é o mesmo do Kaggle (a máquina local tem 3.14, que pode não ter wheels de tudo).
- **cu126:** os builds do PyTorch com CUDA ≥ 12.8 removeram o suporte a GPUs Pascal (a 1080 Ti é sm_61).
  O cu126 ainda inclui sm_60, que é binariamente compatível. Verificado: `torch.cuda.is_available()` = True, 8,3 TFLOPS FP32.

### 2.3 AMP automático — 2026-11-07
**Decisão:** `--amp auto` liga precisão mista só em GPUs com compute capability ≥ 7 (ex.: T4 no Kaggle).

**Por quê:** a 1080 Ti não tem tensor cores e o FP16 dela é muito lento; na Pascal, FP32 é o mais rápido.

### 2.4 Organização do repositório — 2026-11-07
**Decisão:** pacote `src/pa3/` (paths, data, models, engine, metrics, utils) + `scripts/` por etapa;
`runs/` (checkpoints e logs, fora do git) separado de `results/` (tabelas e figuras, versionadas).

**Por quê:**
- O enunciado diz que **toda tabela, curva e imagem da apresentação tem que ser reproduzível a partir do repo**:
  `results/` é esse lastro, e cada figura sai de um script.
- Código reutilizável entre Parte 0, Trilha C, Parte 2 e Parte 3 (mesmo `data.py`, mesmo `train.py`).
- Cada run salva `config.json` (com hash do git), `history.csv` e `best.pt`, então dá para rastrear de onde veio cada número.

### 2.5 Orquestrador retomável com ETA — 2026-11-08
**Decisão:** `scripts/run_parte0.py` roda sweep → finais → avaliação → figuras, pula runs já concluídas e mostra
`[k/N]`, tempo decorrido e ETA global; cada época mostra seu tempo e ETA.

**Por quê:** os treinos levam ~1 h; precisamos de feedback enquanto rodam e de poder retomar se algo cair.

---

## 3. Regras contra vazamento e sobreajuste ao teste

### 3.1 test_public só para reportar — 2026-11-07
**Decisão:** `train.py` lê apenas o orçamento + val; o `test_public` só é lido em `evaluate.py`, que não escreve
nada que o treino consuma. Sweeps nunca são avaliados no test_public e são excluídos do `curva_orcamento.csv`.

**Por quê:** é regra explícita do enunciado. Escolher época, limiar ou arquitetura no test_public é sobreajuste,
e o monitor confere com o `test_private`.

### 3.2 Seleção de época e hiperparâmetros pelo F1 macro no val — 2026-11-07
**Decisão:** salvar o checkpoint da época com maior F1 macro no val (desempate: menor loss de val).

**Por quê:** F1 macro é a métrica principal do PA (pesa cavalo igual a urso); o val é o único conjunto rotulado
permitido para seleção. O desempate pela loss evita escolher uma época instável quando o F1 empata
(o val tem só 202 imagens).

### 3.3 Checagem automática dos dados — 2026-11-07
**Decisão:** `scripts/check_data.py` confere contagens, aninhamento 1 ⊂ 5 ⊂ 20, balanceamento dos orçamentos e
que nenhuma imagem de val/test aparece no train.

**Por quê:** garantir que estamos usando exatamente o pacote do enunciado e que não há vazamento acidental.

---

## 4. Dados e aumentação

### 4.1 Avaliação sem center crop — 2026-11-07
**Decisão:** na avaliação, redimensionar o quadro inteiro para `img_size × img_size` (sem cortar).

**Por quê:** em 25% das imagens o animal ocupa < 2% do quadro e muitas vezes está na borda; um center crop
pode cortá-lo fora. As proporções variam pouco (o corte da faixa de texto muda a altura), então a distorção é pequena.

### 4.2 RandomResizedCrop com escala mínima 0,5 — 2026-11-07
**Decisão:** `RandomResizedCrop(scale=(0.5, 1.0))` em vez do padrão `(0.08, 1.0)`.

**Por quê:** com animais pequenos, um crop agressivo gera muitos exemplos de treino **sem o animal**, rotulados
com a classe, o que ensina o modelo a associar fundo à espécie (o atalho que a Parte 3 investiga).

### 4.3 RandomGrayscale (p = 0,2) e ColorJitter — 2026-11-07
**Decisão:** converter 20% das imagens de treino para tons de cinza, mais jitter de brilho/contraste/saturação.

**Por quê:** muitas fotos são noturnas em infravermelho (sem cor); isso força o modelo a não depender só da cor.

### 4.4 Resolução de 384 px — 2026-11-08
**Decisão:** usar 384 px no resto do PA (escolhido no sweep, ver 5.4).

**Por quê:** ganhou no val nos dois melhores lr (0,442 vs 0,404 em 224 px), coerente com animais ocupando poucos pixels:
em 224 px um animal pequeno vira poucos pixels. O custo é ~2× tempo por época, e o predict em CPU continua muito
abaixo do limite de 10 min.
**Ressalva:** o teto do monitor foi treinado em 224 px; a comparação com ele não é exatamente igual.

---

## 5. Treino e avaliação (Parte 0)

### 5.1 Receita de fine-tuning da ResNet-18 — 2026-11-07
**Decisão:** AdamW, LR diferencial (cabeça 1e-3, backbone ×0,1), weight decay 0,05, cosseno com warmup de 5%,
label smoothing 0,1, batch 32, 60 épocas.

**Por quê:**
- **LR diferencial:** a cabeça é nova (aleatória) e precisa aprender rápido; o backbone já tem boas features
  do ImageNet e um LR alto as destruiria com só 105 imagens.
- **Weight decay e label smoothing:** regularização contra decorar 105 imagens (que acontece em ~10–20 épocas).
- **Cosseno com warmup:** o warmup evita que os primeiros gradientes da cabeça aleatória bagunçem o backbone.
- **60 épocas** bastam: o treino limpo chega a 100% na época 22 e o val estabiliza depois.

### 5.2 CNN pequena do zero com 300 épocas — 2026-11-08 (substitui 150 épocas de 2026-11-07)
**Decisão:** 5 blocos conv-BN-ReLU duplos + GAP + linear (1,88M parâmetros), 300 épocas, lr 1e-3.

**Por quê:** com 150 épocas a melhor época era a 145 e a acurácia de treino só 75%, ou seja, não tinha convergido
e o piso (a) ficava subestimado. Com 300 épocas: F1 0,262 → 0,307.

### 5.3 Mínimo de 240 passos de otimização — 2026-11-08
**Decisão:** se o orçamento for pequeno, aumentar as épocas até completar 240 passos (o mesmo do orçamento 5 com 60 épocas).

**Por quê:** com 1 rótulo/classe (21 imagens = 1 batch/época), 60 épocas seriam só 60 passos, e o piso do
orçamento 1 sairia artificialmente ruim por falta de treino, não por falta de rótulos.

### 5.4 Sweep de hiperparâmetros do piso no val — 2026-11-08
**Decisão:** ResNet-18: lr da cabeça {3e-4, 1e-3, 3e-3} × resolução {224, 320, 384} × 2 seeds; CNN: lr {1e-3, 3e-3}.
Escolha pela média do F1 macro no val. Venceu: ResNet-18 lr 1e-3 / 384 px; CNN lr 1e-3.

**Por quê:** um piso com hiperparâmetros chutados é fraco e **infla artificialmente o ganho da Trilha C**.
Um piso bem ajustado (só no val) torna a comparação honesta.
**Ressalva:** com 202 imagens no val, as diferenças entre configs (0,39–0,44) estão perto do ruído entre seeds.

### 5.5 Curvas com treino "limpo" — 2026-11-08
**Decisão:** a cada época, medir loss e acurácia do conjunto de treino em modo eval, sem aumentação e sem label smoothing.

**Por quê:** a loss de treino "normal" vem com aumentação e label smoothing (piso ~0,65), e a de val não, então
não são comparáveis e "quando começa a decorar" ficava mal medido. Com o treino limpo, a resposta fica objetiva:
95% na época 11, 100% na 22, loss de val parada a partir da ~15.

### 5.6 Seeds e determinismo — 2026-11-08
**Decisão:** seeds 0, 1, 2 no orçamento 5 (média ± desvio) e seed 0 nos orçamentos 1 e 20, como pede a Parte 2;
`cudnn.deterministic = True`, `benchmark = False`.

**Por quê:** com 105 imagens a variância entre seeds é grande (F1 0,424–0,477); uma seed só não basta para afirmar ganho.
O determinismo deixa as runs reproduzíveis na mesma máquina.

---

## 6. Entregáveis

### 6.1 predict.py autocontido — 2026-11-07
**Decisão:** o checkpoint guarda nome do modelo, resolução, lista de classes e pesos; o `predict.py` monta o modelo
**sem baixar pesos** (`pretrained=False`) e roda em CPU.

**Por quê:** o monitor roda sem internet. Testado com `CUDA_VISIBLE_DEVICES=""`: 811 imagens em 17 s, saída idêntica
ao `evaluate.py`, colunas `image_id,label` com os nomes de `classes.csv`.

### 6.2 AI_LOG.md contínuo — 2026-11-07
**Decisão:** registrar episódios de uso de IA ao longo do trabalho, não só no fim.

**Por quê:** é entregável obrigatório e fica mais fiel escrito na hora (inclusive erros, como o `.gitignore` sobrescrito).

---

## 6b. Trilha C — implementação do professor

### 6b.1 Professor via open_clip, só inferência — 2026-11-10
**Decisão:** `scripts/teacher.py` extrai embeddings e logits zero-shot (BioCLIP, SigLIP ou CLIP-L, pesos abertos do HF)
para o `test_public` (só para medir o professor) e para o train sem rótulo. **O val nunca é lido.** Embeddings ficam em
`runs/teacher/` (fora do git); logits, métricas e matriz de confusão em `results/teacher/<nome>/` (versionados).

**Por quê:** o enunciado permite o foundation model só como professor, sem ver o val e fora do modelo entregue.
Salvar os logits desacopla o professor do aluno: o aluno treina sem importar o CLIP.

### 6b.2 Prompts: ensemble de 4 templates, quadro inteiro — 2026-11-10 (resolve item da seção 7)
**Decisão:** média dos embeddings de texto de 4 templates (foto simples, câmera trap, infravermelho noturno, animal pequeno);
imagem redimensionada inteira (sem center crop do preprocess do open_clip), pelo mesmo motivo da 4.1.
**Ressalva:** os templates foram escolhidos a priori, sem ajuste em nenhum conjunto rotulado.

### 6b.3 Compute: professor local (CPU), aluno no Kaggle — 2026-11-10 (ajusta a 2.1; **substituída em parte por 6b.4**: o aluno também rodou em CPU local)
**Decisão:** esta máquina não tem a 1080 Ti (só GPU integrada Intel); o treino do aluno e da Parte 2 roda no Kaggle,
e o código novo chega lá por um branch do repo privado (o notebook clona o branch).

**Por quê:** o professor é só inferência (~7,3k imagens, viável em CPU); o aluno em 384 px sobre 6,5k imagens e os
5 treinos da Parte 2 pedem GPU.

### 6b.4 Professor e aluno rodam localmente em CPU, dados baixados — 2026-11-11 (ajusta a 6b.3)
**Decisão:** o professor foi rodado nesta máquina (CPU, 18 núcleos) com `./scripts/get_data.sh` +
`scripts/teacher.py --splits test_public`, em vez de no Kaggle. Cada professor ViT-B leva ~2 min no `test_public`.

**Por quê:** o fluxo do Kaggle (Dataset privado, Secret do GitHub, branch num repo de terceiros) era um custo alto
para um passo que cabe em CPU. O aluno da Parte 1 também acabou rodando em CPU local (~45 min por treino, ver 6b.10 e 6b.11). O Kaggle/GPU continua sendo a opção para o CLIP-L, para 384 px e para a Parte 2.

### 6b.5 Medição dos professores zero-shot no test_public — 2026-11-11
Só para MEDIR o professor (permitido no enunciado); o val não foi lido. Mesmos 4 prompts e quadro inteiro (6b.2).

| Professor | Parâmetros | F1 macro | Acurácia |
|---|---|---|---|
| Piso (ResNet-18, 5 rótulos/classe) — referência | 11,2M | 0,454 | 0,501 |
| BioCLIP (`imageomics/bioclip`) | 149,6M | 0,304 | 0,356 |
| SigLIP ViT-B/16 (`timm/ViT-B-16-SigLIP`) | 203,2M | **0,381** | **0,435** |
| CLIP-L ViT-L/14 (`laion/CLIP-ViT-L-14-laion2B-s32B-b82K`) | — | **0,451** | **0,492** |

**Leitura:** nenhum zero-shot supera o piso; o CLIP-L o empata (0,451 vs 0,454). O BioCLIP, apesar de treinado em
espécies, foi o pior (erra bird, horse, grey_fox, chipmunk; recall alto e precisão baixa em red_fox/eastern_cottontail).
O SigLIP fica no meio e tem erros diferentes (cottontail com precisão 1,0 mas recall 0,12; chipmunk e bird fracos).
O CLIP-L é o melhor professor medido, mas é o mais caro em CPU (estimativa: ~40 min só para o test_public, medido disputando CPU com outras tarefas, contra ~2 min dos ViT-B),
o que pesa para rodar nas 6.505 imagens do train.
**Consequência:** pseudo-rótulos zero-shot puros provavelmente não vencem o piso; o foco passa a ser o
professor melhorado com os 105 rótulos (item da seção 7). Escolher o professor final continua em aberto.
**Ressalva:** comparar professores no test_public é permitido pelo enunciado só para medir o professor; ainda assim,
a escolha final não deve depender só dele e será confirmada sem usar o val no treino do professor.

### 6b.6 Dependências do SigLIP — 2026-11-11
**Decisão:** extra `teacher` do `pyproject.toml` ganhou `transformers` e `sentencepiece`.

**Por quê:** o tokenizer do SigLIP no open_clip usa `transformers`; sem ele o script falhava
com `ModuleNotFoundError`. O notebook do Kaggle usa `pip install open_clip_torch` e precisaria disso também.

### 6b.7 Professor de trabalho: SigLIP (CLIP-L fica para depois) — 2026-11-11
**Decisão:** o pipeline completo da Trilha C é construído e rodado com o **SigLIP ViT-B/16** como professor.
O train inteiro (6.505 imagens) levou ~17 min em CPU com o SigLIP.

**Por quê:** o CLIP-L tem o melhor zero-shot (0,451 contra 0,381), mas leva ~40 min só para 811 imagens em CPU,
ou seja, várias horas para o train. Validamos o pipeline com o professor barato antes de gastar horas de CPU.

> **⚠ AVISO — rodar com o CLIP-L depois.** Os números da Trilha C até aqui são do **SigLIP**. O CLIP-L deve dar um
> professor melhor (zero-shot 0,451 vs 0,381), então o aluno e a curva da Parte 2 tendem a melhorar com ele.
> Antes da entrega, repetir com o CLIP-L (de preferência numa GPU, Kaggle ou a 1080 Ti):
> ```
> python scripts/teacher.py --name clip-l --splits test_public train_all
> python scripts/teacher_improve.py --name clip-l --budget 5      # e 1, 20
> python scripts/train_student.py --teacher clip-l --budget 5 ...
> ```
> e comparar com o SigLIP. Se o CLIP-L não for rodado, a apresentação deve dizer que o professor usado foi o SigLIP.

### 6b.8 Tratamento do ruído dos pseudo-rótulos — 2026-11-11
O ruído do professor é medido pela matriz de confusão dele (`results/teacher/<nome>/improved_b5/confusion_test_public.*`).
Usamos **quatro** tratamentos combinados, todos justificados abaixo:

1. **Professor melhorado com os 105 rótulos** (`scripts/teacher_improve.py`).
   `log p = (1-α)·log_softmax(zero-shot) + α·log_softmax(probe)`, com o probe sendo uma regressão logística
   (`class_weight=balanced`) sobre os embeddings do professor, treinada nos 105 rótulos do orçamento.
   (α, C) escolhidos por **validação cruzada de 5 dobras dentro dos próprios 105 rótulos**: α = 0,75, C = 100
   (F1 macro de CV 0,535). **Por quê:** o enunciado permite explicitamente "professor melhorado com os 105 rótulos";
   o zero-shot puro está abaixo do piso. **O val não é usado**: o professor não pode vê-lo durante o treino, e o
   test_public só mede.
2. **Filtro por confiança (τ)** *(o sweep do val acabou escolhendo τ = 0, ou seja, sem filtro; ver 6b.11)*: só entram no aluno as imagens sem rótulo com max-prob do professor ≥ τ.
   **Por quê:** a confiança do professor é informativa (no test_public, τ = 0,5 mantém 45% das imagens com 87% de
   acerto; τ = 0,9 mantém 10% com 96%). Mas filtrar por confiança descarta justamente as classes difíceis, que ficam sub-representadas.
3. **Rótulos suaves com temperatura T:** alvo = `softmax(log p / T)`; T > 1 suaviza. **Por quê:** a cross-entropy contra
   a distribuição inteira do professor passa a informação de "classes parecidas" e não força o aluno a ter certeza
   num rótulo possivelmente errado.
4. **Rótulos verdadeiros pesam mais:** as 105 imagens entram com rótulo duro (label smoothing 0,1), peso 1 e
   repetidas 5× por época; as pseudo-amostras têm peso `--pseudo-weight` (padrão 1).

**Escolha de τ e T — o que o enunciado exige e o que foi escolha nossa.** O enunciado **não** manda escolher τ e T na
Trilha C (só a Trilha B diz "a escolha de τ é feita no val"); ele só pede "tratar o ruído". A regra geral é que
seleção de modelo/limiar é no val e nunca no test_public. Decidimos **ajustar τ e T no val** porque deixa a
escolha defendível na apresentação. Isso é escolha nossa, não requisito.
- **Valor inicial a priori:** τ = 0,5 e T = 1 (sem suavização) foram padrões que o Claude pôs em `train_student.py`
  sem critério. **Transparência:** o diagnóstico precisão×cobertura por τ no test_public já tinha sido impresso
  antes dessa escolha (0,5 mantinha 45% das imagens com 87% de acerto), então o 0,5 inicial não é independente do
  test_public. Por isso a escolha final vem do sweep no val, e a tabela do test_public fica só como diagnóstico.
- **Sweep** (`scripts/run_sweep_aluno.py`, seed 0, orçamento 5, 224 px, 15 épocas, retomável):
  (τ, T) ∈ {(0,5; 1), (0; 1), (0,7; 1), (0,5; 2)}, escolhendo pelo F1 macro de val do aluno. Saídas em
  `results/trilha_c/siglip/`. **Ressalva:** o val tem 202 imagens e é uma seed só, então diferenças pequenas
  (< ~0,03 de F1) estão dentro do ruído; nesse caso preferir a configuração mais simples.
- Também fixos sem busca (chutes): `labeled-repeat` = 5, `pseudo-weight` = 1, 15 épocas.
**Não implementado:** correção de rótulo à la SELFIE e re-treino iterativo; ficaram fora por tempo.

### 6b.9 Resultado do professor melhorado (SigLIP, orçamento 5) — 2026-11-11
Medido no test_public (só para medir o professor):

| Professor | F1 macro | Acurácia |
|---|---|---|
| Piso (ResNet-18, 5 rótulos) | 0,454 | 0,501 |
| SigLIP zero-shot | 0,381 | 0,435 |
| **SigLIP melhorado (zero-shot + probe nos 105 rótulos)** | **0,548** | **0,599** |
| Teto | 0,686 | — |

- O professor melhorado **supera o piso em +0,094 de F1** e o zero-shot em +0,167: só o probe leve nos 105 rótulos já fecha
  boa parte da distância piso→teto, sem imagens a mais.
- **Noite (cinza/IR) vs. dia (cor)**, classificadas por saturação das cores: noite F1 0,339 / acc 0,514 (391 imagens) e dia F1 0,413 / acc 0,679 (420).
  Fotos noturnas são ~17 p.p. de acurácia piores, como previsto no enunciado.
- **Melhores classes:** eastern_chipmunk (0,97), white_tailed_deer (0,83), virginia_opossum (0,82), dog (0,77), american_black_bear (0,76).
- **Piores:** eastern_cottontail (0,17), bobcat (0,20), grey_fox (0,23), bird (0,32), domestic_cat (0,32). Em todas elas
  o recall é baixo (0,12–0,25): o professor quase nunca as prevê, então o aluno recebe poucos pseudo-rótulos delas.
  `horse` (4 exemplos): recall 1,0, precisão 0,29.
- **Aviso sobre a "noite":** a classificação noite/dia é uma heurística de cor, não um metadado do dataset.

### 6b.10 Aluno: receita e custo em CPU — 2026-11-11
**Decisão:** `scripts/train_student.py` (separado do `train.py` para não mexer na Parte 0 reproduzível).
ResNet-18 (ImageNet), mesma receita do piso (AdamW, LR diferencial, wd 0,05, cosseno com warmup), 15 épocas,
**224 px** para caber em CPU (o piso usa 384). Seleção de época pelo F1 macro no val.
**Por quê 224 px:** a CPU faz ~113 passos por época com τ = 0,5; 384 px custaria ~3× mais. **Ressalva:** o piso
usa 384 px, então a comparação aluno × piso mistura método e resolução; o ideal é repetir o aluno em 384 px
numa GPU (Kaggle).

### 6b.11 Resultado do sweep de τ e T no val — 2026-11-11
Aluno (SigLIP, orçamento 5, seed 0, 224 px, 15 épocas); F1 macro no **val** na melhor época (`results/trilha_c/siglip/sweep_val.csv`):

| τ | T | Pseudo-rótulos usados | Melhor época | Val F1 | Val acc |
|---|---|---|---|---|---|
| **0 (sem filtro)** | **1** | 6.400 / 6.400 | 2 | **0,516** | 0,520 |
| 0,5 | 1 | 3.095 | 6 | 0,478 | 0,505 |
| 0,5 | 2 | 3.095 | 14 | 0,462 | 0,495 |
| 0,7 | 1 | 1.940 | 3 | 0,459 | 0,495 |

**Escolhida: τ = 0 (sem filtro), T = 1** (`config_escolhida.json`), pelo critério decidido antes (maior F1 no val).
**Leitura:** o filtro por confiança **não ajudou** (quanto mais rígido, pior no val), e a temperatura T = 2 também não.
Hipótese (não testada): o filtro descarta justamente as classes em que o professor é pouco confiante (as difíceis), que
ficam sub-representadas no aluno. **Ressalvas honestas:** val de 202 imagens e uma seed só; as diferenças (0,46–0,52)
são do tamanho do ruído, então dá para afirmar "o filtro não melhorou", mas não "o sem filtro é melhor por X".
A melhor época do vencedor foi a 2, mas o F1 de val dele ficou estável em ~0,49–0,52 até a época 15 (não é um pico isolado).

### 6b.12 Resultado do aluno no test_public (SigLIP, orçamento 5, seed 0) — 2026-11-11
Só reportar; nada foi escolhido aqui. Reproduzir: `scripts/evaluate.py --run runs/aluno_siglip_b5_s0_tau0_T1` e
`scripts/compare_student_teacher.py` → `results/trilha_c/siglip/aluno_vs_professor_b5.csv`.

| Modelo | F1 macro | Acurácia |
|---|---|---|
| Piso (ResNet-18, 5 rótulos, 384 px, média de 3 seeds) | 0,454 ± 0,027 | 0,501 |
| Professor SigLIP zero-shot | 0,381 | 0,435 |
| Professor melhorado (usado para rotular) | 0,548 | 0,599 |
| **Aluno ResNet-18 (224 px, 1 seed)** | **0,533** | **0,599** |
| Teto | 0,686 | — |

- O aluno **supera o piso em +0,079 de F1** (~3 desvios do piso) e **fica 0,015 abaixo do professor**: ele recupera quase
  tudo do professor num modelo de 11,2M de parâmetros que roda em CPU, mas **não o supera no geral**.
- **Onde o aluno supera o professor melhorado** (F1 por classe): grey_fox +0,27, eastern_gray_squirrel +0,15,
  red_fox +0,10, northern_raccoon +0,09, dog +0,08, chicken +0,07.
- **Onde perde:** striped_skunk −0,26, bird −0,20, eastern_cottontail −0,17 (F1 0,00), woodchuck −0,09, horse −0,08.
- **Por quê (hipóteses, não verificadas):** (1) o aluno vê a imagem em si e não só um embedding fixo, então corrige
  classes em que o professor tem recall baixo mas as imagens são separáveis visualmente (grey_fox, red_fox, raccoon);
  (2) o aluno regulariza o ruído ao ver 6.505 imagens: erros do professor que não são sistemáticos se cancelam;
  (3) nas classes em que o professor erra de forma sistemática (cottontail e bird, recall 0,12–0,23), o aluno copia o erro
  e até o amplifica; (4) classes pequenas (horse, 4 exemplos) são puro ruído. Para verificar (1) e (2) faltaria uma
  análise da confusão do aluno e do acerto do professor por imagem.
- **Ressalvas:** (a) o aluno é 224 px e o piso 384 px; (b) uma seed só no aluno (a Parte 2 pede 3 no orçamento 5);
  (c) diferenças por classe com 22–82 imagens são ruidosas, e as de ±0,05 não são conclusivas;
  (d) o CLIP-L (professor melhor) ainda não foi rodado, ver o aviso em 6b.7.

### 6b.13 Leitura da matriz de ruído do professor (SigLIP melhorado, test_public) — 2026-11-11
Matriz = `results/teacher/siglip/improved_b5/confusion_test_public.{png,csv}`; maiores confusões em `top_confusoes.csv`
(reproduzir: `scripts/analisar_ruido.py`). Os números abaixo são contagens de imagens do test_public (811).

**Erros sistemáticos (verdadeiro → previsto):**
- **Cães e galinhas:** chicken → dog (17 de 60) e bird → chicken (14 de 22; só 5 `bird` certos). (Hipótese, não verificada: aves pequenas ou escuras
  parecem galinha para o professor.)
- **Canídeos e raposas:** grey_fox → red_fox (13/39) e → coyote (10/39); só 8 de 39 grey_fox certos. red_fox → coyote (8/43).
  É a mesma família visual, e o professor não separa.
- **Pequenos mamíferos:** eastern_cottontail → coyote (17 de 26!) e → striped_skunk (6); só 3 de 26 certos.
  eastern_fox_squirrel → woodchuck (12/32), striped_skunk → northern_raccoon (12/31).
- **Felinos:** domestic_cat só acerta 12 de 48 e se espalha (gray_squirrel 9, raccoon 8, coyote 7). bobcat acerta 4 de 34
  (→ wild_turkey 7, coyote 5, fox_squirrel 5): os erros dos felinos se espalham por várias classes.
- **Urso → cavalo:** american_black_bear → horse (10 de 82). Foi isso que fez `horse` ter precisão 0,29: são 4 cavalos
  verdadeiros e muitos ursos que caem na classe.
- **Classes-lixeira (colunas mais previstas):** dog (81 previstos), coyote (74), chicken (61), american_crow (58),
  eastern_gray_squirrel (55). Precisão mais baixa: grey_fox 0,27, coyote 0,28, horse 0,29, eastern_cottontail 0,33.

**O exemplo `bird` × `american_crow` do enunciado:** na nossa matriz **não aparece** uma confusão relevante entre
`bird` e `american_crow`. `american_crow` vai bem (F1 0,69; 42 de 63) e erra para chicken/domestic_cat/dog (5, 5, 4);
`bird` não vai para `american_crow`, vai para `chicken`. O professor melhorado mudou isso: no zero-shot puro o
`american_crow` tinha precisão 0,28 (muitas imagens caíam nele), e o probe nos 105 rótulos corrigiu.

**Noite:** acurácia 0,514 nas fotos noturnas (cinza/IR) contra 0,679 nas diurnas (6b.9), ou seja, a noite custa ~17 p.p.
**Animais pequenos:** **não medimos**. O pacote não traz o tamanho do animal (nem máscara/caixa), então não dá para
separar por tamanho sem rotular à mão, o que é proibido. Fica como hipótese: as classes difíceis (cottontail, bobcat,
grey_fox, bird) são em geral pequenas ou noturnas, mas isso não foi verificado.

**O que isso implica para o aluno:** o ruído dos pseudo-rótulos é **estruturado**, não aleatório: aves → galinha, grey_fox →
red_fox/coyote, cottontail → coyote. Um aluno que copia o professor herda esses erros (o aluno foi pior em `bird` −0,20
e `eastern_cottontail` −0,17), e filtrar por confiança não resolveu (6b.11). Hipótese, não verificada: parte desses erros sistemáticos tem confiança
  alta e passa pelo filtro, enquanto o filtro descarta acertos em classes difíceis.

### 6b.14 Pergunta para a apresentação: out-of-distribution (a capivara) — 2026-11-11
**Pergunta:** numa câmera no Uruguai, a primeira capivara seria classificada como uma das 21 classes. O que no método
daria um sinal de que a imagem não pertence a nenhuma delas? **Não implementado** (o enunciado manda só responder).

**Sinais que o nosso método já produz, do mais ao menos fácil:**
1. **Confiança máxima do aluno (softmax).** Baixa confiança → não pertence a nenhuma classe conhecida. Medimos o
   quanto a confiança distingue acerto de erro no test_public: confiança média de **0,55 nos acertos e 0,30 nos erros**,
   **AUROC 0,82** (`results/trilha_c/siglip/confianca_aluno.json`). **Cuidado:** isso mede erro dentro das 21 classes,
   é um **substituto**, não um teste de OOD de verdade; a capivara real não foi testada.
2. **Entropia / margem** entre as duas maiores probabilidades (mesma ideia, mais robusta que só o máximo).
3. **Desacordo entre professor e aluno.** Temos dois modelos de origens diferentes (SigLIP e ResNet-18): se discordam,
   a imagem é suspeita. Pela matriz, já discordam bastante nas classes difíceis, então seria um sinal ruidoso.
4. **Distância no espaço de embeddings do professor** (SigLIP): distância à classe mais próxima (ou vizinho mais próximo
   entre as imagens do train). Uma capivara ficaria longe de todos os agrupamentos. Funciona sem rótulos.
5. **Zero-shot com uma 22ª classe** ("rodent", "unknown animal") no professor, aproveitando que o CLIP/SigLIP entende
   texto. Só serve antes de entregar, porque o modelo entregue (a ResNet-18) não tem o professor.

**Limitações a dizer na apresentação:** (a) o aluno foi treinado com cross-entropy em 21 classes e tende a ser confiante
em qualquer coisa, então só a confiança é um sinal fraco; (b) a capivara é parecida com roedores/mamíferos do dataset
(`woodchuck`, `eastern_gray_squirrel`), exatamente as classes-lixeira, onde o modelo já erra com confiança; (c) o limiar
de rejeição teria de ser calibrado, e sem imagens OOD rotuladas só dá para calibrar pela taxa de erro.

---

## 7. Em aberto (decidir na Trilha C)

- **Qual professor:** a Parte 1 foi fechada com o **SigLIP melhorado** (6b.9, 6b.12). Falta rodar o **CLIP-L** (melhor zero-shot, 6b.5) e decidir o final, comparando alunos no val.
- ~~Prompts do zero-shot~~ → ensemble de 4 templates (ver 6b.2).
- ~~Tratamento do ruído~~ → ver 6b.8 (professor melhorado + filtro τ + rótulos suaves com T). τ e T: escolhidos no val, τ = 0 e T = 1 (6b.11).
- **Cache das imagens pré-redimensionadas** para treinar o aluno nas 6.505 imagens (hoje a GPU fica em ~30% por gargalo de leitura de JPEG).
- ~~Sinal de out-of-distribution~~ → respondido em 6b.14 (sem implementar). [original:] para a pergunta da capivara (ex.: confiança máxima baixa, entropia, distância no espaço de features do professor).
