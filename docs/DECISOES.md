# Decisões do projeto (PA3)

Registro das escolhas que tomamos e por quê. Cada decisão tem data, o que foi decidido e a justificativa.
Quando uma decisão mudar, atualizar aqui (sem apagar a antiga, marcar como **substituída**).

Resumo rápido:

| Tema | Decisão |
|---|---|
| Trilha da Parte 1 | **C — Professor e aluno** |
| Backbone (Parte 0 b em diante) | **ResNet-18, pesos ImageNet (torchvision IMAGENET1K_V1)** |
| Resolução de entrada | **384 px** (escolhida no val) |
| Métrica de seleção | **F1 macro no val** (desempate: menor loss de val) |
| Compute | **GTX 1080 Ti local** (principal) + **Kaggle** (reserva), mesmo código |
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

### 6b.3 Compute: professor local (CPU), aluno no Kaggle — 2026-11-10 (ajusta a 2.1)
**Decisão:** esta máquina não tem a 1080 Ti (só GPU integrada Intel); o treino do aluno e da Parte 2 roda no Kaggle,
e o código novo chega lá por um branch do repo privado (o notebook clona o branch).

**Por quê:** o professor é só inferência (~7,3k imagens, viável em CPU); o aluno em 384 px sobre 6,5k imagens e os
5 treinos da Parte 2 pedem GPU.

---

## 7. Em aberto (decidir na Trilha C)

- **Qual professor:** BioCLIP vs. CLIP/SigLIP, decidido pelo F1 zero-shot medido (o test_public só serve para medir o professor).
- ~~Prompts do zero-shot~~ → ensemble de 4 templates (ver 6b.2).
- **Tratamento do ruído:** filtro por confiança, rótulos suaves com temperatura, ou professor ajustado com os 105 rótulos.
- **Cache das imagens pré-redimensionadas** para treinar o aluno nas 6.505 imagens (hoje a GPU fica em ~30% por gargalo de leitura de JPEG).
- **Sinal de out-of-distribution** para a pergunta da capivara (ex.: confiança máxima baixa, entropia, distância no espaço de features do professor).
