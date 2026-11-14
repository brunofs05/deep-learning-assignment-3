# AI_LOG — uso de IA no PA3

Ferramenta: Claude Code (Claude Opus 5.5), usado no terminal sobre este repositório.

## Episódios

### 2026-11-07 — Leitura do enunciado e infraestrutura
- Pedimos para converter `PA3.pdf` em `docs/PA3.md` e resumir o escopo. A IA recomendou a
  Trilha C (professor → aluno) pelo custo-benefício no prazo; decidimos seguir com ela e com ResNet-18.
- Planejamos juntos a organização do repo (pacote `src/pa3`, scripts por etapa, `runs/` vs `results/`).
- Ponto técnico levantado pela IA: wheels do PyTorch com CUDA ≥ 12.8 não suportam a GTX 1080 Ti
  (Pascal, sm_61) → fixamos o índice cu126; AMP só liga em GPUs com tensor cores.
- A IA escreveu a primeira versão de `data.py`, `train.py`, `evaluate.py` e `predict.py` e um
  `check_data.py` que confere contagens, aninhamento dos orçamentos e ausência de vazamento.
- Decisões que revisamos e mantivemos: seleção de época só no val; eval sem center crop (animais na
  borda); RandomResizedCrop com escala mínima 0,5 por causa dos animais pequenos.

### 2026-11-08 — Revisão crítica da Parte 0 e re-execução na GTX 1080 Ti
Pedimos uma revisão crítica da Parte 0 contra o enunciado. Problemas apontados pela IA e corrigidos:
- **Curva de treino enganosa:** loss/acurácia de treino eram medidas com aumentação e label smoothing,
  e a de val sem, então "quando começa a decorar" ficava mal medido. Passamos a medir o treino *limpo*
  (modo eval, sem aumentação, CE pura) a cada época.
- **CNN do zero subtreinada:** melhor época era a 145/150 com 75% de acurácia de treino → 300 épocas.
- **Piso com hiperparâmetros chutados:** um piso fraco infla o ganho da Trilha C. Fizemos um sweep pequeno
  (lr × resolução 224/320/384, 2 seeds) escolhido **só no val**.
- **Orçamento 1 com só 60 passos de otimização:** adicionamos um mínimo de 240 passos.
- Determinismo do cuDNN, ETA por época e um orquestrador retomável (`scripts/run_parte0.py`).
- Conferimos o `predict.py` no cenário do monitor (CPU, sem GPU): 811 imagens em 17 s, saída idêntica ao `evaluate.py`.
- Incidente: a IA sobrescreveu o `.gitignore` original sem olhar antes; percebemos no `git status` e restauramos.

### 2026-11-13 — Revisão da Parte 1 (branch da Elisa) e Parte 2 na GTX 1080 Ti
- Pedimos à IA (Claude Code) uma revisão cuidadosa da Parte 1 contra o enunciado. Resultado em `docs/REVISAO_PARTE1.md`:
  requisitos atendidos; problemas de coerência/rigor apontados e corrigidos (FP16 forçado na Pascal no `teacher.py`,
  aluno em 224 px × piso em 384 px, professor escolhido olhando o test_public, "cobertura" por classe calculada errado).
- A IA perguntou antes de decidir: escolhemos (1) comparar SigLIP e CLIP-L e escolher por **CV nos 105 rótulos**
  (critério que não toca val nem test), (2) aluno em 384 px, (3) manter τ = 0, T = 1 sem novo sweep.
- Antes de rodar, a IA mediu o hardware (GPU 226 img/s × dataloader 670 img/s em 384 px → GPU é o gargalo, cache
  desnecessário) e montou `scripts/run_parte2.py` com ETA global e monitor de GPU (uso 95–99% medido).
- Verificamos reprodutibilidade: embeddings do SigLIP regenerados na GPU deram o mesmo zero-shot da CPU.
- O pipeline falhou no último passo (pasta inexistente num script da Parte 1); corrigido e re-executado, pulando os treinos.
- A IA investigou o retrocesso de `domestic_cat` e mostrou que vem do professor (6/48 acertos), não do aluno.

### 2026-11-14 — Parte 3 (galeria de falhas) e plano do modelo entregue
- Pedimos à IA um levantamento do que faltava contra o enunciado. Ela apontou que o `weights/model.pt` não existia
  (os checkpoints dos alunos CLIP-L estavam só na máquina da GPU, fora do git) e que a Parte 3 não estava feita.
- A IA percebeu que a máquina de trabalho não tinha GPU NVIDIA (`torch.cuda.is_available()` = False) e não tentou treinar
  em CPU; em vez disso, escrevemos `scripts/galeria.py` agora e deixamos o treino do aluno para uma máquina com GPU
  (passos em `docs/fazer_placa_de_video.md`).
- `scripts/galeria.py`: Grad-CAM implementado à mão (hooks na `layer4`, sem depender do pacote `grad-cam`). Regra de seleção
  fixada antes de ver os resultados: erros = os de maior confiança (um por classe); acertos = sorteio com semente fixa
  (um por classe), para não escolher só os acertos bonitos. Testado com um checkpoint SigLIP antigo em 150 imagens
  (o CAM caía em cerca e vegetação, não no animal); esse teste não vale como resultado.
- Registramos em `docs/DECISOES.md` que o modelo entregue é a seed 1 (melhor val) mesmo com o pior test_public (0,583),
  pois escolher pelo test_public seria sobreajuste.
- Os diagnósticos dos erros e a leitura dos acertos serão escritos por nós depois de ver as imagens do modelo final;
  a IA só gera as figuras e as métricas de apoio.
