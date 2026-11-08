# PA3 — Poucos rótulos: fauna em câmeras trap com dados não rotulados

> Transcrição de `docs/PA3.pdf` para leitura rápida.

| | |
|---|---|
| **Disciplina** | Aprendizado Profundo |
| **Professor** | Dario Oliveira |
| **Monitor** | Erick Brito |
| **Entrega** | 16/11, 23h59 — B41254@fgv.edu.br |
| **Apresentação** | 18/11 e 20/11 (podem mudar; horários a definir) |
| **Formato** | Duplas |

---

## 1. O problema

Câmeras trap são câmeras com sensor de movimento que ficam semanas no mato fotografando o que passa na frente delas. Quando alguém volta para buscar o cartão de memória, encontra milhares de imagens e ninguém para dizer qual bicho está em cada uma. Em 2025 a Tryolabs organizou no Kaggle o **Cup-ybara**, um desafio com vídeos de câmeras trap da fauna uruguaia para uma ONG local, a AMBÁ. A regra que deixava o desafio interessante era que o conjunto de treino não tinha rótulo nenhum — a situação normal de quem faz monitoramento: muito dado e pouquíssima gente para anotar.

No laboratório de dados não rotulados vimos três respostas para isso no MNIST: **Deep k-Means**, **SimCLR com avaliação linear** e **FixMatch com 100 rótulos**. Também vimos o **SwAV** no CIFAR-10. No MNIST tudo funciona bem (dígito ocupa a imagem inteira, fundo preto, classes balanceadas). Numa câmera trap, nada disso vale:

- em **um quarto das imagens, o animal ocupa menos de 2% do quadro**;
- muitas fotos são **noturnas, em infravermelho**, sem cor;
- a mesma câmera fotografa sempre **o mesmo fundo**, e o fundo é um atalho tentador;
- as classes são **desbalanceadas**: no pool há 690 imagens de urso e 45 de cavalo.

**Tarefa:** classificar a espécie em cada imagem tendo **5 rótulos por classe** e alguns milhares de imagens sem rótulo. Vocês escolhem como usar as imagens sem rótulo; o produto final precisa ser um **modelo pequeno, que roda em CPU**.

---

## 2. Dados

**ENA24-detection** (LILA BC, licença CDLA-Permissive-1.0). Câmeras trap do leste da América do Norte, uma espécie por imagem, **21 classes**. Usar **somente** o pacote preparado:

**Dados PA3 (`pa3_dados.zip`, 415 MB):** <https://drive.google.com/file/d/1g7FGqtXOk_ziNKBwQAk8BxjPOaPCnbui/view?usp=sharing>

| Conjunto | Imagens | Rótulo |
|---|---|---|
| `train` | 6.505 | Não, exceto as imagens listadas nos arquivos de orçamento |
| `rotulos_1/5/20.csv` | 21 / 105 / 420 | 1, 5 ou 20 por classe; subconjuntos do train, **aninhados** |
| `val` | 202 | Sim. Até 10 por classe |
| `test_public` | 811 | Sim |
| `test_private` | ~800 | Fica com o monitor |

- Imagens reduzidas para **512 px no lado maior**, sem a faixa de texto da câmera (data, hora, nome), com nomes trocados por código. Por causa do corte, as proporções variam entre câmeras.
- **Orçamento padrão: 5 rótulos por classe.** Os de 1 e 20 servem para a Parte 2.
- O **val é pequeno de propósito** — é nele que se faz **seleção de modelo**.
- O **test_public serve só para reportar números.** Usá-lo para escolher época, limiar ou arquitetura é sobreajuste ao teste, e a diferença aparece no test_private.
- **Cavalo** tem só 45 imagens no treino, 2 no val e 4 no test_public. Com orçamento 20, quase metade fica rotulada; o F1 de cavalo muda muito com um acerto a mais ou a menos. Isso faz parte do problema.

**Split.** O ENA24 não traz local nem horário como metadado, mas a faixa da imagem foi lida e as imagens agrupadas por **câmera e noite** (meio-dia ao meio-dia seguinte). Cada grupo vai inteiro para um único conjunto, então uma rajada/visita do mesmo animal nunca aparece no treino e no teste ao mesmo tempo. A **mesma câmera pode aparecer em conjuntos diferentes** (noites diferentes): o fundo se repete, e aprender o fundo em vez do animal é o atalho que a Parte 3 investiga. Os rótulos do orçamento e do val também vêm de grupos distintos: os 5 rótulos de uma classe vêm de 5 câmeras-noite diferentes. Depois disso a faixa foi cortada.

---

## 3. Regras de engenharia

**Permitido:** PyTorch, torchvision, timm; backbones com pesos de ImageNet (classificação supervisionada); albumentations; sklearn; código dos laboratórios (NT-Xent, perda do FixMatch, SwAV).

**Permitido com condição:** foundation models (CLIP, SigLIP, BioCLIP, e qualquer modelo treinado em espécies/câmeras trap, incluindo MegaDetector e SpeciesNet):
- **só na Trilha C**;
- só com **pesos abertos** rodando no Colab ou localmente;
- **só como professor**: pode olhar o `train` e o `test_public` (para medir o professor), **nunca o `val` durante o treino**, e **não pode fazer parte do modelo entregue**.

**Proibido:**
- rotular imagens à mão (os únicos rótulos humanos são os do pacote);
- API de LLM ou modelo fechado (GPT, Gemini, Claude etc.) para rotular ou classificar;
- baixar o ENA24 original ou qualquer outro dado externo, rotulado ou não;
- treinar ou fazer seleção de modelo no `test_public`.

**Restrições do modelo entregue:**
- **≤ 12 milhões de parâmetros**, contados com `sum(p.numel() for p in model.parameters())` (cabe ResNet-18, MobileNetV3, EfficientNet-B0);
- roda em **CPU, sem internet**;
- classifica o `test_private` em **< 10 minutos** numa máquina comum.

---

## Parte 0 — O piso

1. Treinar um classificador supervisionado **só com `rotulos_5.csv`** (105 imagens), em duas versões:
   - **(a)** uma CNN pequena treinada do zero;
   - **(b)** o backbone que será usado no resto do PA, partindo do ImageNet, com fine-tuning.
2. Reportar **F1 macro** e **acurácia** no `test_public`, mais a **matriz de confusão da versão (b)**. A métrica principal do PA é o **F1 macro** (pesa cavalo igual a urso).
3. Mostrar a **curva de treino e validação** da versão (b). Com 105 imagens, em que época ela começa a decorar?

**Teto:** ResNet-18 do ImageNet com fine-tuning em todos os rótulos do train, época escolhida no val → **0,686 de F1 macro** no `test_public`. Usar como linha do teto na Parte 2 (referência, não limite — dá para chegar perto ou passar). A Parte 1 é sobre quanto da distância piso→teto dá para recuperar sem pedir mais rótulos.

---

## Parte 1 — Usando o que não tem rótulo (escolher **UMA** trilha)

Todas as trilhas usam o **mesmo backbone da Parte 0 (b)** e o **orçamento de 5 por classe**, e se comparam com o piso nas mesmas métricas.

### Trilha A — Auto-supervisionado

Continuar o pré-treino do backbone ImageNet com **SimCLR ou SwAV** sobre as 6.505 imagens do train. Depois, avaliação linear e fine-tuning com os 105 rótulos.

- **Obrigatório:** avaliação linear com três backbones lado a lado: **ImageNet puro**, **ImageNet + auto-supervisionado**, **aleatório + auto-supervisionado**. O pré-treino no domínio vale alguma coisa em cima do ImageNet?
- **Obrigatório:** um **experimento sobre aumentação**. Um random resized crop numa imagem em que o animal ocupa 2% do quadro gera, na maioria das vezes, um par positivo sem animal. O que a rede aprende a aproximar nesse caso? Escolher uma mudança na aumentação, medir e explicar.

### Trilha B — Semi-supervisionado

**FixMatch**, como no laboratório, com os 105 rótulos e o train inteiro como não rotulado.

- **Obrigatório:** **precisão e cobertura dos pseudo-rótulos por classe em função do limiar τ**, medidas rodando o pseudo-rotulador sobre o `test_public` como se não tivesse rótulo. Só diagnóstico: **a escolha de τ é feita no val**.
- **Obrigatório:** com classes desbalanceadas, os pseudo-rótulos tendem a ir para as classes grandes. Mostrar se isso acontece e **testar uma correção** (limiar por classe, alinhamento de distribuição ou outra justificada).

### Trilha C — Professor e aluno

Um **foundation model aberto rotula o train** e um **modelo pequeno (aluno)** é treinado nesses pseudo-rótulos, junto com os 105 rótulos verdadeiros.

- **Obrigatório:** o **F1 zero-shot do professor** no `test_public`, com os nomes das classes como prompt (linha de referência, como o piso).
- **Obrigatório:** a **matriz de confusão do professor** = matriz de ruído dos pseudo-rótulos. Mostrar e dizer onde ele erra sistematicamente (noite, animais pequenos, classes parecidas como `bird` e `american_crow`).
- **Obrigatório:** **tratar o ruído** (filtro por confiança, rótulos suaves com temperatura, correção de rótulo como no SELFIE, ou professor melhorado com os 105 rótulos) e responder: **em quais classes o aluno supera o professor, e por quê?**

### Pergunta para a apresentação (sem implementar)

Se o modelo fosse instalado numa câmera no Uruguai (como no Cup-ybara), a primeira capivara seria classificada como uma das 21 classes, com alguma confiança. **O que no método de vocês daria um sinal de que aquela imagem não pertence a nenhuma das classes?** (detecção de out-of-distribution)

---

## Parte 2 — A curva de orçamento

1. Rodar o **piso (Parte 0 b)** e o **método da trilha** com os três orçamentos: **1, 5 e 20** rótulos por classe. No orçamento de 5, usar **3 seeds** e reportar média ± desvio. Nos outros, uma seed basta.
2. **Gráfico obrigatório:** F1 macro no `test_public` × número de rótulos (**eixo x em log**), uma curva para o piso e outra para o método, com a **linha do teto (0,686)**. Na Trilha C, incluir também o professor.
3. **Ganho por classe no orçamento de 5:** quais classes ganharam mais e quais não ganharam nada?

---

## Parte 3 — Galeria de falhas e atalhos

**Três erros** do modelo final, cada um com:
- a imagem, a classe verdadeira, a prevista e a confiança;
- um **mapa de saliência** (Grad-CAM ou oclusão) mostrando para onde o modelo olhou;
- um **diagnóstico concreto**, ex.: "foto noturna, o animal ocupa 40×30 pixels depois do redimensionamento para 224, e o mapa está todo no tronco da árvore ao fundo".

**Pergunta inversa:** escolher **5 acertos** e mostrar o mapa de saliência. O modelo acertou olhando para o animal ou para o fundo?

---

## 4. Entregáveis

| Arquivo | Descrição |
|---|---|
| Repositório Git | Repositório do trabalho |
| `README.md` | Ambiente, **um comando que treina**, **um comando que avalia** |
| `predict.py` | `python predict.py --imagens PASTA --saida preds.csv`. Escreve CSV com colunas `image_id,label`, usando os nomes de `classes.csv`. Roda em CPU, sem internet, e **imprime o número de parâmetros** do modelo |
| Pesos | Do modelo entregue (link se for grande) |
| `curva_orcamento.csv` | Números da Parte 2: `método, orçamento, seed, F1 macro, acurácia` |
| `AI_LOG.md` | Ver seção 5 |

- O monitor roda o `predict.py` no `test_private`. **Não é competição**: o número privado serve para conferir que o modelo entregue é o apresentado e que não foi ajustado ao `test_public`.
- **Não há relatório escrito. A avaliação é a apresentação.** O repositório dá lastro ao que for afirmado: **toda tabela, curva e imagem mostrada tem que ser reproduzível a partir dele.**

---

## 5. Política de uso de IA

Uso de IA é permitido e esperado. O que não é permitido é entregar algo que vocês não entendem. Subir um **`AI_LOG.md`** descrevendo como a IA foi usada, podendo citar episódios e como as ferramentas ajudaram a resolvê-los.
