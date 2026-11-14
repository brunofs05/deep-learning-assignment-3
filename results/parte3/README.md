# Parte 3 — Galeria de falhas e atalhos

Modelo: **o modelo entregue** (`weights/model.pt` = aluno CLIP-L, orçamento 5, seed 1, escolhida pelo val).
Imagens do `test_public` (só para reportar; nada aqui escolhe modelo). Acurácia do modelo: 0,637; F1 macro: 0,583.

```bash
uv run python scripts/galeria.py --pesos weights/model.pt --caixas results/parte3/caixas_animais.csv
uv run python scripts/vizinhos_galeria.py --pesos weights/model.pt --teacher clip-l
```

## Como as imagens foram escolhidas (regra fixada antes de ver o resultado)
- **3 erros:** os de **maior confiança**, no máximo um por classe verdadeira (erro confiante = falha mais grave).
  Lista dos candidatos em `candidatos_erros.csv`.
- **5 acertos:** **sorteio** com semente 0, no máximo um por classe, para não escolher só os acertos bonitos.
- **Saliência:** Grad-CAM na `layer4` da ResNet-18 (implementação própria em `galeria.py`; conferida contra o pacote
  `grad-cam`, correlação 1,00). A `layer4` tem resolução 12×12 para a entrada de 384×384, então o mapa é borrado.
- **Tamanho do animal:** o ENA24 deste pacote não traz caixas. As caixas de `caixas_animais.csv` foram **estimadas
  olhando as 8 imagens** (com ajuda da IA, ver AI_LOG) e servem só para medir o tamanho do animal na entrada do
  modelo e quanto do CAM cai nele. Nunca entram em treino ou seleção. As imagens do pacote têm 512 px de largura e são
  redimensionadas para **384×384 sem preservar proporção** (sem crop).
- **Atalho:** `vizinhos_galeria.py` busca, para cada imagem, as 5 imagens do **train** mais parecidas no espaço de
  features do próprio modelo (avgpool, cosseno) e mostra com que rótulo cada uma entrou no treino do aluno
  (verdadeiro do `rotulos_5` ou pseudo-rótulo do professor). Ver `vizinhos.png`.

Figuras: `galeria_erros.png`, `galeria_acertos.png` (caixa tracejada = animal), `erro_*.png` (CAM da classe prevista
e da verdadeira), `acerto_*.png`, `vizinhos.png`. Números: `selecao.csv`, `vizinhos.csv`, `vizinhos_resumo_test_public.json`.

## Três erros

| # | Imagem | Verdadeira → prevista | Conf. | Foto | Animal na entrada (384×384) | Pico do CAM | Vizinhas no train |
|---|---|---|---|---|---|---|---|
| 1 | `6cf157171b7364e5` | bird → chicken | 0,96 | tons de cinza | ~49×63 px (2% da imagem) | na ave, mas a região quente se espalha pela tela do galinheiro | 5/5 o mesmo galinheiro, todas "chicken" (pseudo) |
| 2 | `ece8e099eb242e5d` | dog → chicken | 0,78 | tons de cinza | ~54×94 px, meio escondido e cortado pela borda | **fora do animal** (quina da composteira) | 5/5 a mesma composteira, todas "chicken" (1 rótulo **verdadeiro** + 4 pseudo) |
| 3 | `8a42e0eb8dcf4ee8` | domestic_cat → eastern_fox_squirrel | 0,73 | colorida (dia) | ~109×63 px (5%) | dividido entre o gato e o galho horizontal | 5/5 a mesma câmera, todas "eastern_fox_squirrel" (pseudo) |

**Erro 1 — bird → chicken (0,96).** Foto em tons de cinza dentro de um galinheiro; a ave (preta e branca, cauda
longa) ocupa ~49×63 px dos 384×384. O pico do CAM está na ave, mas só 4% da região quente (CAM ≥ 0,5) cai nela: o
resto fica na **tela hexagonal do galinheiro** à direita e no canto superior esquerdo. O CAM da classe verdadeira
(bird) quase não toca a ave (0,6% da massa). As 5 vizinhas no train são **o mesmo galinheiro** (cos 0,97–0,98) e
entraram no treino como "chicken" pelo professor. A ave nelas parece a mesma desta foto. *Diagnóstico:* ruído do
professor (bird → chicken é a maior confusão dele: 14 de 22 no test_public) somado ao cenário: para o modelo,
"galinheiro" = chicken.

**Erro 2 — dog → chicken (0,78).** Foto em tons de cinza; o cachorro claro aparece só pela metade, atrás de uma
composteira e cortado pela borda direita (~54×94 px na entrada). O pico do CAM fica na **quina da composteira**, e
nada da região quente cai no cachorro. O CAM da classe verdadeira (dog) põe 38% da massa no cachorro: o modelo tem
evidência de cachorro, mas a do cenário vence. As 5 vizinhas são **a mesma composteira**, todas "chicken", e uma
delas é um rótulo **verdadeiro** do `rotulos_5`. *Diagnóstico:* atalho de fundo puro. Essa câmera fotografa
galinhas, e a composteira virou sinônimo de "chicken".

**Erro 3 — domestic_cat → eastern_fox_squirrel (0,73).** Foto colorida de dia, gato tigrado andando no chão de
folhas (~109×63 px). O CAM tem dois focos: o gato (pico, 15% da região quente) e o **galho horizontal arqueado** no
meio da imagem, o tipo de lugar onde esquilos aparecem. As 5 vizinhas são a mesma câmera, todas pseudo-rotuladas
"eastern_fox_squirrel" pelo professor, e na maioria delas o animal visível parece ser um gato (preto e branco).
*Diagnóstico:* o professor chamou os gatos dessa câmera de esquilo, e o aluno copiou (mais o cenário). É o mesmo
mecanismo do retrocesso de domestic_cat na Parte 2 (o professor só acerta 6 de 48 gatos; o F1 de gato do aluno cai
de 0,29 para 0,09).

## Pergunta inversa — cinco acertos: animal ou fundo?

| Imagem | Classe | Conf. | Animal na entrada | Pico no animal? | Região quente (CAM ≥ 0,5) no animal | Vizinhas com a mesma classe | Leitura |
|---|---|---|---|---|---|---|---|
| `f43eb3a32122dd83` | dog | 0,84 | 120×199 (16%) | sim | 90% | 5/5 (mesma cena, mesmo cão) | **animal** |
| `ccd002e937cad908` | virginia_opossum | 0,46 | 49×85 (3%) | sim | 23% (massa do CAM 4,1× o acaso) | 5/5 (mesma cena) | **animal** |
| `c0ad7ce642b24c50` | grey_fox | 0,26 | 98×37, borrado | sim | 34% (massa do CAM 3,5× o acaso) | 3/5 (mesma cena, mas 2 são "cottontail") | **animal**: o cenário não decide aqui |
| `5bf96a76149508c6` | coyote | 0,47 | 82×107, borrado e cortado | **não** (arbusto ao fundo) | 10% | 5/5 (mesma cena, cos 0,98–0,99) | **fundo**: acerto pelo motivo errado |
| `69ff348a22c8ff80` | american_black_bear | 0,23 | 144×197, só o dorso, colado na lente | sim (borda do vulto) | 67% | 2/5 (cenas diferentes, vulto escuro na borda) | **animal, pela silhueta** |

**Resposta:** em 3 dos 5 acertos o modelo olhou o animal (cachorro, gambá, raposa), no urso olhou o vulto escuro
perto da lente, e **o coiote foi acertado pelo fundo**. O CAM fica no arbusto, e as 5 vizinhas são a mesma câmera,
todas "coyote". A raposa é o contraexemplo útil: quando a mesma cena aparece no train com classes diferentes
(grey_fox e cottontail), o cenário não basta e o modelo precisa olhar o animal.

## Quanto o atalho pesa no test_public inteiro (`vizinhos_resumo_test_public.json`)
Para cada uma das 811 imagens, pegamos a vizinha mais próxima no train (features do modelo entregue):

- **87,5%** das imagens têm uma vizinha com cosseno ≥ 0,9, quase sempre a **mesma câmera e cena** (o enunciado
  avisa: a mesma câmera aparece em noites diferentes no train e no teste).
- A previsão é igual ao rótulo de treino da vizinha em **68%** das imagens.
- **Acurácia de 0,90 quando a vizinha entrou no treino com o rótulo certo, contra 0,24 quando entrou com o errado.**
  **39% dos erros** repetem exatamente o rótulo da vizinha.

**Leitura (com cuidado):** é correlação, não prova causal. Um bom classificador também concorda com vizinhas certas.
O número revelador é o 0,24: quando a cena foi ensinada com o rótulo errado (pseudo-rótulo ruim ou outra espécie na
mesma câmera), o modelo quase sempre erra. Junto com os CAMs no fundo (erros 2 e 3, coiote), isso indica que boa
parte do desempenho vem de **reconhecer a câmera/cena**. O ruído do professor entra justamente por aí: ele
pseudo-rotula cenas inteiras, e o aluno aprende "esta cena = esta classe".

**Ressalvas:** as caixas são estimadas à vista; o Grad-CAM em 12×12 é grosseiro para animais de ~50 px; "tons de
cinza" vem de uma heurística de cor (IR noturno ou câmera em modo monocromático), não de metadado; as vizinhas usam
o argmax do pseudo-rótulo, mas o aluno treinou com o alvo suave (T = 1).
