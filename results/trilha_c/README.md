# Parte 1 — Trilha C (professor → aluno): análises obrigatórias

Professor: CLIP-L (open_clip, só inferência, fora do modelo entregue). Aluno: ResNet-18, 384 px, orçamento 5
(média de 3 seeds, salvo onde indicado). Tudo medido no test_public **só para reportar**.

## Tratamento do ruído dos pseudo-rótulos
1. **Professor melhorado com os 105 rótulos:** `log p = (1-α)·log_softmax(zero-shot) + α·log_softmax(probe)`, com o
   probe sendo uma regressão logística balanceada nos embeddings do professor. (α, C) por validação cruzada de 5
   dobras **dentro dos 105 rótulos** (sem val, sem test_public). Orçamento 1: α = 0,5 e C = 10 fixos (não há CV).
2. **Filtro por confiança τ e rótulos suaves com temperatura T:** sweep no val (SigLIP, 224 px) escolheu τ = 0 e
   T = 1; não foram re-escolhidos para o CLIP-L em 384 px.
3. **Rótulos verdadeiros pesam mais:** os 105 entram repetidos 5× por época, com label smoothing 0,1.

Não implementado: correção de rótulo à la SELFIE e re-treino iterativo.

## Matriz de ruído do professor
`../teacher/clip-l/improved_b5/confusion_test_public.png` e `top_confusoes.csv`. Erros sistemáticos entre aves e
entre parecidos: bird → chicken (14 de 22), american_crow → wild_turkey (14 de 63) e → chicken (10),
chicken → dog (19 de 60), grey_fox → red_fox (8 de 39), domestic_cat espalhado em woodchuck e northern_raccoon
(7 cada de 48). Com o CLIP-L a noite quase não pesa (acurácia 0,662 à noite × 0,686 de dia; "noite" é heurística de
saturação de cor, não metadado). Animais pequenos: não medido para o professor (o pacote não traz caixas).

## Onde o aluno supera o professor (`clip-l/aluno_vs_professor_b5.csv`)
- **Supera** (nas 3 seeds): american_crow (+0,21), grey_fox (+0,20), chicken (+0,12), virginia_opossum, coyote,
  american_black_bear. **Perde:** eastern_cottontail (−0,35), eastern_fox_squirrel (−0,18), bird, woodchuck,
  striped_skunk, domestic_cat (−0,10 a −0,13).
- **Por quê (hipótese):** o aluno ganha onde o erro do professor é confusão entre vizinhos parecidos
  (corvo/peru/galinha, raposa cinza/vermelha): os 105 rótulos verdadeiros, repetidos, e a média sobre ~6.400
  pseudo-rótulos puxam a fronteira para o lugar certo. Perde em classes pequenas e instáveis (eastern_cottontail
  varia de 0,15 a 0,69 entre seeds; bird tem 22 imagens) e onde o professor já erra de forma sistemática
  (domestic_cat: o aluno copia o ruído estruturado). Diferenças de ±0,05 em classes de 22–82 imagens não são conclusivas.
- **No geral o aluno não supera o professor** (0,625 × 0,641), mas o professor (~428M parâmetros) não pode ir no
  modelo entregue.

## Pergunta da capivara (OOD) — só resposta, sem implementação
Uma capivara seria classificada como uma das 21 classes. Sinais que o **modelo entregue** (sem professor) oferece:
- **Confiança máxima (softmax):** 0,58 nos acertos × 0,28 nos erros, AUROC 0,845 para separar acerto de erro
  (`../parte3/vizinhos_resumo_test_public.json`). Entropia e margem entre as duas maiores probabilidades seguem a
  mesma ideia.
- **Distância às imagens de treino** (features do próprio modelo): 0,95 nos acertos × 0,93 nos erros, AUROC 0,715,
  mais fraca que a confiança.
- **Só antes de entregar** (dependem do professor): desacordo professor × aluno, distância nos embeddings do
  professor, 22ª classe zero-shot ("rodent", "unknown animal").

**Limitações:** (a) isso mede erro dentro das 21 classes, é um substituto; a capivara real não foi testada;
(b) o aluno usa cross-entropy em 21 classes e tende a ser confiante em qualquer coisa; (c) a capivara parece
roedor/mamífero do dataset (woodchuck, esquilos), as classes-lixeira onde o modelo já erra com confiança;
(d) como o modelo reconhece muito a cena (Parte 3), numa câmera nova toda a cena seria desconhecida: a distância ao
train subiria para a câmera inteira, ou seja, detecta mudança de domínio, não classe nova; (e) o limiar de rejeição
teria de ser calibrado, e sem imagens OOD rotuladas só dá para calibrar pela taxa de erro.
