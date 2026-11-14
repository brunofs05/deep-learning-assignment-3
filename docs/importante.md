# Importante — pendências de entrega e o que dizer na apresentação

Resumo do que não pode ser esquecido. Os detalhes e números estão em `DECISOES.md` (seção citada em cada item) e
em `REVISAO_PARTE1.md`. Não há slides: **a avaliação é a apresentação, sobre o repositório.**

## 1. Antes de entregar (prazo: 16/11, 23h59, e-mail para B41254@fgv.edu.br)

- [ ] **Repositório e release são privados.** O link dos pesos no README só abre para quem tem acesso. Dar acesso ao
  monitor, ou mandar o `model.pt` (45 MB) ou um link aberto no e-mail. O monitor precisa dos pesos para rodar o
  `predict.py` no `test_private`.
- [ ] **O trabalho está na branch `trilha-c-professor`, não na `main`.** Fazer o merge na `main` ou dizer a branch no e-mail.
- [ ] **Push feito?** Conferir `git status -sb` e o `origin` antes de enviar.
- [ ] **Conferido em 2026-11-14:** o `model.pt` do release tem o sha256 do README; o `predict.py` em CPU imprime
  11.187.285 parâmetros, classifica as 811 imagens do test_public em ~38 s e dá F1 macro 0,5827 / acurácia 0,6375, igual
  a `results/curva_orcamento.csv`.

## 2. O modelo entregue e o aviso da seed (`DECISOES.md` §7)

- Modelo entregue = **aluno CLIP-L, orçamento 5, seed 1** (`weights/model.pt`).
- A seed foi escolhida pelo **val** (F1 0,603), a única regra permitida. A s1 tem o **pior test_public** das três
  seeds: **0,583**, contra 0,636 (s0) e 0,656 (s2). **Não trocar de seed por isso**: escolher olhando o test_public é
  sobreajuste ao teste, e o `test_private` serve justamente para flagrar isso.
- O val tem 202 imagens, então a seleção por ele é ruidosa. Dizer os dois números: **0,583 para o modelo entregue** e
  **0,625 ± 0,038 (média de 3 seeds)** para o desempenho esperado do método.

## 3. Ressalvas a dizer, se perguntarem

**Parte 0 / piso**
- O teto do monitor (0,686) foi treinado em 224 px; o piso e o aluno usam 384 px, então a comparação não é idêntica (`DECISOES.md` §2).
- Com 202 imagens no val, as diferenças entre configurações do sweep ficam perto do ruído entre seeds.

**Parte 1 / Trilha C**
- **τ = 0 e T = 1** foram escolhidos no val com o professor SigLIP em 224 px e **não foram re-escolhidos** para o CLIP-L
  em 384 px (6b.11, 6c.3).
- **Orçamento 1:** não há validação cruzada com 1 rótulo por classe, então o professor melhorado usa **α = 0,5 e C = 10
  fixos** (6c.4).
- Sem busca de hiperparâmetros do aluno: `labeled-repeat` = 5, `pseudo-weight` = 1, 15 épocas, mesma receita do piso.
- **Prompts do zero-shot:** ensemble de 4 templates escolhidos a priori, sem ajuste em nenhum conjunto rotulado (6b.2).
- **Professor:** escolhido por validação cruzada nos 105 rótulos (CLIP-L 0,598 × SigLIP 0,535), sem tocar no val nem no
  test_public. O test_public só mede o professor (permitido pelo enunciado).
- **"Noite × dia"** e "tons de cinza" são heurísticas de saturação de cor, não metadado do dataset.
- Diferenças por classe com 22–82 imagens são ruidosas; as de ±0,05 não são conclusivas.

**Parte 2**
- **No geral o aluno não supera o professor** (0,625 × 0,641); ele recupera ~74% da distância piso→teto. O professor
  (CLIP-L, ~428M parâmetros) não pode ir no modelo entregue.
- **domestic_cat piora** (0,29 → 0,09): o professor acerta só 6 de 48 gatos e o aluno copia o ruído estruturado.
- A variância entre seeds do aluno é alta (0,583–0,656).

**Parte 3**
- As **caixas dos animais foram estimadas olhando as 8 imagens** (com ajuda da IA); servem só para medir tamanho e quanto
  do CAM cai no animal, nunca entram em treino ou seleção. O pacote não traz caixas.
- O **Grad-CAM da `layer4` é 12×12** para entrada 384×384: é borrado e grosseiro para animais de ~50 px.
- A análise de vizinhos mostra **correlação, não causa**. O número forte: acurácia **0,90** quando a vizinha do train entrou
  com o rótulo certo × **0,24** quando entrou com o errado; 87,5% das imagens do test_public têm vizinha com cosseno ≥ 0,9
  (mesma câmera e cena).
- A regra de seleção foi fixada antes de ver o resultado: erros = os de maior confiança (um por classe); acertos =
  sorteio com semente 0 (um por classe). Em 1 dos 5 acertos (coiote) o modelo acertou **olhando o fundo**.

## 4. Pergunta da capivara (OOD): o que dizer (`DECISOES.md` 6b.14 e 6d.4)

- Sinais disponíveis **no modelo entregue** (que não tem o professor): confiança máxima (0,58 nos acertos × 0,28 nos
  erros, AUROC 0,845), entropia/margem e distância às imagens de treino (AUROC 0,715, mais fraco).
- **Cuidado:** isso mede erro dentro das 21 classes, é um **substituto**; a capivara real não foi testada. A resposta é
  só teórica, como pede o enunciado (sem implementar).
- O aluno foi treinado com cross-entropy em 21 classes e tende a ser confiante em qualquer coisa, então só a confiança
  é um sinal fraco. Numa câmera nova toda a cena é desconhecida, então a distância ao train subiria para a câmera inteira.
- Os sinais que dependem do professor (desacordo professor × aluno, embeddings do professor, 22ª classe zero-shot) só
  servem antes de entregar.

## 5. Política de IA

O `AI_LOG.md` está atualizado. Todos devem saber explicar o que está no repositório: **o enunciado permite IA, mas não
permite entregar o que não se entende.**
