# O que falta fazer na máquina com a placa de vídeo (GTX 1080 Ti)

Tudo o que dava para fazer sem GPU já foi feito (Partes 0, 1, 2, `predict.py`, `scripts/galeria.py`).
Falta só o que depende do **checkpoint do aluno entregue**, que não existe na máquina sem NVIDIA
(`runs/` e `weights/` estão no `.gitignore`). Prazo de entrega: **16/11, 23h59**.

## 0. Preparar
```bash
git pull                      # branch trilha-c-professor
uv sync --extra cam           # o extra "cam" é opcional: a galeria usa Grad-CAM próprio
./scripts/get_data.sh         # só se data/pa3_dados não existir
nvidia-smi                    # confirmar a GPU
```

## 1. Treinar o aluno entregue (~15–20 min na 1080 Ti)
Aluno CLIP-L, orçamento 5, **seed 1** (a de melhor val; regra e aviso em `docs/DECISOES.md` §7).
Os embeddings e pseudo-rótulos do professor já estão versionados em `results/teacher/clip-l/`.
```bash
uv run python scripts/train_student.py --teacher clip-l --budget 5 --seed 1 \
  --tau 0 --temp 1 --img-size 384 --epochs 15 --workers 11 \
  --method aluno_clip-l --run-name aluno_clip-l_b5_s1
uv run python scripts/evaluate.py --run runs/aluno_clip-l_b5_s1 --workers 11
mkdir -p weights && cp runs/aluno_clip-l_b5_s1/best.pt weights/model.pt
```
**Conferir:** val F1 ≈ **0,603** (`runs/aluno_clip-l_b5_s1/best_val.json`) e test_public F1 ≈ **0,583**
(acurácia ≈ 0,6375) — batem com a linha `aluno_clip-l,5,1` de `results/curva_orcamento.csv`.
Se divergir muito, algo mudou na configuração; não trocar de seed olhando o test_public.

> Alternativa: se o `run_parte2.py` for rodado inteiro (`uv run python scripts/run_parte2.py`, ~1h10), ele
> treina as 5 execuções e regera curva, gráficos e análises. Não é necessário para entregar.

## 2. Verificar o `predict.py` (CPU, como o monitor vai rodar)
```bash
CUDA_VISIBLE_DEVICES= uv run python predict.py --imagens data/pa3_dados/images/test_public --saida /tmp/preds.csv
```
**Conferir:** imprime o nº de parâmetros (≈ 11,2 M, ≤ 12 M); 811 imagens em bem menos de 10 min; e o `preds.csv`
é idêntico ao `runs/aluno_clip-l_b5_s1/preds_test_public.csv` (mesmos rótulos).

## 3. Gerar a galeria da Parte 3 (roda em CPU ou GPU; ~1 min)
```bash
uv run python scripts/galeria.py --pesos weights/model.pt
```
Gera em `results/parte3/`: `galeria_erros.png`, `galeria_acertos.png`, `erro_*.png`, `acerto_*.png`,
`selecao.csv`, `candidatos_erros.csv`, `metricas_cam_test_public.csv`.
- Por padrão os 3 erros são os de **maior confiança**, um por classe verdadeira, e os 5 acertos são **sorteados**
  (semente 0), um por classe. Para trocar os erros, olhe `candidatos_erros.csv` e use `--erros ID1 ID2 ID3`.
- **Depois de ver as imagens, escrever os diagnósticos** em `results/parte3/README.md` (o enunciado exige um
  diagnóstico concreto por erro: foto noturna? tamanho do animal depois do resize para 384? CAM no fundo ou no
  animal?). Os números de apoio estão em `selecao.csv` (`noturna`, `cam_area50`, `cam_massa_borda`, `largura`/`altura`).
  O dataset não traz caixa do animal, então o tamanho do animal se estima olhando a imagem.
- Responder também a pergunta inversa: nos 5 acertos, o CAM está no animal ou no fundo?

## 4. Fechar o repositório
- `weights/model.pt` (≈ 45 MB) fica fora do git: subir como release do GitHub ou no Drive e **colocar o link no
  README** (o enunciado aceita link se for grande).
- Acrescentar ao `AI_LOG.md` o episódio da Parte 3 (galeria, Grad-CAM) e o do treino do aluno entregue.
- Atualizar `docs/DECISOES.md` §7 (marcar modelo entregue e Parte 3 como feitos) e commitar `results/parte3/`.
- Conferir o README: comando que treina, comando que avalia, `predict.py`, link dos pesos.

## 5. Opcional (só se sobrar tempo)
- Rodar `scripts/run_parte2.py` completo para regerar tudo na mesma máquina (reprodutibilidade).
- Avaliar o checkpoint na CPU do monitor (tempo no `test_private` estimado em < 1 min; limite é 10 min).
