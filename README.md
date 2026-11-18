# PA3 — Poucos rótulos: fauna em câmeras trap

Classificação de 21 espécies (ENA24) com 5 rótulos por classe + ~6,5k imagens sem rótulo.
Backbone: **ResNet-18 (ImageNet)**. Parte 1: **Trilha C (professor → aluno)**.
Enunciado: [`docs/PA3.md`](docs/PA3.md).

## Ambiente

Gerenciado com [uv](https://docs.astral.sh/uv/) (Python 3.11). O PyTorch vem do índice **cu126**,
o último com suporte a GPUs Pascal (GTX 1080 Ti); funciona também em CPU.

```bash
uv sync                    # cria .venv com tudo
./scripts/get_data.sh      # baixa e descompacta os dados em data/pa3_dados (415 MB)
uv run python scripts/check_data.py   # confere contagens, aninhamento e vazamento
```

Os dados são encontrados automaticamente em `$PA3_DATA`, `/kaggle/input/**` ou `data/pa3_dados`.

**Kaggle:** suba `pa3_dados` como Dataset privado e use [`notebooks/kaggle_runner.ipynb`](notebooks/kaggle_runner.ipynb)
(clona o repo, `pip install -e . --no-deps` e chama os mesmos scripts).

## Treinar (um comando) — o modelo entregue

```bash
uv run python scripts/train_student.py --teacher clip-l --budget 5 --seed 1 --tau 0 --temp 1 --img-size 384 \
  --epochs 15 --method aluno_clip-l --run-name aluno_clip-l_b5_s1
```

Treina o **aluno** ResNet-18 (Trilha C) com os 105 rótulos de `rotulos_5.csv` + os pseudo-rótulos do professor
CLIP-L melhorado, que já estão versionados em `results/teacher/clip-l/improved_b5/` (para refazê-los do zero, ver
"Reproduzir resultados"). Salva em `runs/aluno_clip-l_b5_s1/`: `config.json`, `history.csv` e `best.pt` (melhor época
pelo **F1 macro no val**; o `test_public` nunca é lido no treino). ~12 min na GTX 1080 Ti.
O modelo entregue é esse `best.pt` copiado para `weights/model.pt` (seed 1 = melhor val — a seed é escolhida pelo val, nunca pelo test_public).

O piso (Parte 0) treina com `uv run python scripts/train.py --model resnet18 --budget 5 --seed 0`.

## Avaliar (um comando)

```bash
uv run python scripts/evaluate.py --run runs/aluno_clip-l_b5_s1
```

Gera, na pasta da run, `metrics_test_public.json`, `per_class_test_public.csv`, `preds_test_public.csv` e
`confusion_test_public.png`. Modelo entregue: **F1 macro 0,583, acurácia 0,637** no test_public.

## Predição (entregável)

```bash
python predict.py --imagens PASTA --saida preds.csv   # usa weights/model.pt, CPU, sem internet
```

Imprime o número de parâmetros (11.187.285) e escreve `image_id,label`. 811 imagens em ~40 s numa CPU de 12 núcleos.

**Pesos (`weights/model.pt`, 45 MB, fora do git):** release
[v1.0](https://github.com/brunofs05/deep-learning-assignment-3/releases/tag/v1.0) do repositório.

```bash
mkdir -p weights && gh release download v1.0 -p model.pt -D weights   # ou baixe pelo navegador
sha256sum weights/model.pt   # 12c61be18e53752f2064… (o mesmo de runs/aluno_clip-l_b5_s1/best.pt)
```

## Entregáveis (seção 4 do enunciado)

| Entregável | Onde |
|---|---|
| `README.md` (ambiente, um comando que treina, um que avalia) | este arquivo |
| `predict.py` | [`predict.py`](predict.py): `python predict.py --imagens PASTA --saida preds.csv` (CPU, sem internet, imprime os parâmetros) |
| Pesos do modelo entregue | release [v1.0](https://github.com/brunofs05/deep-learning-assignment-3/releases/tag/v1.0) → `weights/model.pt` |
| `curva_orcamento.csv` | [`results/curva_orcamento.csv`](results/curva_orcamento.csv) (`metodo, orcamento, seed, f1_macro, acuracia`) |
| `AI_LOG.md` | [`AI_LOG.md`](AI_LOG.md) |
| Resultados por parte | `results/parte0/`, `results/teacher/` + `results/trilha_c/` (Parte 1; análises e pergunta da capivara em [`results/trilha_c/README.md`](results/trilha_c/README.md)), `results/parte2/`, `results/parte3/` (cada um com README) |

## Reproduzir resultados

| O quê | Comando | Saída |
|---|---|---|
| Parte 0 (sweep no val + piso) + piso da Parte 2 | `uv run python scripts/run_parte0.py` (~1 h na 1080 Ti, retomável) | `results/parte0/`, `results/curva_orcamento.csv` |
| Parte 1 + 2 (professores, aluno, curva de orçamento) | `uv sync --extra teacher` e `uv run python scripts/run_parte2.py` (~1h10 na 1080 Ti, retomável) | `results/teacher/`, `results/trilha_c/`, `results/parte2/`, `results/curva_orcamento.csv` |
| Parte 3 (galeria de falhas: Grad-CAM, tamanho do animal) | `uv run python scripts/galeria.py --pesos weights/model.pt --caixas results/parte3/caixas_animais.csv` (~2 min em CPU) | `results/parte3/` |
| Parte 3 (atalhos: vizinhos no train) | `uv run python scripts/vizinhos_galeria.py --pesos weights/model.pt` (~3 min em CPU) | `results/parte3/vizinhos*` |

## Estrutura

```
src/pa3/      paths, data, models, engine, metrics, utils
scripts/      get_data, check_data, train, evaluate, plot_curves, collect_results, run_parte*.py
notebooks/    kaggle_runner.ipynb
results/      tabelas e figuras versionadas (lastro da apresentação)
runs/         checkpoints e logs (gitignored)
```
