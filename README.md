# PA3 — Poucos rótulos: fauna em câmeras trap

Classificação de 21 espécies (ENA24) com 5 rótulos por classe + ~6,5k imagens sem rótulo.
Backbone: **ResNet-18 (ImageNet)**. Parte 1: **Trilha C (professor → aluno)**.
Enunciado: [`docs/PA3.md`](docs/PA3.md). Decisões e justificativas: [`docs/DECISOES.md`](docs/DECISOES.md).

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

## Treinar (um comando)

```bash
uv run python scripts/train.py --model resnet18 --budget 5 --seed 0
```

Salva em `runs/<metodo>_b<orçamento>_s<seed>/`: `config.json`, `history.csv` e `best.pt`
(melhor época pelo **F1 macro no val**; o `test_public` nunca é lido no treino).
O log mostra, por época, loss/acurácia do treino *limpo* (sem aumentação) e do val, tempo por época e ETA.

## Avaliar (um comando)

```bash
uv run python scripts/evaluate.py --run runs/piso_resnet18_b5_s0
```

Gera `metrics_test_public.json`, `per_class_test_public.csv`, `preds_test_public.csv` e `confusion_test_public.png`.

## Predição (entregável)

```bash
python predict.py --imagens PASTA --saida preds.csv   # usa weights/model.pt, CPU, sem internet
```

## Reproduzir resultados

| O quê | Comando | Saída |
|---|---|---|
| Parte 0 (sweep no val + piso) + piso da Parte 2 | `uv run python scripts/run_parte0.py` (~1 h na 1080 Ti, retomável) | `results/parte0/`, `results/curva_orcamento.csv` |

## Estrutura

```
src/pa3/      paths, data, models, engine, metrics, utils
scripts/      get_data, check_data, train, evaluate, plot_curves, collect_results, run_parte*.py
notebooks/    kaggle_runner.ipynb
results/      tabelas e figuras versionadas (lastro da apresentação)
runs/         checkpoints e logs (gitignored)
```
