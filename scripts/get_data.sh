#!/usr/bin/env bash
# Baixa e descompacta o pacote de dados do PA3 em data/
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
if [ ! -f data/pa3_dados.zip ]; then
  uv run gdown -q 1g7FGqtXOk_ziNKBwQAk8BxjPOaPCnbui -O data/pa3_dados.zip
fi
unzip -q -n data/pa3_dados.zip -d data/
echo "Dados em $(pwd)/data"
