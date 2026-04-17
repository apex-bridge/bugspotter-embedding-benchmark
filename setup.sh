#!/bin/bash
# Script initialization infrastructure for a benchmark

echo "=== 1. System dependencies setup ==="
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y curl git python3.12 python3.12-venv python3-pip

echo "=== 2. Docker setup ==="
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    rm get-docker.sh
else
    echo "Docker is already set up."
fi

echo "=== 3. Infrastructure start (Ollama, pgvector, Qdrant) ==="
sudo docker compose up -d

echo "Waiting for Ollama to be ready..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Ollama is ready."
    break
  fi
  echo "  Waiting... ($i/30)"
  sleep 5
done

echo "=== 4. Loading Embedding-models in Ollama ==="
MODELS=(
  "all-minilm"
  "nomic-embed-text"
  "snowflake-arctic-embed"
  "mxbai-embed-large"
  "bge-m3"
  "qwen3-embedding"
)

for model in "${MODELS[@]}"; do
  echo ">>> Loading $model..."
  sudo docker compose exec ollama ollama pull "$model"
done

echo "=== Ready! Server is set up. ==="
echo "Don't forget to activate Python: python3 -m venv venv && source venv/bin/activate"