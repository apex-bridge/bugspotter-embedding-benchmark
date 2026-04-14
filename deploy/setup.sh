#!/bin/bash
# =============================================================================
# Server Setup — Full Benchmark Pipeline
#
# Target: CPX42 (16GB RAM) for all 6 models, or CPX32 (8GB) for 5 models (skips qwen3)
#
# Usage:
#   1. Create a server (Hetzner CPX42 recommended), Ubuntu 24.04
#   2. SSH in: ssh root@<ip>
#   3. bash <(curl -sSL https://raw.githubusercontent.com/apex-bridge/bugspotter-embedding-benchmark/main/deploy/setup.sh)
#
# Or copy this script and run manually.
# =============================================================================

set -euo pipefail

echo "============================================"
echo " BugSpotter Embedding Benchmark — Server Setup"
echo " $(date)"
echo "============================================"

# --- 1. System Setup ---
echo ""
echo "=== 1. System dependencies ==="
apt-get update && apt-get upgrade -y
apt-get install -y curl git python3 python3-venv python3-pip jq

# --- 2. Docker ---
echo ""
echo "=== 2. Docker ==="
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
    systemctl enable docker
    systemctl start docker
else
    echo "Docker already installed."
fi

# Install Docker Compose plugin if needed
if ! docker compose version &> /dev/null; then
    apt-get install -y docker-compose-plugin
fi

# --- 3. Clone Repository ---
echo ""
echo "=== 3. Clone repository ==="
REPO_DIR="/opt/benchmark"
if [ -d "$REPO_DIR" ]; then
    echo "Repo already exists, pulling latest..."
    cd "$REPO_DIR" && git pull
else
    git clone https://github.com/apex-bridge/bugspotter-embedding-benchmark.git "$REPO_DIR"
fi
cd "$REPO_DIR"

# --- 4. Start Docker Services ---
echo ""
echo "=== 4. Start Docker services (Ollama + PostgreSQL + Qdrant) ==="
docker compose up -d

echo "Waiting for Ollama to be ready..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is ready."
        break
    fi
    echo "  Waiting... ($i/30)"
    sleep 5
done

# --- 5. Pull Embedding Models ---
echo ""
echo "=== 5. Pull embedding models ==="

# Detect RAM to decide which models to pull
TOTAL_RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
echo "Detected RAM: ${TOTAL_RAM_GB}GB"

MODELS=(
    "all-minilm"
    "nomic-embed-text"
    "mxbai-embed-large"
    "snowflake-arctic-embed"
    "bge-m3"
)

# Only pull qwen3 if we have enough RAM (>12GB free after services)
if [ "$TOTAL_RAM_GB" -ge 14 ]; then
    MODELS+=("qwen3-embedding")
    echo "RAM >= 14GB: including qwen3-embedding (7.6B Q4_K_M)"
else
    echo "RAM < 14GB: skipping qwen3-embedding (needs ~5GB for model alone)"
fi

for model in "${MODELS[@]}"; do
    echo ">>> Pulling $model..."
    docker compose exec -T ollama ollama pull "$model"
done

echo ""
echo "Loaded models:"
curl -s http://localhost:11434/api/tags | jq -r '.models[] | "\(.name)\t\(.details.parameter_size)\t\(.size/1e9 | tostring | .[0:4])GB"'

# --- 6. Python Environment ---
echo ""
echo "=== 6. Python environment ==="
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# --- 7. Generate Dataset ---
echo ""
echo "=== 7. Generate dataset ==="
python data/generate_synthetic.py
python data/generate_pairs.py

# --- 8. Scrape Mozilla Bugzilla (validation dataset) ---
echo ""
echo "=== 8. Scrape Mozilla Bugzilla (validation dataset) ==="
python data/scrape_bugzilla.py --sources mozilla --target-pairs 250 || echo "Bugzilla scrape failed (non-critical), continuing..."

# --- 9. Run Embedding Benchmark ---
echo ""
echo "=== 9. Embed all models ==="

# Build model list from what's actually available
AVAILABLE_MODELS=$(curl -s http://localhost:11434/api/tags | jq -r '.models[].name' | tr '\n' ' ')
echo "Available models: $AVAILABLE_MODELS"

python benchmark/embed_all.py --models $AVAILABLE_MODELS

# --- 10. Evaluate ---
echo ""
echo "=== 10. Evaluate ==="
python benchmark/compute_similarity.py
python benchmark/sweep_threshold.py
python benchmark/mrl_truncation.py

# --- 11. Vector Store Benchmark (E7) ---
echo ""
echo "=== 11. Vector store benchmark ==="

# Load embeddings into all stores
python benchmark/load_pgvector.py
python benchmark/load_qdrant.py
python benchmark/load_chroma.py
python benchmark/load_sqlite_vec.py

# Find best model and run shootout
BEST_MODEL=$(python -c "
import csv
with open('results/raw/model_summary.csv') as f:
    rows = list(csv.DictReader(f))
best = max(rows, key=lambda r: float(r['best_f1']))
print(best['model'])
")
echo "Best model: $BEST_MODEL"

BEST_EMB="results/raw/embeddings_${BEST_MODEL}.json"
if [ -f "$BEST_EMB" ]; then
    python benchmark/vector_store_bench.py --embeddings "$BEST_EMB"
else
    echo "WARNING: $BEST_EMB not found, skipping E7"
fi

# --- 12. Generate Figures ---
echo ""
echo "=== 12. Generate figures ==="
python analysis/results_summary.py

for fig in analysis/fig_*.py; do
    echo "  Running $fig..."
    python "$fig" 2>/dev/null || echo "  Failed: $fig (non-critical)"
done

# --- 13. Summary ---
echo ""
echo "============================================"
echo " BENCHMARK COMPLETE"
echo " $(date)"
echo "============================================"
echo ""
echo "Results:"
echo "  Summary:  results/raw/model_summary.csv"
echo "  Sweep:    results/raw/threshold_sweep.csv"
echo "  MRL:      results/raw/mrl_truncation.csv"
echo "  Stores:   results/raw/vector_store_bench.csv"
echo "  Figures:  results/figures/"
echo "  Report:   results/RESULTS.md"
echo ""
cat results/RESULTS.md
echo ""
echo "To download results:"
echo "  scp -r root@$(hostname -I | awk '{print $1}'):/opt/benchmark/results/ ./results/"
