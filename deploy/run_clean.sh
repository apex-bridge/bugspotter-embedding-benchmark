#!/bin/bash
# =============================================================================
# CLEAN RUN: All experiments from scratch
#
# Target: Hetzner CPX42 (8 vCPU, 16GB RAM), Ubuntu 24.04
# Duration: ~4-5 hours
# Cost: ~€0.20
#
# Usage:
#   1. Create CPX42, SSH in
#   2. apt-get update && apt-get install -y git
#   3. git clone https://github.com/apex-bridge/bugspotter-embedding-benchmark.git /opt/benchmark
#   4. screen -S bench
#   5. bash /opt/benchmark/deploy/run_clean.sh
#   6. Ctrl+A D to detach
#
# DO NOT run via curl | bash — it breaks docker exec and screen.
# =============================================================================

set -euo pipefail

LOG="/root/benchmark_clean.log"
PROGRESS="/root/benchmark_progress.txt"

# Redirect both stdout and stderr to log AND terminal
exec &> >(tee -a "$LOG")

# Progress tracker — writes current step + timestamp to a file
# Check with: cat /root/benchmark_progress.txt
step() {
    echo "[$(date '+%H:%M:%S')] $1" | tee "$PROGRESS"
}

echo "============================================"
echo " CLEAN BENCHMARK RUN"
echo " Started: $(date)"
echo " Host: $(hostname)"
echo " RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo " CPU: $(nproc) cores"
echo "============================================"

# ===== 1. SYSTEM SETUP =====
echo ""
step "1/13 System setup"
echo "=== 1/13. System setup ==="
apt-get update -qq
apt-get install -y curl git python3 python3-venv python3-pip jq 2>&1 | tail -1

if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    sh /tmp/get-docker.sh 2>&1 | tail -3
    rm /tmp/get-docker.sh
fi
echo "Docker: $(docker --version)"

# ===== 2. CLEAN CLONE =====
echo ""
step "2/13 Clean clone"
echo "=== 2/13. Clean clone ==="
REPO_DIR="/opt/benchmark"
rm -rf "$REPO_DIR"
git clone https://github.com/apex-bridge/bugspotter-embedding-benchmark.git "$REPO_DIR"
cd "$REPO_DIR"
echo "Repo: $(git log --oneline -1)"

# ===== 3. DOCKER SERVICES =====
echo ""
step "3/13 Docker services"
echo "=== 3/13. Start Docker services ==="
docker compose down -v 2>/dev/null || true
docker compose up -d 2>&1

echo "Waiting for services..."
sleep 15
for i in $(seq 1 60); do
    OLLAMA_OK="no"; PG_OK="no"; QD_OK="no"
    curl -sf http://localhost:11434/api/tags > /dev/null 2>&1 && OLLAMA_OK="yes"
    docker compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1 && PG_OK="yes"
    curl -sf http://localhost:6333/collections > /dev/null 2>&1 && QD_OK="yes"

    echo "  Check $i: Ollama=$OLLAMA_OK PG=$PG_OK Qdrant=$QD_OK"

    if [ "$OLLAMA_OK" = "yes" ] && [ "$PG_OK" = "yes" ] && [ "$QD_OK" = "yes" ]; then
        echo "All services ready."
        break
    fi
    sleep 5
done

# ===== 4. PULL MODELS =====
echo ""
step "4/13 Pull models"
echo "=== 4/13. Pull embedding models ==="
MODELS=(
    "all-minilm"
    "nomic-embed-text"
    "mxbai-embed-large"
    "snowflake-arctic-embed"
    "bge-m3"
    "qwen3-embedding"
)

for model in "${MODELS[@]}"; do
    echo ">>> Pulling $model..."
    docker exec benchmark-ollama-1 ollama pull "$model" 2>&1
    echo "    Done: $model"
done

echo ""
echo "Loaded models:"
curl -s http://localhost:11434/api/tags | jq -r '.models[] | "  \(.name)  \(.details.parameter_size)"'

# ===== 5. PYTHON ENVIRONMENT =====
echo ""
step "5/13 Python setup"
echo "=== 5/13. Python environment ==="
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "Python: $(python --version)"

# ===== 6. GENERATE DATASET =====
echo ""
step "6/13 Generate dataset"
echo "=== 6/13. Generate dataset ==="
python data/generate_synthetic.py
python data/generate_pairs.py

echo ""
echo "Dataset stats:"
python -c "
import json, csv
from collections import Counter
with open('data/bug_reports.json', encoding='utf-8') as f:
    reports = json.load(f)
with open('data/pairs_ground_truth.csv', encoding='utf-8') as f:
    pairs = list(csv.DictReader(f))
types = Counter(p['pair_type'] for p in pairs)
print(f'  Reports: {len(reports)}')
print(f'  Pairs: {len(pairs)}')
for t, c in sorted(types.items()):
    print(f'    {t}: {c}')
"

# ===== 7. E1: MAIN BENCHMARK =====
echo ""
step "7/13 E1: Embedding 6 models (~3-4h)"
echo "=== 7/13. E1: Main benchmark (6 models) ==="
echo "    This is the longest step (~3-4 hours)..."

AVAILABLE=$(curl -s http://localhost:11434/api/tags | jq -r '.models[].name' | tr '\n' ' ')
echo "    Models: $AVAILABLE"

python benchmark/embed_all.py --models $AVAILABLE

echo ""
echo "=== 7b. Compute similarity + threshold sweep ==="
python benchmark/compute_similarity.py
python benchmark/sweep_threshold.py

echo ""
echo "--- E1 Results ---"
cat results/raw/model_summary.csv

# ===== 8. E3: MRL TRUNCATION =====
echo ""
step "8/13 E3: MRL truncation"
echo "=== 8/13. E3: MRL dimension truncation ==="
python benchmark/mrl_truncation.py

# ===== 9. E4: EMBEDDING STRATEGY =====
echo ""
step "9/13 E4: Embedding strategy"
echo "=== 9/13. E4: Embedding strategy comparison ==="

BEST_MODEL=$(python -c "
import csv
with open('results/raw/model_summary.csv') as f:
    rows = list(csv.DictReader(f))
best = max(rows, key=lambda r: float(r['best_f1']))
print(best['model'])
")
# Ollama model name: replace _latest with :latest
BEST_MODEL_OLLAMA=$(echo "$BEST_MODEL" | sed 's/_latest/:latest/')
echo "    Best model for E4: $BEST_MODEL (Ollama: $BEST_MODEL_OLLAMA)"

# Unload all models to free RAM
for model in $AVAILABLE; do
    curl -s http://localhost:11434/api/generate -d "{\"model\":\"$model\",\"keep_alive\":0}" > /dev/null 2>&1 || true
done
sleep 5

python benchmark/e4_embedding_strategy.py --model "$BEST_MODEL_OLLAMA"

echo ""
echo "--- E4 Results ---"
cat results/raw/embedding_strategy.csv

# ===== 10. E5: HARD NEGATIVES =====
echo ""
step "10/13 E5: Hard negatives"
echo "=== 10/13. E5: Hard negatives deep dive ==="
python analysis/e5_hard_negatives.py

# ===== 11. E7: VECTOR STORE BENCHMARK (real data) =====
echo ""
step "11/13 E7: Vector store bench"
echo "=== 11/13. E7: Vector store benchmark (real embeddings) ==="

BEST_EMB="results/raw/embeddings_${BEST_MODEL}.json"
if [ -f "$BEST_EMB" ]; then
    python benchmark/load_pgvector.py
    python benchmark/load_qdrant.py
    python benchmark/load_chroma.py
    python benchmark/load_sqlite_vec.py
    python benchmark/vector_store_bench.py --embeddings "$BEST_EMB"

    echo ""
    echo "--- E7 Results ---"
    cat results/raw/vector_store_bench.csv
else
    echo "    WARNING: $BEST_EMB not found, skipping E7"
fi

# ===== 12. VECTOR STORE SCALE TEST =====
echo ""
step "12/13 Vector store scale (1K-100K)"
echo "=== 12/13. Vector store scale test (1K/10K/50K/100K) ==="
python benchmark/vector_store_scale.py --scales 1000 10000 50000 100000

echo ""
echo "--- Scale Results ---"
cat results/raw/vector_store_scale.csv

# ===== 13. GENERATE FIGURES =====
echo ""
step "13/13 Generate figures"
echo "=== 13/13. Generate figures ==="
python analysis/results_summary.py

for fig in analysis/fig_*.py; do
    echo "  $(basename $fig)..."
    python "$fig" 2>/dev/null || echo "    Failed: $(basename $fig)"
done

# ===== DONE =====
echo ""
echo "============================================"
echo " BENCHMARK COMPLETE"
echo " Started: $(head -5 $LOG | grep Started)"
echo " Finished: $(date)"
echo "============================================"
echo ""
echo "=== FINAL RESULTS ==="
cat results/RESULTS.md
echo ""
echo "=== ALL OUTPUT FILES ==="
echo "CSVs:"
ls -lh results/raw/*.csv 2>/dev/null
echo ""
echo "Figures:"
ls -lh results/figures/*.png 2>/dev/null
echo ""
echo "Download:"
echo "  scp -i ~/.ssh/id_ed25519_A -r root@$(hostname -I | awk '{print $1}'):/opt/benchmark/results/ ~/results-clean/"
