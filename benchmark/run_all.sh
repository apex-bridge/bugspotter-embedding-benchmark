#!/bin/bash
# Run the complete benchmark pipeline.
# Usage: ./benchmark/run_all.sh [--seed 42]
#
# Prerequisites:
#   - Docker services running (ollama, postgres, qdrant)
#   - Python venv activated with requirements.txt installed
#   - Ollama models pulled (via setup.sh)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Parse --seed argument (default: 42)
SEED=42
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed) SEED="$2"; shift 2 ;;
        *) shift ;;
    esac
done
echo "Random seed: $SEED"

echo "============================================"
echo " BugSpotter Embedding Benchmark Pipeline"
echo "============================================"

# Step 1: Generate dataset
echo ""
echo "=== Step 1: Generate synthetic bug reports ==="
python data/generate_synthetic.py --output data/bug_reports.json --github-input data/github_issues.json --seed $SEED

echo ""
echo "=== Step 1b: Generate ground truth pairs ==="
python data/generate_pairs.py --input data/bug_reports.json --output data/pairs_ground_truth.csv

# Step 2: Generate embeddings for all models
echo ""
echo "=== Step 2: Generate embeddings (all 7 models) ==="
python benchmark/embed_all.py --input data/bug_reports.json --output-dir results/raw

# Step 3: Load into vector stores
echo ""
echo "=== Step 3a: Load into pgvector ==="
python benchmark/load_pgvector.py --embeddings-dir results/raw --reports data/bug_reports.json

echo ""
echo "=== Step 3b: Load into Qdrant ==="
python benchmark/load_qdrant.py --embeddings-dir results/raw --reports data/bug_reports.json

echo ""
echo "=== Step 3c: Load into ChromaDB ==="
python benchmark/load_chroma.py --embeddings-dir results/raw --reports data/bug_reports.json

echo ""
echo "=== Step 3d: Load into sqlite-vec ==="
python benchmark/load_sqlite_vec.py --embeddings-dir results/raw --reports data/bug_reports.json

# Step 4: Compute pairwise similarity
echo ""
echo "=== Step 4: Compute cosine similarity ==="
python benchmark/compute_similarity.py \
    --embeddings-dir results/raw \
    --pairs data/pairs_ground_truth.csv \
    --output results/raw/similarity_scores.csv

# Step 5: Threshold sweep
echo ""
echo "=== Step 5: Threshold sweep ==="
python benchmark/sweep_threshold.py \
    --input results/raw/similarity_scores.csv \
    --output-sweep results/raw/threshold_sweep.csv \
    --output-summary results/raw/model_summary.csv

# Step 5b: BM25/TF-IDF baseline (synthetic dataset)
echo ""
echo "=== Step 5b: BM25/TF-IDF baseline (synthetic) ==="
python benchmark/bm25_baseline.py

# Step 5c: BM25F with 5-fold CV — honest tuned number
# (bm25_baseline's bm25f_tuned is oracle-F1; this is the protocol-correct version)
echo ""
echo "=== Step 5c: BM25F 5-fold CV ==="
python benchmark/bm25f_cv.py

# Step 5d: Mozilla Bugzilla cross-validation (requires data/bugzilla_bugs.json)
if [ -f "data/bugzilla_bugs.json" ]; then
    echo ""
    echo "=== Step 5d: 6-model Bugzilla validation ==="
    python benchmark/bugzilla_validation.py

    echo ""
    echo "=== Step 5e: BM25 + TF-IDF on Bugzilla ==="
    python benchmark/bm25_bugzilla.py
else
    echo ""
    echo "=== Skipping Bugzilla (data/bugzilla_bugs.json missing) ==="
fi

# Step 6: MRL truncation experiment
echo ""
echo "=== Step 6: MRL dimension truncation ==="
python benchmark/mrl_truncation.py \
    --embeddings-dir results/raw \
    --pairs data/pairs_ground_truth.csv \
    --output results/raw/mrl_truncation.csv

# Step 7 (E7): Vector store shootout
# Uses the best model from model_summary.csv
echo ""
echo "=== Step 7 (E7): Vector store shootout ==="
# Find the best model by F1 from the summary
BEST_MODEL=$(python -c "
import csv
with open('results/raw/model_summary.csv') as f:
    rows = list(csv.DictReader(f))
best = max(rows, key=lambda r: float(r['best_f1']))
print(best['model'])
" 2>/dev/null || echo "qwen3_embedding")

echo "Best model: $BEST_MODEL"
BEST_EMB="results/raw/embeddings_${BEST_MODEL}.json"

if [ -f "$BEST_EMB" ]; then
    python benchmark/vector_store_bench.py \
        --embeddings "$BEST_EMB" \
        --output results/raw/vector_store_bench.csv
else
    echo "WARNING: $BEST_EMB not found, skipping E7"
fi

# Generate visualizations
echo ""
echo "=== Generating visualizations ==="
python analysis/results_summary.py 2>/dev/null || echo "Skipping summary (script may not exist yet)"

for fig_script in analysis/fig_*.py; do
    if [ -f "$fig_script" ]; then
        echo "Running $fig_script..."
        python "$fig_script" 2>/dev/null || echo "  Failed: $fig_script"
    fi
done

echo ""
echo "============================================"
echo " Pipeline complete!"
echo " Results: results/raw/"
echo " Figures: results/figures/"
echo "============================================"
