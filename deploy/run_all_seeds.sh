#!/bin/bash
# =============================================================================
# Single-machine reproduction — sequential runs with state cleanup.
# Note: ~15 hours on one machine vs ~5 hours in parallel across 3 VMs.
# For publication-quality reproduction, use 3 separate VMs (see README).
# =============================================================================

set -euo pipefail

for SEED in 42 123 456; do
    echo ""
    echo "============================================"
    echo " Running seed $SEED"
    echo " $(date)"
    echo "============================================"

    # Clean Ollama state between runs
    docker compose restart ollama
    sleep 10

    bash deploy/run_clean.sh --seed $SEED

    # Archive this run's outputs
    mkdir -p results/runs/seed_$SEED
    cp results/raw/model_summary.csv results/runs/seed_$SEED/
    cp results/raw/similarity_scores.csv results/runs/seed_$SEED/
    cp results/raw/threshold_sweep.csv results/runs/seed_$SEED/
    cp results/raw/bm25_summary.csv results/runs/seed_$SEED/ 2>/dev/null || true

    echo "Archived seed $SEED results to results/runs/seed_$SEED/"
done

echo ""
echo "============================================"
echo " Aggregating results"
echo "============================================"
python analysis/aggregate_runs.py

echo ""
echo "Done. See results/aggregated/model_summary_aggregated.csv"
