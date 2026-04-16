"""
Aggregate results from multiple benchmark runs (different seeds/VMs).

Reads model_summary.csv from each seed directory, computes mean ± std
for key metrics, and writes an aggregated summary.

Usage:
    python analysis/aggregate_runs.py

Input:  results/runs/seed_*/model_summary.csv
Output: results/aggregated/model_summary_aggregated.csv

Fails with non-zero exit if fewer than 3 seed directories are present.
"""

import csv
import os
import sys
from glob import glob
from collections import defaultdict

import numpy as np


def main():
    runs_dir = os.path.join("results", "runs")
    output_dir = os.path.join("results", "aggregated")

    # Find all seed directories
    seed_dirs = sorted(glob(os.path.join(runs_dir, "seed_*")))
    if len(seed_dirs) < 3:
        print(f"ERROR: Found {len(seed_dirs)} seed directories, need at least 3.")
        print(f"  Looked in: {runs_dir}")
        print(f"  Found: {seed_dirs}")
        sys.exit(1)

    print(f"Found {len(seed_dirs)} runs: {[os.path.basename(d) for d in seed_dirs]}")

    # Collect metrics per model across runs
    metrics = defaultdict(lambda: defaultdict(list))
    metric_fields = ["best_f1", "best_threshold", "best_precision", "best_recall",
                     "roc_auc", "recall_at_0.90"]

    for seed_dir in seed_dirs:
        summary_path = os.path.join(seed_dir, "model_summary.csv")
        if not os.path.exists(summary_path):
            print(f"  WARNING: {summary_path} not found, skipping")
            continue

        with open(summary_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                model = row["model"].replace("_latest", "")
                for field in metric_fields:
                    if field in row:
                        metrics[model][field].append(float(row[field]))

        # Also check for BM25 summary
        bm25_path = os.path.join(seed_dir, "bm25_summary.csv")
        if os.path.exists(bm25_path):
            with open(bm25_path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    model = row["model"]
                    for field in metric_fields:
                        if field in row:
                            metrics[model][field].append(float(row[field]))

    # Compute aggregated stats
    os.makedirs(output_dir, exist_ok=True)

    agg_fields = ["model", "runs"]
    for field in metric_fields:
        agg_fields.extend([f"{field}_mean", f"{field}_std"])

    rows = []
    for model in sorted(metrics.keys()):
        row = {"model": model, "runs": len(metrics[model].get("best_f1", []))}
        for field in metric_fields:
            values = metrics[model].get(field, [])
            if values:
                row[f"{field}_mean"] = round(np.mean(values), 4)
                row[f"{field}_std"] = round(np.std(values), 4)
            else:
                row[f"{field}_mean"] = ""
                row[f"{field}_std"] = ""
        rows.append(row)

    # Sort by F1 descending
    rows.sort(key=lambda r: -float(r.get("best_f1_mean", 0) or 0))

    output_path = os.path.join(output_dir, "model_summary_aggregated.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nAggregated results ({len(rows)} models):")
    print(f"{'Model':<25} {'F1 mean±std':<18} {'Threshold':<14} {'Recall@0.9':<14} {'Runs'}")
    print("-" * 80)
    for row in rows:
        f1_mean = row.get("best_f1_mean", "")
        f1_std = row.get("best_f1_std", "")
        t_mean = row.get("best_threshold_mean", "")
        r09_mean = row.get("recall_at_0.90_mean", "")
        runs = row.get("runs", "")
        print(f"{row['model']:<25} {f1_mean}±{f1_std:<10} {t_mean:<14} {r09_mean:<14} {runs}")

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
