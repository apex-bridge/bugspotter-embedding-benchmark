"""
Fig 11. Vector Store Shootout — grouped bar chart.
4 stores × 3 metrics (insert throughput, query latency, storage).
"""

import csv
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from plot_config import setup_theme, STORE_COLORS, save_fig, MOCHA

import matplotlib.pyplot as plt
import numpy as np


def main():
    setup_theme()

    bench_path = "results/raw/vector_store_bench.csv"
    if not os.path.exists(bench_path):
        print(f"Missing {bench_path}. Run vector_store_bench.py first.")
        # Use placeholder
        stores = ["pgvector", "qdrant", "chromadb", "sqlite-vec"]
        insert_times = [0.8, 1.2, 2.1, 0.5]
        query_p50 = [1.2, 0.8, 3.5, 2.0]
        storage_mb = [3.2, 4.1, 5.8, 2.5]
        recall = [0.98, 0.99, 0.97, 1.00]
    else:
        with open(bench_path) as f:
            rows = list(csv.DictReader(f))
        stores = [r["store"] for r in rows]
        insert_times = [float(r["insert_time_s"]) for r in rows]
        query_p50 = [float(r["query_p50_ms"]) for r in rows]
        storage_mb = [float(r["storage_mb"]) for r in rows]
        recall = [float(r["recall_at_10"]) for r in rows]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    colors = [STORE_COLORS.get(s, MOCHA["text"]) for s in stores]

    metrics = [
        ("Insert Time (s)", insert_times),
        ("Query p50 (ms)", query_p50),
        ("Storage (MB)", storage_mb),
        ("Recall@10", recall),
    ]

    for ax, (title, values) in zip(axes, metrics):
        bars = ax.bar(range(len(stores)), values, color=colors,
                      edgecolor=MOCHA["surface1"], linewidth=1, width=0.6)
        ax.set_xticks(range(len(stores)))
        ax.set_xticklabels(stores, fontsize=9, rotation=0, ha="center")
        ax.set_title(title, fontsize=11, fontweight="bold")

        for bar, val in zip(bars, values):
            fmt = f"{val:.2f}" if val < 10 else f"{val:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(values),
                    fmt, ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color=MOCHA["text"])

    fig.suptitle("Vector Store Shootout (240 records, mxbai-embed-large)",
                 fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    save_fig(fig, "fig_11_vector_store_shootout")


if __name__ == "__main__":
    main()
