"""
Fig 13. Vector Store Scale Test — query latency by record count.
Shows how each store's performance changes from 1K to 100K records.
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

    scale_path = "results/raw/vector_store_scale.csv"
    if not os.path.exists(scale_path):
        print(f"Missing {scale_path}")
        return

    with open(scale_path) as f:
        rows = list(csv.DictReader(f))

    stores = ["pgvector", "qdrant", "chromadb", "sqlite-vec"]
    store_labels = {"pgvector": "pgvector", "qdrant": "Qdrant",
                    "chromadb": "ChromaDB", "sqlite-vec": "sqlite-vec"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: Query latency p50 ---
    for store in stores:
        store_rows = [r for r in rows if r["store"] == store]
        scales = [int(r["scale"]) for r in store_rows]
        p50 = [float(r["query_p50_ms"]) for r in store_rows]

        color = STORE_COLORS.get(store, MOCHA["text"])
        ax1.plot(scales, p50, color=color, marker="o", linewidth=2.5,
                 markersize=8, label=store_labels[store])

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of Records", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Query Latency p50 (ms)", fontsize=12, fontweight="bold")
    ax1.set_title("Query Latency at Scale", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)

    # Add "real-time threshold" line
    ax1.axhline(y=10, color=MOCHA["yellow"], linestyle=":", alpha=0.5, linewidth=1)
    ax1.annotate("10ms (real-time threshold)", xy=(1200, 12),
                 fontsize=8, color=MOCHA["yellow"], style="italic")

    # --- Right: Insert time ---
    for store in stores:
        store_rows = [r for r in rows if r["store"] == store]
        scales = [int(r["scale"]) for r in store_rows]
        insert = [float(r["insert_time_s"]) for r in store_rows]

        color = STORE_COLORS.get(store, MOCHA["text"])
        ax2.plot(scales, insert, color=color, marker="s", linewidth=2.5,
                 markersize=8, label=store_labels[store])

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of Records", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Insert Time (seconds)", fontsize=12, fontweight="bold")
    ax2.set_title("Insert Time at Scale", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)

    fig.suptitle("Vector Store Performance: 1K to 100K Records (1024 dims, CPX42)",
                 fontsize=15, fontweight="bold", y=1.02)

    plt.tight_layout()
    save_fig(fig, "fig_13_vector_store_scale")


if __name__ == "__main__":
    main()
