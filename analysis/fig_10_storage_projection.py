"""
Fig 10. Storage projection chart.
X = number of bug reports (1K–1M, log scale). Y = storage (GB).
Lines for each model by dimension size.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from plot_config import setup_theme, MODEL_COLORS, save_fig, model_display_name, MOCHA

import matplotlib.pyplot as plt
import numpy as np

MODEL_DIMS = {
    "qwen3_embedding": 1024,
    "qwen3_embedding_4b": 4096,
    "nomic_embed_text": 768,
    "mxbai_embed_large": 1024,
    "bge_m3": 1024,
    "all_minilm": 384,
    "snowflake_arctic_embed": 768,
}


def main():
    setup_theme()

    record_counts = np.array([1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000])
    fig, ax = plt.subplots(figsize=(10, 6))

    for model, dim in sorted(MODEL_DIMS.items(), key=lambda x: x[1]):
        # Storage = records × dims × 4 bytes (float32) + ~20% overhead (index, metadata)
        storage_gb = record_counts * dim * 4 * 1.2 / (1024**3)

        color = MODEL_COLORS.get(model, MOCHA["text"])
        ax.plot(record_counts, storage_gb, color=color, marker="o", markersize=4,
                linewidth=2, label=f"{model_display_name(model)} ({dim}d)")

    # Annotate PG shared_buffers recommendation
    ax.axhline(y=0.25, color=MOCHA["yellow"], linestyle=":", alpha=0.6, linewidth=1.5)
    ax.annotate("256MB shared_buffers (default PG)", xy=(record_counts[-1], 0.25),
                xytext=(-200, 15), textcoords="offset points",
                fontsize=8, color=MOCHA["yellow"], style="italic")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Bug Reports", fontsize=12, fontweight="bold")
    ax.set_ylabel("Storage (GB)", fontsize=12, fontweight="bold")
    ax.set_title("Storage Projection by Model Dimension",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    save_fig(fig, "fig_10_storage_projection")


if __name__ == "__main__":
    main()
