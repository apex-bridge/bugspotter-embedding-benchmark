"""
Fig 6. MRL Truncation: Dims -> F1 line chart.
Lines for each MRL model. X = dimensions, Y = F1. Second axis: storage.
"""

import csv
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from plot_config import setup_theme, MODEL_COLORS, save_fig, model_display_name, MOCHA

import matplotlib.pyplot as plt
import numpy as np


def main():
    setup_theme()

    mrl_path = "results/raw/mrl_truncation.csv"
    if not os.path.exists(mrl_path):
        print(f"Missing {mrl_path}. Run mrl_truncation.py first.")
        return

    with open(mrl_path) as f:
        rows = list(csv.DictReader(f))

    by_model = {}
    for row in rows:
        model = row["model"]
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(row)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    for model, data in sorted(by_model.items()):
        dims = [int(r["dim"]) for r in data]
        f1s = [float(r["f1"]) for r in data]
        storage = [float(r["storage_100k_mb"]) for r in data]

        color = MODEL_COLORS.get(model, MOCHA["text"])
        name = model_display_name(model)

        ax1.plot(dims, f1s, color=color, marker="o", linewidth=2.5,
                 label=f"{name} (F1)", zorder=5, markersize=8)
        ax2.plot(dims, storage, color=color, marker="s", linewidth=1.5,
                 linestyle="--", alpha=0.4, markersize=5)

    ax1.set_xlabel("Embedding Dimensions", fontsize=13, fontweight="bold")
    ax1.set_ylabel("F1 Score", fontsize=13, fontweight="bold", color=MOCHA["blue"])
    ax2.set_ylabel("Storage per 100K (MB)", fontsize=13, fontweight="bold",
                   color=MOCHA["peach"])

    ax1.set_title("MRL Dimension Truncation: F1 vs Storage Trade-off",
                  fontsize=15, fontweight="bold", pad=15)

    # Annotate sweet spot (256 dims)
    ax1.axvline(x=256, color=MOCHA["yellow"], linestyle=":", alpha=0.7, linewidth=1.5)
    ax1.annotate("256 dims\n(sweet spot)", xy=(256, min(f1s)),
                 xytext=(300, min(f1s) - 0.001),
                 fontsize=10, color=MOCHA["yellow"], style="italic")

    ax1.legend(loc="lower right", fontsize=10)

    fig.tight_layout()
    save_fig(fig, "fig_06_mrl_truncation")


if __name__ == "__main__":
    main()
