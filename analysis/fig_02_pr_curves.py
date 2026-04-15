"""
Fig 2. Precision-Recall curves for all 6 models.
Shows optimal threshold point for each model.
"""

import csv
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from plot_config import setup_theme, MODEL_COLORS, save_fig, model_display_name, MOCHA, model_color

import matplotlib.pyplot as plt
import numpy as np


# Fixed label positions in DATA coordinates (recall, precision)
# Spread out around the chart edges so nothing overlaps
LABEL_POSITIONS = {
    "all-minilm": (0.935, 0.997),
    "bge-m3": (0.995, 0.960),
    "mxbai-embed-large": (0.965, 0.960),
    "nomic-embed-text": (0.935, 0.970),
    "qwen3-embedding": (0.995, 0.997),
    "snowflake-arctic-embed": (0.935, 0.955),
}


def main():
    setup_theme()

    sweep_path = "results/raw/threshold_sweep.csv"
    if not os.path.exists(sweep_path):
        print(f"Missing {sweep_path}. Run sweep_threshold.py first.")
        return

    with open(sweep_path) as f:
        rows = list(csv.DictReader(f))

    # Group by model
    by_model = {}
    for row in rows:
        model = row["model"]
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(row)

    fig, ax_zoom = plt.subplots(figsize=(10, 8))

    # Sort models by best F1 descending for consistent legend order
    model_order = []
    for model, data in by_model.items():
        f1s = [float(r["f1"]) for r in data]
        model_order.append((model, max(f1s)))
    model_order.sort(key=lambda x: -x[1])

    linestyles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]

    for idx, (model, _) in enumerate(model_order):
        data = by_model[model]
        recalls = [float(r["recall"]) for r in data]
        precisions = [float(r["precision"]) for r in data]
        f1s = [float(r["f1"]) for r in data]
        thresholds = [float(r["threshold"]) for r in data]

        color = model_color(model)
        label = model_display_name(model)
        best_idx = np.argmax(f1s)
        best_f1 = f1s[best_idx]
        best_t = thresholds[best_idx]

        lw = 3.0 if best_f1 >= 0.987 else 2.0
        ls = linestyles[idx % len(linestyles)]

        ax_zoom.plot(recalls, precisions, color=color,
                    label=f"{label} (F1={best_f1:.3f} @{best_t})",
                    linewidth=lw, alpha=0.9, linestyle=ls)

        ax_zoom.scatter([recalls[best_idx]], [precisions[best_idx]], color=color,
                       s=100, zorder=5, edgecolors="white", linewidths=2, marker="o")


    ax_zoom.set_xlim(0.92, 1.002)
    ax_zoom.set_ylim(0.92, 1.002)
    ax_zoom.set_xlabel("Recall", fontsize=13, fontweight="bold")
    ax_zoom.set_ylabel("Precision", fontsize=13, fontweight="bold")
    ax_zoom.set_title("Precision-Recall Curves",
                      fontsize=15, fontweight="bold", pad=15)
    ax_zoom.legend(loc="lower left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    save_fig(fig, "fig_02_pr_curves")


if __name__ == "__main__":
    main()
