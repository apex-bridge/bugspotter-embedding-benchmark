"""
Fig 2. Precision-Recall curves for all 7 models.
Shows optimal threshold point for each model and the "danger zone" at 0.9.
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

    fig, ax = plt.subplots(figsize=(10, 8))

    for model, data in sorted(by_model.items()):
        recalls = [float(r["recall"]) for r in data]
        precisions = [float(r["precision"]) for r in data]
        f1s = [float(r["f1"]) for r in data]
        thresholds = [float(r["threshold"]) for r in data]

        color = MODEL_COLORS.get(model, MOCHA["text"])
        label = model_display_name(model)

        # Best F1 point
        best_idx = np.argmax(f1s)
        best_f1 = f1s[best_idx]
        best_t = thresholds[best_idx]

        linestyles = ["-", "--", "-.", ":", "-", "--", "-."]
        model_idx = list(sorted(by_model.keys())).index(model)

        # Top 3 models get thicker lines for visibility
        lw = 3.5 if best_f1 >= 0.995 else 2.0

        ax.plot(recalls, precisions, color=color,
                label=f"{label} (F1={best_f1:.3f} @{best_t})",
                linewidth=lw, alpha=0.95,
                linestyle=linestyles[model_idx % len(linestyles)])

        # Mark optimal point with direct label
        ax.scatter([recalls[best_idx]], [precisions[best_idx]], color=color,
                   s=100, zorder=5, edgecolors="white", linewidths=2, marker="o")
        ax.annotate(label, (recalls[best_idx], precisions[best_idx]),
                    textcoords="offset points", xytext=(-40, -15),
                    fontsize=8, fontweight="bold", color=color)

    ax.set_xlabel("Recall", fontsize=13, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=13, fontweight="bold")
    ax.set_title("Precision-Recall Curves",
                 fontsize=15, fontweight="bold", pad=15)
    ax.legend(loc="lower left", fontsize=9)

    # Zoom to the interesting region
    ax.set_xlim(0.85, 1.005)
    ax.set_ylim(0.85, 1.005)

    save_fig(fig, "fig_02_pr_curves")


if __name__ == "__main__":
    main()
