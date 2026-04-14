"""
Fig 4. Cosine distribution violin plot.
For each model: distribution of cosine similarity per pair type (D1-D4).
"""

import csv
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from plot_config import setup_theme, PAIR_COLORS, save_fig, model_display_name, MOCHA

import matplotlib.pyplot as plt
import numpy as np


def main():
    setup_theme()

    scores_path = "results/raw/similarity_scores.csv"
    if not os.path.exists(scores_path):
        print(f"Missing {scores_path}")
        return

    with open(scores_path) as f:
        rows = list(csv.DictReader(f))

    models = sorted(set(r["model"] for r in rows))
    pair_types = ["D1", "D2", "D3", "D4"]

    fig, axes = plt.subplots(1, len(models), figsize=(3 * len(models), 6),
                             sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        data_per_type = []
        colors = []
        labels = []

        for pt in pair_types:
            scores = [float(r["cosine_score"]) for r in rows
                      if r["model"] == model and r["pair_type"] == pt]
            if scores:
                data_per_type.append(scores)
                colors.append(PAIR_COLORS[pt])
                labels.append(pt)

        if data_per_type:
            parts = ax.violinplot(data_per_type, showmedians=True, showextrema=False)
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.7)
            parts["cmedians"].set_color(MOCHA["text"])

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(model_display_name(model), fontsize=9, fontweight="bold")

    axes[0].set_ylabel("Cosine Similarity", fontsize=11, fontweight="bold")

    fig.suptitle("Cosine Similarity Distributions by Pair Type",
                 fontsize=14, fontweight="bold", y=1.02)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=PAIR_COLORS[pt], alpha=0.7, label=f"{pt}")
                       for pt in pair_types]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    save_fig(fig, "fig_04_cosine_violin")


if __name__ == "__main__":
    main()
