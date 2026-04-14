"""
Fig 3. ROC curves + AUC for all models.
"""

import csv
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from plot_config import setup_theme, MODEL_COLORS, save_fig, model_display_name, MOCHA

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


def main():
    setup_theme()

    scores_path = "results/raw/similarity_scores.csv"
    if not os.path.exists(scores_path):
        print(f"Missing {scores_path}")
        return

    with open(scores_path) as f:
        rows = list(csv.DictReader(f))

    by_model = {}
    for row in rows:
        model = row["model"]
        if model not in by_model:
            by_model[model] = {"scores": [], "labels": []}
        by_model[model]["scores"].append(float(row["cosine_score"]))
        by_model[model]["labels"].append(1 if row["label"] == "duplicate" else 0)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Random baseline
    ax.plot([0, 1], [0, 1], linestyle="--", color=MOCHA["overlay0"],
            label="Random (AUC=0.50)", linewidth=1)

    for model in sorted(by_model.keys()):
        data = by_model[model]
        scores = np.array(data["scores"])
        labels = np.array(data["labels"])

        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)

        color = MODEL_COLORS.get(model, MOCHA["text"])
        ax.plot(fpr, tpr, color=color,
                label=f"{model_display_name(model)} (AUC={auc:.3f})",
                linewidth=2, alpha=0.85)

    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title("ROC Curves — Bug Report Deduplication",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    save_fig(fig, "fig_03_roc_curves")


if __name__ == "__main__":
    main()
