"""
Fig 1. Hero chart: F1 vs Latency scatter.
X = median latency, Y = F1 score, size = RAM/params, color = model.
"""

import csv
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from plot_config import setup_theme, MODEL_COLORS, save_fig, model_display_name, model_color, find_file_for_model, MOCHA

import matplotlib.pyplot as plt
import numpy as np

MODEL_PARAMS = {
    "qwen3_embedding": 0.6,
    "qwen3_embedding_4b": 4.0,
    "nomic_embed_text": 0.137,
    "mxbai_embed_large": 0.335,
    "bge_m3": 0.568,
    "all_minilm": 0.022,
    "snowflake_arctic_embed": 0.110,
}


def main():
    setup_theme()

    summary_path = "results/raw/model_summary.csv"
    if not os.path.exists(summary_path):
        print(f"Missing {summary_path}. Run sweep_threshold.py first.")
        return

    with open(summary_path) as f:
        rows = list(csv.DictReader(f))

    # Load latency data
    latency_data = {}
    for row in rows:
        model = row["model"]
        lat_path = find_file_for_model("latency_{model}.csv", model)
        if lat_path:
            with open(lat_path) as f:
                lat_rows = list(csv.DictReader(f))
            warm = [float(r["latency_per_item_ms"]) for r in lat_rows if r["pass_type"] == "warm"]
            latency_data[model] = np.median(warm) if warm else 50.0
        else:
            latency_data[model] = 50.0

    fig, ax = plt.subplots(figsize=(12, 8))

    for row in rows:
        model = row["model"]
        f1 = float(row["best_f1"])
        latency = latency_data.get(model, 50)
        params = MODEL_PARAMS.get(model, 0.1)

        color = model_color(model)
        size = max(params * 500, 80)

        ax.scatter(latency, f1, s=size, c=color, alpha=0.85, edgecolors="white",
                   linewidths=2, zorder=5)
        ax.annotate(model_display_name(model), (latency, f1),
                    textcoords="offset points", xytext=(14, 10),
                    fontsize=11, fontweight="bold", color=color)

    # Zoom Y-axis to show differences clearly
    f1_values = [float(r["best_f1"]) for r in rows]
    y_min = min(f1_values) - 0.002
    y_max = max(f1_values) + 0.002
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Median Latency per Embedding (ms)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Best F1 Score", fontsize=13, fontweight="bold")
    ax.set_title("Embedding Models for Bug Deduplication: F1 vs Latency",
                 fontsize=15, fontweight="bold", pad=15)

    ax.annotate("bubble size = model parameters", xy=(0.02, 0.02),
                xycoords="axes fraction", fontsize=9, color=MOCHA["overlay1"],
                style="italic")

    save_fig(fig, "fig_01_hero_scatter")


if __name__ == "__main__":
    main()
