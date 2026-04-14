"""
Fig 9. Summary table (shareable as image).
Model, Params, F1, AUC, Optimal Threshold, Latency (CPU), RAM, Dims.
"""

import csv
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from plot_config import setup_theme, save_fig, model_display_name, find_file_for_model, normalize_model_name, MOCHA

import matplotlib.pyplot as plt
import numpy as np

MODEL_META = {
    "qwen3-embedding": {"params": "7.6B Q4", "dims": "4096"},
    "qwen3-embedding_4b": {"params": "4B", "dims": "4096"},
    "nomic-embed-text": {"params": "137M", "dims": "768"},
    "mxbai-embed-large": {"params": "335M", "dims": "1024"},
    "bge-m3": {"params": "568M", "dims": "1024"},
    "all-minilm": {"params": "22M", "dims": "384"},
    "snowflake-arctic-embed": {"params": "334M", "dims": "768"},
}


def main():
    setup_theme()

    summary_path = "results/raw/model_summary.csv"
    if not os.path.exists(summary_path):
        print(f"Missing {summary_path}")
        return

    with open(summary_path) as f:
        rows = sorted(list(csv.DictReader(f)), key=lambda r: -float(r["best_f1"]))

    # Build table data
    col_labels = ["Model", "Params", "Dims", "F1", "AUC", "Threshold",
                  "Recall@0.9", "Latency (ms)"]
    cell_data = []
    cell_colors = []

    for row in rows:
        model = row["model"]
        meta = MODEL_META.get(normalize_model_name(model), {"params": "?", "dims": "?"})

        # Try to get latency
        lat_path = find_file_for_model("latency_{model}.csv", model)
        latency = "—"
        if lat_path:
            with open(lat_path) as f:
                lat_rows = list(csv.DictReader(f))
            warm = [float(r["latency_per_item_ms"]) for r in lat_rows if r["pass_type"] == "warm"]
            if warm:
                latency = f"{np.median(warm):.1f}"

        cell_row = [
            model_display_name(model),
            meta["params"],
            meta["dims"],
            f"{float(row['best_f1']):.3f}",
            f"{float(row['roc_auc']):.3f}",
            f"{float(row['best_threshold']):.2f}",
            f"{float(row['recall_at_0.90']):.3f}",
            latency,
        ]
        cell_data.append(cell_row)

        # Color code F1
        f1 = float(row["best_f1"])
        if f1 >= 0.85:
            color = [MOCHA["green"] + "33"] * len(col_labels)
        elif f1 >= 0.75:
            color = [MOCHA["blue"] + "22"] * len(col_labels)
        else:
            color = [MOCHA["surface0"]] * len(col_labels)
        cell_colors.append(color)

    fig, ax = plt.subplots(figsize=(14, len(rows) * 0.6 + 2))
    ax.axis("off")

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=[MOCHA["surface1"]] * len(col_labels),
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_text_props(fontweight="bold", color=MOCHA["text"])
        cell.set_facecolor(MOCHA["surface1"])

    # Style data cells
    for i in range(len(cell_data)):
        for j in range(len(col_labels)):
            cell = table[i + 1, j]
            cell.set_text_props(color=MOCHA["text"])

    ax.set_title("Embedding Model Benchmark Results (sorted by F1)",
                 fontsize=14, fontweight="bold", pad=20, color=MOCHA["text"])

    save_fig(fig, "fig_09_summary_table")


if __name__ == "__main__":
    main()
