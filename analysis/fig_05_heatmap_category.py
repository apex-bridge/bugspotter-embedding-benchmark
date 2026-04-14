"""
Fig 5. Heatmap: Model × Bug Category → F1.
Rows = 7 models, Columns = error types (js_error, network_error, css_ui, etc.).
"""

import csv
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from plot_config import setup_theme, save_fig, model_display_name, MOCHA

import matplotlib.pyplot as plt
import numpy as np
import json


def compute_f1_by_category(scores_path, reports_path, pairs_path):
    """Compute F1 per model per error_type category."""
    with open(reports_path, encoding="utf-8") as f:
        reports = {r["id"]: r for r in json.load(f)}

    with open(pairs_path) as f:
        pairs = {r["pair_id"]: r for r in csv.DictReader(f)}

    with open(scores_path) as f:
        scores = list(csv.DictReader(f))

    # Get model summaries for optimal thresholds
    summary_path = "results/raw/model_summary.csv"
    thresholds = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            for r in csv.DictReader(f):
                thresholds[r["model"]] = float(r["best_threshold"])

    # For each score, determine the error_type of the pair
    results = {}  # model -> error_type -> {tp, fp, fn}

    for row in scores:
        model = row["model"]
        pair = pairs.get(row["pair_id"])
        if not pair:
            continue

        a_report = reports.get(pair["report_a_id"], {})
        error_type = a_report.get("error_type", "unknown")
        threshold = thresholds.get(model, 0.80)

        predicted_dup = float(row["cosine_score"]) >= threshold
        actual_dup = row["label"] == "duplicate"

        if model not in results:
            results[model] = {}
        if error_type not in results[model]:
            results[model][error_type] = {"tp": 0, "fp": 0, "fn": 0}

        if predicted_dup and actual_dup:
            results[model][error_type]["tp"] += 1
        elif predicted_dup and not actual_dup:
            results[model][error_type]["fp"] += 1
        elif not predicted_dup and actual_dup:
            results[model][error_type]["fn"] += 1

    # Compute F1
    f1_matrix = {}
    for model, cats in results.items():
        f1_matrix[model] = {}
        for cat, counts in cats.items():
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            f1_matrix[model][cat] = round(f1, 3)

    return f1_matrix


def main():
    setup_theme()

    scores_path = "results/raw/similarity_scores.csv"
    reports_path = "data/bug_reports.json"
    pairs_path = "data/pairs_ground_truth.csv"

    for p in [scores_path, reports_path, pairs_path]:
        if not os.path.exists(p):
            print(f"Missing {p}")
            return

    f1_matrix = compute_f1_by_category(scores_path, reports_path, pairs_path)
    if not f1_matrix:
        print("No data to plot")
        return

    models = sorted(f1_matrix.keys())
    # Exclude 'github_issue' — these have no duplicate pairs, so F1=0 is misleading
    categories = sorted(cat for cat in set(cat for m in f1_matrix.values() for cat in m.keys())
                        if cat != "github_issue")

    # Build 2D array
    data = np.zeros((len(models), len(categories)))
    for i, model in enumerate(models):
        for j, cat in enumerate(categories):
            data[i, j] = f1_matrix.get(model, {}).get(cat, 0)

    fig, ax = plt.subplots(figsize=(max(10, len(categories) * 1.5), len(models) * 0.8 + 2))

    # Use 0.8-1.0 range to show differences (most values are 0.85-1.0)
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0.8, vmax=1.0)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([c.replace("_", " ").title() for c in categories],
                       rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([model_display_name(m) for m in models], fontsize=9)

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(categories)):
            val = data[i, j]
            color = MOCHA["base"] if val > 0.5 else MOCHA["text"]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="F1 Score", shrink=0.8)
    ax.set_title("F1 by Model × Bug Category", fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    save_fig(fig, "fig_05_heatmap_category")


if __name__ == "__main__":
    main()
