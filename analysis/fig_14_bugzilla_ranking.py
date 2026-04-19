"""
Fig 14. Bugzilla F1 ranking — all 8 methods (6 embeddings + BM25 + TF-IDF) on Mozilla Bugzilla.

Headline finding: BM25 scores 0.954 on Bugzilla, beating 4 of 6 embedding models.
"""

import csv
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from plot_config import setup_theme, save_fig, model_display_name, model_color, MOCHA

import matplotlib.pyplot as plt


def main():
    setup_theme()

    rows = []

    # Load embedding Bugzilla F1s
    emb_path = "results/raw/bugzilla_summary.csv"
    if not os.path.exists(emb_path):
        print(f"Missing {emb_path}. Run bugzilla_validation.py first.")
        return
    with open(emb_path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "model": r["model"],
                "f1": float(r["oracle_f1"]),
                "kind": "embedding",
            })

    # Load BM25/TF-IDF Bugzilla F1s
    bm25_path = "results/raw/bugzilla_bm25_summary.csv"
    if os.path.exists(bm25_path):
        with open(bm25_path) as f:
            for r in csv.DictReader(f):
                rows.append({
                    "model": r["model"],
                    "f1": float(r["oracle_f1"]),
                    "kind": "baseline",
                })
    else:
        print(f"Warning: {bm25_path} missing — run benchmark/bm25_bugzilla.py first.")

    # Sort by F1 descending
    rows.sort(key=lambda r: r["f1"], reverse=True)

    # Display labels
    BASELINE_DISPLAY = {"tfidf_baseline": "TF-IDF", "bm25_baseline": "BM25"}

    def display_name(model, kind):
        if kind == "baseline":
            return BASELINE_DISPLAY.get(model, model)
        return model_display_name(model)

    def color_for(model, kind):
        if kind == "baseline":
            return MOCHA["subtext1"]  # light gray for lexical baselines
        return model_color(model)

    labels = [display_name(r["model"], r["kind"]) for r in rows]
    f1s = [r["f1"] for r in rows]
    colors = [color_for(r["model"], r["kind"]) for r in rows]

    fig, ax = plt.subplots(figsize=(11, 6.5))

    y_positions = list(range(len(rows)))[::-1]  # reverse so top F1 is at top
    bars = ax.barh(y_positions, f1s, color=colors, alpha=0.9,
                    edgecolor="white", linewidth=1.5)

    # F1 value labels at bar ends
    for i, (y, f1, r) in enumerate(zip(y_positions, f1s, rows)):
        label_style = dict(va="center", fontsize=11, fontweight="bold")
        # Hollow arrow/suffix for baselines to mark them visually
        suffix = "  ◆" if r["kind"] == "baseline" else ""
        ax.text(f1 + 0.002, y, f"{f1:.3f}{suffix}",
                color=color_for(r["model"], r["kind"]), **label_style)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=12, fontweight="bold")
    ax.set_xlabel("F1 on Mozilla Bugzilla (350 labeled pairs)",
                  fontsize=12, fontweight="bold")
    ax.set_xlim(0.85, 1.00)
    ax.set_title("Cross-Validation on Mozilla Bugzilla: All Methods Ranked",
                 fontsize=14, fontweight="bold", pad=15)

    # Subtitle note explaining the baseline marker
    ax.annotate("◆ = lexical baseline (BM25, TF-IDF) — no model, no infrastructure",
                xy=(0.02, -0.14), xycoords="axes fraction",
                fontsize=10, color=MOCHA["overlay1"], style="italic")

    # Horizontal "break-even" annotation: where does BM25 sit relative to embeddings?
    bm25_f1 = next((r["f1"] for r in rows if r["model"] == "bm25_baseline"), None)
    if bm25_f1 is not None:
        ax.axvline(bm25_f1, color=MOCHA["subtext1"], linestyle="--",
                   alpha=0.35, linewidth=1, zorder=1)
        ax.annotate(
            "BM25 baseline — beats 4 of 6 embedding models",
            xy=(bm25_f1, len(rows) - 0.5),
            xytext=(bm25_f1 - 0.022, len(rows) - 0.3),
            fontsize=9, color=MOCHA["subtext1"], style="italic",
            ha="center",
        )

    plt.tight_layout()
    save_fig(fig, "fig_14_bugzilla_ranking")


if __name__ == "__main__":
    main()
