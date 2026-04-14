"""
Fig 12. Decision tree flowchart: "Which vector store to choose?"
Redrawn for cleaner visual quality.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from plot_config import setup_theme, save_fig, MOCHA, STORE_COLORS

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def draw_box(ax, x, y, text, color, width=2.4, height=0.7, fontsize=9):
    """Draw a rounded rectangle with text."""
    box = mpatches.FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.2", facecolor=color, edgecolor=MOCHA["surface2"],
        linewidth=2, alpha=0.9, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=MOCHA["text"], zorder=4)


def draw_line(ax, x1, y1, x2, y2, label=""):
    """Draw a line between two points with label."""
    ax.plot([x1, x2], [y1, y2], color=MOCHA["overlay1"], linewidth=2, zorder=1)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.15, my + 0.1, label, fontsize=10, color=MOCHA["text"],
                fontweight="bold", zorder=4)


def main():
    setup_theme()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-0.5, 7.5)
    ax.axis("off")

    # Level 1: First question
    draw_box(ax, 5, 6.5, "Already using\nPostgreSQL?", MOCHA["surface0"], width=3, height=0.9, fontsize=11)

    # Yes -> pgvector
    draw_line(ax, 3.5, 6.05, 2, 5.3, "Yes")
    draw_box(ax, 2, 4.8, "pgvector", STORE_COLORS["pgvector"], width=2.5, height=0.7, fontsize=11)
    ax.text(2, 4.1, "Zero additional infra\nSQL + vectors in one DB\n0.9ms queries", ha="center",
            fontsize=9, color=MOCHA["subtext0"], style="italic")

    # No -> next question
    draw_line(ax, 6.5, 6.05, 8, 5.3, "No")
    draw_box(ax, 8, 4.8, "Need zero\ndependencies?", MOCHA["surface0"], width=3, height=0.9, fontsize=11)

    # Yes -> sqlite-vec
    draw_line(ax, 6.5, 4.35, 5.5, 3.3, "Yes")
    draw_box(ax, 5.5, 2.8, "sqlite-vec", STORE_COLORS["sqlite-vec"], width=2.5, height=0.7, fontsize=11)
    ax.text(5.5, 2.1, "One .db file, 0.5MB RAM\nPerfect for solo dev\n1.05ms queries", ha="center",
            fontsize=9, color=MOCHA["subtext0"], style="italic")

    # No -> next question
    draw_line(ax, 9.5, 4.35, 9.5, 3.3, "No")
    draw_box(ax, 9.5, 2.8, ">100K records?", MOCHA["surface0"], width=2.8, height=0.7, fontsize=11)

    # Yes -> Qdrant
    draw_line(ax, 8.1, 2.35, 7.5, 1.3, "Yes")
    draw_box(ax, 7.5, 0.8, "Qdrant", STORE_COLORS["qdrant"], width=2.2, height=0.7, fontsize=11)
    ax.text(7.5, 0.1, "Rust + SIMD\nPayload filtering\nScales to millions", ha="center",
            fontsize=9, color=MOCHA["subtext0"], style="italic")

    # No -> ChromaDB
    draw_line(ax, 10.9, 2.35, 10.9, 1.3, "No")
    draw_box(ax, 10.9, 0.8, "ChromaDB", STORE_COLORS["chromadb"], width=2.2, height=0.7, fontsize=11)
    ax.text(10.9, 0.1, "pip install chromadb\nQuick prototype\n5 min setup", ha="center",
            fontsize=9, color=MOCHA["subtext0"], style="italic")

    ax.set_title("Which Vector Store Should You Use?",
                 fontsize=18, fontweight="bold", pad=20, color=MOCHA["text"])

    save_fig(fig, "fig_12_decision_tree")


if __name__ == "__main__":
    main()
