"""
Shared plot configuration — Catppuccin Mocha theme for all figures.

Usage:
    from plot_config import setup_theme, COLORS, MODEL_COLORS, save_fig
    setup_theme()
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# Catppuccin Mocha palette
MOCHA = {
    "base": "#1e1e2e",
    "mantle": "#181825",
    "crust": "#11111b",
    "surface0": "#313244",
    "surface1": "#45475a",
    "surface2": "#585b70",
    "overlay0": "#6c7086",
    "overlay1": "#7f849c",
    "text": "#cdd6f4",
    "subtext0": "#a6adc8",
    "subtext1": "#bac2de",
    "red": "#f38ba8",
    "maroon": "#eba0ac",
    "peach": "#fab387",
    "yellow": "#f9e2af",
    "green": "#a6e3a1",
    "teal": "#94e2d5",
    "sky": "#89dceb",
    "sapphire": "#74c7ec",
    "blue": "#89b4fa",
    "lavender": "#b4befe",
    "mauve": "#cba6f7",
    "pink": "#f5c2e7",
    "flamingo": "#f2cdcd",
    "rosewater": "#f5e0dc",
}

# 7 model colors (distinct, accessible) — keys match Ollama model names
MODEL_COLORS = {
    "qwen3-embedding": MOCHA["blue"],
    "qwen3-embedding_4b": MOCHA["sapphire"],
    "nomic-embed-text": MOCHA["green"],
    "mxbai-embed-large": MOCHA["mauve"],
    "bge-m3": MOCHA["peach"],
    "all-minilm": MOCHA["yellow"],
    "snowflake-arctic-embed": MOCHA["teal"],
}

# General-purpose color cycle
COLORS = [
    MOCHA["blue"], MOCHA["green"], MOCHA["mauve"], MOCHA["peach"],
    MOCHA["teal"], MOCHA["yellow"], MOCHA["red"], MOCHA["pink"],
    MOCHA["sapphire"], MOCHA["lavender"], MOCHA["flamingo"], MOCHA["sky"],
]

# Pair type colors
PAIR_COLORS = {
    "D1": MOCHA["green"],
    "D2": MOCHA["blue"],
    "D3": MOCHA["peach"],
    "D4": MOCHA["red"],
}

# Store colors
STORE_COLORS = {
    "pgvector": MOCHA["blue"],
    "qdrant": MOCHA["mauve"],
    "chromadb": MOCHA["green"],
    "sqlite-vec": MOCHA["peach"],
}


def setup_theme():
    """Apply Catppuccin Mocha dark theme to matplotlib."""
    plt.style.use("dark_background")

    mpl.rcParams.update({
        # Figure
        "figure.facecolor": MOCHA["base"],
        "figure.edgecolor": MOCHA["base"],
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.facecolor": MOCHA["base"],
        "savefig.edgecolor": MOCHA["base"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,

        # Axes
        "axes.facecolor": MOCHA["mantle"],
        "axes.edgecolor": MOCHA["surface1"],
        "axes.labelcolor": MOCHA["text"],
        "axes.titlecolor": MOCHA["text"],
        "axes.grid": True,
        "axes.prop_cycle": mpl.cycler(color=COLORS),

        # Grid
        "grid.color": MOCHA["surface0"],
        "grid.alpha": 0.5,
        "grid.linewidth": 0.5,

        # Ticks
        "xtick.color": MOCHA["subtext0"],
        "ytick.color": MOCHA["subtext0"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,

        # Text
        "text.color": MOCHA["text"],
        "font.family": "sans-serif",
        "font.size": 10,

        # Legend
        "legend.facecolor": MOCHA["surface0"],
        "legend.edgecolor": MOCHA["surface1"],
        "legend.fontsize": 8,
        "legend.framealpha": 0.9,

        # Lines
        "lines.linewidth": 2,
        "lines.markersize": 6,
    })


def save_fig(fig, name: str, output_dir: str = "results/figures"):
    """Save figure as PNG @2x and SVG."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, f"{name}.svg"))
    print(f"Saved: {name}.png + {name}.svg")
    plt.close(fig)


def normalize_model_name(name: str) -> str:
    """Strip _latest suffix from Ollama model names for display/lookup."""
    return name.replace("_latest", "").replace(":latest", "")


def find_file_for_model(pattern: str, model: str, directory: str = "results/raw") -> str | None:
    """Find a file matching pattern for a model, trying with and without _latest."""
    import os
    for suffix in ["", "_latest"]:
        path = os.path.join(directory, pattern.format(model=model + suffix))
        if os.path.exists(path):
            return path
    return None


def model_display_name(model_key: str) -> str:
    """Convert model key to display name."""
    key = normalize_model_name(model_key)
    names = {
        "qwen3-embedding": "Qwen3 7.6B",
        "qwen3-embedding_4b": "Qwen3 4B",
        "nomic-embed-text": "nomic v1.5",
        "mxbai-embed-large": "mxbai-large",
        "bge-m3": "BGE-M3",
        "all-minilm": "MiniLM-L6",
        "snowflake-arctic-embed": "Snowflake",
    }
    return names.get(key, key)


def model_color(model_key: str) -> str:
    """Get color for a model, handling _latest suffix."""
    key = normalize_model_name(model_key)
    return MODEL_COLORS.get(key, MOCHA["text"])
