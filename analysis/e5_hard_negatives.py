"""
E5: Hard Negatives Deep Dive.

Analyzes only D3 pairs (different bugs, same component) — the hardest category.
For each model:
  - Confusion matrix at optimal threshold
  - Which specific pairs are misclassified (false positives = different bugs marked as duplicates)
  - F1 breakdown: D3 only vs D2 only vs overall
  - Identifies "deception patterns" — what makes a model confuse two different bugs?

Output:
  results/raw/e5_hard_negatives.csv    — per-model D3 metrics
  results/raw/e5_misclassified.csv     — specific pairs each model gets wrong
  results/figures/fig_e5_confusion.png  — confusion matrices
"""

import csv
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))
from plot_config import setup_theme, model_color, save_fig, model_display_name, normalize_model_name, MOCHA

import numpy as np
import matplotlib.pyplot as plt


def load_data(scores_path, reports_path=None):
    with open(scores_path, encoding="utf-8") as f:
        scores = list(csv.DictReader(f))

    reports = {}
    if reports_path and os.path.exists(reports_path):
        with open(reports_path, encoding="utf-8") as f:
            reports = {r["id"]: r for r in json.load(f)}

    return scores, reports


def load_thresholds(summary_path):
    thresholds = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            for r in csv.DictReader(f):
                thresholds[r["model"]] = float(r["best_threshold"])
    return thresholds


def analyze_model(model, scores, threshold, reports):
    """Analyze one model's performance on different pair types."""
    results = {"model": model}

    for pair_type in ["D1", "D2", "D3", "D4"]:
        type_scores = [s for s in scores if s["model"] == model and s["pair_type"] == pair_type]
        if not type_scores:
            continue

        cosines = np.array([float(s["cosine_score"]) for s in type_scores])
        labels = np.array([1 if s["label"] == "duplicate" else 0 for s in type_scores])
        predicted = (cosines >= threshold).astype(int)

        tp = int(((predicted == 1) & (labels == 1)).sum())
        fp = int(((predicted == 1) & (labels == 0)).sum())
        fn = int(((predicted == 0) & (labels == 1)).sum())
        tn = int(((predicted == 0) & (labels == 0)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        results[f"{pair_type}_tp"] = tp
        results[f"{pair_type}_fp"] = fp
        results[f"{pair_type}_fn"] = fn
        results[f"{pair_type}_tn"] = tn
        results[f"{pair_type}_precision"] = round(prec, 4)
        results[f"{pair_type}_recall"] = round(rec, 4)
        results[f"{pair_type}_f1"] = round(f1, 4)
        results[f"{pair_type}_count"] = len(type_scores)

    return results


def load_pairs_map(base_dir):
    """Load pairs CSV and build pair_id -> {report_a_id, report_b_id} map."""
    # Try multiple locations
    for path in [
        os.path.join(base_dir, "../../data/pairs_ground_truth.csv"),
        "data/pairs_ground_truth.csv",
    ]:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return {r["pair_id"]: r for r in csv.DictReader(f)}
    return {}


def find_misclassified(model, scores, threshold, reports, pairs_map):
    """Find specific D3 pairs that the model misclassifies."""
    misclassified = []

    for pair_type, error_label in [("D3", "false_positive"), ("D2", "false_negative")]:
        type_scores = [s for s in scores if s["model"] == model and s["pair_type"] == pair_type]
        for s in type_scores:
            cosine = float(s["cosine_score"])
            is_dup = s["label"] == "duplicate"
            predicted_dup = cosine >= threshold

            is_mistake = (predicted_dup and not is_dup) if pair_type == "D3" else (not predicted_dup and is_dup)
            if not is_mistake:
                continue

            pair = pairs_map.get(s["pair_id"], {})
            a_id = pair.get("report_a_id", "")
            b_id = pair.get("report_b_id", "")
            a = reports.get(a_id, {})
            b = reports.get(b_id, {})

            misclassified.append({
                "model": model,
                "pair_id": s["pair_id"],
                "error_type": error_label,
                "cosine_score": cosine,
                "threshold": threshold,
                "report_a_id": a_id,
                "report_b_id": b_id,
                "title_a": a.get("title", "")[:80],
                "title_b": b.get("title", "")[:80],
                "component_a": a.get("component", a.get("error_type", "")),
                "component_b": b.get("component", b.get("error_type", "")),
            })

    return misclassified


def plot_confusion_matrices(all_results, output_dir):
    """Plot confusion matrices for D3 pairs per model."""
    setup_theme()

    models = [r["model"] for r in all_results if f"D3_tp" in r]
    n = len(models)
    if n == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (result, ax) in enumerate(zip(all_results, axes[:n])):
        model = result["model"]
        tp = result.get("D3_tp", 0)
        fp = result.get("D3_fp", 0)
        fn = result.get("D3_fn", 0)
        tn = result.get("D3_tn", 0)

        matrix = np.array([[tn, fp], [fn, tp]])
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color=MOCHA["base"] if matrix[i, j] > max(matrix.flatten()) * 0.5 else MOCHA["text"])

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Not-dup", "Dup"], fontsize=9)
        ax.set_yticklabels(["Not-dup", "Dup"], fontsize=9)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        fp = result.get("D3_fp", 0)
        ax.set_title(f"{model_display_name(model)}\nD3: {fp} false positive{'s' if fp != 1 else ''}",
                      fontsize=11, fontweight="bold")

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("D3: False Positive Analysis (600 not-duplicate pairs, same component)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, "fig_e5_confusion", output_dir)


def plot_f1_by_pair_type(all_results, output_dir):
    """Bar chart: F1 per pair type per model."""
    setup_theme()

    models = [r["model"] for r in all_results]
    pair_types = ["D1", "D2"]
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [MOCHA["green"], MOCHA["blue"], MOCHA["peach"]]
    for i, pt in enumerate(pair_types):
        values = [r.get(f"{pt}_f1", 0) for r in all_results]
        bars = ax.bar(x + i * width, values, width, label=pt, color=colors[i],
                      edgecolor=MOCHA["surface1"], linewidth=0.5)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7,
                        color=MOCHA["text"])

    ax.set_xticks(x + width)
    ax.set_xticklabels([model_display_name(r["model"]) for r in all_results], fontsize=10)
    ax.set_ylabel("F1 Score", fontsize=12, fontweight="bold")
    ax.set_title("F1 by Pair Type: Exact Duplicates (D1) vs Paraphrases (D2)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    save_fig(fig, "fig_e5_f1_by_type", output_dir)


def main():
    # Try Hetzner results first, fall back to local
    for base in ["results-hetzner/results-hetzner", "results"]:
        scores_path = os.path.join(base, "raw/similarity_scores.csv")
        summary_path = os.path.join(base, "raw/model_summary.csv")
        if os.path.exists(scores_path):
            break
    else:
        print("No similarity_scores.csv found.")
        return

    reports_path = "data/bug_reports.json"
    output_dir = os.path.join(base, "figures")
    raw_dir = os.path.join(base, "raw")

    print(f"Using data from: {base}")
    scores, reports = load_data(scores_path, reports_path)
    thresholds = load_thresholds(summary_path)

    print(f"Loaded {len(scores)} scores, {len(reports)} reports, {len(thresholds)} thresholds")

    pairs_map = load_pairs_map(base)
    print(f"Loaded {len(pairs_map)} pairs for misclassification lookup")

    models = sorted(set(s["model"] for s in scores))
    all_results = []
    all_misclassified = []

    for model in models:
        threshold = thresholds.get(model, 0.70)
        print(f"\n{'='*50}")
        print(f"Model: {model_display_name(model)} (threshold={threshold})")

        result = analyze_model(model, scores, threshold, reports)
        all_results.append(result)

        # Print D3 vs D2 comparison
        d3_f1 = result.get("D3_f1", 0)
        d2_f1 = result.get("D2_f1", 0)
        d3_fp = result.get("D3_fp", 0)
        d2_fn = result.get("D2_fn", 0)
        print(f"  D2 (semantic dup) F1: {d2_f1:.3f} — missed {d2_fn} real duplicates")
        print(f"  D3 (hard neg)     F1: {d3_f1:.3f} — {d3_fp} false positives (confused different bugs)")

        misclassified = find_misclassified(model, scores, threshold, reports, pairs_map)
        all_misclassified.extend(misclassified)

        fp_count = sum(1 for m in misclassified if m["error_type"] == "false_positive")
        fn_count = sum(1 for m in misclassified if m["error_type"] == "false_negative")
        print(f"  Misclassified: {fp_count} false positives, {fn_count} false negatives")

        # Show worst false positives
        fps = sorted([m for m in misclassified if m["error_type"] == "false_positive"],
                     key=lambda m: -m["cosine_score"])
        if fps:
            print(f"  Worst false positive (cosine={fps[0]['cosine_score']:.3f}):")
            print(f"    A: {fps[0]['title_a']}")
            print(f"    B: {fps[0]['title_b']}")

    # Save results
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if all_results:
        with open(os.path.join(raw_dir, "e5_hard_negatives.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nSaved: {raw_dir}/e5_hard_negatives.csv")

    if all_misclassified:
        with open(os.path.join(raw_dir, "e5_misclassified.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_misclassified[0].keys())
            writer.writeheader()
            writer.writerows(all_misclassified)
        print(f"Saved: {raw_dir}/e5_misclassified.csv ({len(all_misclassified)} rows)")

    # Generate plots
    plot_confusion_matrices(all_results, output_dir)
    plot_f1_by_pair_type(all_results, output_dir)


if __name__ == "__main__":
    main()
