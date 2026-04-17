"""
Single source of truth for all numbers in the article and README.

Run this script to verify that article/README numbers match the raw data.
If any number is wrong, this script shows what it should be.

Usage:
    python analysis/article_numbers.py
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from plot_config import model_display_name, normalize_model_name


def bootstrap_f1(scores, labels, threshold, n=1000, seed=42):
    """Bootstrap 95% CI for F1 at a fixed threshold."""
    rng = np.random.RandomState(seed)
    f1s = []
    for _ in range(n):
        idx = rng.choice(len(scores), size=len(scores), replace=True)
        s, l = scores[idx], labels[idx]
        pred = (s >= threshold).astype(int)
        tp = ((pred == 1) & (l == 1)).sum()
        fp = ((pred == 1) & (l == 0)).sum()
        fn = ((pred == 0) & (l == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1s.append(f1)
    return np.percentile(f1s, 2.5), np.percentile(f1s, 97.5)


def main():
    # Find data
    for base in ["results/runs/seed_42", "results-v2", "results"]:
        scores_path = os.path.join(base, "raw", "similarity_scores.csv") if "runs" not in base else os.path.join(base, "similarity_scores.csv")
        summary_path = os.path.join(base, "raw", "model_summary.csv") if "runs" not in base else os.path.join(base, "model_summary.csv")
        if os.path.exists(scores_path):
            break
    else:
        print("No similarity_scores.csv found")
        return

    # Load scores
    with open(scores_path, encoding="utf-8") as f:
        all_scores = list(csv.DictReader(f))
    print(f"Source: {scores_path} ({len(all_scores)} scores)")

    # Load summary (for thresholds)
    cv_thresholds = {}
    if os.path.exists("results/test_summary.csv"):
        with open("results/test_summary.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cv_thresholds[r["model"]] = {
                    "cv_f1": float(r["cv_f1_mean"]),
                    "cv_threshold": float(r["cv_threshold"]),
                }

    # Group by model
    from collections import defaultdict
    by_model = defaultdict(lambda: {"scores": [], "labels": [], "pair_types": []})
    for s in all_scores:
        m = s["model"]
        by_model[m]["scores"].append(float(s["cosine_score"]))
        by_model[m]["labels"].append(1 if s["label"] == "duplicate" else 0)
        by_model[m]["pair_types"].append(s.get("pair_type", ""))

    # Main table
    print("\n" + "=" * 80)
    print("MAIN TABLE (use these in article and README)")
    print("=" * 80)
    print(f"{'Model':<25} {'CV F1':<8} {'Threshold':<10} {'95% CI':<22} {'Recall@0.9':<10}")
    print("-" * 80)

    embedding_models = [m for m in sorted(by_model.keys()) if "bm25" not in m and "tfidf" not in m]
    baseline_models = [m for m in sorted(by_model.keys()) if "bm25" in m or "tfidf" in m]

    for model in embedding_models:
        data = by_model[model]
        scores = np.array(data["scores"])
        labels = np.array(data["labels"])
        name = normalize_model_name(model)

        cv = cv_thresholds.get(model, {})
        cv_f1 = cv.get("cv_f1", 0)
        threshold = cv.get("cv_threshold", 0.5)

        ci_lo, ci_hi = bootstrap_f1(scores, labels, threshold)
        recall_09 = ((scores >= 0.9) & (labels == 1)).sum() / labels.sum() if labels.sum() > 0 else 0

        print(f"{model_display_name(name):<25} {cv_f1:.3f}   {threshold:<10} [{ci_lo:.3f}, {ci_hi:.3f}]   {recall_09:.1%}")

    print()
    for model in baseline_models:
        data = by_model[model]
        scores = np.array(data["scores"])
        labels = np.array(data["labels"])

        # Sweep for best F1
        best_f1, best_t = 0, 0.5
        for t in np.arange(0.5, 1.0, 0.01):
            pred = (scores >= t).astype(int)
            tp = ((pred == 1) & (labels == 1)).sum()
            fp = ((pred == 1) & (labels == 0)).sum()
            fn = ((pred == 0) & (labels == 1)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            if f1 > best_f1:
                best_f1, best_t = f1, t

        print(f"{model:<25} {best_f1:.3f}   {best_t:<10}")

    print("\n" + "=" * 80)
    print("HARD NEGATIVES (D3 FP / D2 FN)")
    print("=" * 80)

    # Load pairs for pair_type
    pairs_path = "data/pairs_ground_truth.csv"
    if os.path.exists(pairs_path):
        with open(pairs_path, encoding="utf-8") as f:
            pair_types = {p["pair_id"]: p["pair_type"] for p in csv.DictReader(f)}
    else:
        pair_types = {}

    for model in embedding_models:
        cv = cv_thresholds.get(model, {})
        threshold = cv.get("cv_threshold", 0.5)
        name = normalize_model_name(model)

        d3_fp = sum(1 for s in all_scores if s["model"] == model
                    and pair_types.get(s["pair_id"]) == "D3"
                    and float(s["cosine_score"]) >= threshold)
        d2_fn = sum(1 for s in all_scores if s["model"] == model
                    and pair_types.get(s["pair_id"]) == "D2"
                    and float(s["cosine_score"]) < threshold)

        print(f"{model_display_name(name):<25} D3 FP={d3_fp:<4} D2 FN={d2_fn:<4} Total={d3_fp + d2_fn}")

    print("\nVerify these match the article. If not, update the article.")


if __name__ == "__main__":
    main()
