"""
Step 5: Threshold sweep — compute P, R, F1, AUC for each model.

Sweeps cosine similarity threshold from 0.50 to 0.99 with step 0.01.
Computes precision, recall, F1 at each threshold. Finds optimal threshold (max F1).
Also computes ROC-AUC.

Output:
  results/raw/threshold_sweep.csv   — per-model, per-threshold metrics
  results/raw/model_summary.csv     — best threshold, F1, AUC per model
"""

import csv
import os
import argparse

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve


def load_scores(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sweep_thresholds(scores: np.ndarray, labels: np.ndarray,
                     start: float = 0.50, stop: float = 1.00, step: float = 0.01):
    """Sweep threshold and compute metrics at each point."""
    results = []
    for threshold in np.arange(start, stop, step):
        predicted = (scores >= threshold).astype(int)
        tp = int(((predicted == 1) & (labels == 1)).sum())
        fp = int(((predicted == 1) & (labels == 0)).sum())
        fn = int(((predicted == 0) & (labels == 1)).sum())
        tn = int(((predicted == 0) & (labels == 0)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        results.append({
            "threshold": round(threshold, 2),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Threshold sweep")
    parser.add_argument("--input", default="results/raw/similarity_scores.csv")
    parser.add_argument("--output-sweep", default="results/raw/threshold_sweep.csv")
    parser.add_argument("--output-summary", default="results/raw/model_summary.csv")
    args = parser.parse_args()

    all_scores = load_scores(args.input)
    print(f"Loaded {len(all_scores)} similarity scores")

    # Group by model
    by_model = {}
    for row in all_scores:
        model = row["model"]
        if model not in by_model:
            by_model[model] = {"scores": [], "labels": [], "pair_types": []}
        by_model[model]["scores"].append(float(row["cosine_score"]))
        by_model[model]["labels"].append(1 if row["label"] == "duplicate" else 0)
        by_model[model]["pair_types"].append(row["pair_type"])

    sweep_rows = []
    summary_rows = []

    for model, data in sorted(by_model.items()):
        scores = np.array(data["scores"])
        labels = np.array(data["labels"])

        print(f"\n{'='*50}")
        print(f"Model: {model}")
        print(f"  Pairs: {len(scores)} (dup={labels.sum()}, neg={(1-labels).sum()})")

        # Threshold sweep
        results = sweep_thresholds(scores, labels)
        for r in results:
            r["model"] = model
        sweep_rows.extend(results)

        # Find optimal threshold (max F1)
        best = max(results, key=lambda r: r["f1"])
        print(f"  Best threshold: {best['threshold']} => "
              f"P={best['precision']:.3f} R={best['recall']:.3f} F1={best['f1']:.3f}")

        # ROC-AUC
        try:
            auc = roc_auc_score(labels, scores)
        except ValueError:
            auc = 0.0
        print(f"  ROC-AUC: {auc:.4f}")

        # Recall@k (using threshold-based approach)
        # What would we lose at threshold=0.90?
        at_90 = next((r for r in results if r["threshold"] == 0.90), None)
        recall_at_90 = at_90["recall"] if at_90 else 0.0
        print(f"  Recall@0.90: {recall_at_90:.3f} "
              f"({'would miss ' + str(int((1-recall_at_90)*labels.sum())) + ' duplicates' if recall_at_90 < 1 else 'perfect'})")

        summary_rows.append({
            "model": model,
            "best_threshold": best["threshold"],
            "best_f1": best["f1"],
            "best_precision": best["precision"],
            "best_recall": best["recall"],
            "roc_auc": round(auc, 4),
            "recall_at_0.90": round(recall_at_90, 4),
            "total_pairs": len(scores),
            "total_duplicates": int(labels.sum()),
        })

    # Save sweep
    os.makedirs(os.path.dirname(args.output_sweep) or ".", exist_ok=True)
    fieldnames = ["model", "threshold", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]
    with open(args.output_sweep, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sweep_rows)
    print(f"\nThreshold sweep saved to {args.output_sweep}")

    # Save summary
    with open(args.output_summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Model summary saved to {args.output_summary}")


if __name__ == "__main__":
    main()
