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


def archetype_cv_threshold(model_scores, pairs_path="data/pairs_ground_truth.csv",
                           reports_path="data/bug_reports.json", n_folds=5, seed=42):
    """Archetype-level CV: hold out entire archetype groups, not random pairs.

    This tests whether the threshold generalizes to unseen bug types,
    not just unseen pairs of the same bugs. Much harder test.
    """
    import json
    import csv

    # Load group info
    with open(reports_path, encoding="utf-8") as f:
        report_groups = {r["id"]: r.get("group", "") for r in json.load(f)}

    with open(pairs_path, encoding="utf-8") as f:
        pair_info = {p["pair_id"]: p for p in csv.DictReader(f)}

    # Get unique positive-pair groups (those with >1 member)
    from collections import Counter
    group_counts = Counter(report_groups.values())
    positive_groups = sorted(g for g, c in group_counts.items() if c > 1)

    rng = np.random.RandomState(seed)
    rng.shuffle(positive_groups)
    folds = np.array_split(positive_groups, n_folds)

    fold_f1s = []
    fold_thresholds = []

    for i in range(n_folds):
        test_groups = set(folds[i])
        train_groups = set(g for j in range(n_folds) if j != i for g in folds[j])

        # Split pairs: a pair belongs to the fold of its group
        train_idx, test_idx = [], []
        for idx, s in enumerate(model_scores):
            pair = pair_info.get(s["pair_id"], {})
            a_group = report_groups.get(pair.get("report_a_id", ""), "")
            b_group = report_groups.get(pair.get("report_b_id", ""), "")

            # If either report's group is in test set, the pair goes to test
            if a_group in test_groups or b_group in test_groups:
                test_idx.append(idx)
            else:
                train_idx.append(idx)

        if not train_idx or not test_idx:
            continue

        train_scores = np.array([float(model_scores[j]["cosine_score"]) for j in train_idx])
        train_labels = np.array([1 if model_scores[j]["label"] == "duplicate" else 0 for j in train_idx])
        test_scores = np.array([float(model_scores[j]["cosine_score"]) for j in test_idx])
        test_labels = np.array([1 if model_scores[j]["label"] == "duplicate" else 0 for j in test_idx])

        # Pick threshold on train
        train_sweep = sweep_thresholds(train_scores, train_labels)
        best_train = max(train_sweep, key=lambda r: r["f1"])
        threshold = best_train["threshold"]

        # Evaluate on test
        pred = (test_scores >= threshold).astype(int)
        tp = int(((pred == 1) & (test_labels == 1)).sum())
        fp = int(((pred == 1) & (test_labels == 0)).sum())
        fn = int(((pred == 0) & (test_labels == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        fold_f1s.append(f1)
        fold_thresholds.append(threshold)

    return {
        "arch_cv_f1_mean": round(np.mean(fold_f1s), 4) if fold_f1s else 0,
        "arch_cv_f1_std": round(np.std(fold_f1s), 4) if fold_f1s else 0,
        "arch_cv_threshold_mean": round(np.mean(fold_thresholds), 2) if fold_thresholds else 0,
        "n_folds_used": len(fold_f1s),
    }


def cross_validated_threshold(scores, labels, n_folds=5, seed=42):
    """Pick threshold on train folds, evaluate on held-out fold.

    Returns (cv_f1, cv_threshold, cv_precision, cv_recall) — averaged across folds.
    This avoids the test-set contamination of sweeping on the full dataset.
    """
    rng = np.random.RandomState(seed)
    indices = np.arange(len(scores))
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    fold_f1s = []
    fold_thresholds = []
    fold_precisions = []
    fold_recalls = []

    for i in range(n_folds):
        # Train on all folds except i
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])

        train_scores = scores[train_idx]
        train_labels = labels[train_idx]
        test_scores = scores[test_idx]
        test_labels = labels[test_idx]

        # Pick best threshold on train set
        train_sweep = sweep_thresholds(train_scores, train_labels)
        best_train = max(train_sweep, key=lambda r: r["f1"])
        threshold = best_train["threshold"]

        # Evaluate on held-out test set
        predicted = (test_scores >= threshold).astype(int)
        tp = int(((predicted == 1) & (test_labels == 1)).sum())
        fp = int(((predicted == 1) & (test_labels == 0)).sum())
        fn = int(((predicted == 0) & (test_labels == 1)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        fold_f1s.append(f1)
        fold_thresholds.append(threshold)
        fold_precisions.append(prec)
        fold_recalls.append(rec)

    return {
        "cv_f1_mean": round(np.mean(fold_f1s), 4),
        "cv_f1_std": round(np.std(fold_f1s), 4),
        "cv_threshold_mean": round(np.mean(fold_thresholds), 2),
        "cv_precision_mean": round(np.mean(fold_precisions), 4),
        "cv_recall_mean": round(np.mean(fold_recalls), 4),
    }


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

        # Cross-validated threshold selection (avoids train-on-test)
        cv = cross_validated_threshold(scores, labels)
        print(f"  Pair-level CV F1: {cv['cv_f1_mean']:.3f} ± {cv['cv_f1_std']:.3f} "
              f"(threshold={cv['cv_threshold_mean']})")

        # Archetype-level CV (holds out entire bug types — harder test)
        model_scores = [row for row in all_scores if row["model"] == model]
        arch_cv = archetype_cv_threshold(model_scores)
        print(f"  Archetype CV F1: {arch_cv['arch_cv_f1_mean']:.3f} ± {arch_cv['arch_cv_f1_std']:.3f} "
              f"(threshold={arch_cv['arch_cv_threshold_mean']}, {arch_cv['n_folds_used']} folds)")

        summary_rows.append({
            "model": model,
            "best_threshold": best["threshold"],
            "best_f1": best["f1"],
            "best_precision": best["precision"],
            "best_recall": best["recall"],
            "cv_f1_mean": cv["cv_f1_mean"],
            "cv_f1_std": cv["cv_f1_std"],
            "cv_threshold": cv["cv_threshold_mean"],
            "cv_precision": cv["cv_precision_mean"],
            "cv_recall": cv["cv_recall_mean"],
            "arch_cv_f1_mean": arch_cv["arch_cv_f1_mean"],
            "arch_cv_f1_std": arch_cv["arch_cv_f1_std"],
            "arch_cv_threshold": arch_cv["arch_cv_threshold_mean"],
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
