"""
BM25F tuned — but with proper 5-fold cross-validation of the field weights.

The v1 of bm25_baseline.py did grid search over field weights on ALL pairs
and reported the best F1 — i.e. oracle F1 on the same set used for tuning,
which overfits. This script uses the same protocol the embedding models use:
4 folds to pick weights + threshold, 1 held-out fold to measure F1, averaged
across 5 folds.

Usage:
    python benchmark/bm25f_cv.py

Output:
    Prints CV F1 mean ± std and writes results/raw/bm25f_cv_summary.csv
"""

import csv
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from embed_all import prepare_text, extract_console_errors, extract_failed_requests
from bm25_baseline import code_aware_tokenize

WEIGHT_CONFIGS = [
    {"title": 3.0, "description": 1.0, "console": 2.0, "network": 1.5},
    {"title": 4.0, "description": 1.0, "console": 3.0, "network": 2.0},
    {"title": 2.0, "description": 1.5, "console": 2.5, "network": 1.0},
    {"title": 5.0, "description": 1.0, "console": 2.0, "network": 1.0},
    {"title": 3.0, "description": 2.0, "console": 1.0, "network": 1.0},
    {"title": 2.0, "description": 1.0, "console": 3.0, "network": 2.0},
]


def apply_weights(field_scores, config):
    """Weighted combination of per-field scores → single score per pair."""
    weighted = np.zeros(next(iter(field_scores.values())).shape[0])
    total_w = 0
    for field, w in config.items():
        if field in field_scores:
            weighted += w * field_scores[field]
            total_w += w
    if total_w > 0:
        weighted /= total_w
    mn, mx = weighted.min(), weighted.max()
    if mx > mn:
        return (weighted - mn) / (mx - mn)
    return np.zeros_like(weighted)


def sweep_and_eval(scores, labels):
    """Return (best_f1, best_threshold) swept across [0, 1]."""
    best_f1, best_t = 0.0, 0.0
    for t in np.arange(0.0, 1.0, 0.005):
        pred = (scores >= t).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        p = tp / (tp + fp) if tp + fp > 0 else 0
        r = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_f1, best_t


def f1_at(scores, labels, t):
    pred = (scores >= t).astype(int)
    tp = ((pred == 1) & (labels == 1)).sum()
    fp = ((pred == 1) & (labels == 0)).sum()
    fn = ((pred == 0) & (labels == 1)).sum()
    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    return 2 * p * r / (p + r) if p + r > 0 else 0


def main():
    from rank_bm25 import BM25Okapi

    print("Loading data...")
    with open("data/bug_reports.json", encoding="utf-8") as f:
        reports = json.load(f)
    with open("data/pairs_ground_truth.csv", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))

    # Per-report field tokens
    fields = {}
    for r in reports:
        rid = r["id"]
        fields[rid] = {
            "title": code_aware_tokenize(r.get("title", "")),
            "description": code_aware_tokenize(r.get("description", "")),
            "console": code_aware_tokenize(" ".join(extract_console_errors(r.get("console_logs", [])))),
            "network": code_aware_tokenize(" ".join(extract_failed_requests(r.get("network_logs", [])))),
        }
    all_ids = sorted(fields.keys())
    id_to_idx = {rid: i for i, rid in enumerate(all_ids)}

    # Per-field BM25 indexes (once)
    print("Building per-field BM25 indexes...")
    bm25_indexes = {}
    for field_name in ["title", "description", "console", "network"]:
        texts = [fields[rid][field_name] for rid in all_ids]
        if all(len(t) == 0 for t in texts):
            continue
        bm25_indexes[field_name] = BM25Okapi(texts)

    # Per-pair, per-field scores (once)
    print("Computing per-pair, per-field BM25 scores...")
    t0 = time.perf_counter()
    valid_pairs = [p for p in pairs if p["report_a_id"] in id_to_idx and p["report_b_id"] in id_to_idx]
    n = len(valid_pairs)
    field_scores = {f: np.zeros(n) for f in bm25_indexes}
    for i, pair in enumerate(valid_pairs):
        a_idx = id_to_idx[pair["report_a_id"]]
        b_idx = id_to_idx[pair["report_b_id"]]
        for field_name, bm25 in bm25_indexes.items():
            tok_a = fields[all_ids[a_idx]][field_name]
            tok_b = fields[all_ids[b_idx]][field_name]
            if tok_a and tok_b:
                s_ab = bm25.get_scores(tok_a)[b_idx]
                s_ba = bm25.get_scores(tok_b)[a_idx]
                field_scores[field_name][i] = (s_ab + s_ba) / 2.0
    print(f"  {n} pairs x {len(bm25_indexes)} fields in {time.perf_counter()-t0:.1f}s")

    labels = np.array([1 if p["label"] == "duplicate" else 0 for p in valid_pairs])

    # 5-fold CV: for each fold, pick best weight config on train, eval on held-out
    N_FOLDS = 5
    rng = np.random.default_rng(42)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, N_FOLDS)

    # Oracle F1: best across all configs × all thresholds, on ALL pairs (the v1 methodology)
    oracle_f1, oracle_t = 0.0, 0.0
    oracle_cfg = None
    for cfg in WEIGHT_CONFIGS:
        s = apply_weights(field_scores, cfg)
        f1, t = sweep_and_eval(s, labels)
        if f1 > oracle_f1:
            oracle_f1, oracle_t, oracle_cfg = f1, t, cfg
    print(f"\n[Oracle] Best-of-all-configs F1 = {oracle_f1:.4f} @ t={oracle_t:.3f}")
    print(f"         Config: {oracle_cfg}")

    print(f"\nRunning {N_FOLDS}-fold CV of field weights...")
    fold_f1s = []
    fold_chosen_configs = []
    fold_thresholds = []

    for fold_idx in range(N_FOLDS):
        test_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[j] for j in range(N_FOLDS) if j != fold_idx])

        # On train: pick best (config, threshold)
        best_train_f1 = 0
        best_config = None
        best_threshold = 0.5
        for config in WEIGHT_CONFIGS:
            train_scores = apply_weights(
                {f: field_scores[f][train_idx] for f in field_scores}, config
            )
            train_labels = labels[train_idx]
            f1, t = sweep_and_eval(train_scores, train_labels)
            if f1 > best_train_f1:
                best_train_f1, best_config, best_threshold = f1, config, t

        # On test: evaluate F1 at picked (config, threshold)
        test_scores = apply_weights(
            {f: field_scores[f][test_idx] for f in field_scores}, best_config
        )
        test_labels = labels[test_idx]
        test_f1 = f1_at(test_scores, test_labels, best_threshold)

        fold_f1s.append(test_f1)
        fold_chosen_configs.append(best_config)
        fold_thresholds.append(best_threshold)

        print(f"  Fold {fold_idx+1}: train F1={best_train_f1:.4f}  test F1={test_f1:.4f}  "
              f"t={best_threshold:.3f}  weights(title/desc/console/net)="
              f"{best_config['title']:.0f}/{best_config['description']:.0f}/"
              f"{best_config['console']:.0f}/{best_config['network']:.0f}")

    cv_mean = float(np.mean(fold_f1s))
    cv_std = float(np.std(fold_f1s))
    print(f"\n=== BM25F tuned, 5-fold CV ===")
    print(f"CV F1: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Oracle F1 (tuned on all pairs): {oracle_f1:.4f}")
    print(f"Overfitting gap: {oracle_f1 - cv_mean:+.4f}")

    # Save
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/bm25f_cv_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "cv_f1_mean", "cv_f1_std", "oracle_f1",
                    "overfitting_gap", "n_folds", "total_pairs"])
        w.writerow(["bm25f_tuned_cv", round(cv_mean, 4), round(cv_std, 4),
                    round(oracle_f1, 4), round(oracle_f1 - cv_mean, 4), N_FOLDS, n])
    print("\nSaved: results/raw/bm25f_cv_summary.csv")


if __name__ == "__main__":
    main()
