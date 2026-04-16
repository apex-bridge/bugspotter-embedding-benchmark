"""
BM25/TF-IDF baseline for bug report deduplication.

Computes cosine similarity using TF-IDF vectors (a lexical/keyword baseline)
on the same pairs as the embedding models. This answers: "do you even need
embeddings, or would keyword matching work?"

Uses the same prepare_text() function as the embedding pipeline, so the
input text is identical — only the vectorization method differs.

Output: results/raw/similarity_scores.csv (appends BM25 rows)
        results/raw/bm25_summary.csv
"""

import csv
import json
import os
import sys
import time
import argparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

# Reuse prepare_text from embed_all
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from embed_all import prepare_text


def main():
    parser = argparse.ArgumentParser(description="BM25/TF-IDF baseline")
    parser.add_argument("--reports", default="data/bug_reports.json")
    parser.add_argument("--pairs", default="data/pairs_ground_truth.csv")
    parser.add_argument("--output-scores", default="results/raw/similarity_scores.csv",
                        help="Append to existing similarity scores CSV")
    parser.add_argument("--output-summary", default="results/raw/bm25_summary.csv")
    args = parser.parse_args()

    # Load reports
    with open(args.reports, encoding="utf-8") as f:
        reports = json.load(f)
    print(f"Loaded {len(reports)} reports")

    # Build text for each report using the same prepare_text as embeddings
    report_texts = {}
    for r in reports:
        rid = r["id"]
        report_texts[rid] = prepare_text(r)

    # Load pairs
    with open(args.pairs, encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))
    print(f"Loaded {len(pairs)} pairs")

    # Fit TF-IDF on all report texts
    print("\nFitting TF-IDF vectorizer...")
    t0 = time.perf_counter()

    all_ids = sorted(report_texts.keys())
    all_texts = [report_texts[rid] for rid in all_ids]
    id_to_idx = {rid: i for i, rid in enumerate(all_ids)}

    vectorizer = TfidfVectorizer(
        max_features=10000,
        sublinear_tf=True,      # log(1 + tf) — approximates BM25's TF saturation
        norm="l2",
        min_df=1,
        ngram_range=(1, 2),     # unigrams + bigrams for technical terms
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    elapsed = time.perf_counter() - t0
    print(f"TF-IDF: {tfidf_matrix.shape[0]} docs x {tfidf_matrix.shape[1]} features in {elapsed:.2f}s")

    # Compute similarity for all pairs
    print("Computing pairwise similarities...")
    t0 = time.perf_counter()

    results = []
    missing = 0
    for pair in pairs:
        a_id = pair["report_a_id"]
        b_id = pair["report_b_id"]

        if a_id not in id_to_idx or b_id not in id_to_idx:
            missing += 1
            continue

        a_vec = tfidf_matrix[id_to_idx[a_id]]
        b_vec = tfidf_matrix[id_to_idx[b_id]]
        score = float(sklearn_cosine(a_vec, b_vec)[0, 0])

        results.append({
            "pair_id": pair["pair_id"],
            "model": "tfidf_baseline",
            "cosine_score": round(score, 6),
            "label": pair["label"],
            "pair_type": pair["pair_type"],
        })

    elapsed = time.perf_counter() - t0
    print(f"Computed {len(results)} scores in {elapsed:.2f}s ({missing} pairs skipped)")

    # Quick stats
    dup_scores = [r["cosine_score"] for r in results if r["label"] == "duplicate"]
    neg_scores = [r["cosine_score"] for r in results if r["label"] == "not_duplicate"]
    print(f"\nDuplicates:     mean={np.mean(dup_scores):.3f}, min={np.min(dup_scores):.3f}, max={np.max(dup_scores):.3f}")
    print(f"Not-duplicates: mean={np.mean(neg_scores):.3f}, min={np.min(neg_scores):.3f}, max={np.max(neg_scores):.3f}")
    print(f"Separation gap: {np.mean(dup_scores) - np.mean(neg_scores):.3f}")

    # Append to existing similarity scores
    file_exists = os.path.exists(args.output_scores) and os.path.getsize(args.output_scores) > 0
    if file_exists:
        # Remove any existing tfidf_baseline rows first
        with open(args.output_scores, "r", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
        existing = [r for r in existing if r["model"] != "tfidf_baseline"]
        all_rows = existing + results
        with open(args.output_scores, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Appended {len(results)} TF-IDF scores to {args.output_scores}")
    else:
        os.makedirs(os.path.dirname(args.output_scores), exist_ok=True)
        with open(args.output_scores, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved {len(results)} TF-IDF scores to {args.output_scores}")

    # Run threshold sweep for the baseline
    from sweep_threshold import sweep_thresholds
    from sklearn.metrics import roc_auc_score

    scores_arr = np.array([r["cosine_score"] for r in results])
    labels_arr = np.array([1 if r["label"] == "duplicate" else 0 for r in results])

    sweep = sweep_thresholds(scores_arr, labels_arr)
    best_row = max(sweep, key=lambda r: r["f1"])
    auc = roc_auc_score(labels_arr, scores_arr)

    print(f"\nTF-IDF Baseline Results:")
    print(f"  Best F1:     {best_row['f1']:.4f}")
    print(f"  Threshold:   {best_row['threshold']}")
    print(f"  Precision:   {best_row['precision']:.4f}")
    print(f"  Recall:      {best_row['recall']:.4f}")
    print(f"  ROC-AUC:     {auc:.4f}")

    # Compute recall at 0.9 threshold
    tp_at_09 = sum(1 for r in results if r["label"] == "duplicate" and r["cosine_score"] >= 0.9)
    total_dup = sum(1 for r in results if r["label"] == "duplicate")
    recall_at_09 = tp_at_09 / total_dup if total_dup > 0 else 0
    print(f"  Recall@0.9:  {recall_at_09:.4f}")

    # Save summary
    summary = {
        "model": "tfidf_baseline",
        "best_threshold": best_row["threshold"],
        "best_f1": round(best_row["f1"], 4),
        "best_precision": round(best_row["precision"], 4),
        "best_recall": round(best_row["recall"], 4),
        "roc_auc": round(auc, 4),
        "recall_at_0.90": round(recall_at_09, 4),
        "total_pairs": len(results),
        "total_duplicates": total_dup,
    }

    os.makedirs(os.path.dirname(args.output_summary), exist_ok=True)
    with open(args.output_summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)
    print(f"Saved: {args.output_summary}")


if __name__ == "__main__":
    main()
