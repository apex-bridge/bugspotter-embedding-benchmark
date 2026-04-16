"""
Lexical baselines for bug report deduplication: TF-IDF and BM25.

Computes pairwise similarity using two keyword-based methods on the same
pairs as the embedding models. This answers: "do you even need embeddings,
or would keyword matching work?"

Uses the same prepare_text() function as the embedding pipeline, so the
input text is identical — only the vectorization method differs.

Output: results/raw/similarity_scores.csv (appends baseline rows)
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


def run_tfidf(all_ids, all_texts, id_to_idx, pairs):
    """TF-IDF cosine similarity baseline."""
    print("\n=== TF-IDF Baseline ===")
    t0 = time.perf_counter()

    vectorizer = TfidfVectorizer(
        max_features=10000,
        sublinear_tf=True,
        norm="l2",
        min_df=1,
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    elapsed = time.perf_counter() - t0
    print(f"Fitted: {tfidf_matrix.shape[0]} docs x {tfidf_matrix.shape[1]} features in {elapsed:.2f}s")

    results = []
    for pair in pairs:
        a_id, b_id = pair["report_a_id"], pair["report_b_id"]
        if a_id not in id_to_idx or b_id not in id_to_idx:
            continue
        score = float(sklearn_cosine(
            tfidf_matrix[id_to_idx[a_id]],
            tfidf_matrix[id_to_idx[b_id]]
        )[0, 0])
        results.append({
            "pair_id": pair["pair_id"],
            "model": "tfidf_baseline",
            "cosine_score": round(score, 6),
            "label": pair["label"],
            "pair_type": pair["pair_type"],
        })
    return results


def run_bm25(all_ids, all_texts, id_to_idx, pairs):
    """BM25 similarity baseline using rank_bm25."""
    print("\n=== BM25 Baseline ===")
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("rank_bm25 not installed. Run: pip install rank-bm25")
        print("Skipping BM25 baseline.")
        return []

    t0 = time.perf_counter()

    # Tokenize (simple whitespace + lowercase)
    tokenized = [text.lower().split() for text in all_texts]
    bm25 = BM25Okapi(tokenized)
    elapsed = time.perf_counter() - t0
    print(f"Fitted: {len(tokenized)} docs in {elapsed:.2f}s")

    # BM25 returns relevance scores, not similarities in [0,1].
    # We compute BM25 score of doc_b given doc_a as query, then normalize.
    # For pairwise similarity: average score(a→b) and score(b→a), then
    # normalize to [0,1] range across all pairs.
    print("Computing pairwise scores...")
    t0 = time.perf_counter()

    raw_scores = []
    pair_info = []
    for pair in pairs:
        a_id, b_id = pair["report_a_id"], pair["report_b_id"]
        if a_id not in id_to_idx or b_id not in id_to_idx:
            continue

        a_idx = id_to_idx[a_id]
        b_idx = id_to_idx[b_id]

        # Score doc_b using doc_a's tokens as query
        scores_a = bm25.get_scores(tokenized[a_idx])
        score_ab = scores_a[b_idx]

        # Score doc_a using doc_b's tokens as query
        scores_b = bm25.get_scores(tokenized[b_idx])
        score_ba = scores_b[a_idx]

        # Average both directions
        raw_score = (score_ab + score_ba) / 2.0
        raw_scores.append(raw_score)
        pair_info.append(pair)

    elapsed = time.perf_counter() - t0
    print(f"Computed {len(raw_scores)} scores in {elapsed:.2f}s")

    # Normalize to [0, 1] using min-max
    raw_scores = np.array(raw_scores)
    min_s, max_s = raw_scores.min(), raw_scores.max()
    if max_s > min_s:
        normalized = (raw_scores - min_s) / (max_s - min_s)
    else:
        normalized = np.zeros_like(raw_scores)

    results = []
    for i, pair in enumerate(pair_info):
        results.append({
            "pair_id": pair["pair_id"],
            "model": "bm25_baseline",
            "cosine_score": round(float(normalized[i]), 6),
            "label": pair["label"],
            "pair_type": pair["pair_type"],
        })
    return results


def evaluate_baseline(name, results):
    """Run threshold sweep and print results for one baseline."""
    from sweep_threshold import sweep_thresholds
    from sklearn.metrics import roc_auc_score

    scores_arr = np.array([r["cosine_score"] for r in results])
    labels_arr = np.array([1 if r["label"] == "duplicate" else 0 for r in results])

    sweep = sweep_thresholds(scores_arr, labels_arr)
    best = max(sweep, key=lambda r: r["f1"])
    auc = roc_auc_score(labels_arr, scores_arr)

    tp_at_09 = sum(1 for r in results if r["label"] == "duplicate" and r["cosine_score"] >= 0.9)
    total_dup = sum(1 for r in results if r["label"] == "duplicate")
    recall_09 = tp_at_09 / total_dup if total_dup > 0 else 0

    dup_scores = [r["cosine_score"] for r in results if r["label"] == "duplicate"]
    neg_scores = [r["cosine_score"] for r in results if r["label"] == "not_duplicate"]
    print(f"\n{name} Results:")
    print(f"  Duplicates:     mean={np.mean(dup_scores):.3f}, min={np.min(dup_scores):.3f}, max={np.max(dup_scores):.3f}")
    print(f"  Not-duplicates: mean={np.mean(neg_scores):.3f}, min={np.min(neg_scores):.3f}, max={np.max(neg_scores):.3f}")
    print(f"  Best F1:     {best['f1']:.4f} @ threshold {best['threshold']}")
    print(f"  Precision:   {best['precision']:.4f}")
    print(f"  Recall:      {best['recall']:.4f}")
    print(f"  ROC-AUC:     {auc:.4f}")
    print(f"  Recall@0.9:  {recall_09:.4f}")

    return {
        "model": name,
        "best_threshold": best["threshold"],
        "best_f1": round(best["f1"], 4),
        "best_precision": round(best["precision"], 4),
        "best_recall": round(best["recall"], 4),
        "roc_auc": round(auc, 4),
        "recall_at_0.90": round(recall_09, 4),
        "total_pairs": len(results),
        "total_duplicates": total_dup,
    }


def main():
    parser = argparse.ArgumentParser(description="Lexical baselines (TF-IDF + BM25)")
    parser.add_argument("--reports", default="data/bug_reports.json")
    parser.add_argument("--pairs", default="data/pairs_ground_truth.csv")
    parser.add_argument("--output-scores", default="results/raw/similarity_scores.csv")
    parser.add_argument("--output-summary", default="results/raw/bm25_summary.csv")
    args = parser.parse_args()

    # Load reports
    with open(args.reports, encoding="utf-8") as f:
        reports = json.load(f)
    print(f"Loaded {len(reports)} reports")

    # Build text for each report
    report_texts = {}
    for r in reports:
        report_texts[r["id"]] = prepare_text(r)

    # Load pairs
    with open(args.pairs, encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))
    print(f"Loaded {len(pairs)} pairs")

    all_ids = sorted(report_texts.keys())
    all_texts = [report_texts[rid] for rid in all_ids]
    id_to_idx = {rid: i for i, rid in enumerate(all_ids)}

    # Run both baselines
    tfidf_results = run_tfidf(all_ids, all_texts, id_to_idx, pairs)
    bm25_results = run_bm25(all_ids, all_texts, id_to_idx, pairs)

    all_results = tfidf_results + bm25_results

    # Append to existing similarity scores (remove old baseline rows first)
    baseline_models = {"tfidf_baseline", "bm25_baseline"}
    if os.path.exists(args.output_scores) and os.path.getsize(args.output_scores) > 0:
        with open(args.output_scores, "r", encoding="utf-8") as f:
            existing = [r for r in csv.DictReader(f) if r["model"] not in baseline_models]
        all_rows = existing + all_results
    else:
        all_rows = all_results

    os.makedirs(os.path.dirname(args.output_scores) or ".", exist_ok=True)
    with open(args.output_scores, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_results)} baseline scores to {args.output_scores}")

    # Evaluate both
    summaries = []
    if tfidf_results:
        summaries.append(evaluate_baseline("tfidf_baseline", tfidf_results))
    if bm25_results:
        summaries.append(evaluate_baseline("bm25_baseline", bm25_results))

    # Save summary
    os.makedirs(os.path.dirname(args.output_summary) or ".", exist_ok=True)
    with open(args.output_summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Saved: {args.output_summary}")


if __name__ == "__main__":
    main()
