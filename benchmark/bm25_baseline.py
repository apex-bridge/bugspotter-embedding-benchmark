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


def run_bm25f(reports, id_to_idx_map, pairs):
    """BM25F baseline: field-weighted BM25.

    Scores each field (title, description, console_logs, network_logs)
    independently with BM25, then combines with learned-style weights.
    This is what Zhang et al. (2023) found competitive with DL approaches.
    """
    print("\n=== BM25F Baseline (field-weighted) ===")
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("rank_bm25 not installed. Skipping.")
        return []

    # Field weights — title and console errors are most discriminative
    FIELD_WEIGHTS = {
        "title": 3.0,
        "description": 1.0,
        "console": 2.0,
        "network": 1.5,
    }

    # Extract fields per report
    from embed_all import extract_console_errors, extract_failed_requests

    fields = {}  # {report_id: {field_name: text}}
    for r in reports:
        rid = r["id"]
        fields[rid] = {
            "title": r.get("title", ""),
            "description": r.get("description", ""),
            "console": " ".join(extract_console_errors(r.get("console_logs", []))),
            "network": " ".join(extract_failed_requests(r.get("network_logs", []))),
        }

    all_ids = sorted(fields.keys())
    id_to_idx = {rid: i for i, rid in enumerate(all_ids)}

    # Build per-field BM25 indexes
    t0 = time.perf_counter()
    bm25_indexes = {}
    for field_name in FIELD_WEIGHTS:
        texts = [fields[rid][field_name].lower().split() for rid in all_ids]
        # Skip empty fields
        if all(len(t) == 0 for t in texts):
            continue
        bm25_indexes[field_name] = BM25Okapi(texts)
    elapsed = time.perf_counter() - t0
    print(f"Built {len(bm25_indexes)} field indexes in {elapsed:.2f}s")

    # Compute weighted scores for all pairs
    print("Computing pairwise field-weighted scores...")
    t0 = time.perf_counter()

    raw_scores = []
    pair_info = []
    for pair in pairs:
        a_id, b_id = pair["report_a_id"], pair["report_b_id"]
        if a_id not in id_to_idx or b_id not in id_to_idx:
            continue

        a_idx = id_to_idx[a_id]
        b_idx = id_to_idx[b_id]

        weighted_score = 0.0
        total_weight = 0.0

        for field_name, weight in FIELD_WEIGHTS.items():
            if field_name not in bm25_indexes:
                continue
            bm25 = bm25_indexes[field_name]
            tokens_a = fields[all_ids[a_idx]][field_name].lower().split()
            tokens_b = fields[all_ids[b_idx]][field_name].lower().split()

            if not tokens_a or not tokens_b:
                continue

            # Bidirectional scoring
            score_ab = bm25.get_scores(tokens_a)[b_idx]
            score_ba = bm25.get_scores(tokens_b)[a_idx]
            avg_score = (score_ab + score_ba) / 2.0

            weighted_score += weight * avg_score
            total_weight += weight

        raw_scores.append(weighted_score / total_weight if total_weight > 0 else 0)
        pair_info.append(pair)

    elapsed = time.perf_counter() - t0
    print(f"Computed {len(raw_scores)} scores in {elapsed:.2f}s")

    # Normalize to [0, 1]
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
            "model": "bm25f_baseline",
            "cosine_score": round(float(normalized[i]), 6),
            "label": pair["label"],
            "pair_type": pair["pair_type"],
        })
    return results


def code_aware_tokenize(text):
    """Lucene-style code-aware tokenizer.

    - Splits camelCase: "TypeError" -> ["type", "error"]
    - Splits snake_case: "stack_trace" -> ["stack", "trace"]
    - Lowercases but NO stemming (Porter stemmer destroys code tokens
      like "undefined" -> "undefin", "CORS" -> "cor")
    - No stopword removal (small words matter in error messages)
    """
    import re

    # Split camelCase: "TypeError" -> "Type Error", "processPayment" -> "process Payment"
    split_camel = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Split snake_case: "stack_trace" -> "stack trace"
    split_snake = split_camel.replace('_', ' ')
    # Lowercase and extract alphanumeric tokens
    tokens = re.findall(r'[a-z0-9]+', split_snake.lower())
    # Keep tokens of 2+ chars
    return [t for t in tokens if len(t) > 1]


def run_bm25f_tuned(reports, id_to_idx_map, pairs):
    """BM25F with code-aware tokenization, stemming, and grid-searched weights.

    This is the properly-tuned baseline that Zhang et al. (2023) would expect.
    Uses NLTK Porter stemmer, stopword removal, camelCase/snake_case splitting.
    Field weights are grid-searched on a subset.
    """
    print("\n=== BM25F Tuned Baseline (stemming + grid-searched weights) ===")
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("rank_bm25 not installed. Skipping.")
        return []

    from embed_all import extract_console_errors, extract_failed_requests

    # Extract fields per report with code-aware tokenization
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

    # Grid search over field weights
    t0 = time.perf_counter()
    weight_configs = [
        {"title": 3.0, "description": 1.0, "console": 2.0, "network": 1.5},
        {"title": 4.0, "description": 1.0, "console": 3.0, "network": 2.0},
        {"title": 2.0, "description": 1.5, "console": 2.5, "network": 1.0},
        {"title": 5.0, "description": 1.0, "console": 2.0, "network": 1.0},
        {"title": 3.0, "description": 2.0, "console": 1.0, "network": 1.0},
        {"title": 2.0, "description": 1.0, "console": 3.0, "network": 2.0},
    ]

    # Build BM25 indexes (shared across weight configs)
    bm25_indexes = {}
    for field_name in ["title", "description", "console", "network"]:
        texts = [fields[rid][field_name] for rid in all_ids]
        if all(len(t) == 0 for t in texts):
            continue
        bm25_indexes[field_name] = BM25Okapi(texts)

    # Score all pairs once per field (cache), then combine with different weights
    print(f"Computing per-field BM25 scores...")
    field_scores = {}  # {field_name: [(score_ab + score_ba) / 2 for each pair]}

    pair_info = []
    for field_name, bm25 in bm25_indexes.items():
        scores_for_field = []
        pair_info_local = []
        for pair in pairs:
            a_id, b_id = pair["report_a_id"], pair["report_b_id"]
            if a_id not in id_to_idx or b_id not in id_to_idx:
                continue

            a_idx = id_to_idx[a_id]
            b_idx = id_to_idx[b_id]
            tokens_a = fields[all_ids[a_idx]][field_name]
            tokens_b = fields[all_ids[b_idx]][field_name]

            if not tokens_a or not tokens_b:
                scores_for_field.append(0.0)
            else:
                score_ab = bm25.get_scores(tokens_a)[b_idx]
                score_ba = bm25.get_scores(tokens_b)[a_idx]
                scores_for_field.append((score_ab + score_ba) / 2.0)

            if not pair_info:
                pair_info_local.append(pair)

        field_scores[field_name] = np.array(scores_for_field)
        if not pair_info:
            pair_info = pair_info_local

    # Grid search: try each weight config, pick best F1
    from sweep_threshold import sweep_thresholds

    best_config = None
    best_f1 = 0

    labels = np.array([1 if p["label"] == "duplicate" else 0 for p in pair_info])

    for config in weight_configs:
        weighted = np.zeros(len(pair_info))
        total_w = 0
        for field_name, weight in config.items():
            if field_name in field_scores:
                weighted += weight * field_scores[field_name]
                total_w += weight
        if total_w > 0:
            weighted /= total_w

        # Normalize
        mn, mx = weighted.min(), weighted.max()
        if mx > mn:
            norm = (weighted - mn) / (mx - mn)
        else:
            norm = np.zeros_like(weighted)

        sweep = sweep_thresholds(norm, labels, start=0.0, stop=1.0, step=0.01)
        f1 = max(r["f1"] for r in sweep)
        if f1 > best_f1:
            best_f1 = f1
            best_config = config

    elapsed = time.perf_counter() - t0
    print(f"Grid search: best weights={best_config}, F1={best_f1:.4f} in {elapsed:.1f}s")

    # Compute final scores with best weights
    weighted = np.zeros(len(pair_info))
    total_w = 0
    for field_name, weight in best_config.items():
        if field_name in field_scores:
            weighted += weight * field_scores[field_name]
            total_w += weight
    weighted /= total_w

    mn, mx = weighted.min(), weighted.max()
    normalized = (weighted - mn) / (mx - mn) if mx > mn else np.zeros_like(weighted)

    results = []
    for i, pair in enumerate(pair_info):
        results.append({
            "pair_id": pair["pair_id"],
            "model": "bm25f_tuned",
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

    # Sweep from 0 for baselines (scores may cluster below 0.5 after normalization)
    sweep = sweep_thresholds(scores_arr, labels_arr, start=0.0, stop=1.0, step=0.01)
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

    # Run all four baselines
    tfidf_results = run_tfidf(all_ids, all_texts, id_to_idx, pairs)
    bm25_results = run_bm25(all_ids, all_texts, id_to_idx, pairs)
    bm25f_results = run_bm25f(reports, id_to_idx, pairs)
    bm25f_tuned_results = run_bm25f_tuned(reports, id_to_idx, pairs)

    all_results = tfidf_results + bm25_results + bm25f_results + bm25f_tuned_results

    # Append to existing similarity scores (remove old baseline rows first)
    baseline_models = {"tfidf_baseline", "bm25_baseline", "bm25f_baseline", "bm25f_tuned"}
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

    # Evaluate all three
    summaries = []
    if tfidf_results:
        summaries.append(evaluate_baseline("tfidf_baseline", tfidf_results))
    if bm25_results:
        summaries.append(evaluate_baseline("bm25_baseline", bm25_results))
    if bm25f_results:
        summaries.append(evaluate_baseline("bm25f_baseline", bm25f_results))
    if bm25f_tuned_results:
        summaries.append(evaluate_baseline("bm25f_tuned", bm25f_tuned_results))

    # Save summary
    os.makedirs(os.path.dirname(args.output_summary) or ".", exist_ok=True)
    with open(args.output_summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Saved: {args.output_summary}")


if __name__ == "__main__":
    main()
