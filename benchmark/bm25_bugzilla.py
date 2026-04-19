"""Run TF-IDF and BM25 baselines on Mozilla Bugzilla, matching bugzilla_validation.py's protocol.

Outputs:
    results/raw/bugzilla_bm25_summary.csv   — F1 per baseline
    results/raw/bugzilla_bm25_scores.csv    — per-pair scores

Bugzilla bugs have only title + description (no console/network). Skipping BM25F
variants that require structured fields — they'd degenerate to plain BM25.

Usage:
    python benchmark/bm25_bugzilla.py
"""

import csv
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


def prepare_text(bug):
    """Same as bugzilla_validation.py — title + description only."""
    parts = [bug.get("title", "")]
    if bug.get("description"):
        parts.append(bug["description"])
    return " | ".join(filter(None, parts))[:2048]


def sweep_threshold(scores, labels, lo=0.0, hi=1.0, step=0.005):
    scores = np.array(scores); labels = np.array(labels)
    best_f1, best_t, best_p, best_r = 0, 0, 0, 0
    for t in np.arange(lo, hi, step):
        pred = (scores >= t).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        p = tp / (tp + fp) if tp + fp > 0 else 0
        r = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        if f1 > best_f1:
            best_f1, best_t, best_p, best_r = f1, float(t), p, r
    return best_f1, best_t, best_p, best_r


def run_tfidf(bugs, pairs):
    print("=== TF-IDF on Bugzilla ===")
    bug_ids = [b["id"] for b in bugs]
    texts = [prepare_text(b) for b in bugs]
    id_to_idx = {bid: i for i, bid in enumerate(bug_ids)}

    vectorizer = TfidfVectorizer(max_features=10000, sublinear_tf=True, norm="l2",
                                  min_df=1, ngram_range=(1, 2))
    t0 = time.perf_counter()
    mat = vectorizer.fit_transform(texts)
    print(f"Fitted {mat.shape[0]} docs x {mat.shape[1]} features in {time.perf_counter()-t0:.2f}s")

    scores = []
    for pair in pairs:
        a, b = pair["report_a_id"], pair["report_b_id"]
        if a not in id_to_idx or b not in id_to_idx:
            continue
        s = float(sklearn_cosine(mat[id_to_idx[a]], mat[id_to_idx[b]])[0, 0])
        scores.append({
            "pair_id": pair["pair_id"], "model": "tfidf_baseline",
            "score": round(s, 6),
            "label": 1 if pair["label"] == "duplicate" else 0,
            "pair_type": pair["pair_type"],
        })
    return scores


def run_bm25(bugs, pairs):
    print("\n=== BM25 on Bugzilla ===")
    from rank_bm25 import BM25Okapi
    bug_ids = [b["id"] for b in bugs]
    texts = [prepare_text(b) for b in bugs]
    id_to_idx = {bid: i for i, bid in enumerate(bug_ids)}

    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    print(f"Fitted {len(tokenized)} docs")

    raw = []
    pair_order = []
    for pair in pairs:
        a, b = pair["report_a_id"], pair["report_b_id"]
        if a not in id_to_idx or b not in id_to_idx:
            continue
        ai, bi = id_to_idx[a], id_to_idx[b]
        s_ab = bm25.get_scores(tokenized[ai])[bi]
        s_ba = bm25.get_scores(tokenized[bi])[ai]
        raw.append((s_ab + s_ba) / 2.0)
        pair_order.append(pair)

    raw = np.array(raw)
    if raw.max() > raw.min():
        norm = (raw - raw.min()) / (raw.max() - raw.min())
    else:
        norm = np.zeros_like(raw)

    scores = []
    for i, pair in enumerate(pair_order):
        scores.append({
            "pair_id": pair["pair_id"], "model": "bm25_baseline",
            "score": round(float(norm[i]), 6),
            "label": 1 if pair["label"] == "duplicate" else 0,
            "pair_type": pair["pair_type"],
        })
    return scores


def main():
    with open("data/bugzilla_bugs.json", encoding="utf-8") as f:
        bugs = json.load(f)
    with open("data/bugzilla_pairs.csv", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))
    print(f"Bugzilla: {len(bugs)} bugs, {len(pairs)} pairs")

    all_scores = []
    summaries = []

    for runner in [run_tfidf, run_bm25]:
        scores = runner(bugs, pairs)
        all_scores.extend(scores)
        score_vals = [s["score"] for s in scores]
        labels = [s["label"] for s in scores]
        f1, t, p, r = sweep_threshold(score_vals, labels)
        model = scores[0]["model"]
        print(f"  {model}: F1={f1:.4f} @ threshold {t:.3f}  (P={p:.3f}, R={r:.3f}, n={len(scores)})")
        summaries.append({
            "model": model, "oracle_f1": round(f1, 4), "oracle_threshold": round(t, 3),
            "precision": round(p, 4), "recall": round(r, 4),
            "total_pairs": len(scores),
            "total_duplicates": sum(labels),
        })

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/bugzilla_bm25_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_scores[0].keys())
        w.writeheader(); w.writerows(all_scores)
    with open("results/raw/bugzilla_bm25_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summaries[0].keys())
        w.writeheader(); w.writerows(summaries)
    print("\nSaved: results/raw/bugzilla_bm25_summary.csv")


if __name__ == "__main__":
    main()
