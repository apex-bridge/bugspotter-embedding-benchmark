"""
Validate model rankings on Mozilla Bugzilla (independent test set).

Embeds 407 Bugzilla bugs with all 6 models, computes F1 on 350 labeled pairs.
This tests whether our rankings generalize to real multi-author bug reports.

Usage:
    python benchmark/bugzilla_validation.py

Requires: Ollama running, all 6 embedding models pulled.

Output:
    results/raw/bugzilla_similarity_scores.csv  — per-model cosine scores
    results/raw/bugzilla_summary.csv            — per-model F1 at sweep/benchmark thresholds
"""

import csv
import json
import os
import sys
import time
import argparse
import requests
import numpy as np
from glob import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Models (same as main benchmark)
MODELS = [
    "all-minilm",
    "nomic-embed-text",
    "snowflake-arctic-embed",
    "mxbai-embed-large",
    "bge-m3",
    "qwen3-embedding",
]


def prepare_text(bug):
    """Build embedding text from Bugzilla bug (title + description only — plain text data)."""
    parts = [bug.get("title", "")]
    if bug.get("description"):
        parts.append(bug["description"])
    text = " | ".join(filter(None, parts))
    return text[:2048]


def embed_ollama(text, model):
    """Call Ollama /api/embed for a single text."""
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": model, "input": text},
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()
    embeddings = data.get("embeddings", [])
    if embeddings and isinstance(embeddings[0], list):
        return embeddings[0]
    return embeddings


def unload_model(model):
    """Unload a model from Ollama to free memory."""
    try:
        requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=30,
        )
    except Exception:
        pass


def cosine(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bugs", default="data/bugzilla_bugs.json")
    parser.add_argument("--pairs", default="data/bugzilla_pairs.csv")
    parser.add_argument("--output-scores", default="results/raw/bugzilla_similarity_scores.csv")
    parser.add_argument("--output-summary", default="results/raw/bugzilla_summary.csv")
    parser.add_argument("--benchmark-summary", default="results/raw/model_summary.csv",
                        help="Use these thresholds to compute F1 at benchmark threshold")
    args = parser.parse_args()

    with open(args.bugs, encoding="utf-8") as f:
        bugs = json.load(f)
    with open(args.pairs, encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))

    print(f"Bugzilla: {len(bugs)} bugs, {len(pairs)} pairs")

    # Build text for each bug
    texts = {b["id"]: prepare_text(b) for b in bugs}

    # Load benchmark thresholds (to evaluate F1 at our picked thresholds)
    bench_thresholds = {}
    if os.path.exists(args.benchmark_summary):
        with open(args.benchmark_summary, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                # Normalize model name
                name = r["model"].replace("_latest", "").replace(":latest", "")
                bench_thresholds[name] = float(r.get("cv_threshold", r.get("best_threshold", 0.5)))
        print(f"Loaded thresholds: {bench_thresholds}")

    # For each model: embed + compute similarity
    all_scores = []
    summaries = []

    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        t0 = time.perf_counter()
        embeddings = {}
        for i, (bug_id, text) in enumerate(texts.items()):
            try:
                embeddings[bug_id] = embed_ollama(text, model)
            except Exception as e:
                print(f"  Error on {bug_id}: {e}")
                continue
            if (i + 1) % 100 == 0:
                print(f"  Embedded {i+1}/{len(texts)}")

        elapsed = time.perf_counter() - t0
        print(f"Embedded {len(embeddings)}/{len(texts)} bugs in {elapsed:.1f}s")

        # Compute similarity for all pairs
        scores = []
        for p in pairs:
            a, b = p["report_a_id"], p["report_b_id"]
            if a not in embeddings or b not in embeddings:
                continue
            score = cosine(embeddings[a], embeddings[b])
            scores.append({
                "pair_id": p["pair_id"],
                "model": model,
                "cosine_score": round(score, 6),
                "label": p["label"],
            })
        all_scores.extend(scores)

        # Compute F1
        scores_arr = np.array([s["cosine_score"] for s in scores])
        labels_arr = np.array([1 if s["label"] == "duplicate" else 0 for s in scores])

        # Oracle threshold (best F1 on Bugzilla)
        best_f1, best_t = 0, 0
        for t in np.arange(0.0, 1.0, 0.01):
            pred = (scores_arr >= t).astype(int)
            tp = ((pred == 1) & (labels_arr == 1)).sum()
            fp = ((pred == 1) & (labels_arr == 0)).sum()
            fn = ((pred == 0) & (labels_arr == 1)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            if f1 > best_f1:
                best_f1, best_t = f1, t

        # F1 at benchmark threshold
        bench_t = bench_thresholds.get(model, 0.5)
        pred = (scores_arr >= bench_t).astype(int)
        tp = ((pred == 1) & (labels_arr == 1)).sum()
        fp = ((pred == 1) & (labels_arr == 0)).sum()
        fn = ((pred == 0) & (labels_arr == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_at_bench = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        print(f"  Oracle F1: {best_f1:.4f} @ threshold {best_t:.2f}")
        print(f"  F1 at benchmark threshold ({bench_t}): {f1_at_bench:.4f}")

        summaries.append({
            "model": model,
            "oracle_f1": round(best_f1, 4),
            "oracle_threshold": round(best_t, 2),
            "benchmark_threshold": bench_t,
            "f1_at_benchmark_threshold": round(f1_at_bench, 4),
            "total_pairs": len(scores),
            "total_duplicates": int(labels_arr.sum()),
        })

        # Unload to free memory before next model
        unload_model(model)

    # Save results
    os.makedirs(os.path.dirname(args.output_scores) or ".", exist_ok=True)
    with open(args.output_scores, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_scores[0].keys())
        writer.writeheader()
        writer.writerows(all_scores)
    print(f"\nSaved: {args.output_scores}")

    with open(args.output_summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Saved: {args.output_summary}")


if __name__ == "__main__":
    main()
