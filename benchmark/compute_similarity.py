"""
Step 4: Compute pairwise cosine similarity for all ground truth pairs.

For each model's embeddings, compute cosine similarity between every pair
in pairs_ground_truth.csv. Supports both in-memory (numpy) and pgvector modes.

Output: results/raw/similarity_scores.csv
  Columns: pair_id, model, cosine_score, label, pair_type
"""

import json
import csv
import os
import argparse
import time

import numpy as np


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def load_pairs(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_embeddings(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def compute_for_model(model_name: str, embeddings: dict, pairs: list[dict]) -> list[dict]:
    """Compute cosine similarity for all pairs using one model's embeddings."""
    results = []
    missing = 0

    for pair in pairs:
        a_id = pair["report_a_id"]
        b_id = pair["report_b_id"]

        if a_id not in embeddings or b_id not in embeddings:
            missing += 1
            continue

        score = cosine_similarity(embeddings[a_id], embeddings[b_id])
        results.append({
            "pair_id": pair["pair_id"],
            "model": model_name,
            "cosine_score": round(score, 6),
            "label": pair["label"],
            "pair_type": pair["pair_type"],
        })

    if missing > 0:
        print(f"  Warning: {missing} pairs skipped (missing embeddings)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute pairwise cosine similarity")
    parser.add_argument("--embeddings-dir", default="results/raw")
    parser.add_argument("--pairs", default="data/pairs_ground_truth.csv")
    parser.add_argument("--output", default="results/raw/similarity_scores.csv")
    args = parser.parse_args()

    pairs = load_pairs(args.pairs)
    print(f"Loaded {len(pairs)} pairs from {args.pairs}")

    from glob import glob
    emb_files = sorted(glob(os.path.join(args.embeddings_dir, "embeddings_*.json")))
    if not emb_files:
        print("No embedding files found. Run embed_all.py first.")
        return

    all_results = []

    for emb_file in emb_files:
        model_name = os.path.basename(emb_file).replace("embeddings_", "").replace(".json", "")
        print(f"\nComputing: {model_name}")
        embeddings = load_embeddings(emb_file)

        t0 = time.perf_counter()
        results = compute_for_model(model_name, embeddings, pairs)
        elapsed = time.perf_counter() - t0

        print(f"  {len(results)} scores computed in {elapsed:.2f}s")

        # Quick stats
        scores = [r["cosine_score"] for r in results]
        dup_scores = [r["cosine_score"] for r in results if r["label"] == "duplicate"]
        neg_scores = [r["cosine_score"] for r in results if r["label"] == "not_duplicate"]

        if dup_scores and neg_scores:
            print(f"  Duplicates:     mean={np.mean(dup_scores):.3f}, "
                  f"min={np.min(dup_scores):.3f}, max={np.max(dup_scores):.3f}")
            print(f"  Not-duplicates: mean={np.mean(neg_scores):.3f}, "
                  f"min={np.min(neg_scores):.3f}, max={np.max(neg_scores):.3f}")
            print(f"  Separation gap: {np.mean(dup_scores) - np.mean(neg_scores):.3f}")

        all_results.extend(results)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "model", "cosine_score", "label", "pair_type"])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nTotal: {len(all_results)} scores saved to {args.output}")


if __name__ == "__main__":
    main()
