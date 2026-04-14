"""
Step 6 / Experiment E3: MRL Dimension Truncation.

For MRL-capable models (Qwen3, nomic), test embeddings at reduced dimensions:
  full → 512 → 256 → 128

Re-runs the similarity + threshold sweep pipeline at each dimension level.

Output: results/raw/mrl_truncation.csv
"""

import json
import csv
import os
import argparse

import numpy as np

MRL_MODELS = [
    "qwen3-embedding",       # 7.6B Q4, full=4096
    "qwen3-embedding_latest",
    "nomic-embed-text",      # v1.5, full=768
    "nomic-embed-text_latest",
]

TRUNCATION_DIMS = [128, 256, 512]  # + full


def truncate_and_normalize(embedding: list[float], dim: int) -> list[float]:
    """Truncate to dim dimensions and L2-normalize (MRL standard)."""
    vec = np.array(embedding[:dim], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


def sweep_thresholds(scores, labels):
    best_f1 = 0
    best_result = {}
    for threshold in np.arange(0.50, 1.00, 0.01):
        predicted = (scores >= threshold).astype(int)
        tp = ((predicted == 1) & (labels == 1)).sum()
        fp = ((predicted == 1) & (labels == 0)).sum()
        fn = ((predicted == 0) & (labels == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_result = {
                "threshold": round(threshold, 2),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1": round(float(f1), 4),
            }
    return best_result


def main():
    parser = argparse.ArgumentParser(description="MRL dimension truncation experiment")
    parser.add_argument("--embeddings-dir", default="results/raw")
    parser.add_argument("--pairs", default="data/pairs_ground_truth.csv")
    parser.add_argument("--output", default="results/raw/mrl_truncation.csv")
    args = parser.parse_args()

    # Load pairs
    with open(args.pairs, "r", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))
    print(f"Loaded {len(pairs)} pairs")

    results = []

    from glob import glob
    emb_files = sorted(glob(os.path.join(args.embeddings_dir, "embeddings_*.json")))

    for emb_file in emb_files:
        model_name = os.path.basename(emb_file).replace("embeddings_", "").replace(".json", "")
        if model_name not in MRL_MODELS:
            continue

        print(f"\n{'='*50}")
        print(f"Model: {model_name}")

        with open(emb_file, "r") as f:
            embeddings = json.load(f)

        full_dim = len(next(iter(embeddings.values())))
        dims_to_test = [d for d in TRUNCATION_DIMS if d < full_dim] + [full_dim]

        for dim in dims_to_test:
            dim_label = f"{dim}" if dim < full_dim else f"{dim} (full)"
            print(f"\n  Dimension: {dim_label}")

            # Truncate all embeddings
            truncated = {
                rid: truncate_and_normalize(emb, dim)
                for rid, emb in embeddings.items()
            }

            # Compute cosine similarity for all pairs
            scores_list = []
            labels_list = []
            for pair in pairs:
                a_id = pair["report_a_id"]
                b_id = pair["report_b_id"]
                if a_id not in truncated or b_id not in truncated:
                    continue
                score = cosine_similarity(truncated[a_id], truncated[b_id])
                scores_list.append(score)
                labels_list.append(1 if pair["label"] == "duplicate" else 0)

            scores = np.array(scores_list)
            labels = np.array(labels_list)

            best = sweep_thresholds(scores, labels)
            print(f"    Best: t={best['threshold']} F1={best['f1']:.3f} "
                  f"P={best['precision']:.3f} R={best['recall']:.3f}")

            # Estimate storage per 100K records
            storage_mb = (dim * 4 * 100_000) / (1024 * 1024)  # float32

            results.append({
                "model": model_name,
                "dim": dim,
                "is_full": dim == full_dim,
                "best_threshold": best["threshold"],
                "f1": best["f1"],
                "precision": best["precision"],
                "recall": best["recall"],
                "storage_100k_mb": round(storage_mb, 1),
                "f1_loss_vs_full": 0.0,  # filled below
            })

    # Compute F1 loss vs full for each model
    for model_name in MRL_MODELS:
        model_results = [r for r in results if r["model"] == model_name]
        full_f1 = next((r["f1"] for r in model_results if r["is_full"]), 0)
        for r in model_results:
            r["f1_loss_vs_full"] = round(full_f1 - r["f1"], 4)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if results:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {args.output}")
    else:
        print("\nNo MRL models found in embeddings. Run embed_all.py first.")


if __name__ == "__main__":
    main()
