"""
Step 3b: Load embeddings into Qdrant.

Creates a collection per model with HNSW (m=16, ef_construct=200).
"""

import json
import os
import argparse
import time
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, HnswConfigDiff,
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")


def load_model_embeddings(client: QdrantClient, model_name: str,
                          embeddings: dict, reports: dict):
    """Create collection and load embeddings for one model."""
    collection = f"bugs_{model_name.replace(':', '_').replace('/', '_').replace('-', '_')}"
    dim = len(next(iter(embeddings.values())))

    # Recreate collection
    if client.collection_exists(collection):
        client.delete_collection(collection)

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
    )

    # Build points
    points = []
    for report_id, emb in embeddings.items():
        report = reports.get(report_id, {})
        points.append(PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, report_id)),
            vector=emb,
            payload={
                "report_id": report_id,
                "group": report.get("group", ""),
                "error_type": report.get("error_type", ""),
                "url": report.get("url", ""),
            },
        ))

    # Batch upsert
    t0 = time.perf_counter()
    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=collection, points=points[i:i+batch_size])
    insert_time = time.perf_counter() - t0

    info = client.get_collection(collection)
    print(f"  [{collection}] {info.points_count} points, dim={dim}")
    print(f"    Insert: {insert_time:.2f}s")
    return {"collection": collection, "points": len(points), "dim": dim,
            "insert_time_s": insert_time}


def main():
    parser = argparse.ArgumentParser(description="Load embeddings into Qdrant")
    parser.add_argument("--embeddings-dir", default="results/raw")
    parser.add_argument("--reports", default="data/bug_reports.json")
    parser.add_argument("--qdrant-url", default=None)
    args = parser.parse_args()

    url = args.qdrant_url or QDRANT_URL
    client = QdrantClient(url=url)

    with open(args.reports, "r", encoding="utf-8") as f:
        reports_list = json.load(f)
    reports = {r["id"]: r for r in reports_list}

    from glob import glob
    emb_files = sorted(glob(os.path.join(args.embeddings_dir, "embeddings_*.json")))
    if not emb_files:
        print("No embedding files found. Run embed_all.py first.")
        return

    for emb_file in emb_files:
        model_name = os.path.basename(emb_file).replace("embeddings_", "").replace(".json", "")
        print(f"\nLoading: {model_name}")
        with open(emb_file, "r") as f:
            embeddings = json.load(f)
        load_model_embeddings(client, model_name, embeddings, reports)

    print("\nDone. All models loaded into Qdrant.")


if __name__ == "__main__":
    main()
