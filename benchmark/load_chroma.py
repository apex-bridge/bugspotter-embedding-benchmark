"""
Step 3c: Load embeddings into ChromaDB (embedded/persistent mode).

Creates a collection per model with HNSW (M=16, construction_ef=200, search_ef=100).
"""

import json
import os
import argparse
import time

import chromadb
from chromadb.config import Settings

CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma")


def load_model_embeddings(client, model_name: str, embeddings: dict, reports: dict):
    """Create collection and load embeddings for one model."""
    collection_name = f"bugs_{model_name.replace(':', '_').replace('/', '_').replace('-', '_')}"

    # Delete if exists
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={
            "hnsw:M": 16,
            "hnsw:construction_ef": 200,
            "hnsw:search_ef": 100,
            "hnsw:space": "cosine",
        },
    )

    # Prepare data
    ids = []
    embs = []
    metadatas = []

    for report_id, emb in embeddings.items():
        report = reports.get(report_id, {})
        ids.append(report_id)
        embs.append(emb)
        metadatas.append({
            "group": report.get("group", ""),
            "error_type": report.get("error_type", ""),
            "url": report.get("url", ""),
        })

    # Batch add (ChromaDB recommends batches <= 5000)
    t0 = time.perf_counter()
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            embeddings=embs[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
        )
    insert_time = time.perf_counter() - t0

    count = collection.count()
    print(f"  [{collection_name}] {count} documents")
    print(f"    Insert: {insert_time:.2f}s")
    return {"collection": collection_name, "count": count, "insert_time_s": insert_time}


def main():
    parser = argparse.ArgumentParser(description="Load embeddings into ChromaDB")
    parser.add_argument("--embeddings-dir", default="results/raw")
    parser.add_argument("--reports", default="data/bug_reports.json")
    parser.add_argument("--chroma-dir", default=None)
    args = parser.parse_args()

    persist_dir = args.chroma_dir or CHROMA_DIR
    client = chromadb.PersistentClient(path=persist_dir)

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

    print(f"\nDone. All models loaded into ChromaDB at {persist_dir}")


if __name__ == "__main__":
    main()
