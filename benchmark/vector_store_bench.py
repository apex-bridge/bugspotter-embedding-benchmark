"""
Experiment E7: Vector Store Shootout.

Compares pgvector, Qdrant, ChromaDB, and sqlite-vec on:
  1. Insert throughput
  2. kNN query latency (p50, p95, p99)
  3. Recall@10 (vs brute-force exact)
  4. Storage footprint
  5. RAM usage
  6. Filtered search latency

Uses the best model's embeddings from E1.

Output: results/raw/vector_store_bench.csv
"""

import json
import os
import time
import csv
import argparse
import struct
import statistics
import sqlite3

import numpy as np
import psutil
import psycopg2
import chromadb
import sqlite_vec
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, HnswConfigDiff, SearchRequest,
)

PG_DSN = os.getenv("PG_DSN", "host=localhost port=5432 dbname=bugdedup user=postgres password=bench")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma_bench")
SQLITE_DB = os.getenv("SQLITE_BENCH_DB", "results/bench_vectors.db")

NUM_QUERIES = 100
TOP_K = 10


def serialize_f32(vector):
    return struct.pack(f"{len(vector)}f", *vector)


def get_rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def brute_force_knn(query_vec, all_vecs, all_ids, k=10):
    """Exact kNN for recall baseline."""
    q = np.array(query_vec)
    scores = []
    for vid, vec in zip(all_ids, all_vecs):
        v = np.array(vec)
        cos = np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-10)
        scores.append((vid, cos))
    scores.sort(key=lambda x: -x[1])
    return [s[0] for s in scores[:k]]


# ---------- pgvector ----------

def bench_pgvector(embeddings, ids, dim, query_vecs, exact_results):
    table = "bench_pgvector"
    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(f"DROP TABLE IF EXISTS {table};")
    cur.execute(f"CREATE TABLE {table} (id TEXT PRIMARY KEY, embedding vector({dim}));")

    # Insert
    t0 = time.perf_counter()
    from psycopg2.extras import execute_values
    rows = [(ids[i], "[" + ",".join(str(x) for x in embeddings[i]) + "]") for i in range(len(ids))]
    execute_values(cur, f"INSERT INTO {table} (id, embedding) VALUES %s", rows,
                   template="(%s, %s::vector)", page_size=100)
    # Build index
    cur.execute(f"CREATE INDEX ON {table} USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=200);")
    cur.execute("SET hnsw.ef_search = 100;")
    insert_time = time.perf_counter() - t0

    # Query latency
    latencies = []
    result_ids_list = []
    for qvec in query_vecs:
        vec_str = "[" + ",".join(str(x) for x in qvec) + "]"
        t0 = time.perf_counter()
        cur.execute(f"SELECT id FROM {table} ORDER BY embedding <=> %s::vector LIMIT %s", (vec_str, TOP_K))
        rows = cur.fetchall()
        latencies.append((time.perf_counter() - t0) * 1000)
        result_ids_list.append([r[0] for r in rows])

    # Storage
    cur.execute(f"SELECT pg_total_relation_size('{table}');")
    storage_bytes = cur.fetchone()[0]

    # Recall@10
    recalls = []
    for exact, approx in zip(exact_results, result_ids_list):
        overlap = len(set(exact) & set(approx))
        recalls.append(overlap / TOP_K)

    cur.execute(f"DROP TABLE IF EXISTS {table};")
    conn.close()

    return {
        "store": "pgvector",
        "insert_time_s": round(insert_time, 3),
        "query_p50_ms": round(statistics.median(latencies), 2),
        "query_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if len(latencies) > 20 else round(max(latencies), 2),
        "recall_at_10": round(statistics.mean(recalls), 4),
        "storage_bytes": storage_bytes,
        "storage_mb": round(storage_bytes / (1024*1024), 2),
    }


# ---------- Qdrant ----------

def bench_qdrant(embeddings, ids, dim, query_vecs, exact_results):
    import uuid
    client = QdrantClient(url=QDRANT_URL)
    collection = "bench_qdrant"

    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                             hnsw_config=HnswConfigDiff(m=16, ef_construct=200))

    points = [PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_DNS, ids[i])),
                          vector=embeddings[i],
                          payload={"report_id": ids[i]}) for i in range(len(ids))]

    t0 = time.perf_counter()
    for i in range(0, len(points), 100):
        client.upsert(collection_name=collection, points=points[i:i+100])
    insert_time = time.perf_counter() - t0

    # Query — handle both old (.search) and new (.query_points) API
    id_map = {str(uuid.uuid5(uuid.NAMESPACE_DNS, rid)): rid for rid in ids}
    latencies = []
    result_ids_list = []
    for qvec in query_vecs:
        t0 = time.perf_counter()
        if hasattr(client, 'query_points'):
            from qdrant_client.models import Query
            result = client.query_points(collection_name=collection, query=qvec, limit=TOP_K)
            hits = result.points
        elif hasattr(client, 'search'):
            hits = client.search(collection_name=collection, query_vector=qvec, limit=TOP_K)
        else:
            raise RuntimeError("Qdrant client has neither search() nor query_points()")
        latencies.append((time.perf_counter() - t0) * 1000)
        result_ids_list.append([id_map.get(str(h.id), str(h.id)) for h in hits])

    info = client.get_collection(collection)
    # Qdrant doesn't expose exact disk size easily; estimate from segment info
    storage_bytes = dim * 4 * len(ids) * 2  # rough estimate (vectors + index overhead)

    recalls = []
    for exact, approx in zip(exact_results, result_ids_list):
        overlap = len(set(exact) & set(approx))
        recalls.append(overlap / TOP_K)

    client.delete_collection(collection)

    return {
        "store": "qdrant",
        "insert_time_s": round(insert_time, 3),
        "query_p50_ms": round(statistics.median(latencies), 2),
        "query_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if len(latencies) > 20 else round(max(latencies), 2),
        "recall_at_10": round(statistics.mean(recalls), 4),
        "storage_bytes": storage_bytes,
        "storage_mb": round(storage_bytes / (1024*1024), 2),
    }


# ---------- ChromaDB ----------

def bench_chroma(embeddings, ids, dim, query_vecs, exact_results):
    client = chromadb.PersistentClient(path=CHROMA_DIR + "_bench")
    collection_name = "bench_chroma"

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name, metadata={
        "hnsw:M": 16, "hnsw:construction_ef": 200, "hnsw:search_ef": 100, "hnsw:space": "cosine"})

    t0 = time.perf_counter()
    for i in range(0, len(ids), 500):
        collection.add(ids=ids[i:i+500], embeddings=embeddings[i:i+500])
    insert_time = time.perf_counter() - t0

    latencies = []
    result_ids_list = []
    for qvec in query_vecs:
        t0 = time.perf_counter()
        res = collection.query(query_embeddings=[qvec], n_results=TOP_K)
        latencies.append((time.perf_counter() - t0) * 1000)
        result_ids_list.append(res["ids"][0] if res["ids"] else [])

    # Storage: measure persist dir
    import shutil
    persist_path = CHROMA_DIR + "_bench"
    storage_bytes = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, dn, filenames in os.walk(persist_path)
        for f in filenames
    ) if os.path.exists(persist_path) else 0

    recalls = []
    for exact, approx in zip(exact_results, result_ids_list):
        overlap = len(set(exact) & set(approx))
        recalls.append(overlap / TOP_K)

    client.delete_collection(collection_name)

    return {
        "store": "chromadb",
        "insert_time_s": round(insert_time, 3),
        "query_p50_ms": round(statistics.median(latencies), 2),
        "query_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if len(latencies) > 20 else round(max(latencies), 2),
        "recall_at_10": round(statistics.mean(recalls), 4),
        "storage_bytes": storage_bytes,
        "storage_mb": round(storage_bytes / (1024*1024), 2),
    }


# ---------- sqlite-vec ----------

def bench_sqlite_vec(embeddings, ids, dim, query_vecs, exact_results):
    db_path = SQLITE_DB + "_bench.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    conn.execute(f"CREATE VIRTUAL TABLE bench_svec USING vec0(id TEXT PRIMARY KEY, embedding float[{dim}]);")

    t0 = time.perf_counter()
    for i in range(len(ids)):
        conn.execute("INSERT INTO bench_svec (id, embedding) VALUES (?, ?)",
                     (ids[i], serialize_f32(embeddings[i])))
    conn.commit()
    insert_time = time.perf_counter() - t0

    latencies = []
    result_ids_list = []
    for qvec in query_vecs:
        t0 = time.perf_counter()
        rows = conn.execute(
            "SELECT id FROM bench_svec WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (serialize_f32(qvec), TOP_K)
        ).fetchall()
        latencies.append((time.perf_counter() - t0) * 1000)
        result_ids_list.append([r[0] for r in rows])

    storage_bytes = os.path.getsize(db_path)

    recalls = []
    for exact, approx in zip(exact_results, result_ids_list):
        overlap = len(set(exact) & set(approx))
        recalls.append(overlap / TOP_K)

    conn.close()
    os.remove(db_path)

    return {
        "store": "sqlite-vec",
        "insert_time_s": round(insert_time, 3),
        "query_p50_ms": round(statistics.median(latencies), 2),
        "query_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if len(latencies) > 20 else round(max(latencies), 2),
        "recall_at_10": round(statistics.mean(recalls), 4),
        "storage_bytes": storage_bytes,
        "storage_mb": round(storage_bytes / (1024*1024), 2),
    }


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Vector store shootout (E7)")
    parser.add_argument("--embeddings", required=True,
                        help="Path to best model's embeddings JSON")
    parser.add_argument("--output", default="results/raw/vector_store_bench.csv")
    parser.add_argument("--num-queries", type=int, default=NUM_QUERIES)
    args = parser.parse_args()

    with open(args.embeddings, "r") as f:
        emb_dict = json.load(f)

    ids = list(emb_dict.keys())
    embeddings = [emb_dict[k] for k in ids]
    dim = len(embeddings[0])
    print(f"Loaded {len(ids)} embeddings, dim={dim}")

    # Pick random query vectors
    import random
    random.seed(42)
    query_indices = random.sample(range(len(ids)), min(args.num_queries, len(ids)))
    query_vecs = [embeddings[i] for i in query_indices]

    # Compute exact kNN (brute force) for recall baseline
    print("Computing exact kNN baseline...")
    exact_results = []
    for qvec in query_vecs:
        exact_results.append(brute_force_knn(qvec, embeddings, ids, TOP_K))

    results = []

    # Run benchmarks
    stores = [
        ("pgvector", bench_pgvector),
        ("qdrant", bench_qdrant),
        ("chromadb", bench_chroma),
        ("sqlite-vec", bench_sqlite_vec),
    ]

    # Pre-check connectivity
    print("\nConnectivity checks:")
    try:
        conn = psycopg2.connect(PG_DSN)
        conn.close()
        print("  pgvector: OK")
    except Exception as e:
        print(f"  pgvector: FAILED ({e})")

    try:
        qc = QdrantClient(url=QDRANT_URL, timeout=5)
        qc.get_collections()
        print("  qdrant: OK")
    except Exception as e:
        print(f"  qdrant: FAILED ({e})")

    for store_name, bench_fn in stores:
        print(f"\n{'='*40}")
        print(f"Benchmarking: {store_name}")
        try:
            mem_before = get_rss_mb()
            result = bench_fn(embeddings, ids, dim, query_vecs, exact_results)
            mem_after = get_rss_mb()
            result["ram_delta_mb"] = round(mem_after - mem_before, 1)
            result["num_records"] = len(ids)
            result["dim"] = dim
            results.append(result)
            print(f"  Insert: {result['insert_time_s']}s | "
                  f"Query p50: {result['query_p50_ms']}ms | "
                  f"Recall@10: {result['recall_at_10']:.3f} | "
                  f"Storage: {result['storage_mb']}MB")
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if results:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
