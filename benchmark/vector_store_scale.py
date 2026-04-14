"""
Vector Store Scale Test: 1K / 10K / 50K / 100K records.

Generates synthetic embeddings (same dimensions as mxbai-embed-large = 1024)
and benchmarks insert + query performance across all 4 stores at each scale.

This answers: "At what point do you need a dedicated vector DB?"

Output: results/raw/vector_store_scale.csv
"""

import json
import os
import time
import csv
import argparse
import struct
import statistics
import sqlite3
import random

import numpy as np
import psutil
import psycopg2
from psycopg2.extras import execute_values
import chromadb
import sqlite_vec
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, HnswConfigDiff,
)

PG_DSN = os.getenv("PG_DSN", "host=localhost port=5432 dbname=bugdedup user=postgres password=bench")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma_scale")
SQLITE_DB = os.getenv("SQLITE_SCALE_DB", "results/scale_vectors.db")

NUM_QUERIES = 100
TOP_K = 10
DIM = 1024
SCALES = [1_000, 10_000, 50_000, 100_000]


def serialize_f32(vector):
    return struct.pack(f"{len(vector)}f", *vector)


def generate_embeddings(n, dim):
    """Generate n random unit vectors of given dimension."""
    vecs = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms
    return vecs


def pick_queries(embeddings, n_queries):
    indices = random.sample(range(len(embeddings)), min(n_queries, len(embeddings)))
    return [embeddings[i].tolist() for i in indices]


# ---------- pgvector ----------

def bench_pgvector(ids, embeddings, query_vecs, scale):
    table = f"scale_pg_{scale}"
    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(f"DROP TABLE IF EXISTS {table};")
    cur.execute(f"CREATE TABLE {table} (id TEXT PRIMARY KEY, embedding vector({DIM}));")

    # Insert
    t0 = time.perf_counter()
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        rows = [(ids[j], "[" + ",".join(str(x) for x in embeddings[j]) + "]")
                for j in range(i, min(i + batch_size, len(ids)))]
        execute_values(cur, f"INSERT INTO {table} (id, embedding) VALUES %s",
                       rows, template="(%s, %s::vector)", page_size=500)

    # Build HNSW index
    cur.execute(f"""CREATE INDEX ON {table} USING hnsw
        (embedding vector_cosine_ops) WITH (m=16, ef_construction=200);""")
    cur.execute("SET hnsw.ef_search = 100;")
    insert_time = time.perf_counter() - t0

    # Warmup
    vec_str = "[" + ",".join(str(x) for x in query_vecs[0]) + "]"
    cur.execute(f"SELECT id FROM {table} ORDER BY embedding <=> %s::vector LIMIT %s", (vec_str, TOP_K))
    cur.fetchall()

    # Query
    latencies = []
    for qvec in query_vecs:
        vec_str = "[" + ",".join(str(x) for x in qvec) + "]"
        t0 = time.perf_counter()
        cur.execute(f"SELECT id FROM {table} ORDER BY embedding <=> %s::vector LIMIT %s", (vec_str, TOP_K))
        cur.fetchall()
        latencies.append((time.perf_counter() - t0) * 1000)

    # Storage
    cur.execute(f"SELECT pg_total_relation_size('{table}');")
    storage_bytes = cur.fetchone()[0]

    cur.execute(f"DROP TABLE IF EXISTS {table};")
    conn.close()

    return {
        "store": "pgvector",
        "scale": scale,
        "insert_time_s": round(insert_time, 2),
        "query_p50_ms": round(statistics.median(latencies), 2),
        "query_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
        "query_p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 2),
        "storage_mb": round(storage_bytes / (1024 * 1024), 1),
    }


# ---------- Qdrant ----------

def bench_qdrant(ids, embeddings, query_vecs, scale):
    import uuid
    client = QdrantClient(url=QDRANT_URL, timeout=120)
    collection = f"scale_qd_{scale}"

    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(collection,
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(m=16, ef_construct=200))

    # Insert
    t0 = time.perf_counter()
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        points = [PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, ids[j])),
            vector=embeddings[j].tolist(),
            payload={"report_id": ids[j]}
        ) for j in range(i, min(i + batch_size, len(ids)))]
        client.upsert(collection_name=collection, points=points)
    insert_time = time.perf_counter() - t0

    # Warmup
    if hasattr(client, 'query_points'):
        client.query_points(collection_name=collection, query=query_vecs[0], limit=TOP_K)
    elif hasattr(client, 'search'):
        client.search(collection_name=collection, query_vector=query_vecs[0], limit=TOP_K)

    # Query
    latencies = []
    for qvec in query_vecs:
        t0 = time.perf_counter()
        if hasattr(client, 'query_points'):
            client.query_points(collection_name=collection, query=qvec, limit=TOP_K)
        else:
            client.search(collection_name=collection, query_vector=qvec, limit=TOP_K)
        latencies.append((time.perf_counter() - t0) * 1000)

    # Storage estimate
    info = client.get_collection(collection)
    storage_bytes = DIM * 4 * scale * 2  # rough estimate

    client.delete_collection(collection)

    return {
        "store": "qdrant",
        "scale": scale,
        "insert_time_s": round(insert_time, 2),
        "query_p50_ms": round(statistics.median(latencies), 2),
        "query_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
        "query_p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 2),
        "storage_mb": round(storage_bytes / (1024 * 1024), 1),
    }


# ---------- ChromaDB ----------

def bench_chroma(ids, embeddings, query_vecs, scale):
    persist_dir = f"{CHROMA_DIR}_scale_{scale}"
    client = chromadb.PersistentClient(path=persist_dir)
    collection_name = f"scale_ch_{scale}"

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name, metadata={
        "hnsw:M": 16, "hnsw:construction_ef": 200, "hnsw:search_ef": 100, "hnsw:space": "cosine"})

    # Insert
    t0 = time.perf_counter()
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end],
            embeddings=[embeddings[j].tolist() for j in range(i, end)],
        )
    insert_time = time.perf_counter() - t0

    # Warmup
    collection.query(query_embeddings=[query_vecs[0]], n_results=TOP_K)

    # Query
    latencies = []
    for qvec in query_vecs:
        t0 = time.perf_counter()
        collection.query(query_embeddings=[qvec], n_results=TOP_K)
        latencies.append((time.perf_counter() - t0) * 1000)

    # Storage
    storage_bytes = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, dn, filenames in os.walk(persist_dir)
        for f in filenames
    ) if os.path.exists(persist_dir) else 0

    client.delete_collection(collection_name)
    import shutil
    shutil.rmtree(persist_dir, ignore_errors=True)

    return {
        "store": "chromadb",
        "scale": scale,
        "insert_time_s": round(insert_time, 2),
        "query_p50_ms": round(statistics.median(latencies), 2),
        "query_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
        "query_p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 2),
        "storage_mb": round(storage_bytes / (1024 * 1024), 1),
    }


# ---------- sqlite-vec ----------

def bench_sqlite_vec(ids, embeddings, query_vecs, scale):
    db_path = f"results/scale_svec_{scale}.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.execute(f"CREATE VIRTUAL TABLE scale_sv USING vec0(id TEXT PRIMARY KEY, embedding float[{DIM}]);")

    # Insert
    t0 = time.perf_counter()
    for i in range(len(ids)):
        conn.execute("INSERT INTO scale_sv (id, embedding) VALUES (?, ?)",
                     (ids[i], serialize_f32(embeddings[i])))
        if i % 5000 == 0:
            conn.commit()
    conn.commit()
    insert_time = time.perf_counter() - t0

    # Warmup
    conn.execute("SELECT id FROM scale_sv WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                 (serialize_f32(query_vecs[0]), TOP_K)).fetchall()

    # Query
    latencies = []
    for qvec in query_vecs:
        t0 = time.perf_counter()
        conn.execute("SELECT id FROM scale_sv WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                     (serialize_f32(qvec), TOP_K)).fetchall()
        latencies.append((time.perf_counter() - t0) * 1000)

    storage_bytes = os.path.getsize(db_path)
    conn.close()
    os.remove(db_path)

    return {
        "store": "sqlite-vec",
        "scale": scale,
        "insert_time_s": round(insert_time, 2),
        "query_p50_ms": round(statistics.median(latencies), 2),
        "query_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
        "query_p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 2),
        "storage_mb": round(storage_bytes / (1024 * 1024), 1),
    }


# ---------- Main ----------

def main():
    global DIM

    parser = argparse.ArgumentParser(description="Vector store scale test")
    parser.add_argument("--output", default="results/raw/vector_store_scale.csv")
    parser.add_argument("--scales", nargs="*", type=int, default=SCALES)
    parser.add_argument("--dim", type=int, default=DIM)
    args = parser.parse_args()

    DIM = args.dim
    random.seed(42)
    np.random.seed(42)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    stores = [
        ("pgvector", bench_pgvector),
        ("qdrant", bench_qdrant),
        ("chromadb", bench_chroma),
        ("sqlite-vec", bench_sqlite_vec),
    ]

    all_results = []

    for scale in args.scales:
        print(f"\n{'='*60}")
        print(f"Scale: {scale:,} records, {DIM} dims")
        print(f"{'='*60}")

        # Generate data
        print("  Generating embeddings...")
        embeddings = generate_embeddings(scale, DIM)
        ids = [f"rec_{i:06d}" for i in range(scale)]
        query_vecs = pick_queries(embeddings, NUM_QUERIES)

        for store_name, bench_fn in stores:
            print(f"\n  {store_name}...")
            try:
                result = bench_fn(ids, embeddings, query_vecs, scale)
                all_results.append(result)
                print(f"    Insert: {result['insert_time_s']}s | "
                      f"Query p50: {result['query_p50_ms']}ms | "
                      f"p95: {result['query_p95_ms']}ms | "
                      f"Storage: {result['storage_mb']}MB")
            except Exception as e:
                import traceback
                print(f"    FAILED: {e}")
                traceback.print_exc()

    # Save
    if all_results:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nSaved: {args.output}")

        # Print summary table
        print(f"\n{'='*60}")
        print(f"{'Store':<15} {'Scale':>8} {'Insert':>8} {'p50':>8} {'p95':>8} {'Storage':>8}")
        print(f"{'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for r in all_results:
            print(f"{r['store']:<15} {r['scale']:>8,} {r['insert_time_s']:>7.1f}s "
                  f"{r['query_p50_ms']:>7.1f}ms {r['query_p95_ms']:>7.1f}ms "
                  f"{r['storage_mb']:>6.1f}MB")


if __name__ == "__main__":
    main()
