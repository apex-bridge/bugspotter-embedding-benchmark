"""
Step 3a: Load embeddings into PostgreSQL + pgvector.

Creates a table per model with HNSW index (m=16, ef_construction=200).
"""

import json
import os
import argparse
import time

import psycopg2
from psycopg2.extras import execute_values

PG_DSN = os.getenv("PG_DSN", "host=localhost port=5432 dbname=bugdedup user=postgres password=bench")


def get_connection():
    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = True
    return conn


def init_pgvector(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")


def load_model_embeddings(conn, model_name: str, embeddings: dict, reports: dict):
    """Create table and load embeddings for one model."""
    table = f"bugs_{model_name.replace(':', '_').replace('/', '_').replace('-', '_')}"
    original_dim = len(next(iter(embeddings.values())))

    # pgvector max indexed dimensions is 2000 — truncate if needed
    PGVECTOR_MAX_DIM = 2000
    truncated = original_dim > PGVECTOR_MAX_DIM
    dim = min(original_dim, PGVECTOR_MAX_DIM)
    if truncated:
        print(f"  NOTE: Truncating {original_dim}d -> {dim}d (pgvector limit)")

    with conn.cursor() as cur:
        # Drop and recreate
        cur.execute(f"DROP TABLE IF EXISTS {table};")
        cur.execute(f"""
            CREATE TABLE {table} (
                id TEXT PRIMARY KEY,
                embedding vector({dim}),
                metadata JSONB
            );
        """)

        # Batch insert
        rows = []
        for report_id, emb in embeddings.items():
            report = reports.get(report_id, {})
            metadata = {
                "group": report.get("group", ""),
                "error_type": report.get("error_type", ""),
                "url": report.get("url", ""),
            }
            # Truncate and L2-normalize if needed
            if truncated:
                import numpy as np
                vec = np.array(emb[:dim], dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                emb = vec.tolist()
            vec_str = "[" + ",".join(str(x) for x in emb) + "]"
            rows.append((report_id, vec_str, json.dumps(metadata)))

        t0 = time.perf_counter()
        execute_values(
            cur,
            f"INSERT INTO {table} (id, embedding, metadata) VALUES %s",
            rows,
            template="(%s, %s::vector, %s::jsonb)",
            page_size=100,
        )
        insert_time = time.perf_counter() - t0

        # Create HNSW index (dim is always <= 2000 after truncation)
        t0 = time.perf_counter()
        cur.execute(f"""
            CREATE INDEX ON {table} USING hnsw
            (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 200);
        """)
        cur.execute("SET hnsw.ef_search = 100;")
        index_type = "hnsw"
        index_time = time.perf_counter() - t0

    print(f"  [{table}] {len(rows)} rows, dim={dim}, index={index_type}")
    print(f"    Insert: {insert_time:.2f}s | Index build: {index_time:.2f}s")
    return {"table": table, "rows": len(rows), "dim": dim,
            "insert_time_s": insert_time, "index_time_s": index_time,
            "index_type": index_type}


def main():
    parser = argparse.ArgumentParser(description="Load embeddings into pgvector")
    parser.add_argument("--embeddings-dir", default="results/raw")
    parser.add_argument("--reports", default="data/bug_reports.json")
    args = parser.parse_args()

    with open(args.reports, "r", encoding="utf-8") as f:
        reports_list = json.load(f)
    reports = {r["id"]: r for r in reports_list}

    conn = get_connection()
    init_pgvector(conn)

    # Find all embedding files
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
        load_model_embeddings(conn, model_name, embeddings, reports)

    conn.close()
    print("\nDone. All models loaded into pgvector.")


if __name__ == "__main__":
    main()
