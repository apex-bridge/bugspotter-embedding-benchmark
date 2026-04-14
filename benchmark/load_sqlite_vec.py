"""
Step 3d: Load embeddings into sqlite-vec.

Creates a virtual table per model. Brute-force exact search on small datasets.
"""

import json
import os
import argparse
import time
import struct
import sqlite3

import sqlite_vec

SQLITE_DB = os.getenv("SQLITE_DB", "results/bugdedup.db")


def serialize_f32(vector: list[float]) -> bytes:
    """Serialize a float list to bytes for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


def load_model_embeddings(conn: sqlite3.Connection, model_name: str,
                          embeddings: dict, reports: dict):
    """Create virtual table and load embeddings for one model."""
    table = f"bugs_{model_name.replace(':', '_').replace('/', '_').replace('-', '_')}"
    dim = len(next(iter(embeddings.values())))

    cur = conn.cursor()

    # Drop if exists
    cur.execute(f"DROP TABLE IF EXISTS {table};")
    cur.execute(f"DROP TABLE IF EXISTS {table}_meta;")

    # Create vec0 virtual table
    cur.execute(f"""
        CREATE VIRTUAL TABLE {table} USING vec0(
            id TEXT PRIMARY KEY,
            embedding float[{dim}]
        );
    """)

    # Metadata table (vec0 doesn't support extra columns)
    cur.execute(f"""
        CREATE TABLE {table}_meta (
            id TEXT PRIMARY KEY,
            group_id TEXT,
            error_type TEXT,
            url TEXT
        );
    """)

    # Insert
    t0 = time.perf_counter()
    for report_id, emb in embeddings.items():
        cur.execute(
            f"INSERT INTO {table} (id, embedding) VALUES (?, ?)",
            (report_id, serialize_f32(emb)),
        )
        report = reports.get(report_id, {})
        cur.execute(
            f"INSERT INTO {table}_meta (id, group_id, error_type, url) VALUES (?, ?, ?, ?)",
            (report_id, report.get("group", ""), report.get("error_type", ""), report.get("url", "")),
        )
    conn.commit()
    insert_time = time.perf_counter() - t0

    count = cur.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    print(f"  [{table}] {count} rows, dim={dim}")
    print(f"    Insert: {insert_time:.2f}s")
    return {"table": table, "count": count, "dim": dim, "insert_time_s": insert_time}


def main():
    parser = argparse.ArgumentParser(description="Load embeddings into sqlite-vec")
    parser.add_argument("--embeddings-dir", default="results/raw")
    parser.add_argument("--reports", default="data/bug_reports.json")
    parser.add_argument("--db", default=None)
    args = parser.parse_args()

    db_path = args.db or SQLITE_DB
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

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
        load_model_embeddings(conn, model_name, embeddings, reports)

    conn.close()
    print(f"\nDone. All models loaded into sqlite-vec at {db_path}")


if __name__ == "__main__":
    main()
