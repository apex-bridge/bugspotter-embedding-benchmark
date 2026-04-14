"""
E4: Embedding Text Strategy Comparison.

Tests 4 different ways to build embedding text from a bug report:
  A: title only
  B: title + description
  C: title + description + first console error (BugSpotter default)
  D: title + description + all console errors + network logs + stack trace + env

Uses the best model from E1 (mxbai-embed-large).
Re-embeds all reports with each strategy, computes similarity, sweeps thresholds.

Output:
  results/raw/embedding_strategy.csv — F1 per strategy
"""

import json
import csv
import os
import time
import argparse
import re

import requests
import numpy as np

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
REQUEST_TIMEOUT = 600
BATCH_SIZE = 10


def strategy_a(report):
    """Title only."""
    return report.get("title", "")


def strategy_b(report):
    """Title + description."""
    parts = [report.get("title", "")]
    if report.get("description"):
        parts.append(report["description"])
    return " | ".join(filter(None, parts))


def strategy_c(report):
    """Title + description + first console error. (Simplified BugSpotter default)"""
    parts = [report.get("title", "")]
    if report.get("description"):
        parts.append(report["description"])
    logs = report.get("console_logs", [])
    if logs:
        first = logs[0] if isinstance(logs[0], str) else logs[0].get("message", "")
        if first:
            parts.append(first)
    return " | ".join(filter(None, parts))


def strategy_d(report):
    """Everything: title + desc + all console errors + network logs + stack trace + browser/OS/page.
    Matches BugSpotter's full build_embedding_text()."""
    parts = [report.get("title", "")]

    if report.get("description"):
        parts.append(report["description"])

    # All console errors (up to 5)
    for log in (report.get("console_logs") or [])[:5]:
        if isinstance(log, str):
            parts.append(log)
        elif isinstance(log, dict):
            msg = log.get("message", "")
            stack = log.get("stack", "")
            if msg:
                parts.append(msg)
            if stack:
                parts.extend(stack.strip().split("\n")[:3])

    # Network logs (up to 3)
    for req in (report.get("network_logs") or [])[:3]:
        status = req.get("status", 200)
        if status >= 400 or status == 0:
            parts.append(f"{req.get('method', 'GET')} {req.get('url', '')} returned {status} (took {req.get('duration', 0)}ms)")

    # Stack trace
    if report.get("stack_trace"):
        parts.extend(report["stack_trace"].strip().split("\n")[:3])

    # Environment
    env_parts = []
    if report.get("browser"):
        env_parts.append(f"Browser: {report['browser']}")
    if report.get("url"):
        from urllib.parse import urlparse
        path = urlparse(report["url"]).path
        if path and path != "/":
            env_parts.append(f"Page: {path}")
    if env_parts:
        parts.append(" | ".join(env_parts))

    text = " | ".join(filter(None, parts))
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    return text[:2048]


STRATEGIES = {
    "A: Title only": strategy_a,
    "B: Title + Desc": strategy_b,
    "C: Title + Desc + Console": strategy_c,
    "D: All (BugSpotter full)": strategy_d,
}


def embed_batch(model, texts):
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": model, "input": texts},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


def sweep_best_f1(scores, labels):
    best_f1, best_result = 0, {}
    for threshold in np.arange(0.40, 1.00, 0.01):
        predicted = (scores >= threshold).astype(int)
        tp = ((predicted == 1) & (labels == 1)).sum()
        fp = ((predicted == 1) & (labels == 0)).sum()
        fn = ((predicted == 0) & (labels == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_result = {"threshold": round(threshold, 2),
                           "precision": round(float(prec), 4),
                           "recall": round(float(rec), 4),
                           "f1": round(float(f1), 4)}
    return best_result


def main():
    parser = argparse.ArgumentParser(description="E4: Embedding strategy comparison")
    parser.add_argument("--model", default="mxbai-embed-large",
                        help="Model to test strategies with")
    parser.add_argument("--reports", default="data/bug_reports.json")
    parser.add_argument("--pairs", default="data/pairs_ground_truth.csv")
    parser.add_argument("--output", default="results/raw/embedding_strategy.csv")
    args = parser.parse_args()

    with open(args.reports, encoding="utf-8") as f:
        reports = json.load(f)
    with open(args.pairs, encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))

    # Normalize model name: "mxbai-embed-large_latest" -> "mxbai-embed-large:latest"
    model = args.model.replace("_latest", ":latest")
    if ":latest" not in model and "_" not in model:
        model = model  # already clean
    print(f"Model: {model}")
    print(f"Reports: {len(reports)}, Pairs: {len(pairs)}")

    # Warmup
    print("Warming up model...")
    embed_batch(model, ["warmup"])

    results = []

    for strategy_name, strategy_fn in STRATEGIES.items():
        print(f"\n{'='*50}")
        print(f"Strategy: {strategy_name}")

        # Build texts
        texts = [strategy_fn(r) for r in reports]
        ids = [r["id"] for r in reports]

        # Show sample
        print(f"  Sample: {texts[0][:100]}...")
        avg_len = np.mean([len(t.split()) for t in texts])
        print(f"  Avg length: {avg_len:.0f} words")

        # Embed in batches
        embeddings = {}
        t0 = time.perf_counter()
        for i in range(0, len(texts), BATCH_SIZE):
            batch_ids = ids[i:i+BATCH_SIZE]
            batch_texts = texts[i:i+BATCH_SIZE]
            embs = embed_batch(model, batch_texts)
            for bid, emb in zip(batch_ids, embs):
                embeddings[bid] = emb
            if i % 100 == 0:
                print(f"  [{i}/{len(texts)}]", end="\r", flush=True)
        embed_time = time.perf_counter() - t0
        print(f"  Embedded {len(embeddings)} reports in {embed_time:.1f}s")

        # Compute cosine for all pairs
        scores_list, labels_list = [], []
        for pair in pairs:
            a_id, b_id = pair["report_a_id"], pair["report_b_id"]
            if a_id not in embeddings or b_id not in embeddings:
                continue
            score = cosine_similarity(embeddings[a_id], embeddings[b_id])
            scores_list.append(score)
            labels_list.append(1 if pair["label"] == "duplicate" else 0)

        scores = np.array(scores_list)
        labels = np.array(labels_list)

        best = sweep_best_f1(scores, labels)
        print(f"  Best: F1={best['f1']:.3f} @ threshold={best['threshold']}")

        results.append({
            "strategy": strategy_name,
            "f1": best["f1"],
            "precision": best["precision"],
            "recall": best["recall"],
            "threshold": best["threshold"],
            "avg_text_words": round(avg_len, 1),
            "embed_time_s": round(embed_time, 1),
        })

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {args.output}")

    # Summary
    print("\n" + "="*50)
    print("STRATEGY COMPARISON:")
    for r in sorted(results, key=lambda x: -x["f1"]):
        print(f"  {r['strategy']:30s} F1={r['f1']:.3f}  threshold={r['threshold']}  ({r['avg_text_words']:.0f} words)")


if __name__ == "__main__":
    main()
