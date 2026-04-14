"""
Step 2: Generate embeddings for all bug reports through all 7 Ollama models.

For each model:
  - POST /api/embed with batches of 10
  - Measure wall-clock latency, record cold/warm start
  - Run 3 passes, take median latency (warm)
  - Save embeddings + timing data to results/raw/

Output:
  results/raw/embeddings_{model_name}.json  — {id: embedding_vector}
  results/raw/latency_{model_name}.csv      — per-batch timing
"""

import json
import time
import os
import csv
import argparse
import statistics

import requests
import psutil

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

MODELS = [
    "qwen3-embedding",       # 7.6B Q4_K_M — largest, SOTA
    "nomic-embed-text",      # 137M F16
    "mxbai-embed-large",     # 334M F16
    "bge-m3",                # 567M F16
    "all-minilm",            # 23M F16 — baseline
    "snowflake-arctic-embed", # 334M F16
]

BATCH_SIZE = 10
NUM_PASSES = 3
REQUEST_TIMEOUT = 600  # 10 min — large models on CPU are very slow

# Models >1B params get smaller batches to avoid timeouts on CPU
LARGE_MODEL_BATCH_SIZE = 1


def extract_console_errors(console_logs, max_errors=5):
    """Extract error/warn messages from console logs.
    Mirrors bugspotter-intelligence/utils/log_extractor.py exactly."""
    if not console_logs:
        return []
    errors = []
    for log in console_logs:
        # Support both dict format (BugSpotter native) and plain string (GitHub scraped)
        if isinstance(log, dict):
            level = log.get("level", "").lower()
            if level not in ("error", "warn"):
                continue
            message = log.get("message", "")
            stack = log.get("stack", "")
            parts = [message]
            if stack:
                stack_lines = stack.strip().split("\n")[:3]
                parts.extend(stack_lines)
            errors.append(" | ".join(parts))
        elif isinstance(log, str) and log.strip():
            errors.append(log.strip())
    return errors[:max_errors]


def extract_failed_requests(network_logs, max_requests=3):
    """Extract failed HTTP requests (status >= 400).
    Mirrors bugspotter-intelligence/utils/log_extractor.py exactly."""
    if not network_logs:
        return []
    failed = []
    for req in network_logs:
        status = req.get("status", 200)
        if status >= 400:
            method = req.get("method", "GET")
            url = req.get("url", "")
            duration = req.get("duration", 0)
            failed.append(f"{method} {url} returned {status} (took {duration}ms)")
    return failed[:max_requests]


def extract_environment_info(metadata):
    """Extract browser, OS, page path from metadata.
    Mirrors bugspotter-intelligence/utils/log_extractor.py exactly."""
    if not metadata:
        return ""
    parts = []
    if browser := metadata.get("browser"):
        parts.append(f"Browser: {browser}")
    if os_name := metadata.get("os"):
        parts.append(f"OS: {os_name}")
    if url := metadata.get("url"):
        from urllib.parse import urlparse
        path = urlparse(url).path
        if path and path != "/":
            parts.append(f"Page: {path}")
    return " | ".join(parts)


def prepare_text(report: dict) -> str:
    """Build embedding text matching BugSpotter's build_embedding_text() exactly.

    Format: title | description | console_errors (up to 5) |
            failed_requests (up to 3) | Browser: X | OS: Y | Page: /path
    """
    import re

    parts = [report.get("title", "")]

    if report.get("description"):
        parts.append(report["description"])

    # Console errors (up to 5, with first 3 stack trace lines each)
    console_errors = extract_console_errors(report.get("console_logs", []))
    parts.extend(console_errors)

    # Failed network requests (up to 3)
    failed_reqs = extract_failed_requests(report.get("network_logs", []))
    parts.extend(failed_reqs)

    # Environment info: browser, OS, page path
    # Build metadata dict from report fields (GitHub/synthetic have flat fields)
    metadata = report.get("metadata")
    if not metadata:
        metadata = {}
        if report.get("browser"):
            metadata["browser"] = report["browser"]
        if report.get("url"):
            metadata["url"] = report["url"]
    env = extract_environment_info(metadata)
    if env:
        parts.append(env)

    text = " | ".join(filter(None, parts))

    # Remove ANSI codes
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    return text[:2048]  # Safety truncation


def embed_batch(model: str, texts: list[str]) -> tuple[list, float]:
    """Call Ollama /api/embed and return embeddings + latency."""
    t0 = time.perf_counter()
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": model, "input": texts},
        timeout=REQUEST_TIMEOUT,
    )
    latency = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    return data["embeddings"], latency


def get_memory_usage_mb() -> float:
    """Current process RSS in MB."""
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 * 1024)


def run_model(model: str, reports: list[dict], output_dir: str, text_strategy: str = "default"):
    """Embed all reports with one model, running NUM_PASSES passes."""
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"{'='*60}")

    # Prepare texts
    texts = [prepare_text(r) for r in reports]
    ids = [r["id"] for r in reports]

    # Warmup: load the model into Ollama memory before timed runs
    print(f"  Warming up model (this may take a while on first load)...")
    try:
        embed_batch(model, ["warmup test"])
        print(f"  Model loaded successfully.")
    except Exception as e:
        print(f"  WARNING: warmup failed — {e}")
        print(f"  Skipping this model.")
        return {}

    # Detect large models — use batch size of 1 to avoid timeouts
    warmup_emb, _ = embed_batch(model, ["size probe"])
    emb_dim = len(warmup_emb[0])
    is_large = emb_dim > 1024 or "4b" in model or "7b" in model or "qwen3" in model.lower()
    batch_size = LARGE_MODEL_BATCH_SIZE if is_large else BATCH_SIZE
    if is_large:
        print(f"  Large model detected (dim={emb_dim}). Using batch_size={batch_size}")

    # Batching
    batches = []
    for i in range(0, len(texts), batch_size):
        batches.append((ids[i:i+batch_size], texts[i:i+batch_size]))

    all_pass_latencies = []
    final_embeddings = {}
    num_passes = 2 if is_large else NUM_PASSES

    for pass_idx in range(num_passes):
        pass_type = "cold" if pass_idx == 0 else "warm"
        print(f"  Pass {pass_idx+1}/{NUM_PASSES} ({pass_type})...")

        pass_latencies = []
        pass_embeddings = {}
        mem_before = get_memory_usage_mb()

        for batch_num, (batch_ids, batch_texts) in enumerate(batches, 1):
            if batch_num % 10 == 1 or batch_size == 1:
                done = (batch_num - 1) * batch_size
                print(f"    [{done}/{len(texts)}] ...", end="\r", flush=True)
            try:
                embeddings, latency = embed_batch(model, batch_texts)
                pass_latencies.append({
                    "batch_size": len(batch_texts),
                    "latency_s": latency,
                    "latency_per_item_ms": (latency / len(batch_texts)) * 1000,
                    "pass": pass_idx + 1,
                    "pass_type": pass_type,
                })
                for bid, emb in zip(batch_ids, embeddings):
                    pass_embeddings[bid] = emb
            except Exception as e:
                print(f"    ERROR: {e}")
                # Retry once with smaller batch
                for single_id, single_text in zip(batch_ids, batch_texts):
                    try:
                        embs, lat = embed_batch(model, [single_text])
                        pass_latencies.append({
                            "batch_size": 1,
                            "latency_s": lat,
                            "latency_per_item_ms": lat * 1000,
                            "pass": pass_idx + 1,
                            "pass_type": pass_type,
                        })
                        pass_embeddings[single_id] = embs[0]
                    except Exception as e2:
                        print(f"    SKIP {single_id}: {e2}")

        mem_after = get_memory_usage_mb()
        total_time = sum(l["latency_s"] for l in pass_latencies)
        throughput = len(texts) / total_time if total_time > 0 else 0

        print(f"    Total: {total_time:.1f}s | Throughput: {throughput:.1f} emb/s | "
              f"RAM delta: {mem_after - mem_before:.1f}MB")

        all_pass_latencies.extend(pass_latencies)

        # Keep embeddings from last pass (warm)
        if pass_embeddings:
            final_embeddings = pass_embeddings

    # Compute summary stats from warm passes
    warm_latencies = [l["latency_per_item_ms"] for l in all_pass_latencies if l["pass_type"] == "warm"]
    if warm_latencies:
        sorted_lat = sorted(warm_latencies)
        n = len(sorted_lat)
        summary = {
            "model": model,
            "total_reports": len(reports),
            "embedding_dim": len(next(iter(final_embeddings.values()))) if final_embeddings else 0,
            "median_latency_ms": statistics.median(warm_latencies),
            "p95_latency_ms": sorted_lat[int(n * 0.95)] if n > 20 else sorted_lat[-1],
            "p99_latency_ms": sorted_lat[int(n * 0.99)] if n > 100 else sorted_lat[-1],
            "throughput_emb_per_s": len(reports) / sum(
                l["latency_s"] for l in all_pass_latencies
                if l["pass_type"] == "warm" and l["pass"] == NUM_PASSES
            ) if any(l["pass"] == NUM_PASSES for l in all_pass_latencies) else 0,
        }
        print(f"  Summary: dim={summary['embedding_dim']}, "
              f"median={summary['median_latency_ms']:.1f}ms, "
              f"throughput={summary['throughput_emb_per_s']:.1f} emb/s")

    # Save embeddings
    model_safe = model.replace(":", "_").replace("/", "_")
    emb_path = os.path.join(output_dir, f"embeddings_{model_safe}.json")
    with open(emb_path, "w") as f:
        json.dump(final_embeddings, f)
    print(f"  Saved {len(final_embeddings)} embeddings to {emb_path}")

    # Save latency data
    lat_path = os.path.join(output_dir, f"latency_{model_safe}.csv")
    if all_pass_latencies:
        with open(lat_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_pass_latencies[0].keys())
            writer.writeheader()
            writer.writerows(all_pass_latencies)
        print(f"  Saved latency data to {lat_path}")

    return final_embeddings


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for all models")
    parser.add_argument("--input", default="data/bug_reports.json")
    parser.add_argument("--output-dir", default="results/raw")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Specific models to run (default: all 7)")
    parser.add_argument("--ollama-url", default=None)
    args = parser.parse_args()

    if args.ollama_url:
        global OLLAMA_URL
        OLLAMA_URL = args.ollama_url

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f:
        reports = json.load(f)
    print(f"Loaded {len(reports)} bug reports from {args.input}")

    models = args.models or MODELS

    for model in models:
        # Unload any previously loaded model to ensure clean state
        try:
            requests.post(f"{OLLAMA_URL}/api/generate",
                          json={"model": model, "keep_alive": 0}, timeout=10)
        except Exception:
            pass
        time.sleep(2)  # Let Ollama fully release memory

        try:
            run_model(model, reports, args.output_dir)
        except Exception as e:
            print(f"\nFAILED: {model} — {e}")
            continue

    print(f"\n{'='*60}")
    print("All models processed. Results in:", args.output_dir)


if __name__ == "__main__":
    main()
