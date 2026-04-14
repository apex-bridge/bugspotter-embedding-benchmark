# Embedding Models for Bug Report Deduplication

We benchmarked 6 self-hosted embedding models for duplicate bug report detection. All models run locally via [Ollama](https://ollama.com/) on a single CPU server (€25/mo). No data leaves your network.

**Full write-up:** [I Benchmarked 6 Embedding Models for Bug Report Deduplication. Here's What Actually Works.](https://dev.to/bugspotter/embedding-models-bug-dedup)

## Key Findings

- **A 335M model matched a 7.6B model.** mxbai-embed-large (F1=0.996) vs Qwen3 (F1=0.995) — 22x smaller, 10x faster.
- **Threshold 0.9 is a trap.** At cosine ≥ 0.9, recall drops to 8–42%. Optimal thresholds range from 0.55 to 0.75, different for every model.
- **Machine-captured metadata > human descriptions.** Console errors, network logs, and stack traces improved F1 from 0.86 to 0.995.
- **You probably don't need a dedicated vector DB.** pgvector was faster than Qdrant at bug-tracker scale (0.9ms vs 4.5ms).

## Models Tested

| Model | Params | Dims | F1 | Latency |
|-------|--------|------|----|---------|
| mxbai-embed-large | 335M | 1024 | 0.996 | 186ms |
| bge-m3 | 568M | 1024 | 0.995 | 221ms |
| qwen3-embedding | 7.6B | 4096 | 0.995 | 1,932ms |
| all-minilm | 22M | 384 | 0.992 | 13ms |
| snowflake-arctic-embed | 334M | 768 | 0.982 | 188ms |
| nomic-embed-text | 137M | 768 | 0.979 | 69ms |

Vector stores: pgvector, Qdrant, ChromaDB, sqlite-vec — all compared at 240 and up to 100K records.

## Quick Start

**Requirements:** Docker, Python 3.10+, ~16GB RAM (8GB if skipping qwen3)

```bash
git clone https://github.com/apex-bridge/bugspotter-embedding-benchmark.git
cd bugspotter-embedding-benchmark

# Start infrastructure
docker compose up -d

# Pull embedding models
./setup.sh

# Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the full pipeline (~4-5 hours on 8 vCPU)
./benchmark/run_all.sh
```

Or run steps individually:

```bash
# 1. Generate dataset (540 bug reports, ~2900 labeled pairs)
python data/generate_synthetic.py
python data/generate_pairs.py

# 2. Embed with all models
python benchmark/embed_all.py

# 3. Load into vector stores
python benchmark/load_pgvector.py
python benchmark/load_qdrant.py
python benchmark/load_chroma.py
python benchmark/load_sqlite_vec.py

# 4. Evaluate
python benchmark/compute_similarity.py
python benchmark/sweep_threshold.py

# 5. Additional experiments
python benchmark/mrl_truncation.py          # MRL dimension truncation
python benchmark/e4_embedding_strategy.py   # What text to embed?
python benchmark/vector_store_bench.py      # Vector store comparison
python benchmark/vector_store_scale.py      # Scale test (1K–100K)

# 6. Generate all figures
python analysis/results_summary.py
for f in analysis/fig_*.py; do python "$f"; done
```

## One-Click Cloud Setup

Spin up a Hetzner CPX42 (8 vCPU, 16GB RAM, €25/mo), SSH in, and run:

```bash
bash <(curl -sSL https://raw.githubusercontent.com/apex-bridge/bugspotter-embedding-benchmark/main/deploy/setup.sh)
```

This installs everything, pulls models, generates the dataset, and runs the full benchmark. Results in `results/`.

## Repository Structure

```
├── setup.sh                    # Pull Ollama models + init services
├── docker-compose.yml          # Ollama + pgvector + Qdrant
├── requirements.txt            # Python dependencies (pinned ranges)
├── data/
│   ├── scrape_github.py        # Scrape GitHub Issues (300 reports)
│   ├── generate_synthetic.py   # Synthetic bug reports (200 + 40 SDK captures)
│   └── generate_pairs.py       # Ground truth pairs (D1–D4)
├── benchmark/
│   ├── run_all.sh              # Full pipeline orchestrator
│   ├── embed_all.py            # Generate embeddings (6 models via Ollama)
│   ├── compute_similarity.py   # Pairwise cosine similarity
│   ├── sweep_threshold.py      # Threshold sweep (P/R/F1/AUC)
│   ├── e4_embedding_strategy.py # What text to embed? (title vs full capture)
│   ├── mrl_truncation.py       # Matryoshka dimension truncation
│   ├── vector_store_bench.py   # 4-store comparison (real data)
│   ├── vector_store_scale.py   # Scale test (1K–100K synthetic)
│   ├── load_pgvector.py        # Load embeddings into PostgreSQL
│   ├── load_qdrant.py          # Load embeddings into Qdrant
│   ├── load_chroma.py          # Load embeddings into ChromaDB
│   └── load_sqlite_vec.py      # Load embeddings into sqlite-vec
├── analysis/
│   ├── plot_config.py          # Catppuccin Mocha theme for all figures
│   ├── results_summary.py      # Generate markdown results tables
│   ├── e5_hard_negatives.py    # Hard negatives deep dive
│   └── fig_*.py                # 13 figure scripts (hero scatter, PR curves, etc.)
└── deploy/
    ├── setup.sh                # One-click Hetzner setup
    └── run_clean.sh            # Full clean run with logging
```

## Dataset

540 bug reports from 3 sources:
- **300** real GitHub Issues (React, Next.js, VS Code, Angular, Vue, Svelte, Tailwind CSS)
- **200** synthetic (20 bug archetypes × 10 paraphrases each)
- **40** BugSpotter SDK captures (structured: console errors, network logs, stack traces)

~2,900 labeled pairs across 4 difficulty levels:
- **D1** — Exact duplicates (sanity check)
- **D2** — Semantic duplicates / paraphrases (main test)
- **D3** — Hard negatives: different bugs, same component (critical category)
- **D4** — Easy negatives: completely different bugs

## Hardware

Tested on Hetzner CPX42: 8 vCPU (AMD EPYC), 16GB RAM, 320GB NVMe, Ubuntu 24.04. Total cost: ~€0.20 for a full run.

## Notes

- **Ollama version:** We used Ollama v0.6.2. Ollama has had embedding consistency issues across versions ([#3777](https://github.com/ollama/ollama/issues/3777), [#4207](https://github.com/ollama/ollama/issues/4207)). Pin the version in production.
- **Python:** Tested with Python 3.12 on Ubuntu 24.04.
- **PostgreSQL password:** Defaults to `bench`. Override with `POSTGRES_PASSWORD` env var.

## License

MIT

---

*Built for [BugSpotter](https://github.com/apex-bridge/bugspotter) — the self-hosted bug reporting platform.*
