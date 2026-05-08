# RAG Optimization Project
**Caching and Retrieval Strategies for Latency–Quality Trade-offs**

---

## Project Structure

```
rag_project/
├── src/
│   ├── data_prep.py     # Download & chunk HotpotQA
│   ├── retriever.py     # ChromaDB dense retriever + adaptive complexity
│   ├── cache.py         # ExactCache + SemanticCache
│   ├── generator.py     # OpenAI gpt-4o-mini generation
│   ├── pipeline.py      # Baseline, V1, V2, V3 variants
│   └── evaluator.py     # EM, F1, Recall@k, latency stats
├── experiments/
│   └── run_experiments.py   # Full experiment runner
├── data/                    # Auto-created: QA pairs, chunks, ChromaDB
├── results/                 # Auto-created: JSON results + summaries
├── quickstart.py            # Smoke test (run this first)
├── requirements.txt
└── .env.example
```

---

## Pipeline Variants

| Variant | Cache | Retrieval |
|---------|-------|-----------|
| Baseline | None | Dense, fixed top-k |
| V1 | Exact match | Dense, fixed top-k |
| V2 | Exact match | Dense, variable top-k (sweep) |
| V3 | Exact match | Adaptive top-k (complexity-based) |

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Run smoke test (downloads 50 samples, tests all variants)
python quickstart.py

# 4. Run full experiments (200 samples, all variants + V2 top-k sweep)
python experiments/run_experiments.py --n_eval 200
```

## Testing

```bash
# Run unit + lightweight integration tests
pytest -q
```

These tests are mock/fake based and do not overwrite `data/` or `results/`.
Integration coverage includes an end-to-end offline flow:
`pipeline.run(...) -> evaluate_results(...)`.

---

## Adaptive Retrieval (V3)

Query complexity is estimated from mean cosine distance between the query
embedding and the top-20 retrieved chunks:

- **Low complexity** (specific query, high similarity) → `top_k = 3`
- **High complexity** (vague query, low similarity) → `top_k = 10`

Threshold default: `0.55` (tunable in `pipeline.py`).

---

## Evaluation Metrics

| Category | Metric |
|----------|--------|
| Answer quality | Exact Match, Token F1 |
| Retrieval quality | Recall@k |
| Latency | Mean, P50, P95 (ms) |
| Cache | Hit rate |

Results are saved to `results/summaries.json` for Pareto analysis.

---

## Reproducible Experiment Checklist

To keep experiment outputs reproducible across runs:

1. Use the same dataset split and sample size (`--n_eval`).
2. Keep model settings unchanged (`gpt-4o-mini`, `temperature=0`, same embedding model).
3. Keep adaptive thresholds in `src/pipeline.py` unchanged.
4. Do not modify existing files under `data/` or `results/` before comparison.
5. Save each run's `results/summaries.json` with a timestamped copy for traceability.

---

## Troubleshooting

- `OPENAI_API_KEY not set`
  - Copy `.env.example` to `.env`, then set `OPENAI_API_KEY=...`.
- `FileNotFoundError: data/chunks.json`
  - Run `python quickstart.py` first to generate initial data files.
- ChromaDB index seems stale or incomplete
  - Ensure `data/chunks.json` matches your intended dataset, then rerun indexing.
- Slow first run
  - The first execution includes embedding/index warm-up; repeated runs should be faster.
- No `results/summaries.json`
  - Run `python experiments/run_experiments.py --n_eval 200` and check terminal errors.

---

## Week-by-Week Plan

| Week | Task |
|------|------|
| 1 | `python quickstart.py` — verify pipeline end-to-end |
| 2 | Run baseline, lock evaluation pipeline |
| 3 | Run V1 (exact cache), analyze hit rate vs latency |
| 4 | Run V2 top-k sweep, draw latency–quality curve |
| 5 | Run V3 adaptive, compare against V2 best config |
| 6 | Pareto analysis, write report |
