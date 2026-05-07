"""
quickstart.py
End-to-end smoke test with a single hard-coded question.
Run this first to verify your setup is working before running full experiments.

Usage:
    cd rag_project
    python quickstart.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Verify API key
if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not set.")
    print("Copy .env.example to .env and add your key.")
    sys.exit(1)

print("Step 1/4 — Preparing data...")
from src.data_prep import load_hotpotqa, chunk_passages
import json, os

os.makedirs("data", exist_ok=True)
if not os.path.exists("data/qa_pairs.json"):
    samples = load_hotpotqa(n_samples=50)   # small for smoke test
    chunks  = chunk_passages(samples)
    with open("data/qa_pairs.json", "w") as f: json.dump(samples, f)
    with open("data/chunks.json",   "w") as f: json.dump(chunks, f)
    print(f"  Saved {len(samples)} QA pairs, {len(chunks)} chunks.")
else:
    print("  data/qa_pairs.json already exists — skipping download.")

print("\nStep 2/4 — Building vector index...")
from src.retriever import DenseRetriever
retriever = DenseRetriever()
retriever.index_chunks()

print("\nStep 3/4 — Running a single question through all variants...")
from src.generator import Generator
from src.pipeline import BaselinePipeline, V1Pipeline, V3Pipeline

generator = Generator()
question = "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"

for PipelineClass, kwargs in [
    (BaselinePipeline, {"top_k": 5}),
    (V1Pipeline,       {"top_k": 5}),
    (V3Pipeline,       {}),
]:
    pipeline = PipelineClass(retriever, generator, **kwargs)
    result = pipeline.run(question)
    print(f"\n  [{pipeline.name}]")
    print(f"    answer      : {result['answer'][:120]}")
    print(f"    total ms    : {result['total_latency_ms']:.1f}")
    print(f"    top_k used  : {result['top_k_used']}")
    print(f"    cache hit   : {result['cache_hit']}")

print("\nStep 4/4 — Quick metric check...")
from src.evaluator import evaluate_results
with open("data/qa_pairs.json") as f:
    qa_pairs = json.load(f)

baseline = BaselinePipeline(retriever, generator, top_k=5)
results = [baseline.run(q["question"]) for q in qa_pairs[:10]]
metrics = evaluate_results(results, qa_pairs[:10])
print("  Baseline metrics (10 samples):")
for k, v in metrics.items():
    print(f"    {k}: {v}")

print("\n✓ Smoke test complete! Run the full experiment with:")
print("  python experiments/run_experiments.py --n_eval 200")
