"""
run_experiments.py
Run all four pipeline variants on a subset of HotpotQA
and save raw results + evaluation summaries.

Usage:
    cd rag_project
    python experiments/run_experiments.py --n_eval 200 --top_k 5
"""

import argparse
import json
import os
import sys
import time
from dotenv import load_dotenv

# Make sure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from src.retriever import DenseRetriever
from src.generator import Generator
from src.pipeline import BaselinePipeline, V1Pipeline, V2Pipeline, V3Pipeline
from src.evaluator import evaluate_results, save_results

TOP_K_VALUES = [1, 3, 5, 10, 20]  # for V2 sweep


def load_qa_pairs(path: str = "data/qa_pairs.json", n: int = 200) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data[:n]


def run_variant(pipeline, questions: list[str], variant_name: str) -> list[dict]:
    print(f"\n{'='*50}")
    print(f"Running variant: {variant_name}  ({len(questions)} queries)")
    print("="*50)
    results = []
    for i, q in enumerate(questions):
        try:
            if variant_name == "v2_variable_topk":
                # V2 uses fixed top_k=5 for fair comparison; sweep done separately
                r = pipeline.run(q, top_k=5)
            else:
                r = pipeline.run(q)
            results.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(questions)} done")
        except Exception as e:
            print(f"  [ERROR] q={q[:60]}... → {e}")
    return results


def run_topk_sweep(questions: list[str], retriever, generator) -> dict:
    """V2 top-k sweep across TOP_K_VALUES."""
    print(f"\n{'='*50}")
    print("Running V2 top-k sweep")
    print("="*50)
    sweep_results = {}
    pipeline = V2Pipeline(retriever, generator)
    for k in TOP_K_VALUES:
        print(f"  top-k = {k}")
        results = []
        for q in questions:
            try:
                results.append(pipeline.run(q, top_k=k))
            except Exception as e:
                print(f"    [ERROR] {e}")
        sweep_results[k] = results
    return sweep_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_eval", type=int, default=200, help="Number of QA pairs to evaluate")
    parser.add_argument("--top_k", type=int, default=5, help="Fixed top-k for Baseline/V1/V3")
    parser.add_argument("--skip_sweep", action="store_true", help="Skip V2 top-k sweep")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    print("Initialising retriever and generator...")
    retriever = DenseRetriever()
    retriever.index_chunks()          # no-op if already indexed
    generator = Generator()

    qa_pairs = load_qa_pairs(n=args.n_eval)
    questions = [item["question"] for item in qa_pairs]

    os.makedirs("results", exist_ok=True)

    # ------------------------------------------------------------------
    # Run variants
    # ------------------------------------------------------------------
    summaries = {}

    # Baseline
    baseline = BaselinePipeline(retriever, generator, top_k=args.top_k)
    baseline_results = run_variant(baseline, questions, "baseline")
    save_results(baseline_results, "results/baseline_results.json")
    summaries["baseline"] = evaluate_results(baseline_results, qa_pairs)

    # V1 – exact cache
    v1 = V1Pipeline(retriever, generator, top_k=args.top_k)
    v1_results = run_variant(v1, questions, "v1_exact_cache")
    save_results(v1_results, "results/v1_results.json")
    summaries["v1_exact_cache"] = evaluate_results(v1_results, qa_pairs)

    # V3 – adaptive retrieval
    v3 = V3Pipeline(retriever, generator)
    v3_results = run_variant(v3, questions, "v3_adaptive")
    save_results(v3_results, "results/v3_results.json")
    summaries["v3_adaptive"] = evaluate_results(v3_results, qa_pairs)


    # V2 – top-k sweep
    if not args.skip_sweep:
        sweep = run_topk_sweep(questions, retriever, generator)
        sweep_summaries = {}
        for k, res in sweep.items():
            save_results(res, f"results/v2_topk{k}_results.json")
            sweep_summaries[f"topk_{k}"] = evaluate_results(res, qa_pairs)
        summaries["v2_topk_sweep"] = sweep_summaries

    # ------------------------------------------------------------------
    # Print & save summaries
    # ------------------------------------------------------------------
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    for name, s in summaries.items():
        print(f"\n[{name}]")
        if isinstance(s, dict) and "n_samples" in s:
            for k, v in s.items():
                print(f"  {k}: {v}")
        else:
            # sweep sub-dict
            for subname, ss in s.items():
                print(f"  [{subname}] F1={ss['f1']} | mean={ss['mean_total_latency_ms']}ms | p50={ss['p50_total_latency_ms']}ms | p95={ss['p95_total_latency_ms']}ms")

    with open("results/summaries.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print("\nSummaries saved → results/summaries.json")


if __name__ == "__main__":
    main()