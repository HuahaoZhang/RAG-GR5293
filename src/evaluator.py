"""
evaluator.py
Evaluation metrics for the RAG project:
  - Exact Match (EM)
  - Token-level F1
  - Recall@k  (chunk-level)
  - Latency stats
  - Pareto helpers
"""

import re
import string
import json
import os
from collections import Counter
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Text normalisation (SQuAD / HotpotQA style)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, ground_truth: str) -> int:
    return int(_normalize(prediction) == _normalize(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gt_tokens = _normalize(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Retrieval quality
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_chunks: list[dict], ground_truth_answer: str, k: int) -> float:
    """
    Approximate Recall@k: fraction of retrieved chunks (up to k)
    that contain any token from the ground-truth answer.
    """
    gt_tokens = set(_normalize(ground_truth_answer).split())
    hits = 0
    for chunk in retrieved_chunks[:k]:
        chunk_tokens = set(_normalize(chunk["text"]).split())
        if gt_tokens & chunk_tokens:
            hits += 1
    return hits / max(len(retrieved_chunks[:k]), 1)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_results(results: list[dict], qa_pairs: list[dict]) -> dict:
    """
    Compare pipeline outputs against ground-truth QA pairs.

    Args:
        results  : list of pipeline run() outputs
        qa_pairs : list of dicts with 'question' and 'answer' keys

    Returns a summary dict with mean metrics.
    """
    gt_map = {item["question"]: item["answer"] for item in qa_pairs}

    em_scores, f1_scores, recall_scores = [], [], []
    retrieval_latencies, gen_latencies, total_latencies = [], [], []
    cache_hits = 0

    for r in results:
        gt = gt_map.get(r["question"], "")
        em_scores.append(exact_match(r["answer"], gt))
        f1_scores.append(f1_score(r["answer"], gt))
        recall_scores.append(recall_at_k(r["chunks"], gt, k=r["top_k_used"]))

        retrieval_latencies.append(r["retrieval_latency_ms"])
        gen_latencies.append(r["generation_latency_ms"])
        total_latencies.append(r["total_latency_ms"])

        if r.get("cache_hit"):
            cache_hits += 1

    def _latency_inlier_mask(values: list[float], iqr_multiplier: float = 1.5) -> np.ndarray:
        """Return True for inliers using Tukey's IQR fences."""
        arr = np.array(values, dtype=float)
        if arr.size < 4:
            return np.ones(arr.size, dtype=bool)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            return np.ones(arr.size, dtype=bool)
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        return (arr >= lower) & (arr <= upper)

    # Remove extreme latency outliers so one long-tail request does not dominate summary stats.
    total_mask = _latency_inlier_mask(total_latencies)
    retrieval_arr = np.array(retrieval_latencies, dtype=float)
    gen_arr = np.array(gen_latencies, dtype=float)
    total_arr = np.array(total_latencies, dtype=float)
    retrieval_inliers = retrieval_arr[total_mask]
    gen_inliers = gen_arr[total_mask]
    total_inliers = total_arr[total_mask]

    # Safety fallback: if all points are filtered, keep original arrays.
    if total_inliers.size == 0:
        retrieval_inliers = retrieval_arr
        gen_inliers = gen_arr
        total_inliers = total_arr

    n_outliers = int(total_arr.size - total_inliers.size)
    n = len(results)
    return {
        "n_samples": n,
        "exact_match": round(np.mean(em_scores), 4),
        "f1": round(np.mean(f1_scores), 4),
        "recall_at_k": round(np.mean(recall_scores), 4),
        "mean_retrieval_latency_ms": round(float(np.mean(retrieval_inliers)), 2),
        "mean_generation_latency_ms": round(float(np.mean(gen_inliers)), 2),
        "mean_total_latency_ms": round(float(np.mean(total_inliers)), 2),
        "p50_total_latency_ms": round(float(np.percentile(total_inliers, 50)), 2),
        "p95_total_latency_ms": round(float(np.percentile(total_inliers, 95)), 2),
        "cache_hit_rate": round(cache_hits / n, 4) if n > 0 else 0.0,
        "latency_outliers_removed": n_outliers,
    }


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_results(results: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results → {path}")


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)
