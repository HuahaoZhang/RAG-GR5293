"""
pipeline.py
Four RAG pipeline variants from the proposal:

  Baseline : No cache, dense retrieval, fixed top-k
  V1       : Semantic cache, dense retrieval, fixed top-k
  V2       : Semantic cache, dense retrieval, varying top-k (passed per call)
  V3       : Semantic cache, adaptive top-k (complexity-based)

Each variant's run() returns a unified result dict for easy comparison.
"""

import time
from typing import Optional

from src.retriever import DenseRetriever
from src.generator import Generator
from src.cache import SemanticCache, ExactCache


# ---------------------------------------------------------------------------
# Adaptive retrieval config
# ---------------------------------------------------------------------------
ADAPTIVE_TOP_K_SIMPLE = 3      # simple queries
ADAPTIVE_TOP_K_MEDIUM = 5      # medium queries
ADAPTIVE_TOP_K_COMPLEX = 10    # complex queries
COMPLEXITY_THRESHOLD_SIMPLE = 0.569   # below this → simple top-k
COMPLEXITY_THRESHOLD_COMPLEX = 0.603  # above/equal this → complex top-k
SEMANTIC_THRESHOLD = 0.92     # cosine similarity threshold for cache hit


# ---------------------------------------------------------------------------
# Shared embed_fn factory
# ---------------------------------------------------------------------------

def _make_embed_fn(retriever: DenseRetriever):
    """Wrap ChromaDB's OpenAI embed function for use in SemanticCache."""
    def embed_fn(query: str):
        return retriever.embed_fn([query])[0]
    return embed_fn


# ---------------------------------------------------------------------------
# Base pipeline (shared logic)
# ---------------------------------------------------------------------------

class _BasePipeline:
    def __init__(self, retriever: DenseRetriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator

    def _build_result(
        self,
        question: str,
        retrieval_result: dict,
        gen_result: dict,
        cache_hit: bool,
        top_k_used: int,
    ) -> dict:
        total_latency = retrieval_result["latency_ms"] + gen_result["latency_ms"]
        return {
            "question": question,
            "answer": gen_result["answer"],
            "chunks": retrieval_result["chunks"],
            "retrieval_latency_ms": retrieval_result["latency_ms"],
            "generation_latency_ms": gen_result["latency_ms"],
            "total_latency_ms": total_latency,
            "cache_hit": cache_hit,
            "top_k_used": top_k_used,
        }


# ---------------------------------------------------------------------------
# Baseline: no cache, fixed top-k
# ---------------------------------------------------------------------------

class BaselinePipeline(_BasePipeline):
    name = "baseline"

    def __init__(self, retriever: DenseRetriever, generator: Generator, top_k: int = 5):
        super().__init__(retriever, generator)
        self.top_k = top_k

    def run(self, question: str) -> dict:
        retrieval = self.retriever.retrieve(question, top_k=self.top_k)
        gen = self.generator.generate(question, retrieval["chunks"])
        return self._build_result(question, retrieval, gen, cache_hit=False, top_k_used=self.top_k)


# ---------------------------------------------------------------------------
# V1: semantic cache + fixed top-k
# ---------------------------------------------------------------------------

class V1Pipeline(_BasePipeline):
    name = "v1_semantic_cache"

    def __init__(
        self,
        retriever: DenseRetriever,
        generator: Generator,
        top_k: int = 5,
        threshold: float = SEMANTIC_THRESHOLD,
    ):

        super().__init__(retriever, generator)
        self.top_k = top_k
        self.cache = ExactCache()

    def run(self, question: str) -> dict:
        cached = self.cache.get(question)
        if cached:
            return {**cached, "cache_hit": True}

        retrieval = self.retriever.retrieve(question, top_k=self.top_k)
        gen = self.generator.generate(question, retrieval["chunks"])
        result = self._build_result(question, retrieval, gen, cache_hit=False, top_k_used=self.top_k)
        self.cache.set(question, result)
        return result

    def cache_stats(self) -> dict:
        return self.cache.stats()


# ---------------------------------------------------------------------------
# V2: semantic cache + variable top-k (passed per call)
# ---------------------------------------------------------------------------

class V2Pipeline(_BasePipeline):
    name = "v2_variable_topk"

    def __init__(
        self,
        retriever: DenseRetriever,
        generator: Generator,
        threshold: float = SEMANTIC_THRESHOLD,
    ):
        super().__init__(retriever, generator)
        self.cache = ExactCache()

    def run(self, question: str, top_k: int = 5) -> dict:
        cached = self.cache.get(question)
        if cached and cached.get("top_k_used") == top_k:
            return {**cached, "cache_hit": True}

        retrieval = self.retriever.retrieve(question, top_k=top_k)
        gen = self.generator.generate(question, retrieval["chunks"])
        result = self._build_result(question, retrieval, gen, cache_hit=False, top_k_used=top_k)
        self.cache.set(question, result)
        return result

    def cache_stats(self) -> dict:
        return self.cache.stats()


# ---------------------------------------------------------------------------
# V3: semantic cache + adaptive top-k
# ---------------------------------------------------------------------------

class V3Pipeline(_BasePipeline):
    name = "v3_adaptive"

    def __init__(
        self,
        retriever: DenseRetriever,
        generator: Generator,
        complexity_threshold_simple: float = COMPLEXITY_THRESHOLD_SIMPLE,
        complexity_threshold_complex: float = COMPLEXITY_THRESHOLD_COMPLEX,
        top_k_simple: int = ADAPTIVE_TOP_K_SIMPLE,
        top_k_medium: int = ADAPTIVE_TOP_K_MEDIUM,
        top_k_complex: int = ADAPTIVE_TOP_K_COMPLEX,
        cache_threshold: float = SEMANTIC_THRESHOLD,
    ):
        super().__init__(retriever, generator)
        self.complexity_threshold_simple = complexity_threshold_simple
        self.complexity_threshold_complex = complexity_threshold_complex
        self.top_k_simple = top_k_simple
        self.top_k_medium = top_k_medium
        self.top_k_complex = top_k_complex
        self.cache = ExactCache()

    def run(self, question: str) -> dict:
        cached = self.cache.get(question)
        if cached:
            return {**cached, "cache_hit": True}

        complexity = self.retriever.query_complexity_score(question)
        if complexity < self.complexity_threshold_simple:
            top_k = self.top_k_simple
        elif complexity < self.complexity_threshold_complex:
            top_k = self.top_k_medium
        else:
            top_k = self.top_k_complex

        retrieval = self.retriever.retrieve(question, top_k=top_k)
        gen = self.generator.generate(question, retrieval["chunks"])
        result = self._build_result(question, retrieval, gen, cache_hit=False, top_k_used=top_k)
        result["complexity_score"] = complexity
        self.cache.set(question, result)
        return result

    def cache_stats(self) -> dict:
        return self.cache.stats()