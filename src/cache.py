"""
cache.py
Two-level query cache:
  - ExactCache  : exact string match (V1, V2, V3)
  - SemanticCache: embedding cosine-similarity match (optional extension)
"""

import time
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Exact Cache (V1 / V2 / V3)
# ---------------------------------------------------------------------------

class ExactCache:
    """In-memory exact-match cache keyed on query string."""

    def __init__(self):
        self._store: dict[str, dict] = {}  # query → cached result
        self.hits = 0
        self.misses = 0

    def get(self, query: str) -> Optional[dict]:
        result = self._store.get(query)
        if result is not None:
            self.hits += 1
            return result
        self.misses += 1
        return None

    def set(self, query: str, result: dict):
        self._store[query] = result

    def clear(self):
        self._store.clear()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._store)

    def stats(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "cache_size": self.size,
        }

    def reset_stats(self):
        self.hits = 0
        self.misses = 0


# ---------------------------------------------------------------------------
# Semantic Cache (optional extension)
# ---------------------------------------------------------------------------

class SemanticCache:
    """
    Embedding-based cache. Reuses a cached result when cosine similarity
    between the new query embedding and a stored query embedding exceeds
    `threshold`.

    Requires an embed_fn: Callable[[str], list[float]].
    """

    def __init__(self, embed_fn, threshold: float = 0.92):
        self.embed_fn = embed_fn
        self.threshold = threshold
        self._keys: list[str] = []           # original query strings
        self._embeddings: list[np.ndarray] = []
        self._values: list[dict] = []
        self.hits = 0
        self.misses = 0

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def get(self, query: str) -> Optional[dict]:
        if not self._embeddings:
            self.misses += 1
            return None

        q_emb = np.array(self.embed_fn(query))
        sims = [self._cosine(q_emb, e) for e in self._embeddings]
        best_idx = int(np.argmax(sims))

        if sims[best_idx] >= self.threshold:
            self.hits += 1
            return self._values[best_idx]

        self.misses += 1
        return None

    def set(self, query: str, result: dict):
        q_emb = np.array(self.embed_fn(query))
        self._keys.append(query)
        self._embeddings.append(q_emb)
        self._values.append(result)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "cache_size": len(self._keys),
            "threshold": self.threshold,
        }
        
    def reset_stats(self):       # ← 加这个
        self.hits = 0
        self.misses = 0