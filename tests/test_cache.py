from src.cache import ExactCache, SemanticCache


def test_exact_cache_tracks_hits_misses_and_hit_rate():
    cache = ExactCache()
    payload = {"answer": "Paris"}

    assert cache.get("q1") is None
    cache.set("q1", payload)
    assert cache.get("q1") == payload

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["cache_size"] == 1
    assert abs(stats["hit_rate"] - 0.5) < 1e-9


def test_semantic_cache_hit_and_reset_stats():
    vectors = {
        "capital of france": [1.0, 0.0, 0.0],
        "what is france capital": [0.99, 0.01, 0.0],
        "speed of light": [0.0, 1.0, 0.0],
    }

    def embed_fn(query: str):
        return vectors[query]

    cache = SemanticCache(embed_fn=embed_fn, threshold=0.95)
    cache.set("capital of france", {"answer": "Paris"})

    hit = cache.get("what is france capital")
    miss = cache.get("speed of light")

    assert hit == {"answer": "Paris"}
    assert miss is None
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 1

    cache.reset_stats()
    assert cache.stats()["hits"] == 0
    assert cache.stats()["misses"] == 0
