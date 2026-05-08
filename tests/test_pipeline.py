import sys
import types

# Keep tests lightweight: avoid importing heavy runtime deps (chromadb/openai).
if "src.retriever" not in sys.modules:
    retriever_module = types.ModuleType("src.retriever")
    retriever_module.DenseRetriever = object
    sys.modules["src.retriever"] = retriever_module

if "src.generator" not in sys.modules:
    generator_module = types.ModuleType("src.generator")
    generator_module.Generator = object
    sys.modules["src.generator"] = generator_module

from src.pipeline import BaselinePipeline, V1Pipeline, V2Pipeline, V3Pipeline


class FakeRetriever:
    def __init__(self):
        self.retrieve_calls = 0

    def retrieve(self, query: str, top_k: int = 5):
        self.retrieve_calls += 1
        return {
            "chunks": [{"text": f"context for {query}", "title": "t1", "score": 0.9}],
            "latency_ms": float(top_k),
        }

    def query_complexity_score(self, query: str):
        if "simple" in query:
            return 0.2
        if "complex" in query:
            return 0.8
        return 0.58


class FakeGenerator:
    def __init__(self):
        self.generate_calls = 0

    def generate(self, question: str, chunks: list[dict]):
        self.generate_calls += 1
        return {"answer": f"ans:{question}", "latency_ms": 10.0}


def test_baseline_pipeline_runs_without_cache():
    retriever = FakeRetriever()
    generator = FakeGenerator()
    pipeline = BaselinePipeline(retriever, generator, top_k=5)

    result = pipeline.run("hello")
    assert result["cache_hit"] is False
    assert result["top_k_used"] == 5
    assert retriever.retrieve_calls == 1
    assert generator.generate_calls == 1


def test_v1_pipeline_cache_hit_on_second_call():
    retriever = FakeRetriever()
    generator = FakeGenerator()
    pipeline = V1Pipeline(retriever, generator, top_k=5)

    first = pipeline.run("repeat")
    second = pipeline.run("repeat")

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert retriever.retrieve_calls == 1
    assert generator.generate_calls == 1


def test_v2_pipeline_cache_depends_on_top_k():
    retriever = FakeRetriever()
    generator = FakeGenerator()
    pipeline = V2Pipeline(retriever, generator)

    first = pipeline.run("repeat", top_k=3)
    second = pipeline.run("repeat", top_k=3)
    third = pipeline.run("repeat", top_k=5)

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert third["cache_hit"] is False
    assert retriever.retrieve_calls == 2


def test_v3_pipeline_uses_adaptive_top_k():
    retriever = FakeRetriever()
    generator = FakeGenerator()
    pipeline = V3Pipeline(retriever, generator)

    simple = pipeline.run("simple question")
    medium = pipeline.run("medium question")
    complex_q = pipeline.run("complex question")

    assert simple["top_k_used"] == 3
    assert medium["top_k_used"] == 5
    assert complex_q["top_k_used"] == 10
