import sys
import types

from src.evaluator import evaluate_results

if "src.retriever" not in sys.modules:
    retriever_module = types.ModuleType("src.retriever")
    retriever_module.DenseRetriever = object
    sys.modules["src.retriever"] = retriever_module

if "src.generator" not in sys.modules:
    generator_module = types.ModuleType("src.generator")
    generator_module.Generator = object
    sys.modules["src.generator"] = generator_module

from src.pipeline import BaselinePipeline


class FakeRetriever:
    def retrieve(self, query: str, top_k: int = 5):
        chunks = [
            {"text": "Paris is the capital of France.", "title": "France", "score": 0.95},
            {"text": "Berlin is the capital of Germany.", "title": "Germany", "score": 0.70},
        ][:top_k]
        return {"chunks": chunks, "latency_ms": 5.0}


class FakeGenerator:
    def generate(self, question: str, chunks: list[dict]):
        # Simple deterministic behavior for integration testing.
        answer = "Paris" if "france" in question.lower() else "I don't know"
        return {"answer": answer, "latency_ms": 8.0}


def test_end_to_end_pipeline_to_evaluator_flow():
    retriever = FakeRetriever()
    generator = FakeGenerator()
    pipeline = BaselinePipeline(retriever, generator, top_k=2)

    qa_pairs = [{"question": "What is the capital of France?", "answer": "Paris"}]
    results = [pipeline.run(qa_pairs[0]["question"])]
    summary = evaluate_results(results, qa_pairs)

    assert results[0]["answer"] == "Paris"
    assert results[0]["top_k_used"] == 2
    assert summary["n_samples"] == 1
    assert summary["exact_match"] == 1.0
    assert summary["f1"] == 1.0
