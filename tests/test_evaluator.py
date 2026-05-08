from src.evaluator import evaluate_results, exact_match, f1_score


def test_exact_match_and_f1_are_normalized():
    assert exact_match("The Paris.", "paris") == 1
    assert f1_score("new york city", "new york") > 0
    assert f1_score("tokyo", "paris") == 0.0


def test_evaluate_results_returns_expected_summary_fields():
    qa_pairs = [
        {"question": "q1", "answer": "Paris"},
        {"question": "q2", "answer": "1945"},
    ]
    results = [
        {
            "question": "q1",
            "answer": "Paris",
            "chunks": [{"text": "Paris is the capital of France."}],
            "retrieval_latency_ms": 10.0,
            "generation_latency_ms": 20.0,
            "total_latency_ms": 30.0,
            "cache_hit": False,
            "top_k_used": 1,
        },
        {
            "question": "q2",
            "answer": "1945",
            "chunks": [{"text": "World War II ended in 1945."}],
            "retrieval_latency_ms": 15.0,
            "generation_latency_ms": 25.0,
            "total_latency_ms": 40.0,
            "cache_hit": True,
            "top_k_used": 1,
        },
    ]

    summary = evaluate_results(results, qa_pairs)
    assert summary["n_samples"] == 2
    assert summary["exact_match"] == 1.0
    assert summary["f1"] == 1.0
    assert summary["cache_hit_rate"] == 0.5
    assert "p95_total_latency_ms" in summary
