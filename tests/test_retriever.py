import json
from pathlib import Path

from app.knowledge import load_chunks
from app.retriever import BM25Retriever, build_retriever, tokenize

GOLDEN = Path(__file__).parent.parent / "evals" / "golden_set.jsonl"


def test_knowledge_base_loads_with_citations(chunks):
    assert len(chunks) > 20
    for c in chunks:
        assert c.title and c.url.startswith("http") and c.section and c.text
        assert len(c.text) <= 1200  # chunk_size + overlap slack


def test_tokenize_drops_stopwords():
    assert tokenize("What did Lahiru do at Dialog Axiata?") == ["dialog", "axiata"]


def test_bm25_returns_relevant_chunk_first(retriever):
    results = retriever.search("random forest parking spots MAE", k=3)
    assert results and results[0][0].source_id == "projects-ml"
    assert "parking" in results[0][0].text.lower()


def test_bm25_empty_query_returns_nothing(retriever):
    assert retriever.search("the and of", k=5) == []


def test_retrieval_hit_rate_on_golden_set(retriever):
    golden = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    hits = 0
    for item in golden:
        sources = [c.source_id for c, _ in retriever.search(item["question"], 5)]
        hits += any(s in item["expected_sources"] for s in sources)
    assert hits / len(golden) >= 0.85, f"hit rate {hits}/{len(golden)}"


def test_build_retriever_rejects_unknown_backend():
    import pytest

    with pytest.raises(ValueError):
        build_retriever("faiss", chunks=load_chunks())


def test_bm25_is_deterministic(chunks):
    a = BM25Retriever(chunks).search("SageMaker churn XGBoost", 4)
    b = BM25Retriever(chunks).search("SageMaker churn XGBoost", 4)
    assert [c.id for c, _ in a] == [c.id for c, _ in b]
