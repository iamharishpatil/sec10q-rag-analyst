from sec_rag.retrieval import RetrievalFilters, RetrievalResult
from sec_rag.retrievers import HybridRetriever


class FakeRetriever:
    def __init__(self, results: tuple[RetrievalResult, ...]) -> None:
        self.results = results
        self.calls: list[tuple[str, int, RetrievalFilters | None]] = []

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: RetrievalFilters | None = None,
    ) -> tuple[RetrievalResult, ...]:
        self.calls.append((query, top_k, filters))
        return self.results[:top_k]


def _result(chunk_id: str, score: float, backend: str, ticker: str = "AAPL") -> RetrievalResult:
    return RetrievalResult(
        rank=1,
        score=score,
        backend=backend,
        dense_score=score if backend == "dense" else None,
        bm25_score=score if backend == "bm25" else None,
        chunk={
            "chunk_id": chunk_id,
            "ticker": ticker,
            "source_filename": f"{ticker}.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "text": f"{ticker} revenue",
        },
    )


def test_hybrid_retriever_merges_duplicate_chunks_and_preserves_components() -> None:
    dense = FakeRetriever((_result("shared", 0.9, "dense"), _result("dense-only", 0.8, "dense")))
    bm25 = FakeRetriever((_result("shared", 3.0, "bm25"), _result("bm25-only", 2.0, "bm25")))
    retriever = HybridRetriever(dense, bm25, dense_weight=0.65, bm25_weight=0.35)

    results = retriever.search("revenue", top_k=3)

    shared = next(result for result in results if result.chunk["chunk_id"] == "shared")
    assert len(results) == 3
    assert shared.backend == "hybrid"
    assert shared.dense_score == 0.9
    assert shared.bm25_score == 3.0


def test_hybrid_retriever_passes_filters_to_children() -> None:
    filters = RetrievalFilters(ticker="MSFT")
    dense = FakeRetriever((_result("dense", 0.9, "dense", ticker="MSFT"),))
    bm25 = FakeRetriever((_result("bm25", 3.0, "bm25", ticker="MSFT"),))
    retriever = HybridRetriever(dense, bm25)

    retriever.search("revenue", top_k=1, filters=filters)

    assert dense.calls[0][2] == filters
    assert bm25.calls[0][2] == filters
