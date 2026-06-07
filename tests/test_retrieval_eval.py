from sec_rag.retrieval_eval import expected_source_labels, is_source_hit, result_source_labels
from sec_rag.vector_index import SearchResult


def test_expected_source_labels_extracts_ticker_and_full_source() -> None:
    labels = expected_source_labels("*2023 Q3 MSFT*, *AAPL*")

    assert "2023 Q3 MSFT" in labels
    assert "MSFT" in labels
    assert "AAPL" in labels


def test_result_source_labels_uses_ticker_and_filename() -> None:
    result = SearchResult(
        rank=1,
        score=0.9,
        chunk={"ticker": "MSFT", "source_filename": "2023 Q3 MSFT.pdf"},
    )

    assert result_source_labels(result) == {"MSFT", "2023 Q3 MSFT"}


def test_is_source_hit_matches_expected_labels() -> None:
    result = SearchResult(
        rank=1,
        score=0.9,
        chunk={"ticker": "MSFT", "source_filename": "2023 Q3 MSFT.pdf"},
    )

    assert is_source_hit({"AAPL", "MSFT"}, (result,))
    assert not is_source_hit({"AAPL"}, (result,))
