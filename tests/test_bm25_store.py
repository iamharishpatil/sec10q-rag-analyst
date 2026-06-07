from pathlib import Path

import pytest

from sec_rag.bm25_store import load_bm25_index, search_bm25_index, tokenize, write_bm25_index


def _chunks() -> tuple[dict, ...]:
    return (
        {
            "chunk_id": "aapl",
            "ticker": "AAPL",
            "source_filename": "2023 Q2 AAPL.pdf",
            "page_number": 10,
            "chunk_index": 0,
            "text": "Apple net sales services iphone revenue",
        },
        {
            "chunk_id": "msft",
            "ticker": "MSFT",
            "source_filename": "2023 Q2 MSFT.pdf",
            "page_number": 20,
            "chunk_index": 0,
            "text": "Microsoft cloud revenue azure office",
        },
    )


def test_tokenize_normalizes_terms() -> None:
    assert tokenize("Net Sales, Q2-2023!") == ["net", "sales", "q2", "2023"]


def test_bm25_index_round_trip_and_search(tmp_path: Path) -> None:
    manifest = write_bm25_index(tmp_path, _chunks())
    index = load_bm25_index(tmp_path)

    results = search_bm25_index("apple iphone sales", index=index, top_k=1)

    assert manifest["chunk_rows"] == 2
    assert results[0].chunk["chunk_id"] == "aapl"
    assert results[0].backend == "bm25"
    assert results[0].bm25_score is not None


def test_bm25_search_supports_ticker_filter(tmp_path: Path) -> None:
    write_bm25_index(tmp_path, _chunks())
    index = load_bm25_index(tmp_path)

    results = search_bm25_index("revenue", index=index, top_k=2, ticker="MSFT")

    assert len(results) == 1
    assert results[0].chunk["ticker"] == "MSFT"


def test_bm25_search_validates_top_k(tmp_path: Path) -> None:
    write_bm25_index(tmp_path, _chunks())
    index = load_bm25_index(tmp_path)

    with pytest.raises(ValueError):
        search_bm25_index("revenue", index=index, top_k=0)
