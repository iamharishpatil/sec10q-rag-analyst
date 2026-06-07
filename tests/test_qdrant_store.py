import numpy as np
import pytest

from sec_rag.qdrant_store import (
    DEFAULT_COLLECTION,
    create_local_client,
    recreate_collection,
    search_chunks,
    ticker_filter,
    upsert_chunks,
)


def _chunks() -> tuple[dict, ...]:
    return (
        {
            "chunk_id": "aapl",
            "ticker": "AAPL",
            "source_filename": "2023 Q1 AAPL.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "text": "apple revenue",
        },
        {
            "chunk_id": "msft",
            "ticker": "MSFT",
            "source_filename": "2023 Q1 MSFT.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "text": "microsoft revenue",
        },
    )


def test_qdrant_local_upsert_and_search(tmp_path) -> None:
    client = create_local_client(tmp_path)
    recreate_collection(client, DEFAULT_COLLECTION, vector_size=2)
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    rows = upsert_chunks(client, DEFAULT_COLLECTION, embeddings=embeddings, chunks=_chunks())
    results = search_chunks(
        client,
        DEFAULT_COLLECTION,
        query_embedding=np.asarray([0.9, 0.1], dtype=np.float32),
        top_k=1,
    )

    assert rows == 2
    assert results[0].chunk["chunk_id"] == "aapl"


def test_qdrant_search_supports_ticker_filter(tmp_path) -> None:
    client = create_local_client(tmp_path)
    recreate_collection(client, DEFAULT_COLLECTION, vector_size=2)
    embeddings = np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    upsert_chunks(client, DEFAULT_COLLECTION, embeddings=embeddings, chunks=_chunks())

    results = search_chunks(
        client,
        DEFAULT_COLLECTION,
        query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        top_k=2,
        ticker="MSFT",
    )

    assert len(results) == 1
    assert results[0].chunk["ticker"] == "MSFT"


def test_upsert_chunks_validates_lengths(tmp_path) -> None:
    client = create_local_client(tmp_path)
    recreate_collection(client, DEFAULT_COLLECTION, vector_size=2)

    with pytest.raises(ValueError):
        upsert_chunks(
            client,
            DEFAULT_COLLECTION,
            embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
            chunks=_chunks(),
        )


def test_ticker_filter_normalizes_ticker() -> None:
    payload_filter = ticker_filter("msft")

    assert payload_filter.must[0].match.value == "MSFT"
