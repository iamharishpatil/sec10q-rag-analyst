import json
from pathlib import Path

import numpy as np
import pytest

from sec_rag.vector_index import load_index, search_index, write_index


def _chunks() -> tuple[dict, ...]:
    return (
        {"chunk_id": "a", "ticker": "AAPL", "source_filename": "2023 Q1 AAPL.pdf", "text": "apple"},
        {"chunk_id": "m", "ticker": "MSFT", "source_filename": "2023 Q1 MSFT.pdf", "text": "microsoft"},
    )


def test_search_index_returns_top_k_by_cosine_score() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    query = np.asarray([0.9, 0.1], dtype=np.float32)

    results = search_index(query, embeddings=embeddings, chunks=_chunks(), top_k=1)

    assert len(results) == 1
    assert results[0].chunk["chunk_id"] == "a"


def test_search_index_validates_inputs() -> None:
    with pytest.raises(ValueError):
        search_index(np.asarray([1.0]), np.asarray([[1.0]]), _chunks(), top_k=0)
    with pytest.raises(ValueError):
        search_index(np.asarray([1.0]), np.asarray([[1.0]]), _chunks(), top_k=1)


def test_write_and_load_index_round_trip(tmp_path: Path) -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    write_index(tmp_path, embeddings=embeddings, chunks=_chunks(), model_name="test-model")
    loaded_embeddings, loaded_chunks, manifest = load_index(tmp_path)

    assert np.array_equal(loaded_embeddings, embeddings)
    assert loaded_chunks == _chunks()
    assert manifest["model_name"] == "test-model"
    assert manifest["embedding_rows"] == 2
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["embedding_dim"] == 2
