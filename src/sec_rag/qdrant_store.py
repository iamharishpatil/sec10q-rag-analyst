"""Qdrant vector database helpers for production-shaped retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams


DEFAULT_COLLECTION = "sec_10q_chunks"


@dataclass(frozen=True)
class QdrantSearchResult:
    """One Qdrant retrieval result."""

    rank: int
    score: float
    chunk: dict


def create_local_client(path: Path) -> QdrantClient:
    """Create a local persistent Qdrant client."""
    root = path.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(root))


def recreate_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    """Create a fresh cosine-similarity collection."""
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    embeddings: np.ndarray,
    chunks: Iterable[dict],
    batch_size: int = 128,
) -> int:
    """Upsert chunk vectors and metadata payloads into Qdrant."""
    chunk_records = tuple(chunks)
    if len(embeddings) != len(chunk_records):
        raise ValueError("embeddings and chunks must have the same length")

    total = 0
    for start in range(0, len(chunk_records), batch_size):
        batch_chunks = chunk_records[start : start + batch_size]
        batch_vectors = embeddings[start : start + batch_size]
        points = [
            PointStruct(
                id=start + offset,
                vector=vector.tolist(),
                payload=chunk,
            )
            for offset, (vector, chunk) in enumerate(zip(batch_vectors, batch_chunks, strict=True))
        ]
        client.upsert(collection_name=collection_name, points=points)
        total += len(points)
    return total


def search_chunks(
    client: QdrantClient,
    collection_name: str,
    query_embedding: np.ndarray,
    top_k: int = 5,
    ticker: str | None = None,
) -> tuple[QdrantSearchResult, ...]:
    """Search Qdrant and return chunk payloads with scores."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    query_filter = ticker_filter(ticker) if ticker else None
    raw_results = _query_points(
        client=client,
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=top_k,
        query_filter=query_filter,
    )
    return tuple(
        QdrantSearchResult(
            rank=rank,
            score=float(result.score),
            chunk=dict(result.payload or {}),
        )
        for rank, result in enumerate(raw_results, start=1)
    )


def ticker_filter(ticker: str) -> Filter:
    """Build a Qdrant payload filter for one ticker."""
    return Filter(
        must=[
            FieldCondition(
                key="ticker",
                match=MatchValue(value=ticker.upper()),
            )
        ]
    )


def _query_points(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    limit: int,
    query_filter: Filter | None,
):
    """Call the available Qdrant search API across client versions."""
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
        )
        return response.points

    return client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=limit,
    )
