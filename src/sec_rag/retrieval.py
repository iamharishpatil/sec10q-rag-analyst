"""Retriever interfaces and shared retrieval result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RetrievalFilters:
    """Metadata filters applied at retrieval time."""

    ticker: str | None = None


@dataclass(frozen=True)
class RetrievalResult:
    """One standardized retrieval result."""

    rank: int
    score: float
    chunk: dict
    backend: str
    dense_score: float | None = None
    bm25_score: float | None = None


class Retriever(Protocol):
    """Search interface implemented by all retrievers."""

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: RetrievalFilters | None = None,
    ) -> tuple[RetrievalResult, ...]:
        """Return top-k chunks for a query."""
