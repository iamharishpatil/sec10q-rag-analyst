"""Baseline retrieval evaluation helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from sec_rag.qna import split_source_docs
from typing import Protocol


class RetrievalResult(Protocol):
    chunk: dict


TICKER_PATTERN = re.compile(r"\b(AAPL|AMZN|INTC|MSFT|NVDA)\b")


@dataclass(frozen=True)
class RetrievalEvalResult:
    """Aggregate source-document retrieval evaluation."""

    total_questions: int
    hits: int
    recall_at_k: float


def expected_source_labels(source_docs: str) -> set[str]:
    """Normalize Q&A source-doc labels into comparable labels."""
    labels: set[str] = set()
    for source_doc in split_source_docs(source_docs):
        upper = source_doc.upper()
        labels.add(upper)
        labels.update(TICKER_PATTERN.findall(upper))
    return labels


def result_source_labels(result: RetrievalResult) -> set[str]:
    """Return comparable labels from a search result chunk."""
    chunk = result.chunk
    labels = {
        str(chunk.get("ticker", "")).upper(),
        str(chunk.get("source_filename", "")).upper().replace(".PDF", ""),
    }
    return {label for label in labels if label}


def is_source_hit(expected_labels: set[str], results: tuple[RetrievalResult, ...]) -> bool:
    """Return whether retrieved results match any expected source label."""
    if not expected_labels:
        return False
    for result in results:
        if expected_labels & result_source_labels(result):
            return True
    return False


def summarize_retrieval_hits(total_questions: int, hits: int) -> RetrievalEvalResult:
    """Build an aggregate recall result."""
    return RetrievalEvalResult(
        total_questions=total_questions,
        hits=hits,
        recall_at_k=hits / total_questions if total_questions else 0.0,
    )
