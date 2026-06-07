"""Q&A dataset profiling for SEC 10-Q evaluation data."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QnaProfile:
    """Summary of a Q&A CSV used for RAG evaluation."""

    path: Path
    row_count: int
    columns: tuple[str, ...]
    question_type_counts: tuple[tuple[str, int], ...]
    source_chunk_type_counts: tuple[tuple[str, int], ...]
    source_doc_counts: tuple[tuple[str, int], ...]


def profile_qna_csv(path: Path) -> QnaProfile:
    """Profile the KG-RAG Q&A CSV schema and label distributions."""
    csv_path = path.expanduser().resolve()
    question_type_counts: Counter[str] = Counter()
    source_chunk_type_counts: Counter[str] = Counter()
    source_doc_counts: Counter[str] = Counter()
    row_count = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = tuple(reader.fieldnames or ())
        for row in reader:
            row_count += 1
            question_type_counts[row.get("Question Type", "") or "(blank)"] += 1
            source_chunk_type_counts[row.get("Source Chunk Type", "") or "(blank)"] += 1
            for source_doc in split_source_docs(row.get("Source Docs", "")):
                source_doc_counts[source_doc] += 1

    return QnaProfile(
        path=csv_path,
        row_count=row_count,
        columns=columns,
        question_type_counts=sorted_counts(question_type_counts),
        source_chunk_type_counts=sorted_counts(source_chunk_type_counts),
        source_doc_counts=sorted_counts(source_doc_counts),
    )


def split_source_docs(value: str) -> tuple[str, ...]:
    """Split the KG-RAG source-docs field into normalized source labels."""
    cleaned = value.replace("*", "").strip()
    if not cleaned:
        return tuple()
    parts = [part.strip() for part in cleaned.replace(";", ",").split(",")]
    return tuple(part for part in parts if part)


def sorted_counts(counter: Counter[str]) -> tuple[tuple[str, int], ...]:
    """Return counts sorted by frequency descending, then label ascending."""
    return tuple(sorted(counter.items(), key=lambda item: (-item[1], item[0])))
