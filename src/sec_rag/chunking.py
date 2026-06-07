"""Chunk page-level SEC filing records into retrievable text units."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ChunkRecord:
    """A retrievable chunk derived from one parsed page."""

    chunk_id: str
    document_id: str
    source_path: str
    source_filename: str
    ticker: str
    company: str
    year: int | None
    quarter: int | None
    filing_type: str
    page_number: int
    total_pages: int
    chunk_index: int
    word_count: int
    char_count: int
    text: str


@dataclass(frozen=True)
class ChunkingSummary:
    """Summary of chunking output for one or more page files."""

    page_records: int
    chunks: int
    skipped_empty_pages: int
    average_words_per_chunk: float


def chunk_page_record(
    page: dict,
    chunk_size: int = 350,
    overlap: int = 50,
) -> tuple[ChunkRecord, ...]:
    """Split a page record into word-based overlapping chunks."""
    validate_chunking_params(chunk_size=chunk_size, overlap=overlap)

    words = page.get("text", "").split()
    if not words:
        return tuple()

    chunks: list[ChunkRecord] = []
    step = chunk_size - overlap
    chunk_index = 0

    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            break
        text = " ".join(chunk_words)
        chunk_id = f"{page['document_id']}_p{page['page_number']}_c{chunk_index}"
        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id,
                document_id=page["document_id"],
                source_path=page["source_path"],
                source_filename=page["source_filename"],
                ticker=page["ticker"],
                company=page["company"],
                year=page["year"],
                quarter=page["quarter"],
                filing_type=page["filing_type"],
                page_number=page["page_number"],
                total_pages=page["total_pages"],
                chunk_index=chunk_index,
                word_count=len(chunk_words),
                char_count=len(text),
                text=text,
            )
        )
        chunk_index += 1
        if start + chunk_size >= len(words):
            break

    return tuple(chunks)


def chunk_page_records(
    pages: Iterable[dict],
    chunk_size: int = 350,
    overlap: int = 50,
) -> tuple[ChunkRecord, ...]:
    """Chunk many page records."""
    chunks: list[ChunkRecord] = []
    for page in pages:
        chunks.extend(chunk_page_record(page, chunk_size=chunk_size, overlap=overlap))
    return tuple(chunks)


def read_page_records_jsonl(path: Path) -> tuple[dict, ...]:
    """Read parsed page records from JSONL."""
    input_path = path.expanduser().resolve()
    records = []
    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return tuple(records)


def write_chunk_records_jsonl(chunks: Iterable[ChunkRecord], output_path: Path) -> int:
    """Write chunk records as JSONL and return written row count."""
    output = output_path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
            count += 1
    return count


def summarize_chunks(page_count: int, chunks: Iterable[ChunkRecord]) -> ChunkingSummary:
    """Summarize chunking output."""
    chunk_records = tuple(chunks)
    total_words = sum(chunk.word_count for chunk in chunk_records)
    return ChunkingSummary(
        page_records=page_count,
        chunks=len(chunk_records),
        skipped_empty_pages=max(page_count - len({chunk.page_number for chunk in chunk_records}), 0),
        average_words_per_chunk=total_words / len(chunk_records) if chunk_records else 0.0,
    )


def validate_chunking_params(chunk_size: int, overlap: int) -> None:
    """Validate fixed-size chunking parameters."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
