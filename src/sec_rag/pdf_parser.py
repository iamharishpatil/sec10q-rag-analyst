"""Page-aware PDF parsing for SEC filing ingestion."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import fitz

from sec_rag.documents import metadata_from_pdf_path


@dataclass(frozen=True)
class PageRecord:
    """Text extracted from one page of one source PDF."""

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
    text: str


@dataclass(frozen=True)
class PdfParseSummary:
    """Extraction quality summary for one parsed PDF."""

    document_id: str
    source_filename: str
    total_pages: int
    empty_pages: int
    average_chars_per_page: float
    shortest_page_chars: int
    longest_page_chars: int


def parse_pdf_pages(path: Path) -> tuple[PageRecord, ...]:
    """Extract text and page metadata from a PDF."""
    pdf_path = path.expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF does not exist: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path}")

    records: list[PageRecord] = []
    metadata = metadata_from_pdf_path(pdf_path)

    with fitz.open(pdf_path) as document:
        total_pages = document.page_count
        for page_index in range(total_pages):
            page = document.load_page(page_index)
            records.append(
                PageRecord(
                    document_id=metadata.document_id,
                    source_path=str(pdf_path),
                    source_filename=metadata.source_filename,
                    ticker=metadata.ticker,
                    company=metadata.company,
                    year=metadata.year,
                    quarter=metadata.quarter,
                    filing_type=metadata.filing_type,
                    page_number=page_index + 1,
                    total_pages=total_pages,
                    text=page.get_text("text").strip(),
                )
            )

    return tuple(records)


def summarize_page_records(records: Iterable[PageRecord]) -> PdfParseSummary:
    """Summarize page extraction quality for one PDF."""
    pages = tuple(records)
    if not pages:
        return PdfParseSummary(
            document_id="",
            source_filename="",
            total_pages=0,
            empty_pages=0,
            average_chars_per_page=0.0,
            shortest_page_chars=0,
            longest_page_chars=0,
        )

    lengths = [len(page.text) for page in pages]
    return PdfParseSummary(
        document_id=pages[0].document_id,
        source_filename=pages[0].source_filename,
        total_pages=len(pages),
        empty_pages=sum(1 for length in lengths if length == 0),
        average_chars_per_page=sum(lengths) / len(lengths),
        shortest_page_chars=min(lengths),
        longest_page_chars=max(lengths),
    )


def write_page_records_jsonl(records: Iterable[PageRecord], output_path: Path) -> int:
    """Write page records as JSONL and return the number of written rows."""
    output = output_path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            count += 1
    return count
