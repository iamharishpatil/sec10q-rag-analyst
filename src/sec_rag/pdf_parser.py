"""Page-aware PDF parsing for SEC filing ingestion."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import fitz


@dataclass(frozen=True)
class PageRecord:
    """Text extracted from one page of one source PDF."""

    document_id: str
    source_path: str
    page_number: int
    total_pages: int
    text: str


def document_id_from_path(path: Path) -> str:
    """Create a stable document id from a PDF path."""
    return path.stem.replace(" ", "_").lower()


def parse_pdf_pages(path: Path) -> tuple[PageRecord, ...]:
    """Extract text and page metadata from a PDF."""
    pdf_path = path.expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF does not exist: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path}")

    records: list[PageRecord] = []
    document_id = document_id_from_path(pdf_path)

    with fitz.open(pdf_path) as document:
        total_pages = document.page_count
        for page_index in range(total_pages):
            page = document.load_page(page_index)
            records.append(
                PageRecord(
                    document_id=document_id,
                    source_path=str(pdf_path),
                    page_number=page_index + 1,
                    total_pages=total_pages,
                    text=page.get_text("text").strip(),
                )
            )

    return tuple(records)


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
