"""Document metadata normalization for SEC 10-Q filings."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


TICKER_TO_COMPANY = {
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "INTC": "Intel",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
}

FILING_TYPE_10Q = "10-Q"
FILENAME_PATTERN = re.compile(r"^(?P<year>\d{4})\s+Q(?P<quarter>[1-4])\s+(?P<ticker>[A-Z]{2,5})$")


@dataclass(frozen=True)
class DocumentMetadata:
    """Normalized metadata for one SEC filing document."""

    document_id: str
    source_filename: str
    ticker: str
    company: str
    year: int | None
    quarter: int | None
    filing_type: str


def document_id_from_path(path: Path) -> str:
    """Create a stable document id from a document path."""
    return path.stem.replace(" ", "_").lower()


def metadata_from_pdf_path(path: Path) -> DocumentMetadata:
    """Extract normalized metadata from a SEC 10-Q PDF filename."""
    stem = path.stem
    match = FILENAME_PATTERN.match(stem)
    if match:
        ticker = match.group("ticker")
        year = int(match.group("year"))
        quarter = int(match.group("quarter"))
    else:
        ticker = ""
        year = None
        quarter = None

    return DocumentMetadata(
        document_id=document_id_from_path(path),
        source_filename=path.name,
        ticker=ticker,
        company=TICKER_TO_COMPANY.get(ticker, ""),
        year=year,
        quarter=quarter,
        filing_type=FILING_TYPE_10Q,
    )
