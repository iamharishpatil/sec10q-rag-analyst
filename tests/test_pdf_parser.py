import json
from pathlib import Path

import fitz

from sec_rag.pdf_parser import parse_pdf_pages, summarize_page_records, write_page_records_jsonl


def _make_pdf(path: Path, page_texts: list[str]) -> None:
    document = fitz.open()
    for text in page_texts:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_parse_pdf_pages_returns_page_metadata(tmp_path: Path) -> None:
    pdf_path = tmp_path / "2023 Q2 MSFT.pdf"
    _make_pdf(pdf_path, ["Revenue increased.", "Risk factors updated."])

    records = parse_pdf_pages(pdf_path)

    assert len(records) == 2
    assert records[0].document_id == "2023_q2_msft"
    assert records[0].source_path == str(pdf_path.resolve())
    assert records[0].source_filename == "2023 Q2 MSFT.pdf"
    assert records[0].ticker == "MSFT"
    assert records[0].company == "Microsoft"
    assert records[0].year == 2023
    assert records[0].quarter == 2
    assert records[0].filing_type == "10-Q"
    assert records[0].page_number == 1
    assert records[0].total_pages == 2
    assert "Revenue increased" in records[0].text


def test_parse_pdf_pages_handles_textless_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "empty.pdf"
    _make_pdf(pdf_path, [""])

    records = parse_pdf_pages(pdf_path)

    assert len(records) == 1
    assert records[0].text == ""


def test_summarize_page_records_reports_extraction_quality(tmp_path: Path) -> None:
    pdf_path = tmp_path / "2023 Q1 AAPL.pdf"
    _make_pdf(pdf_path, ["Revenue increased.", ""])

    summary = summarize_page_records(parse_pdf_pages(pdf_path))

    assert summary.document_id == "2023_q1_aapl"
    assert summary.source_filename == "2023 Q1 AAPL.pdf"
    assert summary.total_pages == 2
    assert summary.empty_pages == 1
    assert summary.longest_page_chars > summary.shortest_page_chars


def test_write_page_records_jsonl_writes_valid_json(tmp_path: Path) -> None:
    pdf_path = tmp_path / "filing.pdf"
    output_path = tmp_path / "out" / "filing.pages.jsonl"
    _make_pdf(pdf_path, ["Revenue increased."])
    records = parse_pdf_pages(pdf_path)

    count = write_page_records_jsonl(records, output_path)

    assert count == 1
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["document_id"] == "filing"
    assert rows[0]["page_number"] == 1
