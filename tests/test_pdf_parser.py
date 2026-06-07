import json
from pathlib import Path

import fitz

from sec_rag.pdf_parser import document_id_from_path, parse_pdf_pages, write_page_records_jsonl


def _make_pdf(path: Path, page_texts: list[str]) -> None:
    document = fitz.open()
    for text in page_texts:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_document_id_from_path_normalizes_stem() -> None:
    assert document_id_from_path(Path("AAPL 10-Q.pdf")) == "aapl_10-q"


def test_parse_pdf_pages_returns_page_metadata(tmp_path: Path) -> None:
    pdf_path = tmp_path / "filing.pdf"
    _make_pdf(pdf_path, ["Revenue increased.", "Risk factors updated."])

    records = parse_pdf_pages(pdf_path)

    assert len(records) == 2
    assert records[0].document_id == "filing"
    assert records[0].source_path == str(pdf_path.resolve())
    assert records[0].page_number == 1
    assert records[0].total_pages == 2
    assert "Revenue increased" in records[0].text


def test_parse_pdf_pages_handles_textless_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "empty.pdf"
    _make_pdf(pdf_path, [""])

    records = parse_pdf_pages(pdf_path)

    assert len(records) == 1
    assert records[0].text == ""


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
