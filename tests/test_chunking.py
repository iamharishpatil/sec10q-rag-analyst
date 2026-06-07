import json
from pathlib import Path

import pytest

from sec_rag.chunking import (
    chunk_page_record,
    chunk_page_records,
    read_page_records_jsonl,
    summarize_chunks,
    validate_chunking_params,
    write_chunk_records_jsonl,
)


def _page_record(text: str = "one two three four five six") -> dict:
    return {
        "document_id": "2023_q2_msft",
        "source_path": "data/raw/sec-10-q/data/sec-10-q/docs/2023 Q2 MSFT.pdf",
        "source_filename": "2023 Q2 MSFT.pdf",
        "ticker": "MSFT",
        "company": "Microsoft",
        "year": 2023,
        "quarter": 2,
        "filing_type": "10-Q",
        "page_number": 4,
        "total_pages": 74,
        "text": text,
    }


def test_chunk_page_record_preserves_metadata() -> None:
    chunks = chunk_page_record(_page_record(), chunk_size=4, overlap=1)

    assert chunks[0].chunk_id == "2023_q2_msft_p4_c0"
    assert chunks[0].document_id == "2023_q2_msft"
    assert chunks[0].source_filename == "2023 Q2 MSFT.pdf"
    assert chunks[0].ticker == "MSFT"
    assert chunks[0].company == "Microsoft"
    assert chunks[0].page_number == 4
    assert chunks[0].text == "one two three four"


def test_chunk_page_record_uses_word_overlap() -> None:
    chunks = chunk_page_record(_page_record(), chunk_size=4, overlap=1)

    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "four five six",
    ]


def test_chunk_page_record_skips_empty_text() -> None:
    assert chunk_page_record(_page_record(text="")) == tuple()


def test_chunk_page_records_chunks_many_pages() -> None:
    chunks = chunk_page_records(
        [_page_record("one two three"), _page_record("four five six")],
        chunk_size=10,
        overlap=0,
    )

    assert len(chunks) == 2


def test_validate_chunking_params_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        validate_chunking_params(chunk_size=0, overlap=0)
    with pytest.raises(ValueError):
        validate_chunking_params(chunk_size=10, overlap=-1)
    with pytest.raises(ValueError):
        validate_chunking_params(chunk_size=10, overlap=10)


def test_write_and_read_chunk_jsonl_flow(tmp_path: Path) -> None:
    chunks = chunk_page_record(_page_record(), chunk_size=4, overlap=1)
    output_path = tmp_path / "chunks.jsonl"

    count = write_chunk_records_jsonl(chunks, output_path)

    assert count == 2
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["chunk_id"] == "2023_q2_msft_p4_c0"


def test_read_page_records_jsonl(tmp_path: Path) -> None:
    input_path = tmp_path / "pages.jsonl"
    input_path.write_text(json.dumps(_page_record()) + "\n", encoding="utf-8")

    records = read_page_records_jsonl(input_path)

    assert records == (_page_record(),)


def test_summarize_chunks_reports_counts() -> None:
    chunks = chunk_page_records([_page_record(), _page_record(text="")], chunk_size=4, overlap=1)

    summary = summarize_chunks(page_count=2, chunks=chunks)

    assert summary.page_records == 2
    assert summary.chunks == 2
    assert summary.skipped_empty_pages == 1
    assert summary.average_words_per_chunk > 0
