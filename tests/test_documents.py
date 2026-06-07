from pathlib import Path

from sec_rag.documents import document_id_from_path, metadata_from_pdf_path


def test_document_id_from_path_normalizes_stem() -> None:
    assert document_id_from_path(Path("2023 Q1 AAPL.pdf")) == "2023_q1_aapl"


def test_metadata_from_pdf_path_extracts_sec_10q_filename_fields() -> None:
    metadata = metadata_from_pdf_path(Path("2023 Q2 MSFT.pdf"))

    assert metadata.document_id == "2023_q2_msft"
    assert metadata.source_filename == "2023 Q2 MSFT.pdf"
    assert metadata.ticker == "MSFT"
    assert metadata.company == "Microsoft"
    assert metadata.year == 2023
    assert metadata.quarter == 2
    assert metadata.filing_type == "10-Q"


def test_metadata_from_pdf_path_handles_unknown_filename_pattern() -> None:
    metadata = metadata_from_pdf_path(Path("filing.pdf"))

    assert metadata.document_id == "filing"
    assert metadata.ticker == ""
    assert metadata.company == ""
    assert metadata.year is None
    assert metadata.quarter is None
