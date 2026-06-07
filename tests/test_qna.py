from pathlib import Path

from sec_rag.qna import profile_qna_csv, split_source_docs


def test_split_source_docs_normalizes_markup_and_commas() -> None:
    assert split_source_docs("*AAPL*, *MSFT*") == ("AAPL", "MSFT")


def test_profile_qna_csv_counts_key_fields(tmp_path: Path) -> None:
    qna_path = tmp_path / "qna_data.csv"
    qna_path.write_text(
        "Question,Source Docs,Question Type,Source Chunk Type,Answer\n"
        "Q1,*AAPL*,Single-Doc RAG,Text,A1\n"
        "Q2,\"*AAPL*, *MSFT*\",Multi-Doc RAG,Table,A2\n",
        encoding="utf-8",
    )

    profile = profile_qna_csv(qna_path)

    assert profile.row_count == 2
    assert profile.columns == (
        "Question",
        "Source Docs",
        "Question Type",
        "Source Chunk Type",
        "Answer",
    )
    assert dict(profile.question_type_counts) == {"Multi-Doc RAG": 1, "Single-Doc RAG": 1}
    assert dict(profile.source_chunk_type_counts) == {"Table": 1, "Text": 1}
    assert dict(profile.source_doc_counts) == {"AAPL": 2, "MSFT": 1}
