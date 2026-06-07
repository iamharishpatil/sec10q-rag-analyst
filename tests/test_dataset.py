from pathlib import Path

import pytest

from sec_rag.dataset import (
    dataset_missing_message,
    discover_dataset_files,
    profile_csv,
    profile_extensions,
    relative_paths,
)


def test_discover_dataset_files_groups_relevant_extensions(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    pdf = tmp_path / "docs" / "aapl-10q.pdf"
    csv_file = tmp_path / "qa.csv"
    readme = tmp_path / "README.md"
    pdf.write_text("fake pdf", encoding="utf-8")
    csv_file.write_text("question,answer\nWhat?,That.\n", encoding="utf-8")
    readme.write_text("notes", encoding="utf-8")

    inventory = discover_dataset_files(tmp_path)

    assert inventory.pdfs == (pdf.resolve(),)
    assert inventory.qa_files == (csv_file.resolve(),)
    assert inventory.other_files == (readme.resolve(),)


def test_discover_dataset_files_requires_existing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_dataset_files(tmp_path / "missing")


def test_profile_csv_counts_rows_and_samples(tmp_path: Path) -> None:
    csv_file = tmp_path / "qa.csv"
    csv_file.write_text(
        "question,answer\n"
        "What was revenue?,Revenue increased.\n"
        "What risks?,Supply constraints.\n",
        encoding="utf-8",
    )

    profile = profile_csv(csv_file, sample_size=1)

    assert profile.columns == ("question", "answer")
    assert profile.row_count == 2
    assert profile.sample_rows == (
        {"question": "What was revenue?", "answer": "Revenue increased."},
    )


def test_relative_paths_limits_and_renders_from_root(tmp_path: Path) -> None:
    paths = [tmp_path / "a.txt", tmp_path / "b.txt", tmp_path / "c.txt"]

    rendered = relative_paths(paths, tmp_path, limit=2)

    assert rendered == ("a.txt", "b.txt")


def test_dataset_missing_message_points_to_acquisition_command(tmp_path: Path) -> None:
    message = dataset_missing_message(tmp_path / "missing")

    assert "python scripts\\acquire_dataset.py" in message


def test_profile_extensions_counts_all_discovered_file_types(tmp_path: Path) -> None:
    (tmp_path / "a.pdf").write_text("pdf", encoding="utf-8")
    (tmp_path / "qa.csv").write_text("question,answer\nq,a\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("notes", encoding="utf-8")

    inventory = discover_dataset_files(tmp_path)
    profile = profile_extensions(inventory)

    assert {item.extension: item.count for item in profile} == {
        ".csv": 1,
        ".md": 1,
        ".pdf": 1,
    }
