"""Dataset discovery helpers for SEC filing experiments."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PDF_EXTENSIONS = {".pdf"}
QA_EXTENSIONS = {".csv", ".json", ".jsonl"}


@dataclass(frozen=True)
class FileInventory:
    """Summary of relevant files found in a dataset directory."""

    root: Path
    pdfs: tuple[Path, ...]
    qa_files: tuple[Path, ...]
    other_files: tuple[Path, ...]


@dataclass(frozen=True)
class CsvProfile:
    """Small, dependency-free profile of a CSV file."""

    path: Path
    columns: tuple[str, ...]
    row_count: int
    sample_rows: tuple[dict[str, str], ...]


def discover_dataset_files(data_dir: Path) -> FileInventory:
    """Find PDFs, likely Q&A files, and other files under ``data_dir``."""
    root = data_dir.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {root}")

    pdfs: list[Path] = []
    qa_files: list[Path] = []
    other_files: list[Path] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in PDF_EXTENSIONS:
            pdfs.append(path)
        elif suffix in QA_EXTENSIONS:
            qa_files.append(path)
        else:
            other_files.append(path)

    return FileInventory(
        root=root,
        pdfs=tuple(pdfs),
        qa_files=tuple(qa_files),
        other_files=tuple(other_files),
    )


def profile_csv(path: Path, sample_size: int = 5) -> CsvProfile:
    """Return columns, row count, and a few sample rows for a CSV file."""
    sample_rows: list[dict[str, str]] = []
    row_count = 0

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = tuple(reader.fieldnames or ())
        for row in reader:
            row_count += 1
            if len(sample_rows) < sample_size:
                sample_rows.append({key: value for key, value in row.items()})

    return CsvProfile(
        path=path,
        columns=columns,
        row_count=row_count,
        sample_rows=tuple(sample_rows),
    )


def sample_json_records(path: Path, sample_size: int = 5) -> tuple[dict, ...]:
    """Read a few records from a JSON or JSONL file."""
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    records.append(json.loads(line))
                if len(records) >= sample_size:
                    break
        return tuple(records)

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, list):
        return tuple(item for item in payload[:sample_size] if isinstance(item, dict))
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                return tuple(item for item in value[:sample_size] if isinstance(item, dict))
        return (payload,)
    return tuple()


def relative_paths(paths: Iterable[Path], root: Path, limit: int = 10) -> tuple[str, ...]:
    """Render paths relative to a root for compact console output."""
    rendered = []
    for path in paths:
        try:
            rendered.append(str(path.relative_to(root)))
        except ValueError:
            rendered.append(str(path))
        if len(rendered) >= limit:
            break
    return tuple(rendered)
