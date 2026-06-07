"""Dataset discovery helpers for SEC filing experiments."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PDF_EXTENSIONS = {".pdf"}
QA_EXTENSIONS = {".csv", ".json", ".jsonl"}
KG_RAG_REPO_URL = "https://github.com/VectorInstitute/kg-rag.git"
EXCLUDED_DIR_NAMES = {".git", ".github", "__pycache__", "node_modules"}
QA_FILENAME_HINTS = ("qna", "qa", "question")


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


@dataclass(frozen=True)
class ExtensionProfile:
    """File counts grouped by extension."""

    extension: str
    count: int


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
        if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
            continue
        suffix = path.suffix.lower()
        if suffix in PDF_EXTENSIONS:
            pdfs.append(path)
        elif suffix in QA_EXTENSIONS and is_likely_qa_file(path):
            qa_files.append(path)
        else:
            other_files.append(path)

    return FileInventory(
        root=root,
        pdfs=tuple(pdfs),
        qa_files=tuple(qa_files),
        other_files=tuple(other_files),
    )


def is_likely_qa_file(path: Path) -> bool:
    """Return whether a structured file likely contains question-answer data."""
    name = path.name.lower()
    return path.suffix.lower() in QA_EXTENSIONS and any(hint in name for hint in QA_FILENAME_HINTS)


def dataset_missing_message(data_dir: Path) -> str:
    """Return actionable setup guidance for a missing dataset directory."""
    return (
        f"Dataset directory does not exist: {data_dir}\n"
        "Acquire the dataset first:\n"
        "  python scripts\\acquire_dataset.py\n"
        "Then rerun the command."
    )


def find_git_executable() -> str:
    """Return a usable git executable path for CLI-based dataset acquisition."""
    executable = shutil.which("git")
    if executable:
        return executable

    windows_candidates = (
        Path("C:/Program Files/Git/cmd/git.exe"),
        Path("C:/Program Files/Git/bin/git.exe"),
        Path("C:/Program Files (x86)/Git/cmd/git.exe"),
    )
    for candidate in windows_candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError("Git executable not found. Install Git or add it to PATH.")


def clone_kg_rag_dataset(target_dir: Path, repo_url: str = KG_RAG_REPO_URL) -> Path:
    """Clone the KG-RAG repository into ``target_dir`` if it is not present."""
    target = target_dir.expanduser().resolve()
    if target.exists() and any(target.iterdir()):
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    git_executable = find_git_executable()
    subprocess.run(
        [git_executable, "clone", "--depth", "1", repo_url, str(target)],
        check=True,
    )
    return target


def profile_extensions(inventory: FileInventory) -> tuple[ExtensionProfile, ...]:
    """Count all discovered files by extension."""
    counter: Counter[str] = Counter()
    for path in (*inventory.pdfs, *inventory.qa_files, *inventory.other_files):
        counter[path.suffix.lower() or "(no extension)"] += 1
    return tuple(
        ExtensionProfile(extension=extension, count=count)
        for extension, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
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
