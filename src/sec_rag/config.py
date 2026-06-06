"""Project configuration helpers."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem paths used by the SEC 10-Q RAG project."""

    root: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path


def get_project_paths(root: Path | None = None) -> ProjectPaths:
    """Return conventional project paths without creating directories."""
    project_root = root or Path.cwd()
    data_dir = project_root / "data"
    return ProjectPaths(
        root=project_root,
        data_dir=data_dir,
        raw_dir=data_dir / "raw",
        processed_dir=data_dir / "processed",
    )
