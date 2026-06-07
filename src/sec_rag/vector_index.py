"""Simple local vector index for baseline dense retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


EMBEDDINGS_FILENAME = "embeddings.npy"
METADATA_FILENAME = "chunks.jsonl"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class SearchResult:
    """One dense retrieval result."""

    rank: int
    score: float
    chunk: dict


def read_chunk_records_jsonl(input_dir: Path) -> tuple[dict, ...]:
    """Read all chunk records from a directory of chunk JSONL files."""
    records: list[dict] = []
    for path in sorted(input_dir.expanduser().resolve().glob("*.chunks.jsonl")):
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    records.append(json.loads(line))
    return tuple(records)


def write_index(
    output_dir: Path,
    embeddings: np.ndarray,
    chunks: Iterable[dict],
    model_name: str,
) -> None:
    """Write embeddings, chunk metadata, and a manifest."""
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    chunk_records = tuple(chunks)

    np.save(output / EMBEDDINGS_FILENAME, np.asarray(embeddings, dtype=np.float32))
    with (output / METADATA_FILENAME).open("w", encoding="utf-8") as file:
        for chunk in chunk_records:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    manifest = {
        "model_name": model_name,
        "embedding_rows": int(len(embeddings)),
        "chunk_rows": len(chunk_records),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
    }
    (output / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def load_index(index_dir: Path) -> tuple[np.ndarray, tuple[dict, ...], dict]:
    """Load embeddings, chunk metadata, and manifest."""
    root = index_dir.expanduser().resolve()
    embeddings = np.load(root / EMBEDDINGS_FILENAME)
    chunks = []
    with (root / METADATA_FILENAME).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))
    manifest = json.loads((root / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    return embeddings, tuple(chunks), manifest


def search_index(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    chunks: tuple[dict, ...],
    top_k: int = 5,
) -> tuple[SearchResult, ...]:
    """Return top-k cosine results from normalized vectors."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if len(embeddings) != len(chunks):
        raise ValueError("embeddings and chunks must have the same length")

    scores = embeddings @ query_embedding
    k = min(top_k, len(scores))
    top_indices = np.argsort(scores)[::-1][:k]
    return tuple(
        SearchResult(rank=rank, score=float(scores[index]), chunk=chunks[int(index)])
        for rank, index in enumerate(top_indices, start=1)
    )
