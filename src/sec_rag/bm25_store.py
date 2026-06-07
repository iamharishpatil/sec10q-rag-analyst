"""BM25 lexical index storage and search helpers."""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from rank_bm25 import BM25Okapi

from sec_rag.retrieval import RetrievalResult


BM25_FILENAME = "bm25.pkl"
CHUNKS_FILENAME = "chunks.jsonl"
MANIFEST_FILENAME = "manifest.json"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class BM25Index:
    """Loaded BM25 index and chunk payloads."""

    bm25: BM25Okapi
    chunks: tuple[dict, ...]
    manifest: dict


def tokenize(text: str) -> list[str]:
    """Tokenize text for lexical retrieval."""
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def write_bm25_index(output_dir: Path, chunks: Iterable[dict]) -> dict:
    """Build and persist a BM25 index from chunk records."""
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    chunk_records = tuple(chunks)
    tokenized_corpus = [tokenize(chunk.get("text", "")) for chunk in chunk_records]
    bm25 = BM25Okapi(tokenized_corpus)

    with (output / BM25_FILENAME).open("wb") as file:
        pickle.dump(bm25, file)

    with (output / CHUNKS_FILENAME).open("w", encoding="utf-8") as file:
        for chunk in chunk_records:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    manifest = {
        "backend": "bm25",
        "chunk_rows": len(chunk_records),
    }
    (output / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_bm25_index(index_dir: Path) -> BM25Index:
    """Load a persisted BM25 index."""
    root = index_dir.expanduser().resolve()
    with (root / BM25_FILENAME).open("rb") as file:
        bm25 = pickle.load(file)

    chunks = []
    with (root / CHUNKS_FILENAME).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    manifest = json.loads((root / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    return BM25Index(bm25=bm25, chunks=tuple(chunks), manifest=manifest)


def search_bm25_index(
    query: str,
    index: BM25Index,
    top_k: int = 5,
    ticker: str | None = None,
) -> tuple[RetrievalResult, ...]:
    """Search a BM25 index and return standardized retrieval results."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    query_tokens = tokenize(query)
    scores = np.asarray(index.bm25.get_scores(query_tokens), dtype=np.float32)
    candidates = [
        (row_index, float(score))
        for row_index, score in enumerate(scores)
        if _matches_ticker(index.chunks[row_index], ticker)
    ]
    candidates.sort(key=lambda item: item[1], reverse=True)
    top_candidates = candidates[: min(top_k, len(candidates))]

    return tuple(
        RetrievalResult(
            rank=rank,
            score=score,
            chunk=index.chunks[row_index],
            backend="bm25",
            bm25_score=score,
        )
        for rank, (row_index, score) in enumerate(top_candidates, start=1)
    )


def _matches_ticker(chunk: dict, ticker: str | None) -> bool:
    if ticker is None:
        return True
    return str(chunk.get("ticker", "")).upper() == ticker.upper()
