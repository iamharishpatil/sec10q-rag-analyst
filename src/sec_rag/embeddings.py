"""Embedding model helpers for local retrieval experiments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from typing import Any


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Any:
    """Load an open-source sentence-transformers embedding model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(
    texts: Sequence[str],
    model: Any,
    batch_size: int = 32,
) -> np.ndarray:
    """Embed texts as L2-normalized float32 vectors."""
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def embed_query(query: str, model: Any) -> np.ndarray:
    """Embed one query as a normalized float32 vector."""
    return embed_texts([query], model=model, batch_size=1)[0]
