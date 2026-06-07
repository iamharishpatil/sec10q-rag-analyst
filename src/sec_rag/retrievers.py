"""Concrete retriever classes for dense, lexical, and hybrid retrieval."""

from __future__ import annotations

from pathlib import Path

from sec_rag.bm25_store import load_bm25_index, search_bm25_index
from sec_rag.embeddings import DEFAULT_EMBEDDING_MODEL, embed_query, load_embedding_model
from sec_rag.qdrant_store import DEFAULT_COLLECTION, create_local_client, search_chunks
from sec_rag.retrieval import RetrievalFilters, RetrievalResult, Retriever
from sec_rag.vector_index import load_index, search_index


class QdrantRetriever:
    """Dense retriever backed by local Qdrant storage."""

    def __init__(
        self,
        index_dir: Path,
        collection: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.index_dir = index_dir
        self.collection = collection
        self.model_name = model_name
        self.client = create_local_client(index_dir)
        self.model = load_embedding_model(model_name)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: RetrievalFilters | None = None,
    ) -> tuple[RetrievalResult, ...]:
        query_embedding = embed_query(query, model=self.model)
        ticker = filters.ticker if filters else None
        results = search_chunks(
            self.client,
            collection_name=self.collection,
            query_embedding=query_embedding,
            top_k=top_k,
            ticker=ticker,
        )
        return tuple(
            RetrievalResult(
                rank=result.rank,
                score=result.score,
                chunk=result.chunk,
                backend="dense",
                dense_score=result.score,
            )
            for result in results
        )


class NumpyRetriever:
    """Dense teaching baseline backed by NumPy arrays."""

    def __init__(self, index_dir: Path) -> None:
        self.embeddings, self.chunks, self.manifest = load_index(index_dir)
        self.model_name = self.manifest["model_name"]
        self.model = load_embedding_model(self.model_name)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: RetrievalFilters | None = None,
    ) -> tuple[RetrievalResult, ...]:
        query_embedding = embed_query(query, model=self.model)
        chunks = self.chunks
        embeddings = self.embeddings
        if filters and filters.ticker:
            selected = [
                index
                for index, chunk in enumerate(chunks)
                if str(chunk.get("ticker", "")).upper() == filters.ticker.upper()
            ]
            chunks = tuple(chunks[index] for index in selected)
            embeddings = embeddings[selected]

        results = search_index(query_embedding, embeddings=embeddings, chunks=chunks, top_k=top_k)
        return tuple(
            RetrievalResult(
                rank=result.rank,
                score=result.score,
                chunk=result.chunk,
                backend="numpy",
                dense_score=result.score,
            )
            for result in results
        )


class BM25Retriever:
    """Lexical retriever backed by BM25."""

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.index = load_bm25_index(index_dir)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: RetrievalFilters | None = None,
    ) -> tuple[RetrievalResult, ...]:
        ticker = filters.ticker if filters else None
        return search_bm25_index(query, index=self.index, top_k=top_k, ticker=ticker)


class HybridRetriever:
    """Hybrid retriever that fuses dense and BM25 results."""

    def __init__(
        self,
        dense_retriever: Retriever,
        bm25_retriever: Retriever,
        dense_weight: float = 0.65,
        bm25_weight: float = 0.35,
        candidate_multiplier: int = 5,
    ) -> None:
        if dense_weight < 0 or bm25_weight < 0:
            raise ValueError("retriever weights must be non-negative")
        if dense_weight + bm25_weight == 0:
            raise ValueError("at least one retriever weight must be positive")
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.candidate_multiplier = candidate_multiplier

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: RetrievalFilters | None = None,
    ) -> tuple[RetrievalResult, ...]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        candidate_k = max(top_k * self.candidate_multiplier, top_k)
        dense_results = self.dense_retriever.search(query, top_k=candidate_k, filters=filters)
        bm25_results = self.bm25_retriever.search(query, top_k=candidate_k, filters=filters)
        dense_scores = _normalize_scores(dense_results)
        bm25_scores = _normalize_scores(bm25_results)

        merged: dict[str, dict] = {}
        for result in dense_results:
            chunk_id = _chunk_key(result.chunk)
            merged.setdefault(chunk_id, {"chunk": result.chunk, "dense": None, "bm25": None})
            merged[chunk_id]["dense"] = result.score
        for result in bm25_results:
            chunk_id = _chunk_key(result.chunk)
            merged.setdefault(chunk_id, {"chunk": result.chunk, "dense": None, "bm25": None})
            merged[chunk_id]["bm25"] = result.score

        fused = []
        for chunk_id, payload in merged.items():
            dense_component = dense_scores.get(chunk_id, 0.0)
            bm25_component = bm25_scores.get(chunk_id, 0.0)
            score = self.dense_weight * dense_component + self.bm25_weight * bm25_component
            fused.append((score, payload["chunk"], payload["dense"], payload["bm25"]))

        fused.sort(key=lambda item: item[0], reverse=True)
        return tuple(
            RetrievalResult(
                rank=rank,
                score=float(score),
                chunk=chunk,
                backend="hybrid",
                dense_score=dense_score,
                bm25_score=bm25_score,
            )
            for rank, (score, chunk, dense_score, bm25_score) in enumerate(fused[:top_k], start=1)
        )


def _normalize_scores(results: tuple[RetrievalResult, ...]) -> dict[str, float]:
    if not results:
        return {}
    raw_scores = [result.score for result in results]
    minimum = min(raw_scores)
    maximum = max(raw_scores)
    if maximum == minimum:
        return {_chunk_key(result.chunk): 1.0 for result in results}
    return {
        _chunk_key(result.chunk): (result.score - minimum) / (maximum - minimum)
        for result in results
    }


def _chunk_key(chunk: dict) -> str:
    return str(chunk.get("chunk_id") or f"{chunk.get('source_filename')}:{chunk.get('page_number')}")


def create_retriever(
    backend: str,
    qdrant_index_dir: Path,
    bm25_index_dir: Path,
    collection: str = DEFAULT_COLLECTION,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> Retriever:
    """Create a retriever from a backend name."""
    normalized = "dense" if backend == "qdrant" else backend
    if normalized == "dense":
        return QdrantRetriever(qdrant_index_dir, collection=collection, model_name=model_name)
    if normalized == "numpy":
        return NumpyRetriever(qdrant_index_dir)
    if normalized == "bm25":
        return BM25Retriever(bm25_index_dir)
    if normalized == "hybrid":
        return HybridRetriever(
            dense_retriever=QdrantRetriever(
                qdrant_index_dir,
                collection=collection,
                model_name=model_name,
            ),
            bm25_retriever=BM25Retriever(bm25_index_dir),
        )
    raise ValueError(f"Unsupported retriever backend: {backend}")
