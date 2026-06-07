"""Application-level pipeline orchestration."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from sec_rag.bm25_store import write_bm25_index
from sec_rag.chunking import (
    chunk_page_records,
    read_page_records_jsonl,
    summarize_chunks,
    write_chunk_records_jsonl,
)
from sec_rag.config import get_project_paths
from sec_rag.dataset import discover_dataset_files, dataset_missing_message, relative_paths
from sec_rag.embeddings import DEFAULT_EMBEDDING_MODEL, embed_texts, load_embedding_model
from sec_rag.pdf_parser import parse_pdf_pages, summarize_page_records, write_page_records_jsonl
from sec_rag.qdrant_store import DEFAULT_COLLECTION, create_local_client, recreate_collection, upsert_chunks
from sec_rag.retrieval import RetrievalFilters, RetrievalResult
from sec_rag.retrieval_eval import expected_source_labels, is_source_hit, summarize_retrieval_hits
from sec_rag.retrievers import create_retriever
from sec_rag.vector_index import read_chunk_records_jsonl, write_index


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the current SEC 10-Q RAG pipeline."""

    raw_dir: Path
    pages_dir: Path
    chunks_dir: Path
    qdrant_index_dir: Path
    bm25_index_dir: Path
    qna_path: Path
    retriever_backend: str = "hybrid"
    collection: str = DEFAULT_COLLECTION
    model_name: str = DEFAULT_EMBEDDING_MODEL
    batch_size: int = 32
    top_k: int = 5
    query: str = "What was Apple's total net sales in Q2 2023?"
    ticker: str | None = None
    limit_pdfs: int | None = None
    eval_limit: int | None = None
    skip_parse: bool = False
    skip_chunk: bool = False
    skip_index: bool = False
    skip_retrieve: bool = False
    skip_eval: bool = False

    @classmethod
    def default(cls) -> "PipelineConfig":
        """Build default config from conventional project paths."""
        paths = get_project_paths()
        return cls(
            raw_dir=paths.sec_10q_raw_dir,
            pages_dir=paths.parsed_pages_dir,
            chunks_dir=paths.chunks_dir,
            qdrant_index_dir=paths.indexes_dir / "qdrant",
            bm25_index_dir=paths.indexes_dir / "bm25",
            qna_path=paths.sec_10q_raw_dir / "data" / "sec-10-q" / "qna_data.csv",
        )


@dataclass(frozen=True)
class PipelineResult:
    """Summary of one pipeline run."""

    parsed_pages: int = 0
    empty_pages: int = 0
    chunks: int = 0
    skipped_empty_pages: int = 0
    qdrant_rows: int = 0
    bm25_rows: int = 0
    retrieved: tuple[RetrievalResult, ...] = tuple()
    evaluated_questions: int = 0
    retrieval_hits: int = 0
    recall_at_k: float = 0.0


class PipelineRunner:
    """Run parse, chunk, index, retrieve, and evaluate stages."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run(self) -> PipelineResult:
        """Run enabled pipeline stages in order."""
        if not self.config.raw_dir.exists():
            raise FileNotFoundError(dataset_missing_message(self.config.raw_dir))

        parsed_pages = 0
        empty_pages = 0
        chunks = 0
        skipped_empty_pages = 0
        qdrant_rows = 0
        bm25_rows = 0
        retrieved: tuple[RetrievalResult, ...] = tuple()
        evaluated_questions = 0
        retrieval_hits = 0
        recall_at_k = 0.0

        if not self.config.skip_parse:
            parsed_pages, empty_pages = self.parse_pages()
        if not self.config.skip_chunk:
            chunks, skipped_empty_pages = self.chunk_pages()
        if not self.config.skip_index:
            qdrant_rows, bm25_rows = self.build_indexes()
        if not self.config.skip_retrieve:
            retrieved = self.retrieve()
        if not self.config.skip_eval:
            evaluated_questions, retrieval_hits, recall_at_k = self.evaluate()

        return PipelineResult(
            parsed_pages=parsed_pages,
            empty_pages=empty_pages,
            chunks=chunks,
            skipped_empty_pages=skipped_empty_pages,
            qdrant_rows=qdrant_rows,
            bm25_rows=bm25_rows,
            retrieved=retrieved,
            evaluated_questions=evaluated_questions,
            retrieval_hits=retrieval_hits,
            recall_at_k=recall_at_k,
        )

    def parse_pages(self) -> tuple[int, int]:
        """Parse PDFs into page JSONL files."""
        inventory = discover_dataset_files(self.config.raw_dir)
        pdfs = inventory.pdfs[: self.config.limit_pdfs] if self.config.limit_pdfs else inventory.pdfs
        if not pdfs:
            raise FileNotFoundError(f"No PDF files found under: {inventory.root}")

        total_pages = 0
        total_empty_pages = 0
        print(f"Parsing {len(pdfs)} PDF(s) from {inventory.root}")
        for pdf_path in pdfs:
            records = parse_pdf_pages(pdf_path)
            summary = summarize_page_records(records)
            output_path = self.config.pages_dir / f"{pdf_path.stem}.pages.jsonl"
            written = write_page_records_jsonl(records, output_path)
            total_pages += written
            total_empty_pages += summary.empty_pages
            rel_pdf = relative_paths([pdf_path], inventory.root, limit=1)[0]
            print(
                "  - "
                f"{rel_pdf}: {written} page record(s), "
                f"{summary.empty_pages} empty page(s), "
                f"{summary.average_chars_per_page:.0f} avg chars/page"
            )
        print(f"Output directory: {self.config.pages_dir}")
        print(f"Total page records: {total_pages}")
        print(f"Total empty pages: {total_empty_pages}")
        return total_pages, total_empty_pages

    def chunk_pages(self) -> tuple[int, int]:
        """Chunk parsed page JSONL files."""
        input_files = tuple(sorted(self.config.pages_dir.glob("*.pages.jsonl")))
        if self.config.limit_pdfs is not None:
            input_files = input_files[: self.config.limit_pdfs]
        if not input_files:
            raise FileNotFoundError(f"No page JSONL files found under: {self.config.pages_dir}")

        total_chunks = 0
        total_empty_pages = 0
        print(f"Chunking {len(input_files)} parsed page file(s) from {self.config.pages_dir}")
        for input_file in input_files:
            pages = read_page_records_jsonl(input_file)
            chunks = chunk_page_records(pages)
            output_file = self.config.chunks_dir / input_file.name.replace(
                ".pages.jsonl",
                ".chunks.jsonl",
            )
            write_chunk_records_jsonl(chunks, output_file)
            summary = summarize_chunks(page_count=len(pages), chunks=chunks)
            total_chunks += summary.chunks
            total_empty_pages += summary.skipped_empty_pages
            print(
                "  - "
                f"{input_file.name}: {summary.page_records} page(s), "
                f"{summary.chunks} chunk(s), "
                f"{summary.skipped_empty_pages} empty page(s), "
                f"{summary.average_words_per_chunk:.0f} avg words/chunk"
            )
        print(f"Output directory: {self.config.chunks_dir}")
        print(f"Total chunks: {total_chunks}")
        print(f"Total skipped empty pages: {total_empty_pages}")
        return total_chunks, total_empty_pages

    def build_indexes(self) -> tuple[int, int]:
        """Build indexes needed by the configured retriever backend."""
        chunks = read_chunk_records_jsonl(self.config.chunks_dir)
        if not chunks:
            raise FileNotFoundError(f"No chunk records found under: {self.config.chunks_dir}")

        backend = _normalize_backend(self.config.retriever_backend)
        qdrant_rows = 0
        bm25_rows = 0
        if backend in {"dense", "hybrid", "numpy"}:
            qdrant_rows = self.build_dense_index(chunks)
        if backend in {"bm25", "hybrid"}:
            bm25_rows = self.build_bm25_index(chunks)
        return qdrant_rows, bm25_rows

    def build_dense_index(self, chunks: tuple[dict, ...]) -> int:
        """Build the dense Qdrant index, plus NumPy baseline if requested."""
        print(f"Loading embedding model: {self.config.model_name}")
        model = load_embedding_model(self.config.model_name)
        texts = [chunk["text"] for chunk in chunks]
        print(f"Embedding {len(texts)} chunk(s)")
        embeddings = embed_texts(texts, model=model, batch_size=self.config.batch_size)

        if _normalize_backend(self.config.retriever_backend) == "numpy":
            write_index(
                self.config.qdrant_index_dir,
                embeddings=embeddings,
                chunks=chunks,
                model_name=self.config.model_name,
            )
            print(f"NumPy index written to: {self.config.qdrant_index_dir}")
        else:
            client = create_local_client(self.config.qdrant_index_dir)
            recreate_collection(
                client,
                collection_name=self.config.collection,
                vector_size=embeddings.shape[1],
            )
            rows = upsert_chunks(
                client,
                self.config.collection,
                embeddings=embeddings,
                chunks=chunks,
            )
            print(f"Qdrant collection: {self.config.collection}")
            print(f"Upserted rows: {rows}")
        print(f"Rows: {len(chunks)}")
        print(f"Dimensions: {embeddings.shape[1]}")
        return len(chunks)

    def build_bm25_index(self, chunks: tuple[dict, ...]) -> int:
        """Build the BM25 index."""
        manifest = write_bm25_index(self.config.bm25_index_dir, chunks)
        print(f"BM25 index written to: {self.config.bm25_index_dir}")
        print(f"BM25 rows: {manifest['chunk_rows']}")
        return int(manifest["chunk_rows"])

    def retrieve(self) -> tuple[RetrievalResult, ...]:
        """Run a retrieval smoke query."""
        retriever = create_retriever(
            self.config.retriever_backend,
            qdrant_index_dir=self.config.qdrant_index_dir,
            bm25_index_dir=self.config.bm25_index_dir,
            collection=self.config.collection,
            model_name=self.config.model_name,
        )
        results = retriever.search(
            self.config.query,
            top_k=self.config.top_k,
            filters=RetrievalFilters(ticker=self.config.ticker),
        )
        print_retrieval_results(
            query=self.config.query,
            backend=_normalize_backend(self.config.retriever_backend),
            results=results,
        )
        return results

    def evaluate(self) -> tuple[int, int, float]:
        """Evaluate retrieval with source-document Recall@k."""
        retriever = create_retriever(
            self.config.retriever_backend,
            qdrant_index_dir=self.config.qdrant_index_dir,
            bm25_index_dir=self.config.bm25_index_dir,
            collection=self.config.collection,
            model_name=self.config.model_name,
        )
        total = 0
        hits = 0
        with self.config.qna_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if self.config.eval_limit is not None and total >= self.config.eval_limit:
                    break
                expected = expected_source_labels(row.get("Source Docs", ""))
                results = retriever.search(row["Question"], top_k=self.config.top_k)
                total += 1
                hits += int(is_source_hit(expected, results))

        summary = summarize_retrieval_hits(total_questions=total, hits=hits)
        print(f"Questions: {summary.total_questions}")
        print(f"Backend: {_normalize_backend(self.config.retriever_backend)}")
        print(f"Hits@{self.config.top_k}: {summary.hits}")
        print(f"Source Recall@{self.config.top_k}: {summary.recall_at_k:.3f}")
        return summary.total_questions, summary.hits, summary.recall_at_k


def print_retrieval_results(
    query: str,
    backend: str,
    results: tuple[RetrievalResult, ...],
) -> None:
    """Print retrieval results in a terminal-friendly format."""
    print(f"Query: {query}")
    print(f"Backend: {backend}")
    for result in results:
        chunk = result.chunk
        preview = " ".join(chunk["text"].split()[:45])
        score_parts = [f"score={result.score:.4f}"]
        if result.dense_score is not None:
            score_parts.append(f"dense={result.dense_score:.4f}")
        if result.bm25_score is not None:
            score_parts.append(f"bm25={result.bm25_score:.4f}")
        print(
            f"\n#{result.rank} {' '.join(score_parts)} "
            f"{chunk['source_filename']} page={chunk['page_number']} "
            f"chunk={chunk['chunk_index']}"
        )
        print(f"chunk_id={chunk['chunk_id']}")
        print(preview)


def _normalize_backend(backend: str) -> str:
    return "dense" if backend == "qdrant" else backend
