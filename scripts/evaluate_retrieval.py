"""Evaluate dense retrieval with source-document Recall@k."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sec_rag.config import get_project_paths
from sec_rag.embeddings import DEFAULT_EMBEDDING_MODEL, embed_query, load_embedding_model
from sec_rag.qdrant_store import DEFAULT_COLLECTION, create_local_client, search_chunks
from sec_rag.retrieval_eval import expected_source_labels, is_source_hit, summarize_retrieval_hits
from sec_rag.vector_index import load_index, search_index


def build_parser() -> argparse.ArgumentParser:
    paths = get_project_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qna-path",
        default=paths.sec_10q_raw_dir / "data" / "sec-10-q" / "qna_data.csv",
        type=Path,
        help="Path to Q&A CSV.",
    )
    parser.add_argument(
        "--index-dir",
        default=paths.indexes_dir / "qdrant",
        type=Path,
        help="Directory containing dense index files.",
    )
    parser.add_argument("--backend", choices=("qdrant", "numpy"), default="qdrant")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model-name", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    embeddings = None
    chunks = None
    model_name = args.model_name
    client = None
    if args.backend == "qdrant":
        client = create_local_client(args.index_dir)
    else:
        embeddings, chunks, manifest = load_index(args.index_dir)
        model_name = manifest["model_name"]

    model = load_embedding_model(model_name)

    total = 0
    hits = 0
    with open(args.qna_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if args.limit is not None and total >= args.limit:
                break
            expected = expected_source_labels(row.get("Source Docs", ""))
            query_embedding = embed_query(row["Question"], model=model)
            if args.backend == "qdrant":
                assert client is not None
                results = search_chunks(
                    client,
                    collection_name=args.collection,
                    query_embedding=query_embedding,
                    top_k=args.top_k,
                )
            else:
                assert embeddings is not None
                assert chunks is not None
                results = search_index(
                    query_embedding,
                    embeddings=embeddings,
                    chunks=chunks,
                    top_k=args.top_k,
                )
            total += 1
            hits += int(is_source_hit(expected, results))

    summary = summarize_retrieval_hits(total_questions=total, hits=hits)
    print(f"Questions: {summary.total_questions}")
    print(f"Backend: {args.backend}")
    print(f"Hits@{args.top_k}: {summary.hits}")
    print(f"Source Recall@{args.top_k}: {summary.recall_at_k:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
