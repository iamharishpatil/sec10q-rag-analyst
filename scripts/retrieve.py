"""Retrieve top-k SEC 10-Q chunks for a query."""

from __future__ import annotations

import argparse
from pathlib import Path

from sec_rag.config import get_project_paths
from sec_rag.embeddings import DEFAULT_EMBEDDING_MODEL, embed_query, load_embedding_model
from sec_rag.qdrant_store import DEFAULT_COLLECTION, create_local_client, search_chunks
from sec_rag.vector_index import load_index, search_index


def build_parser() -> argparse.ArgumentParser:
    paths = get_project_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Question or search query.")
    parser.add_argument(
        "--index-dir",
        default=paths.indexes_dir / "qdrant",
        type=Path,
        help="Directory containing dense index files.",
    )
    parser.add_argument("--backend", choices=("qdrant", "numpy"), default="qdrant")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model-name", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--ticker", default=None, help="Optional ticker metadata filter.")
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model_name = args.model_name
    if args.backend == "numpy":
        embeddings, chunks, manifest = load_index(args.index_dir)
        model_name = manifest["model_name"]

    model = load_embedding_model(model_name)
    query_embedding = embed_query(args.query, model=model)

    if args.backend == "qdrant":
        client = create_local_client(args.index_dir)
        results = search_chunks(
            client,
            collection_name=args.collection,
            query_embedding=query_embedding,
            top_k=args.top_k,
            ticker=args.ticker,
        )
    else:
        results = search_index(query_embedding, embeddings=embeddings, chunks=chunks, top_k=args.top_k)

    print(f"Query: {args.query}")
    print(f"Model: {model_name}")
    print(f"Backend: {args.backend}")
    for result in results:
        chunk = result.chunk
        preview = " ".join(chunk["text"].split()[:45])
        print(
            f"\n#{result.rank} score={result.score:.4f} "
            f"{chunk['source_filename']} page={chunk['page_number']} chunk={chunk['chunk_index']}"
        )
        print(f"chunk_id={chunk['chunk_id']}")
        print(preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
