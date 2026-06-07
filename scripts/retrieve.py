"""Retrieve top-k SEC 10-Q chunks for a query."""

from __future__ import annotations

import argparse
from pathlib import Path

from sec_rag.pipeline import PipelineConfig, print_retrieval_results
from sec_rag.retrieval import RetrievalFilters
from sec_rag.retrievers import create_retriever


def build_parser() -> argparse.ArgumentParser:
    defaults = PipelineConfig.default()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Question or search query.")
    parser.add_argument(
        "--index-dir",
        default=None,
        type=Path,
        help="Backward-compatible dense/NumPy index directory.",
    )
    parser.add_argument("--qdrant-index-dir", default=defaults.qdrant_index_dir, type=Path)
    parser.add_argument("--bm25-index-dir", default=defaults.bm25_index_dir, type=Path)
    parser.add_argument(
        "--backend",
        choices=("dense", "qdrant", "bm25", "hybrid", "numpy"),
        default="dense",
    )
    parser.add_argument("--collection", default=defaults.collection)
    parser.add_argument("--model-name", default=defaults.model_name)
    parser.add_argument("--ticker", default=None, help="Optional ticker metadata filter.")
    parser.add_argument("--top-k", type=int, default=defaults.top_k)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    qdrant_index_dir = args.index_dir if args.index_dir is not None else args.qdrant_index_dir
    retriever = create_retriever(
        args.backend,
        qdrant_index_dir=qdrant_index_dir,
        bm25_index_dir=args.bm25_index_dir,
        collection=args.collection,
        model_name=args.model_name,
    )
    results = retriever.search(
        args.query,
        top_k=args.top_k,
        filters=RetrievalFilters(ticker=args.ticker),
    )
    backend = "dense" if args.backend == "qdrant" else args.backend
    print_retrieval_results(query=args.query, backend=backend, results=results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
