"""Evaluate retrieval with source-document Recall@k."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from sec_rag.pipeline import PipelineConfig, PipelineRunner


def build_parser() -> argparse.ArgumentParser:
    defaults = PipelineConfig.default()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qna-path", default=defaults.qna_path, type=Path)
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
    parser.add_argument("--top-k", type=int, default=defaults.top_k)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    qdrant_index_dir = args.index_dir if args.index_dir is not None else args.qdrant_index_dir
    config = replace(
        PipelineConfig.default(),
        qna_path=args.qna_path,
        qdrant_index_dir=qdrant_index_dir,
        bm25_index_dir=args.bm25_index_dir,
        retriever_backend=args.backend,
        collection=args.collection,
        model_name=args.model_name,
        top_k=args.top_k,
        eval_limit=args.limit,
    )
    PipelineRunner(config).evaluate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
