"""Run the current SEC 10-Q RAG pipeline stages in order."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from sec_rag.pipeline import PipelineConfig, PipelineRunner


def build_parser() -> argparse.ArgumentParser:
    defaults = PipelineConfig.default()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=defaults.raw_dir, type=Path)
    parser.add_argument("--pages-dir", default=defaults.pages_dir, type=Path)
    parser.add_argument("--chunks-dir", default=defaults.chunks_dir, type=Path)
    parser.add_argument("--qdrant-index-dir", default=defaults.qdrant_index_dir, type=Path)
    parser.add_argument("--bm25-index-dir", default=defaults.bm25_index_dir, type=Path)
    parser.add_argument("--qna-path", default=defaults.qna_path, type=Path)
    parser.add_argument(
        "--retriever",
        choices=("dense", "qdrant", "bm25", "hybrid", "numpy"),
        default=defaults.retriever_backend,
        help="Retriever backend used for indexing, retrieval, and evaluation.",
    )
    parser.add_argument("--model-name", default=defaults.model_name)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--limit-pdfs", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=defaults.top_k)
    parser.add_argument("--query", default=defaults.query)
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--skip-parse", action="store_true")
    parser.add_argument("--skip-chunk", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--skip-retrieve", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = replace(
        PipelineConfig.default(),
        raw_dir=args.raw_dir,
        pages_dir=args.pages_dir,
        chunks_dir=args.chunks_dir,
        qdrant_index_dir=args.qdrant_index_dir,
        bm25_index_dir=args.bm25_index_dir,
        qna_path=args.qna_path,
        retriever_backend=args.retriever,
        model_name=args.model_name,
        batch_size=args.batch_size,
        limit_pdfs=args.limit_pdfs,
        eval_limit=args.eval_limit,
        top_k=args.top_k,
        query=args.query,
        ticker=args.ticker,
        skip_parse=args.skip_parse,
        skip_chunk=args.skip_chunk,
        skip_index=args.skip_index,
        skip_retrieve=args.skip_retrieve,
        skip_eval=args.skip_eval,
    )
    PipelineRunner(config).run()
    print("\nPipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
