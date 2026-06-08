"""Ask a SEC 10-Q question and generate a cited answer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sec_rag.answering import AnswerGenerator, GroundedAnswer, GroundedAnswerSchema
from sec_rag.llm import DEFAULT_GROQ_MODEL, GroqProvider
from sec_rag.pipeline import PipelineConfig
from sec_rag.retrieval import RetrievalFilters
from sec_rag.retrievers import create_retriever


def build_parser() -> argparse.ArgumentParser:
    defaults = PipelineConfig.default()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Question to answer.")
    parser.add_argument(
        "--provider",
        choices=("groq",),
        default="groq",
        help="Hosted LLM provider.",
    )
    parser.add_argument("--llm-model", default=DEFAULT_GROQ_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--no-strict-schema",
        action="store_true",
        help="Use best-effort JSON Schema mode instead of strict schema mode.",
    )
    parser.add_argument(
        "--response-format",
        choices=("json-schema", "json-object", "none"),
        default="json-schema",
        help="Groq response format. Use json-object for models that do not support json-schema.",
    )
    parser.add_argument(
        "--retriever",
        choices=("dense", "qdrant", "bm25", "hybrid", "numpy"),
        default="dense",
    )
    parser.add_argument("--qdrant-index-dir", default=defaults.qdrant_index_dir, type=Path)
    parser.add_argument("--bm25-index-dir", default=defaults.bm25_index_dir, type=Path)
    parser.add_argument("--embedding-model", default=defaults.model_name)
    parser.add_argument("--collection", default=defaults.collection)
    parser.add_argument("--ticker", default=None, help="Optional ticker metadata filter.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=12_000)
    return parser


def main() -> int:
    configure_stdout()
    args = build_parser().parse_args()
    retriever = create_retriever(
        args.retriever,
        qdrant_index_dir=args.qdrant_index_dir,
        bm25_index_dir=args.bm25_index_dir,
        collection=args.collection,
        model_name=args.embedding_model,
    )
    provider = GroqProvider(
        model=args.llm_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        response_format=build_response_format(args.response_format, strict=not args.no_strict_schema),
    )
    generator = AnswerGenerator(
        retriever=retriever,
        llm_provider=provider,
        top_k=args.top_k,
        max_context_chars=args.max_context_chars,
    )
    answer = generator.answer(args.query, filters=RetrievalFilters(ticker=args.ticker))
    print_answer(answer)
    return 0


def configure_stdout() -> None:
    """Prefer UTF-8 terminal output for model responses on Windows."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def build_response_format(format_name: str, strict: bool = True) -> dict | None:
    """Build the Groq response_format payload for the selected mode."""
    if format_name == "json-schema":
        return GroundedAnswerSchema.groq_response_format(strict=strict)
    if format_name == "json-object":
        return {"type": "json_object"}
    if format_name == "none":
        return None
    raise ValueError(f"Unsupported response format: {format_name}")


def print_answer(answer: GroundedAnswer) -> None:
    """Print a cited answer in a terminal-friendly format."""
    print(f"Question: {answer.question}")
    print(f"Provider: {answer.provider}")
    print(f"Model: {answer.model}")
    print(f"Abstained: {answer.abstained}")
    print("\nAnswer:")
    print(answer.answer)
    print("\nCitations:")
    if not answer.citations:
        print("  none")
    for citation in answer.citations:
        print(
            f"  - {citation.source_id}: {citation.source_filename} "
            f"page={citation.page_number} chunk_id={citation.chunk_id}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
