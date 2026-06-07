"""Run the current SEC 10-Q RAG pipeline stages in order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from sec_rag.config import get_project_paths


def build_parser() -> argparse.ArgumentParser:
    paths = get_project_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        default=paths.sec_10q_raw_dir,
        type=Path,
        help="Directory containing the raw SEC 10-Q dataset.",
    )
    parser.add_argument(
        "--pages-dir",
        default=paths.parsed_pages_dir,
        type=Path,
        help="Directory for parsed page JSONL records.",
    )
    parser.add_argument(
        "--chunks-dir",
        default=paths.chunks_dir,
        type=Path,
        help="Directory for chunk JSONL records.",
    )
    parser.add_argument(
        "--index-dir",
        default=paths.indexes_dir / "qdrant",
        type=Path,
        help="Directory for the local Qdrant index.",
    )
    parser.add_argument(
        "--limit-pdfs",
        type=int,
        default=None,
        help="Optional PDF limit for smoke testing.",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Optional Q&A row limit for smoke testing.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--query",
        default="What was Apple's total net sales in Q2 2023?",
        help="Retrieval smoke-test query to run after indexing.",
    )
    parser.add_argument(
        "--ticker",
        default=None,
        help="Optional ticker filter for the retrieval smoke test.",
    )
    parser.add_argument("--skip-parse", action="store_true")
    parser.add_argument("--skip-chunk", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--skip-retrieve", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser


def run_step(label: str, command: list[str]) -> None:
    """Run one pipeline command and fail fast on errors."""
    print(f"\n== {label} ==", flush=True)
    print(format_command(command), flush=True)
    subprocess.run(command, check=True)


def format_command(command: list[str]) -> str:
    """Format a command for readable terminal output."""
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def main() -> int:
    args = build_parser().parse_args()
    if not args.raw_dir.exists():
        print(f"Raw dataset directory does not exist: {args.raw_dir}")
        print("Run dataset acquisition first:")
        print("  python scripts\\acquire_dataset.py")
        return 1

    python = sys.executable

    if not args.skip_parse:
        command = [
            python,
            "scripts\\parse_pdfs.py",
            "--input-dir",
            str(args.raw_dir),
            "--output-dir",
            str(args.pages_dir),
        ]
        if args.limit_pdfs is not None:
            command.extend(["--limit", str(args.limit_pdfs)])
        run_step("Parse PDFs into page records", command)

    if not args.skip_chunk:
        command = [
            python,
            "scripts\\chunk_pages.py",
            "--input-dir",
            str(args.pages_dir),
            "--output-dir",
            str(args.chunks_dir),
        ]
        if args.limit_pdfs is not None:
            command.extend(["--limit", str(args.limit_pdfs)])
        run_step("Chunk page records", command)

    if not args.skip_index:
        run_step(
            "Build Qdrant vector index",
            [
                python,
                "scripts\\build_index.py",
                "--chunks-dir",
                str(args.chunks_dir),
                "--index-dir",
                str(args.index_dir),
            ],
        )

    if not args.skip_retrieve:
        command = [
            python,
            "scripts\\retrieve.py",
            "--query",
            args.query,
            "--top-k",
            str(args.top_k),
            "--index-dir",
            str(args.index_dir),
        ]
        if args.ticker:
            command.extend(["--ticker", args.ticker])
        run_step("Retrieve top-k chunks", command)

    if not args.skip_eval:
        command = [
            python,
            "scripts\\evaluate_retrieval.py",
            "--top-k",
            str(args.top_k),
            "--index-dir",
            str(args.index_dir),
        ]
        if args.eval_limit is not None:
            command.extend(["--limit", str(args.eval_limit)])
        run_step("Evaluate source-document Recall@k", command)

    print("\nPipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
