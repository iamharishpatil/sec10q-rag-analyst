"""Chunk parsed SEC 10-Q page JSONL files into retrievable JSONL chunks."""

from __future__ import annotations

import argparse
from pathlib import Path

from sec_rag.chunking import (
    chunk_page_records,
    read_page_records_jsonl,
    summarize_chunks,
    write_chunk_records_jsonl,
)
from sec_rag.config import get_project_paths


def build_parser() -> argparse.ArgumentParser:
    paths = get_project_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=paths.parsed_pages_dir,
        help="Directory containing parsed page JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.processed_dir / "chunks",
        help="Directory where chunk JSONL files should be written.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=350,
        help="Maximum words per chunk.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlapping words between adjacent chunks from the same page.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of page JSONL files to chunk.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.input_dir.exists():
        print(f"Parsed pages directory does not exist: {args.input_dir}")
        print("Run PDF parsing first:")
        print("  python scripts\\parse_pdfs.py --input-dir data\\raw\\sec-10-q --output-dir data\\processed\\pages")
        return 1

    input_files = tuple(sorted(args.input_dir.glob("*.pages.jsonl")))
    if args.limit is not None:
        input_files = input_files[: args.limit]
    if not input_files:
        print(f"No page JSONL files found under: {args.input_dir}")
        return 1

    total_pages = 0
    total_chunks = 0
    total_empty_pages = 0
    print(f"Chunking {len(input_files)} parsed page file(s) from {args.input_dir}")
    for input_file in input_files:
        pages = read_page_records_jsonl(input_file)
        chunks = chunk_page_records(pages, chunk_size=args.chunk_size, overlap=args.overlap)
        output_file = args.output_dir / input_file.name.replace(".pages.jsonl", ".chunks.jsonl")
        write_chunk_records_jsonl(chunks, output_file)
        summary = summarize_chunks(page_count=len(pages), chunks=chunks)
        total_pages += summary.page_records
        total_chunks += summary.chunks
        total_empty_pages += summary.skipped_empty_pages
        print(
            "  - "
            f"{input_file.name}: {summary.page_records} page(s), "
            f"{summary.chunks} chunk(s), "
            f"{summary.skipped_empty_pages} empty page(s), "
            f"{summary.average_words_per_chunk:.0f} avg words/chunk"
        )

    print(f"Output directory: {args.output_dir}")
    print(f"Total page records: {total_pages}")
    print(f"Total chunks: {total_chunks}")
    print(f"Total skipped empty pages: {total_empty_pages}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
