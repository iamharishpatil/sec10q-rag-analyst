"""Parse SEC 10-Q PDFs into page-aware JSONL records."""

from __future__ import annotations

import argparse
from pathlib import Path

from sec_rag.config import get_project_paths
from sec_rag.dataset import discover_dataset_files, dataset_missing_message, relative_paths
from sec_rag.pdf_parser import parse_pdf_pages, write_page_records_jsonl


def build_parser() -> argparse.ArgumentParser:
    paths = get_project_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=paths.sec_10q_raw_dir,
        help="Directory containing SEC 10-Q PDFs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.parsed_pages_dir,
        help="Directory where parsed JSONL page files should be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of PDFs to parse.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.input_dir.exists():
        print(dataset_missing_message(args.input_dir))
        return 1

    inventory = discover_dataset_files(args.input_dir)
    pdfs = inventory.pdfs[: args.limit] if args.limit is not None else inventory.pdfs
    if not pdfs:
        print(f"No PDF files found under: {inventory.root}")
        return 1

    total_pages = 0
    print(f"Parsing {len(pdfs)} PDF(s) from {inventory.root}")
    for pdf_path in pdfs:
        records = parse_pdf_pages(pdf_path)
        output_path = args.output_dir / f"{pdf_path.stem}.pages.jsonl"
        written = write_page_records_jsonl(records, output_path)
        total_pages += written
        rel_pdf = relative_paths([pdf_path], inventory.root, limit=1)[0]
        print(f"  - {rel_pdf}: {written} page record(s)")

    print(f"Output directory: {args.output_dir}")
    print(f"Total page records: {total_pages}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
