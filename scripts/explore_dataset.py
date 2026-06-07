"""Explore a SEC 10-Q dataset before building the RAG pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from sec_rag.dataset import (
    discover_dataset_files,
    dataset_missing_message,
    profile_extensions,
    profile_csv,
    relative_paths,
    sample_json_records,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/sec-10-q"),
        help="Directory containing SEC 10-Q PDFs and Q&A files.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3,
        help="Number of sample records to print from each Q&A file.",
    )
    parser.add_argument(
        "--list-limit",
        type=int,
        default=10,
        help="Maximum number of sample paths to print per file group.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.data_dir.exists():
        print(dataset_missing_message(args.data_dir))
        return 1

    inventory = discover_dataset_files(args.data_dir)

    print(f"Dataset root: {inventory.root}")
    print(f"PDF files: {len(inventory.pdfs)}")
    print(f"Q&A-like files: {len(inventory.qa_files)}")
    print(f"Other files: {len(inventory.other_files)}")

    extension_profile = profile_extensions(inventory)
    if extension_profile:
        print("\nFile types:")
        for item in extension_profile[: args.list_limit]:
            print(f"  - {item.extension}: {item.count}")

    if inventory.pdfs:
        print("\nSample PDFs:")
        for path in relative_paths(inventory.pdfs, inventory.root, limit=args.list_limit):
            print(f"  - {path}")
    else:
        print("\nNo PDF files found.")

    if inventory.qa_files:
        print("\nQ&A file profiles:")
        for path in inventory.qa_files[: args.list_limit]:
            print(f"\n{path.relative_to(inventory.root)}")
            if path.suffix.lower() == ".csv":
                profile = profile_csv(path, sample_size=args.sample_size)
                print(f"  Rows: {profile.row_count}")
                print(f"  Columns: {', '.join(profile.columns) or '(none)'}")
                for index, row in enumerate(profile.sample_rows, start=1):
                    print(f"  Sample {index}: {row}")
            else:
                records = sample_json_records(path, sample_size=args.sample_size)
                print(f"  Sample records: {len(records)}")
                for index, record in enumerate(records, start=1):
                    print(f"  Sample {index}: {record}")
    else:
        print("\nNo CSV, JSON, or JSONL Q&A-like files found.")

    if inventory.other_files:
        print("\nSample other files:")
        for path in relative_paths(inventory.other_files, inventory.root, limit=args.list_limit):
            print(f"  - {path}")

    print("\nNext learning question:")
    print("  Which metadata fields must survive parsing so answers can cite evidence?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
