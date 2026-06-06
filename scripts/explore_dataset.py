"""Explore a SEC 10-Q dataset before building the RAG pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sec_rag.dataset import (
    discover_dataset_files,
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    inventory = discover_dataset_files(args.data_dir)

    print(f"Dataset root: {inventory.root}")
    print(f"PDF files: {len(inventory.pdfs)}")
    print(f"Q&A-like files: {len(inventory.qa_files)}")
    print(f"Other files: {len(inventory.other_files)}")

    if inventory.pdfs:
        print("\nSample PDFs:")
        for path in relative_paths(inventory.pdfs, inventory.root, limit=10):
            print(f"  - {path}")

    if inventory.qa_files:
        print("\nQ&A file profiles:")
        for path in inventory.qa_files:
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

    print("\nNext learning question:")
    print("  Which metadata fields must survive parsing so answers can cite evidence?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
