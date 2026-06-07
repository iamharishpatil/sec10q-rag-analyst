"""Profile the SEC 10-Q Q&A benchmark CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from sec_rag.config import get_project_paths
from sec_rag.qna import profile_qna_csv


def build_parser() -> argparse.ArgumentParser:
    default_path = get_project_paths().sec_10q_raw_dir / "data" / "sec-10-q" / "qna_data.csv"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qna-path",
        type=Path,
        default=default_path,
        help="Path to the Q&A CSV to profile.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of values to print for each distribution.",
    )
    return parser


def print_counts(title: str, counts: tuple[tuple[str, int], ...], limit: int) -> None:
    print(f"\n{title}:")
    if not counts:
        print("  (none)")
        return
    for label, count in counts[:limit]:
        print(f"  - {label}: {count}")


def main() -> int:
    args = build_parser().parse_args()
    if not args.qna_path.exists():
        print(f"Q&A file does not exist: {args.qna_path}")
        print("Run dataset acquisition first:")
        print("  python scripts\\acquire_dataset.py")
        return 1

    profile = profile_qna_csv(args.qna_path)
    print(f"Q&A path: {profile.path}")
    print(f"Rows: {profile.row_count}")
    print(f"Columns: {', '.join(profile.columns)}")
    print_counts("Question types", profile.question_type_counts, args.limit)
    print_counts("Source chunk types", profile.source_chunk_type_counts, args.limit)
    print_counts("Source docs", profile.source_doc_counts, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
