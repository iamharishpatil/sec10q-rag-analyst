"""Acquire the KG-RAG SEC 10-Q dataset repository locally."""

from __future__ import annotations

import argparse
from pathlib import Path

from sec_rag.config import get_project_paths
from sec_rag.dataset import KG_RAG_REPO_URL, clone_kg_rag_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=get_project_paths().sec_10q_raw_dir,
        help="Local directory where the KG-RAG repository should be cloned.",
    )
    parser.add_argument(
        "--repo-url",
        default=KG_RAG_REPO_URL,
        help="Git repository URL to clone.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    target = clone_kg_rag_dataset(args.target_dir, repo_url=args.repo_url)
    print(f"Dataset repository available at: {target}")
    print("Run:")
    print(f"  python scripts\\explore_dataset.py --data-dir {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
