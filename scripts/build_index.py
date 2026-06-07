"""Build a local dense vector index from chunk JSONL files."""

from __future__ import annotations

import argparse
from pathlib import Path

from sec_rag.config import get_project_paths
from sec_rag.embeddings import DEFAULT_EMBEDDING_MODEL, embed_texts, load_embedding_model
from sec_rag.qdrant_store import DEFAULT_COLLECTION, create_local_client, recreate_collection, upsert_chunks
from sec_rag.vector_index import read_chunk_records_jsonl, write_index


def build_parser() -> argparse.ArgumentParser:
    paths = get_project_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks-dir",
        default=paths.chunks_dir,
        type=Path,
        help="Directory containing chunk JSONL files.",
    )
    parser.add_argument(
        "--index-dir",
        default=paths.indexes_dir / "qdrant",
        type=Path,
        help="Directory where the index should be written.",
    )
    parser.add_argument(
        "--backend",
        choices=("qdrant", "numpy"),
        default="qdrant",
        help="Index backend to build.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="Qdrant collection name.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Sentence-transformers model name.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    chunks = read_chunk_records_jsonl(args.chunks_dir)
    if not chunks:
        print(f"No chunk records found under: {args.chunks_dir}")
        print("Run chunking first:")
        print("  python scripts\\chunk_pages.py --input-dir data\\processed\\pages --output-dir data\\processed\\chunks")
        return 1

    print(f"Loading embedding model: {args.model_name}")
    model = load_embedding_model(args.model_name)
    texts = [chunk["text"] for chunk in chunks]
    print(f"Embedding {len(texts)} chunk(s)")
    embeddings = embed_texts(texts, model=model, batch_size=args.batch_size)

    if args.backend == "qdrant":
        client = create_local_client(args.index_dir)
        recreate_collection(client, collection_name=args.collection, vector_size=embeddings.shape[1])
        rows = upsert_chunks(client, args.collection, embeddings=embeddings, chunks=chunks)
        print(f"Qdrant collection: {args.collection}")
        print(f"Upserted rows: {rows}")
    else:
        write_index(args.index_dir, embeddings=embeddings, chunks=chunks, model_name=args.model_name)

    print(f"Index written to: {args.index_dir}")
    print(f"Rows: {len(chunks)}")
    print(f"Dimensions: {embeddings.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
