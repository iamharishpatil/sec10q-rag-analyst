# Open-Source SEC 10-Q RAG Analyst

Build a local retrieval-augmented generation assistant that answers questions
about SEC 10-Q filings with grounded citations.

## Project Goal

By the end of this project, we will have a working portfolio project that:

- Ingests SEC 10-Q filings.
- Extracts page-aware text and metadata.
- Chunks filings into retrievable passages.
- Builds dense and hybrid retrieval indexes.
- Uses an open-weight local LLM to answer questions from retrieved context.
- Returns citations and abstains when evidence is missing.
- Evaluates retrieval and answer faithfulness.
- Ships as a FastAPI service with a simple UI and Docker setup.

## Learning Contract

Every implementation module includes:

- Generic notes on the topic.
- Questions we solve while building.
- Potential interview questions.
- Working code or measurable project output.

## Planned Stack

- Language: Python 3.10+
- PDF parsing: PyMuPDF first, table extraction later
- Embeddings: sentence-transformers with open-source embedding models
- Retrieval: Qdrant vector database first, then BM25 for hybrid retrieval
- LLM runtime: Ollama or llama.cpp with open-weight models
- API: FastAPI
- Evaluation: custom Recall@k first, RAGAS/DeepEval later
- Packaging: Docker

## Repository Layout

```text
.
|-- docs/
|   |-- implementation_plan.md
|   |-- setup.md
|   `-- classes/
|       |-- 01_rag_and_sec_10q.md
|       `-- 02_dataset_exploration.md
|-- scripts/
|   |-- acquire_dataset.py
|   |-- build_index.py
|   |-- chunk_pages.py
|   |-- evaluate_retrieval.py
|   |-- explore_dataset.py
|   |-- parse_pdfs.py
|   |-- profile_qna.py
|   `-- retrieve.py
|-- src/
|   `-- sec_rag/
|       |-- chunking.py
|       |-- embeddings.py
|       |-- __init__.py
|       |-- config.py
|       |-- documents.py
|       |-- dataset.py
|       |-- pdf_parser.py
|       |-- qdrant_store.py
|       |-- qna.py
|       |-- retrieval_eval.py
|       `-- vector_index.py
|-- tests/
|-- AGENTS.md
|-- pyproject.toml
`-- README.md
```

## Current Status

The first runnable script is:

```powershell
python scripts/explore_dataset.py --data-dir data/raw/sec-10-q
```

Use it after downloading or cloning the dataset into `data/raw/sec-10-q`.

Acquire the dataset repository locally:

```powershell
python scripts\acquire_dataset.py
```

Parse PDFs into page-aware JSONL records:

```powershell
python scripts\parse_pdfs.py --input-dir data\raw\sec-10-q --output-dir data\processed\pages
```

Chunk parsed page records into retrievable JSONL chunks:

```powershell
python scripts\chunk_pages.py --input-dir data\processed\pages --output-dir data\processed\chunks
```

Build the Qdrant vector database index:

```powershell
python scripts\build_index.py --chunks-dir data\processed\chunks --index-dir data\indexes\qdrant
```

Retrieve chunks for a query:

```powershell
python scripts\retrieve.py --query "What was Apple's total net sales in Q2 2023?"
```

Retrieve with a metadata filter:

```powershell
python scripts\retrieve.py --query "What was Microsoft's revenue?" --ticker MSFT
```

Evaluate source-document Recall@k:

```powershell
python scripts\evaluate_retrieval.py --top-k 5
```

Current full-dataset dense retrieval baseline:

- Backend: local Qdrant collection `sec_10q_chunks`
- Index size: 1,935 chunk vectors
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Metric: Source-document Recall@5 = 0.949 over 195 Q&A rows

Local Qdrant storage should be accessed by one process at a time. For concurrent
retrieval workloads, run Qdrant as a server instead of local file-backed mode.

Profile the primary Q&A benchmark:

```powershell
python scripts\profile_qna.py
```

## Quickstart

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/explore_dataset.py --help
python scripts/chunk_pages.py --help
python scripts/build_index.py --help
python scripts/retrieve.py --help
python scripts/evaluate_retrieval.py --help
python scripts/parse_pdfs.py --help
python scripts/profile_qna.py --help
```
