# Setup Guide

This guide prepares the local development environment for the SEC 10-Q RAG
Analyst project.

## Recommended Workflow

Use VS Code as the editor and Codex as the coding mentor/agent.

Suggested pattern:

1. Read the class note.
2. Implement the matching feature.
3. Run a small verification command.
4. Update notes with what we learned.
5. Add interview questions and answers.

## Create A Virtual Environment

From the project root:

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Upgrade packaging tools:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

## Install Dependencies

Install the project and development tools:

```powershell
python -m pip install -r requirements-dev.txt
```

Later modules will add packages such as:

- `rank-bm25` for keyword retrieval.
- `fastapi` and `uvicorn` for serving.

## Open In VS Code

From the project root:

```powershell
code .
```

In VS Code:

- Open the Codex panel.
- Use Chat mode for concepts and planning.
- Use Agent mode for implementation.
- Keep `docs/classes` open while coding so Codex has teaching context.

## Next Command We Will Run

After creating the environment, acquire the dataset:

```powershell
python scripts\acquire_dataset.py
```

Then run:

```text
scripts/explore_dataset.py
```

The script will inspect the SEC 10-Q dataset once downloaded:

```powershell
python scripts/explore_dataset.py --data-dir data/raw/sec-10-q
```

Parse PDFs into page-aware records:

```powershell
python scripts\parse_pdfs.py --input-dir data\raw\sec-10-q --output-dir data\processed\pages
```

Chunk parsed pages:

```powershell
python scripts\chunk_pages.py --input-dir data\processed\pages --output-dir data\processed\chunks
```

Build the Qdrant vector database index:

```powershell
python scripts\build_index.py --chunks-dir data\processed\chunks --index-dir data\indexes\qdrant
```

Run retrieval:

```powershell
python scripts\retrieve.py --query "What was Apple's total net sales in Q2 2023?"
```

Run filtered retrieval:

```powershell
python scripts\retrieve.py --query "What was Microsoft's revenue?" --ticker MSFT
```

Evaluate source-document retrieval:

```powershell
python scripts\evaluate_retrieval.py --top-k 5
```

Expected current retrieval baseline after indexing all chunks:

```text
Questions: 195
Backend: qdrant
Hits@5: 185
Source Recall@5: 0.949
```

Local Qdrant mode uses file locking. Run index/retrieval/evaluation commands
sequentially, or use a Qdrant server for concurrent access.

Profile the Q&A benchmark:

```powershell
python scripts\profile_qna.py
```
