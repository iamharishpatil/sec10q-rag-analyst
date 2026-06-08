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

Run the current pipeline end to end:

```powershell
python scripts\run_pipeline.py --retriever hybrid
```

For a faster smoke test:

```powershell
python scripts\run_pipeline.py --limit-pdfs 1 --eval-limit 5 --pages-dir data\processed\smoke-pages --chunks-dir data\processed\smoke-chunks --qdrant-index-dir data\indexes\smoke-qdrant --bm25-index-dir data\indexes\smoke-bm25
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
python scripts\build_index.py --backend dense
```

Build the BM25 lexical index:

```powershell
python scripts\build_index.py --backend bm25
```

Build both indexes for hybrid retrieval:

```powershell
python scripts\build_index.py --backend hybrid
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
python scripts\evaluate_retrieval.py --backend dense --top-k 5
python scripts\evaluate_retrieval.py --backend bm25 --top-k 5
python scripts\evaluate_retrieval.py --backend hybrid --top-k 5
```

Expected current retrieval baseline after indexing all chunks:

```text
Questions: 195
Dense Hits@5: 185, Source Recall@5: 0.949
BM25 Hits@5: 151, Source Recall@5: 0.774
Hybrid Hits@5: 184, Source Recall@5: 0.944
```

Local Qdrant mode uses file locking. Run index/retrieval/evaluation commands
sequentially, or use a Qdrant server for concurrent access.

## Hosted LLM Setup

This project uses Groq as the first hosted open-model provider. The API key must
come from the environment and must not be committed.

Set the key on Windows PowerShell:

```powershell
setx GROQ_API_KEY "your_new_key_here"
```

Close and reopen the terminal, then verify without printing the key:

```powershell
python -c "import os; print('GROQ_API_KEY set:', bool(os.getenv('GROQ_API_KEY')))"
```

Ask a cited question:

```powershell
python scripts\ask.py --retriever dense --temperature 0 --query "What was Apple's total net sales in Q2 2023?"
```

Expected smoke shape:

```text
Provider: groq
Model: openai/gpt-oss-20b
Abstained: False
Answer: $94,836 million
Citations: 2023 Q2 AAPL.pdf page 19
```

Controllable LLM parameters:

- `--llm-model`: hosted Groq model. Default is `openai/gpt-oss-20b` because it supports strict structured output.
- `--temperature`: randomness. Default is `0.0` for deterministic financial answers.
- `--max-tokens`: answer token budget. Default is `700`.
- `--top-p`: nucleus sampling. Default is `1.0`.
- `--no-strict-schema`: use best-effort JSON Schema mode instead of strict mode.

The production prompt is stored as code in `src/sec_rag/prompts/financial_qa.py`.
It uses:

- Delimited retrieved context.
- Edge-case few-shot examples for direct answers, insufficient evidence, and conflicting context.
- Abstention rules.
- Citation rules.
- Strict Pydantic schema validation.

The prompt does not request visible chain-of-thought. The model is instructed to
check support internally and return only the final schema-compliant answer.

Profile the Q&A benchmark:

```powershell
python scripts\profile_qna.py
```
