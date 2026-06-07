# Implementation Plan

## North Star

Create a local SEC 10-Q RAG analyst that can answer investor-style questions
with cited evidence from quarterly filings.

## Milestones

### 0. Setup

Outcome:

- Repo scaffold. Done.
- Python environment. Next user action.
- Dependency plan. Done.
- Project notes. Done.

Learning:

- What a production-style ML project structure looks like.
- Why reproducibility matters.

Interview angle:

- "How would you structure an ML application repository?"
- "Why pin dependencies?"

### 1. Dataset Exploration

Outcome:

- Download or clone the KG-RAG SEC 10-Q dataset. Done locally via ignored `data/raw`.
- Inspect PDFs and Q&A files. Done.
- Build a small data profile. Done.
- First exploration script. Done.

Current dataset profile:

- 20 SEC 10-Q PDF files.
- 7 likely Q&A CSV files.
- Primary Q&A file: `qna_data.csv` with 195 rows.
- Companies include Apple, Amazon, Intel, Microsoft, and NVIDIA.

Learning:

- What SEC 10-Q filings contain.
- How domain data influences retrieval design.

Interview angle:

- "Why is domain understanding important in RAG?"
- "What metadata would you keep for financial documents?"

### 2. PDF Parsing

Outcome:

- Extract page-aware text from filings. Done.
- Preserve document ID, source path, page number, and total pages. Done.
- Preserve company, ticker, year, quarter, filing type, and source filename. Done.
- Add extraction quality summaries: page count, empty pages, and average characters per page. Done.
- Preserve section hints. Pending.

Current parser profile:

- Full dataset parse command is available through `scripts/parse_pdfs.py`.
- Page records are written as JSONL under ignored `data/processed/pages`.
- Q&A profiling is available through `scripts/profile_qna.py`.

Learning:

- Why PDF extraction is messy.
- Why citations depend on metadata quality.

Interview angle:

- "What can go wrong when parsing PDFs for RAG?"
- "How do you preserve citation provenance?"

### 3. Chunking

Outcome:

- Implement fixed-size, overlapping chunks. Done.
- Preserve source document and page metadata on every chunk. Done.
- Add section-aware chunking when possible. Pending.

Current chunking profile:

- Page-local word chunking is available through `scripts/chunk_pages.py`.
- Default chunk size is 350 words with 50-word overlap.
- Chunk records are written as JSONL under ignored `data/processed/chunks`.

Learning:

- Chunk size tradeoffs.
- Recall versus precision.

Interview angle:

- "How do you choose chunk size?"
- "Why use overlap?"

### 4. Embeddings And Indexing

Outcome:

- Generate embeddings. Done with an open-source sentence-transformers model.
- Build the first vector database. Done with local Qdrant.
- Keep a NumPy cosine path. Done as an inspectable baseline.
- Evaluate Recall@k. Done as source-document Recall@k baseline.

Current retrieval profile:

- Qdrant index build command is available through `scripts/build_index.py`.
- Top-k retrieval command is available through `scripts/retrieve.py`.
- Source-document retrieval evaluation is available through `scripts/evaluate_retrieval.py`.
- Current Qdrant collection: `sec_10q_chunks`.
- Current index size: 1,935 chunk vectors with 384 embedding dimensions.
- Full Qdrant evaluation: Source-document Recall@5 = 0.949 over 195 Q&A rows.
- Local Qdrant mode should be used sequentially; Qdrant server is the next production step for concurrent access.

Learning:

- Embedding spaces.
- Similarity search.
- Retrieval metrics.

Interview angle:

- "What is Recall@k?"
- "Why normalize embeddings?"

### 5. Hybrid Retrieval And Reranking

Outcome:

- Add BM25.
- Combine lexical and dense retrieval.
- Add reranking.

Learning:

- Why exact terms and numbers are hard for dense-only search.
- How reranking improves precision.

Interview angle:

- "Why hybrid retrieval?"
- "Where does reranking fit in a RAG pipeline?"

### 6. Local LLM Answering

Outcome:

- Run an open-weight local LLM.
- Generate cited answers from retrieved chunks.
- Enforce structured output.

Learning:

- Prompt boundaries.
- Abstention.
- JSON/schema validation.

Interview angle:

- "How do you reduce hallucinations?"
- "Why use structured output?"

### 7. Evaluation And Guardrails

Outcome:

- Evaluate retrieval and generation separately.
- Detect missing citations and unsupported claims.

Learning:

- Faithfulness.
- Answer correctness.
- Failure analysis.

Interview angle:

- "How do you evaluate RAG?"
- "What is the difference between retrieval failure and generation failure?"

### 8. API, UI, And Deployment

Outcome:

- FastAPI service.
- Simple query UI.
- Dockerized local deployment.

Learning:

- Serving ML systems.
- Latency and monitoring basics.

Interview angle:

- "How would you deploy this system?"
- "What would you monitor in production?"

### 9. Showcase

Outcome:

- README, architecture diagram, metrics, demo examples, resume bullets.

Learning:

- Communicating engineering impact.

Interview angle:

- "Walk me through your project."
- "What tradeoffs did you make?"
