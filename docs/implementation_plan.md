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

- Add BM25. Done.
- Combine lexical and dense retrieval. Done.
- Refactor pipeline and retrieval into OOP application classes. Done.
- Add reranking. Pending.

Current hybrid retrieval profile:

- `PipelineRunner` runs the current pipeline through Python classes instead of subprocess orchestration.
- `Retriever` interface supports dense, BM25, hybrid, and NumPy baseline retrieval.
- BM25 index stores 1,935 chunk rows under ignored `data/indexes/bm25`.
- Dense Qdrant Recall@5: 0.949 over 195 Q&A rows.
- BM25 Recall@5: 0.774 over 195 Q&A rows.
- Hybrid 0.65/0.35 Recall@5: 0.944 over 195 Q&A rows.
- Dense remains the best source-document Recall@5 baseline; hybrid is retained for exact-token/numeric retrieval experiments.

Learning:

- Why exact terms and numbers are hard for dense-only search.
- How reranking improves precision.

Interview angle:

- "Why hybrid retrieval?"
- "Where does reranking fit in a RAG pipeline?"

### 6. Local LLM Answering

Outcome:

- Run a hosted open-model LLM through Groq because local inference is not available. Done.
- Generate cited answers from retrieved chunks. Done.
- Enforce structured JSON-shaped output. Done.
- Add local Ollama support later if hardware becomes available. Pending.

Current answer-generation profile:

- `LLMProvider` interface supports provider isolation.
- `GroqProvider` reads `GROQ_API_KEY` from the environment.
- `GroqProvider` uses `temperature=0.0` by default for deterministic financial QA.
- `AnswerGenerator` retrieves context and asks the LLM for cited JSON validated by Pydantic.
- `scripts/ask.py` provides the user-facing command.
- Strict Groq JSON Schema mode is enabled from `GroundedAnswerSchema.model_json_schema()`.
- Smoke query answered Apple Q2 2023 net sales as `$94,836 million` with citations to `2023 Q2 AAPL.pdf` pages 19 and 10.

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
