# Class 02: Dataset Exploration

## Class Goal

Learn how to inspect a domain dataset before building a RAG system.

The habit we are practicing: do not jump straight to embeddings. First understand
the files, labels, questions, and source documents.

## Why Dataset Exploration Matters

RAG quality depends on the data pipeline. If we do not understand the dataset,
we will make weak choices about parsing, chunking, metadata, retrieval, and
evaluation.

For SEC 10-Q filings, exploration helps us answer:

- Which companies and quarters are included?
- Are filings stored as PDFs, text, HTML, or a mix?
- Where are the question-answer pairs?
- Do answers include evidence pages?
- Are questions factual, numerical, comparative, or qualitative?
- Do important answers live in prose, tables, or footnotes?

## Implementation Target

We add a script:

```powershell
python scripts/explore_dataset.py --data-dir data/raw/sec-10-q
```

The script reports:

- Number of PDF files.
- Number of likely Q&A files.
- Sample filing paths.
- CSV columns and sample rows.
- JSON/JSONL sample records when present.

This is intentionally simple. The goal is to create the first observable output
from the project before adding ML complexity.

## Generic Notes

### Dataset Profiling

Dataset profiling is the process of summarizing what exists in the data. In ML
engineering, this is often more important than choosing a model early.

Good profiling asks:

- What are the records?
- What fields are present?
- What fields are missing?
- What is the label or ground truth?
- How noisy is the data?
- What should be measured?

### Ground Truth

Ground truth is the reference answer used for evaluation. In our project, the
Q&A file acts as ground truth, but we still need to inspect whether it contains
page evidence, exact answers, or only free-form answer text.

### Metadata

Metadata is information about a chunk besides the chunk text itself. For this
project, metadata is not optional. Citations depend on it.

Useful metadata:

- Company name.
- Ticker.
- Filing type.
- Filing date or quarter.
- Document path.
- Page number.
- Section name.
- Chunk ID.

## Questions We Solve

1. Why inspect files before building embeddings?
2. What makes a Q&A pair useful for RAG evaluation?
3. What metadata should we preserve?
4. Why are financial filings harder than normal articles?

## Answers

### 1. Why inspect files before building embeddings?

Embeddings only encode the text we give them. If parsing loses pages, tables, or
sections, the vector index will preserve those mistakes. Exploration catches
that early.

### 2. What makes a Q&A pair useful for RAG evaluation?

A useful Q&A pair has a clear question, a verified answer, and ideally evidence
location. Evidence location lets us measure whether retrieval found the right
source, not just whether the final answer sounds plausible.

### 3. What metadata should we preserve?

At minimum: document ID, company, filing period, page number, section, and chunk
ID. Without this, the system cannot produce trustworthy citations.

### 4. Why are financial filings harder than normal articles?

Financial filings mix prose, tables, footnotes, legal language, repeated
headings, and numeric values. Dense retrieval can miss exact numbers or retrieve
similar boilerplate from the wrong section.

## Interview Questions

### Beginner

1. What is dataset profiling?
2. What is ground truth?
3. Why do we need metadata in a RAG system?

### Intermediate

1. How would you inspect a new document QA dataset?
2. What fields would you want in a financial RAG benchmark?
3. How can missing metadata hurt citations?

### Advanced

1. How would you design an evaluation set for financial question answering?
2. How would you detect whether answers come from tables versus prose?
3. How would you split evaluation questions into tuning and holdout sets?

## Homework

After running the exploration script, summarize:

```text
The dataset contains X filings and Y Q&A records. The most important metadata
fields for citation are ____. The main risk I see for retrieval is ____.
```
