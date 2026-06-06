# Class 01: RAG And SEC 10-Q Filings

## Class Goal

Understand what we are building, why RAG is useful, and how this project maps
to real interview skills.

## What We Are Building

We are building a financial-document question answering system.

Input:

- A user's investor-style question.
- A collection of SEC 10-Q filings.

Output:

- A grounded answer.
- Citations pointing to source pages or chunks.
- An abstention when the evidence is missing.

Example question:

```text
What were the main risk factors discussed in Nvidia's latest 10-Q?
```

Desired behavior:

```text
The answer should summarize only retrieved evidence and cite the filing pages.
If the retrieved context does not include the answer, the assistant should say
it could not find the information.
```

## What Is A 10-Q?

A 10-Q is a quarterly report filed by public companies in the United States.
It usually includes:

- Financial statements.
- Management discussion and analysis.
- Risk factors.
- Legal proceedings.
- Market and business updates.
- Footnotes and tables.

For our project, this matters because the answer may live in normal prose,
tables, footnotes, or repeated boilerplate. A good RAG system must retrieve the
right evidence before the LLM can answer well.

## What Is RAG?

RAG means retrieval-augmented generation.

Instead of asking an LLM to answer from memory, we:

1. Retrieve relevant source text from a trusted document collection.
2. Give only that source text to the LLM.
3. Ask the LLM to answer using the retrieved evidence.
4. Return citations so the answer can be checked.

This makes the system more useful for factual, domain-specific tasks.

## Core Pipeline

```text
PDF filings
  -> parse text and metadata
  -> chunk documents
  -> embed chunks
  -> build vector index
  -> retrieve relevant chunks
  -> optionally rerank
  -> generate cited answer
  -> evaluate retrieval and answer quality
```

## Why This Is Interview-Strong

This project lets you explain:

- Data ingestion.
- PDF parsing.
- Chunking strategies.
- Embeddings.
- Vector search.
- Hybrid retrieval.
- Reranking.
- Prompt design.
- Structured outputs.
- Hallucination control.
- Evaluation metrics.
- API deployment.

That is the full path from ML concept to working AI product.

## Questions We Solve In This Class

1. Why not just ask the LLM directly?
2. Why do citations matter?
3. Why is retrieval quality more important than prompt cleverness?
4. Why are financial filings harder than normal web pages?
5. What would make this project impressive on a resume?

## Answers

### 1. Why not just ask the LLM directly?

The LLM may not know the exact filing, may have stale knowledge, or may invent
facts. RAG anchors the answer in retrieved documents.

### 2. Why do citations matter?

Financial answers must be auditable. Citations let a user verify the answer
against the source filing.

### 3. Why is retrieval quality more important than prompt cleverness?

If the right evidence is missing, the LLM cannot reliably answer. Good prompts
help, but retrieval determines whether the system has the facts.

### 4. Why are financial filings harder than normal web pages?

They contain tables, footnotes, repeated sections, dense language, numeric
values, and legal wording. The same phrase can appear in many irrelevant places.

### 5. What makes this resume-worthy?

The project has a real domain, measurable metrics, open-source models, citations,
guardrails, and deployment. It is not just a chatbot.

## Interview Questions

### Beginner

1. What is RAG?
2. Why do we use embeddings?
3. What is a vector database?
4. Why do we chunk documents?

### Intermediate

1. How do you choose chunk size?
2. What is Recall@k?
3. Why can dense retrieval fail on financial questions?
4. What is hybrid retrieval?

### Advanced

1. How would you evaluate hallucination in a RAG system?
2. How would you handle table-heavy financial filings?
3. How would you debug a wrong answer?
4. How would you design this system for production?

## First Implementation Target

In the next class, we will create the environment and add a dataset exploration
script that:

- Locates downloaded dataset files.
- Lists available filings.
- Reads the Q&A CSV.
- Prints question categories and sample records.

## Homework

Before the next implementation step, be ready to explain this in your own words:

```text
RAG improves factual answering by retrieving trusted source context before
generation. In this project, citations and evaluation matter because financial
answers must be auditable.
```
