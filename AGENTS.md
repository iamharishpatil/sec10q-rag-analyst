# Codex Project Instructions

You are helping build an interview-focused ML engineering portfolio project.

## Teaching Style

For every major change:

- Explain the concept before or alongside the implementation.
- Prefer simple first principles before framework abstractions.
- Add notes that a beginner can revise before interviews.
- Include questions and interview prompts where useful.

## Engineering Style

- Use open-source and free tools only for the core implementation.
- Keep source files small and readable.
- Prefer measurable experiments over opinions.
- Do not hide retrieval internals; expose chunks, scores, metadata, and citations.
- Use page-aware metadata everywhere citations are involved.
- Chunking outputs must preserve source document and page metadata for citation tracing.
- Retrieval phases should be production-shaped: use a vector database, metadata payloads, filters, and explicit metrics; keep simple baselines only for explanation/debugging.
- Use OOP where it improves boundaries: orchestration, interchangeable retrieval strategies, and service-style dependencies. Keep pure transformations as functions.

## Verification

- Add tests for pure functions.
- Add small smoke tests for scripts.
- Track metrics before changing retrieval strategies.
- Compare retrieval backends with the same benchmark before choosing defaults.

## Phase Completion

At the end of every completed phase:

- Update project-facing docs when commands, dependencies, or architecture change.
- Update `docs/implementation_plan.md` with completed outcomes and remaining gaps.
- Update local-only class notes under `docs/classes/`; these files are intentionally ignored by Git.
- Run tests and lint before committing.
- Commit only project code, tests, setup files, and project-facing docs.
- Do not commit raw data, processed data, local lessons, virtual environments, caches, or package metadata.

## Project Principle

If a choice cannot be explained in an interview, simplify it or document it.
