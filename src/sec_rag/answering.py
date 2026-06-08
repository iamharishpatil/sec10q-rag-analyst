"""Grounded answer generation from retrieved SEC filing chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from sec_rag.llm import LLMProvider
from sec_rag.retrieval import RetrievalFilters, RetrievalResult, Retriever


SYSTEM_PROMPT = """You are a financial RAG assistant.
Answer only from the provided SEC 10-Q context.
If the context does not contain enough evidence, abstain.
Return only valid JSON matching the provided response schema.
"""


class AnswerCitationSchema(BaseModel):
    """Pydantic schema for one answer citation."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(description="Context source id, for example S1.")
    source_filename: str = Field(description="Source SEC filing filename.")
    page_number: int = Field(description="One-indexed source page number.")
    chunk_id: str = Field(description="Retrieved chunk identifier.")


class GroundedAnswerSchema(BaseModel):
    """Pydantic schema for grounded answer output."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="Short answer grounded only in the provided context.")
    citations: list[AnswerCitationSchema] = Field(
        description="Citations that support the answer. Use an empty list when abstaining."
    )
    abstained: bool = Field(description="True when the context is insufficient to answer.")

    @classmethod
    def groq_response_format(cls, strict: bool = True) -> dict:
        """Return Groq JSON Schema response_format from the Pydantic schema."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "grounded_answer",
                "strict": strict,
                "schema": cls.model_json_schema(),
            },
        }


@dataclass(frozen=True)
class AnswerCitation:
    """One citation selected by the answer generator."""

    source_id: str
    source_filename: str
    page_number: int
    chunk_id: str


@dataclass(frozen=True)
class GroundedAnswer:
    """Structured answer generated from retrieved context."""

    question: str
    answer: str
    citations: tuple[AnswerCitation, ...]
    abstained: bool
    model: str
    provider: str
    raw_response: str
    retrieved: tuple[RetrievalResult, ...]


class AnswerGenerator:
    """Generate cited answers from retriever results."""

    def __init__(
        self,
        retriever: Retriever,
        llm_provider: LLMProvider,
        top_k: int = 5,
        max_context_chars: int = 12_000,
    ) -> None:
        self.retriever = retriever
        self.llm_provider = llm_provider
        self.top_k = top_k
        self.max_context_chars = max_context_chars

    def answer(
        self,
        question: str,
        filters: RetrievalFilters | None = None,
    ) -> GroundedAnswer:
        """Retrieve context and generate a grounded answer."""
        retrieved = self.retriever.search(question, top_k=self.top_k, filters=filters)
        messages = build_answer_messages(
            question=question,
            results=retrieved,
            max_context_chars=self.max_context_chars,
        )
        response = self.llm_provider.generate(messages)
        parsed = parse_answer_json(response.content)
        return GroundedAnswer(
            question=question,
            answer=parsed["answer"],
            citations=parsed["citations"],
            abstained=parsed["abstained"],
            model=response.model,
            provider=response.provider,
            raw_response=response.content,
            retrieved=retrieved,
        )


def build_answer_messages(
    question: str,
    results: tuple[RetrievalResult, ...],
    max_context_chars: int = 12_000,
) -> list[dict[str, str]]:
    """Build chat messages for grounded answer generation."""
    context = build_context(results, max_context_chars=max_context_chars)
    user_prompt = f"""Question:
{question}

Context:
{context}
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_context(
    results: tuple[RetrievalResult, ...],
    max_context_chars: int = 12_000,
) -> str:
    """Render retrieved chunks as citation-ready context blocks."""
    blocks: list[str] = []
    used_chars = 0
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        source_id = f"S{index}"
        text = str(chunk.get("text", "")).strip()
        block = (
            f"[{source_id}]\n"
            f"source_filename: {chunk.get('source_filename')}\n"
            f"page_number: {chunk.get('page_number')}\n"
            f"chunk_id: {chunk.get('chunk_id')}\n"
            f"text: {text}\n"
        )
        if used_chars + len(block) > max_context_chars:
            break
        blocks.append(block)
        used_chars += len(block)
    return "\n".join(blocks)


def parse_answer_json(content: str) -> dict:
    """Parse and validate LLM JSON output with Pydantic."""
    payload = GroundedAnswerSchema.model_validate_json(_extract_json_object(content))
    citations = tuple(_parse_citation(citation) for citation in payload.citations)
    return {
        "answer": payload.answer.strip(),
        "citations": citations,
        "abstained": payload.abstained,
    }


def _parse_citation(payload: AnswerCitationSchema) -> AnswerCitation:
    return AnswerCitation(
        source_id=payload.source_id,
        source_filename=payload.source_filename,
        page_number=payload.page_number,
        chunk_id=payload.chunk_id,
    )


def _extract_json_object(content: str) -> str:
    text = content.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("LLM response did not contain a JSON object")
    return match.group(0)
