"""Grounded answer generation from retrieved SEC filing chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from sec_rag.llm import LLMProvider
from sec_rag.prompts.financial_qa import FinancialQAPrompt, build_context as build_context
from sec_rag.retrieval import RetrievalFilters, RetrievalResult, Retriever


class AnswerCitationSchema(BaseModel):
    """Pydantic schema for one answer citation."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(description="Context source id, for example S1.")
    source_filename: str = Field(description="Source SEC filing filename.")
    page_number: int = Field(description="One-indexed source page number.")
    chunk_id: str = Field(description="Retrieved chunk identifier.")

    @field_validator("source_id", "source_filename", "chunk_id")
    @classmethod
    def require_non_empty_string(cls, value: str, info: ValidationInfo) -> str:
        """Reject blank citation fields."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{info.field_name} must not be empty")
        return cleaned

    @field_validator("source_id")
    @classmethod
    def require_source_id_shape(cls, value: str) -> str:
        """Require source ids from the rendered context format."""
        if not re.fullmatch(r"S[1-9][0-9]*", value):
            raise ValueError("source_id must look like S1, S2, ...")
        return value

    @field_validator("page_number")
    @classmethod
    def require_positive_page_number(cls, value: int) -> int:
        """Reject invalid page numbers."""
        if value <= 0:
            raise ValueError("page_number must be positive")
        return value


class GroundedAnswerSchema(BaseModel):
    """Pydantic schema for grounded answer output."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="Short answer grounded only in the provided context.")
    citations: list[AnswerCitationSchema] = Field(
        description="Citations that support the answer. Use an empty list when abstaining."
    )
    abstained: bool = Field(description="True when the context is insufficient to answer.")

    @field_validator("answer")
    @classmethod
    def require_non_empty_answer(cls, value: str) -> str:
        """Reject blank answers."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("answer must not be empty")
        return cleaned

    @model_validator(mode="after")
    def validate_abstention_citations(self) -> "GroundedAnswerSchema":
        """Keep abstention and citation behavior consistent."""
        if self.abstained and self.citations:
            raise ValueError("abstained answers must not include citations")
        if not self.abstained and not self.citations:
            raise ValueError("non-abstained answers must include at least one citation")
        return self

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
        prompt: FinancialQAPrompt | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm_provider = llm_provider
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.prompt = prompt or FinancialQAPrompt(max_context_chars=max_context_chars)

    def answer(
        self,
        question: str,
        filters: RetrievalFilters | None = None,
    ) -> GroundedAnswer:
        """Retrieve context and generate a grounded answer."""
        retrieved = self.retriever.search(question, top_k=self.top_k, filters=filters)
        messages = self.prompt.build_messages(question=question, results=retrieved)
        response = self.llm_provider.generate(messages)
        parsed = parse_answer_json(response.content, retrieved=retrieved)
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


def parse_answer_json(content: str, retrieved: tuple[RetrievalResult, ...] = tuple()) -> dict:
    """Parse and validate LLM JSON output with Pydantic."""
    payload = GroundedAnswerSchema.model_validate_json(_extract_json_object(content))
    if retrieved:
        validate_citations_against_retrieved(payload.citations, retrieved)
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


def validate_citations_against_retrieved(
    citations: list[AnswerCitationSchema],
    retrieved: tuple[RetrievalResult, ...],
) -> None:
    """Validate that cited source ids and metadata match retrieved chunks."""
    sources = {
        f"S{index}": result.chunk
        for index, result in enumerate(retrieved, start=1)
    }
    for citation in citations:
        chunk = sources.get(citation.source_id)
        if chunk is None:
            raise ValueError(f"citation source_id was not retrieved: {citation.source_id}")
        expected = {
            "source_filename": str(chunk.get("source_filename", "")),
            "page_number": int(chunk.get("page_number", 0)),
            "chunk_id": str(chunk.get("chunk_id", "")),
        }
        actual = {
            "source_filename": citation.source_filename,
            "page_number": citation.page_number,
            "chunk_id": citation.chunk_id,
        }
        if actual != expected:
            raise ValueError(f"citation metadata mismatch for {citation.source_id}")


def _extract_json_object(content: str) -> str:
    text = content.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("LLM response did not contain a JSON object")
    return match.group(0)
