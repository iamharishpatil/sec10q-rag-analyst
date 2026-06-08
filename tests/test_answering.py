import json

import pytest
from pydantic import ValidationError

from sec_rag.answering import (
    AnswerGenerator,
    GroundedAnswerSchema,
    build_context,
    parse_answer_json,
)
from sec_rag.llm import MockLLMProvider
from sec_rag.retrieval import RetrievalFilters, RetrievalResult


class FakeRetriever:
    def __init__(self, results: tuple[RetrievalResult, ...]) -> None:
        self.results = results
        self.calls: list[tuple[str, int, RetrievalFilters | None]] = []

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: RetrievalFilters | None = None,
    ) -> tuple[RetrievalResult, ...]:
        self.calls.append((query, top_k, filters))
        return self.results[:top_k]


def _result() -> RetrievalResult:
    return RetrievalResult(
        rank=1,
        score=0.9,
        backend="dense",
        dense_score=0.9,
        chunk={
            "chunk_id": "2023_q2_aapl_p10_c0",
            "source_filename": "2023 Q2 AAPL.pdf",
            "ticker": "AAPL",
            "page_number": 10,
            "chunk_index": 0,
            "text": "Apple total net sales were $94,836 million.",
        },
    )


def test_build_context_renders_citation_metadata() -> None:
    context = build_context((_result(),))

    assert '<source id="S1"' in context
    assert 'source_filename="2023 Q2 AAPL.pdf"' in context
    assert 'page_number="10"' in context
    assert "Apple total net sales" in context


def test_parse_answer_json_accepts_plain_json() -> None:
    payload = {
        "answer": "Apple total net sales were $94,836 million.",
        "citations": [
            {
                "source_id": "S1",
                "source_filename": "2023 Q2 AAPL.pdf",
                "page_number": 10,
                "chunk_id": "2023_q2_aapl_p10_c0",
            }
        ],
        "abstained": False,
    }

    parsed = parse_answer_json(json.dumps(payload), retrieved=(_result(),))

    assert parsed["answer"] == "Apple total net sales were $94,836 million."
    assert parsed["citations"][0].source_id == "S1"
    assert not parsed["abstained"]


def test_grounded_answer_schema_rejects_non_abstained_answer_without_citations() -> None:
    with pytest.raises(ValidationError):
        GroundedAnswerSchema.model_validate(
            {
                "answer": "Apple total net sales were $94,836 million.",
                "citations": [],
                "abstained": False,
            }
        )


def test_grounded_answer_schema_rejects_abstained_answer_with_citations() -> None:
    with pytest.raises(ValidationError):
        GroundedAnswerSchema.model_validate(
            {
                "answer": "The context is insufficient.",
                "citations": [
                    {
                        "source_id": "S1",
                        "source_filename": "2023 Q2 AAPL.pdf",
                        "page_number": 10,
                        "chunk_id": "2023_q2_aapl_p10_c0",
                    }
                ],
                "abstained": True,
            }
        )


def test_parse_answer_json_rejects_citation_metadata_mismatch() -> None:
    payload = {
        "answer": "Apple total net sales were $94,836 million.",
        "citations": [
            {
                "source_id": "S1",
                "source_filename": "wrong.pdf",
                "page_number": 10,
                "chunk_id": "2023_q2_aapl_p10_c0",
            }
        ],
        "abstained": False,
    }

    with pytest.raises(ValueError, match="citation metadata mismatch"):
        parse_answer_json(json.dumps(payload), retrieved=(_result(),))


def test_grounded_answer_schema_builds_groq_response_format() -> None:
    response_format = GroundedAnswerSchema.groq_response_format()

    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["additionalProperties"] is False
    assert "answer" in response_format["json_schema"]["schema"]["properties"]


def test_answer_generator_retrieves_and_calls_llm() -> None:
    llm = MockLLMProvider(
        json.dumps(
            {
                "answer": "Apple total net sales were $94,836 million.",
                "citations": [
                    {
                        "source_id": "S1",
                        "source_filename": "2023 Q2 AAPL.pdf",
                        "page_number": 10,
                        "chunk_id": "2023_q2_aapl_p10_c0",
                    }
                ],
                "abstained": False,
            }
        )
    )
    retriever = FakeRetriever((_result(),))
    generator = AnswerGenerator(retriever=retriever, llm_provider=llm, top_k=1)

    answer = generator.answer("What were Apple total net sales?")

    assert retriever.calls[0][0] == "What were Apple total net sales?"
    assert answer.answer == "Apple total net sales were $94,836 million."
    assert answer.citations[0].chunk_id == "2023_q2_aapl_p10_c0"
    assert "<retrieved_context>" in llm.messages[1]["content"]
    assert "<example name=\"insufficient_evidence\">" in llm.messages[1]["content"]
