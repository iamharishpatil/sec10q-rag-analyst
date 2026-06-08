from sec_rag.prompts.financial_qa import EDGE_CASE_EXAMPLES, FinancialQAPrompt, build_context
from sec_rag.retrieval import RetrievalResult


def _result() -> RetrievalResult:
    return RetrievalResult(
        rank=1,
        score=0.9,
        backend="dense",
        chunk={
            "chunk_id": "2023_q2_aapl_p19_c0",
            "source_filename": "2023 Q2 AAPL.pdf",
            "page_number": 19,
            "text": "Total net sales were $94,836 million.",
        },
    )


def test_prompt_uses_delimited_context_blocks() -> None:
    context = build_context((_result(),))

    assert context.startswith('<source id="S1"')
    assert 'source_filename="2023 Q2 AAPL.pdf"' in context
    assert 'chunk_id="2023_q2_aapl_p19_c0"' in context
    assert context.strip().endswith("</source>")


def test_prompt_includes_edge_case_few_shots() -> None:
    messages = FinancialQAPrompt().build_messages("What were net sales?", (_result(),))
    user_message = messages[1]["content"]

    assert "<examples>" in user_message
    assert '<example name="direct_numeric_answer">' in EDGE_CASE_EXAMPLES
    assert '<example name="insufficient_evidence">' in user_message
    assert '<example name="conflicting_context">' in user_message


def test_prompt_can_disable_examples() -> None:
    messages = FinancialQAPrompt(include_examples=False).build_messages(
        "What were net sales?",
        (_result(),),
    )

    assert "<examples>" not in messages[1]["content"]
