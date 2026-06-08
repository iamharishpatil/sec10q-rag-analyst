"""Production prompt builder for SEC financial question answering."""

from __future__ import annotations

from dataclasses import dataclass

from sec_rag.retrieval import RetrievalResult


SYSTEM_PROMPT = """You are a financial RAG assistant for SEC 10-Q filings.

<task>
Answer the user's question using only the retrieved SEC filing context.
</task>

<grounding_rules>
- Use only facts explicitly present in the retrieved context.
- Do not use outside knowledge.
- Do not estimate, infer, annualize, or calculate unless the context directly supports it.
- If the context is insufficient, set abstained=true.
- Check support internally, but do not include reasoning steps in the response.
</grounding_rules>

<citation_rules>
- Every non-abstained answer must include at least one citation.
- Citations must use source_id values from the provided context, such as S1.
- Citation fields must exactly match the cited context metadata.
- If retrieved sources conflict, abstain unless one source directly answers the question.
</citation_rules>

<output_rules>
Return only valid JSON matching the provided response schema.
</output_rules>
"""


EDGE_CASE_EXAMPLES = """<examples>
<example name="direct_numeric_answer">
<question>What were Apple's total net sales?</question>
<context>
<source id="S1" source_filename="2023 Q2 AAPL.pdf" page_number="10" chunk_id="aapl_p10_c0">
Apple total net sales were $94,836 million.
</source>
</context>
<answer_json>{"answer":"Apple's total net sales were $94,836 million.","citations":[{"source_id":"S1","source_filename":"2023 Q2 AAPL.pdf","page_number":10,"chunk_id":"aapl_p10_c0"}],"abstained":false}</answer_json>
</example>

<example name="insufficient_evidence">
<question>What will Apple's total net sales be next year?</question>
<context>
<source id="S1" source_filename="2023 Q2 AAPL.pdf" page_number="10" chunk_id="aapl_p10_c0">
Apple total net sales were $94,836 million for the reported period.
</source>
</context>
<answer_json>{"answer":"The provided context does not contain enough evidence to answer.","citations":[],"abstained":true}</answer_json>
</example>

<example name="conflicting_context">
<question>What were Microsoft's revenues?</question>
<context>
<source id="S1" source_filename="2023 Q1 MSFT.pdf" page_number="7" chunk_id="msft_q1_p7_c0">
Revenue was $52.9 billion for the quarter.
</source>
<source id="S2" source_filename="2023 Q2 MSFT.pdf" page_number="7" chunk_id="msft_q2_p7_c0">
Revenue was $56.2 billion for the quarter.
</source>
</context>
<answer_json>{"answer":"The retrieved context contains multiple period-specific revenue values and the question does not specify the period.","citations":[],"abstained":true}</answer_json>
</example>
</examples>
"""


@dataclass(frozen=True)
class FinancialQAPrompt:
    """Build chat messages for grounded financial QA."""

    include_examples: bool = True
    max_context_chars: int = 12_000

    def build_messages(
        self,
        question: str,
        results: tuple[RetrievalResult, ...],
    ) -> list[dict[str, str]]:
        """Build chat messages from a question and retrieved context."""
        context = build_context(results, max_context_chars=self.max_context_chars)
        examples = f"\n{EDGE_CASE_EXAMPLES}" if self.include_examples else ""
        user_prompt = f"""{examples}
<retrieved_context>
{context}
</retrieved_context>

<user_question>
{question}
</user_question>
"""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]


def build_context(
    results: tuple[RetrievalResult, ...],
    max_context_chars: int = 12_000,
) -> str:
    """Render retrieved chunks as delimited citation-ready context."""
    blocks: list[str] = []
    used_chars = 0
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        source_id = f"S{index}"
        text = str(chunk.get("text", "")).strip()
        block = (
            f'<source id="{source_id}" '
            f'source_filename="{chunk.get("source_filename")}" '
            f'page_number="{chunk.get("page_number")}" '
            f'chunk_id="{chunk.get("chunk_id")}">\n'
            f"{text}\n"
            "</source>\n"
        )
        if used_chars + len(block) > max_context_chars:
            break
        blocks.append(block)
        used_chars += len(block)
    return "\n".join(blocks)
