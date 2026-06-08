"""LLM provider interfaces and hosted Groq implementation."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


@dataclass(frozen=True)
class LLMResponse:
    """Raw text returned by an LLM provider."""

    content: str
    model: str
    provider: str


class LLMProvider:
    """Minimal provider interface for chat-style LLM calls."""

    def generate(self, messages: Sequence[Mapping[str, str]]) -> LLMResponse:
        """Generate a response from chat messages."""
        raise NotImplementedError


class GroqProvider(LLMProvider):
    """Groq-hosted open-model provider."""

    def __init__(
        self,
        model: str = DEFAULT_GROQ_MODEL,
        api_key: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 700,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set")

    def generate(self, messages: Sequence[Mapping[str, str]]) -> LLMResponse:
        from groq import Groq

        client = Groq(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[dict(message) for message in messages],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content or ""
        return LLMResponse(content=content, model=self.model, provider="groq")


class MockLLMProvider(LLMProvider):
    """Deterministic provider used in tests."""

    def __init__(self, content: str, model: str = "mock-model") -> None:
        self.content = content
        self.model = model
        self.messages: tuple[Mapping[str, str], ...] = tuple()

    def generate(self, messages: Sequence[Mapping[str, str]]) -> LLMResponse:
        self.messages = tuple(messages)
        return LLMResponse(content=self.content, model=self.model, provider="mock")
