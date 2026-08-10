"""Request/response schemas for the chat API.

These are the *wire* contract and are deliberately separate from the internal
``app.llm.base.Message`` dataclass. Keeping them apart means we can change the
internal representation (Phase 3 adds retrieved context to it) without
altering the public API, and vice versa.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.core.config import ProviderName

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: Role
    content: str = Field(min_length=1, max_length=100_000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=200)
    # None => use the configured default provider (Ollama).
    provider: ProviderName | None = None
    model: str | None = Field(default=None, max_length=200)
    max_tokens: int | None = Field(default=None, ge=1, le=32_000)

    @field_validator("messages")
    @classmethod
    def must_contain_a_non_system_message(
        cls, v: list[ChatMessage]
    ) -> list[ChatMessage]:
        # Every provider rejects a request whose only content is a system
        # prompt. Catching it here returns a clean 422 describing the problem,
        # instead of a 502 wrapping some provider's error string.
        if all(m.role == "system" for m in v):
            raise ValueError("at least one non-system message is required")
        return v


class UsageOut(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class ChatDoneOut(BaseModel):
    """Payload of the terminal `done` SSE event."""

    provider: str
    model: str
    usage: UsageOut
    finish_reason: str | None = None
    latency_ms: float
    # Always present so clients can branch on it without a null check. The
    # richer cache fields (similarity, age, tokens_saved) only appear on hits,
    # which is why a hit is built as a raw dict rather than through this model.
    cached: bool = False
