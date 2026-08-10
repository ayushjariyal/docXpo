"""The LLMProvider interface and the normalized types it speaks.

The whole point of this module is that the three providers have genuinely
different streaming wire formats, and everything above this layer should be
unaware of that:

    Ollama     newline-delimited JSON; each line is a full object with a
               `message.content` fragment and a `done` flag. Token counts
               arrive on the final line as prompt_eval_count / eval_count.
    OpenAI     SSE; `choices[0].delta.content` fragments, terminated by a
               literal `[DONE]`. Usage is *omitted by default* and only
               appears if you opt in.
    Anthropic  SSE with typed events; text arrives as content_block_delta,
               and usage is split across message_start (input) and
               message_delta (output).

Each provider collapses its format into the same two-event stream:

    TextDelta(...)   zero or more, one per fragment
    StreamDone(...)  exactly one, last, carrying usage and the resolved model

Guaranteeing "exactly one StreamDone, always last" is what lets Phase 4's
cache and Phase 6's metrics have a single place to hook in, regardless of
which provider served the request.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts. Phase 6 turns these into cost."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class TextDelta:
    """One fragment of assistant text. Not necessarily one token."""

    text: str


@dataclass(frozen=True, slots=True)
class StreamDone:
    """Terminal event. Always emitted exactly once, always last."""

    model: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None


# A tagged union rather than one event class with optional fields: the consumer
# does `isinstance(ev, StreamDone)` and the type checker then knows `usage`
# exists. With a single class, every access to `usage` would need a None check.
StreamEvent = TextDelta | StreamDone


@dataclass(frozen=True, slots=True)
class ChatResult:
    """Fully accumulated response, for non-streaming callers."""

    text: str
    model: str
    usage: Usage
    finish_reason: str | None = None


class LLMProvider(ABC):
    """One provider. Instances are long-lived and hold a connection pool.

    Only ``stream_chat`` is abstract. ``complete()`` is derived from it, so a
    new provider implements one method and gets both APIs — and streaming and
    non-streaming can never drift out of sync, because there is only one
    code path.
    """

    name: ClassVar[str]

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Model used when the caller doesn't name one."""

    @abstractmethod
    def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a completion.

        Note this is declared as a plain method returning an AsyncIterator
        rather than as an `async def` generator. That matters: calling an async
        generator function does *not* execute any of its body, so a provider
        that is unreachable would not raise until the first `__anext__()`.
        Declaring the return type this way keeps that explicit for callers who
        want to surface connection failures before streaming starts
        (see the "priming" comment in the chat route).
        """

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """Non-streaming convenience wrapper, built on stream_chat."""
        parts: list[str] = []
        done: StreamDone | None = None

        async for event in self.stream_chat(
            messages, model=model, max_tokens=max_tokens
        ):
            if isinstance(event, TextDelta):
                parts.append(event.text)
            else:
                done = event

        if done is None:
            # Contract violation by a provider implementation, not a user error.
            raise RuntimeError(f"{self.name}: stream ended without a StreamDone event")

        return ChatResult(
            text="".join(parts),
            model=done.model,
            usage=done.usage,
            finish_reason=done.finish_reason,
        )

    async def check_credentials(self) -> None:
        """Verify this provider is actually usable, cheaply.

        Called at startup for the default provider. Implementations should make
        the cheapest possible authenticated call -- never a generation, which
        would cost tokens on every boot and every deploy.

        Must raise a ``ProviderError`` subclass on failure. The default is a
        no-op so a provider that has no cheap check simply opts out.
        """
        return None

    @abstractmethod
    async def aclose(self) -> None:
        """Release the underlying HTTP connection pool."""


def split_system(messages: Sequence[Message]) -> tuple[str | None, list[Message]]:
    """Separate system messages from the conversation.

    Ollama and OpenAI take the system prompt as a message with role="system";
    Anthropic takes it as a *top-level parameter* and rejects the role
    entirely. Providers that need the split call this; the caller above never
    has to know which convention applies.

    Multiple system messages are joined rather than dropped, so nothing the
    caller sent is silently discarded.
    """
    system_parts = [m.content for m in messages if m.role == "system"]
    rest = [m for m in messages if m.role != "system"]
    return ("\n\n".join(system_parts) if system_parts else None, rest)
