"""Chat orchestration.

Sits between the router (HTTP concerns) and the providers (vendor concerns).
Right now it resolves the provider, converts wire types to domain types, and
times the request. It exists as its own layer because the later phases all
attach here rather than to the router:

    Phase 3  retrieve chunks and prepend them as context
    Phase 4  check the semantic cache before calling a provider
    Phase 6  persist the usage/latency numbers this already measures

None of those should require touching the web layer.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence

from app.core.logging import get_logger
from app.llm.base import LLMProvider, Message, StreamDone, StreamEvent, TextDelta
from app.llm.registry import ProviderRegistry
from app.schemas.chat import ChatMessage

log = get_logger(__name__)


class ChatService:
    def __init__(self, registry: ProviderRegistry) -> None:
        self._registry = registry

    async def resolve_provider(self, name: str | None) -> LLMProvider:
        return await self._registry.get(name)

    async def stream(
        self,
        provider: LLMProvider,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Yield normalized events, measuring wall-clock latency.

        Latency is measured to the *last* token, not the first. Time-to-first-
        token is the better UX metric and Phase 6 will record both; total
        latency is what the p95 aggregate needs.
        """
        domain_messages = [Message(role=m.role, content=m.content) for m in messages]
        started = time.perf_counter()
        token_count = 0

        async for event in provider.stream_chat(
            domain_messages, model=model, max_tokens=max_tokens
        ):
            if isinstance(event, TextDelta):
                token_count += 1
                yield event
            else:
                elapsed_ms = (time.perf_counter() - started) * 1000
                log.info(
                    "chat_completed",
                    provider=provider.name,
                    model=event.model,
                    input_tokens=event.usage.input_tokens,
                    output_tokens=event.usage.output_tokens,
                    chunks=token_count,
                    latency_ms=round(elapsed_ms, 2),
                    finish_reason=event.finish_reason,
                )
                # Re-emit with nothing changed; the route needs the timing,
                # which it reads from its own clock. Kept as a separate branch
                # so the logging stays in the service, not the router.
                yield event

    @staticmethod
    def is_done(event: StreamEvent) -> bool:
        return isinstance(event, StreamDone)
