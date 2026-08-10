"""Anthropic provider, via the official `anthropic` SDK.

Two things differ structurally from the other providers, and this class exists
mostly to hide them:

1. **System prompts are a top-level parameter**, not a message role. Sending
   ``{"role": "system", ...}`` in the messages array is rejected. We pull them
   out with ``split_system``.

2. **max_tokens is required**, not optional. The abstraction always supplies a
   value so this provider is never the odd one out.

Usage is also split across two events -- input tokens on `message_start`,
output tokens on `message_delta` -- which the SDK's `.stream()` helper
accumulates for us; `get_final_message()` returns the merged totals.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import ClassVar

import anthropic
from anthropic import AsyncAnthropic

from app.core.config import Settings
from app.llm.base import (
    LLMProvider,
    Message,
    StreamDone,
    StreamEvent,
    TextDelta,
    Usage,
    split_system,
)
from app.llm.errors import (
    ProviderAuthError,
    ProviderBadRequest,
    ProviderError,
    ProviderNotConfigured,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
)


class AnthropicProvider(LLMProvider):
    name: ClassVar[str] = "anthropic"

    def __init__(self, settings: Settings) -> None:
        if settings.anthropic_api_key is None:
            raise ProviderNotConfigured(
                "ANTHROPIC_API_KEY is not set", provider=self.name
            )

        self._model = settings.anthropic_model
        self._max_tokens = settings.llm_max_tokens
        self._client = AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
            timeout=settings.llm_timeout_seconds,
            max_retries=2,
        )

    @property
    def default_model(self) -> str:
        return self._model

    async def check_credentials(self) -> None:
        """Cheap auth + model check: GET /v1/models/{id}.

        Chosen over a one-token generation because it costs nothing, and it
        validates two things at once -- that the key is accepted, and that the
        configured model ID actually exists and is available to this account
        (a typo like "claude-sonnet-4.6" fails here rather than at 3am).
        """
        try:
            await self._client.models.retrieve(self._model)
        except anthropic.APIError as exc:
            raise self._translate(exc) from exc

    def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        return self._stream(messages, model or self._model, max_tokens)

    async def _stream(
        self,
        messages: Sequence[Message],
        model: str,
        max_tokens: int | None,
    ) -> AsyncIterator[StreamEvent]:
        system, conversation = split_system(messages)

        # `system=None` is not accepted; omit the key entirely when unset.
        extra: dict[str, object] = {"system": system} if system else {}

        try:
            async with self._client.messages.stream(
                model=model,
                # Required by this API. On current models max_tokens caps
                # thinking *plus* visible text, so a value tuned for the
                # answer alone can truncate the response.
                max_tokens=max_tokens or self._max_tokens,
                messages=[
                    {"role": m.role, "content": m.content} for m in conversation
                ],
                **extra,  # type: ignore[arg-type]
            ) as stream:
                # text_stream yields only assistant text, skipping the
                # thinking/tool blocks we don't surface in a chat gateway.
                async for text in stream.text_stream:
                    if text:
                        yield TextDelta(text)

                # Available only after the stream is exhausted; the SDK has
                # accumulated usage from message_start and message_delta by now.
                final = await stream.get_final_message()

            yield StreamDone(
                model=final.model,
                usage=Usage(
                    input_tokens=final.usage.input_tokens or 0,
                    output_tokens=final.usage.output_tokens or 0,
                ),
                finish_reason=final.stop_reason,
            )

        except anthropic.APIError as exc:
            raise self._translate(exc) from exc

    def _translate(self, exc: anthropic.APIError) -> ProviderError:
        if isinstance(
            exc, anthropic.AuthenticationError | anthropic.PermissionDeniedError
        ):
            return ProviderAuthError(
                f"Anthropic rejected our credentials: {exc}", provider=self.name
            )
        if isinstance(exc, anthropic.RateLimitError):
            return ProviderRateLimited(
                f"Anthropic rate limit: {exc}", provider=self.name
            )
        if isinstance(exc, anthropic.APITimeoutError):
            return ProviderTimeout(f"Anthropic timed out: {exc}", provider=self.name)
        if isinstance(exc, anthropic.APIConnectionError):
            return ProviderUnavailable(
                f"Cannot reach Anthropic: {exc}", provider=self.name
            )
        if isinstance(exc, anthropic.NotFoundError | anthropic.BadRequestError):
            return ProviderBadRequest(
                f"Anthropic rejected the request: {exc}", provider=self.name
            )
        return ProviderError(f"Anthropic error: {exc}", provider=self.name)

    async def aclose(self) -> None:
        await self._client.close()
