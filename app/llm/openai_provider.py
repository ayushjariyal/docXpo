"""OpenAI provider, via the official `openai` SDK.

Module is named `openai_provider` rather than `openai` on purpose: a module
named `openai.py` inside a package still shadows the installed SDK for any
sibling module doing `import openai`, which produces a baffling ImportError.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import ClassVar

import openai
from openai import AsyncOpenAI

from app.core.config import Settings
from app.llm.base import LLMProvider, Message, StreamDone, StreamEvent, TextDelta, Usage
from app.llm.errors import (
    ProviderAuthError,
    ProviderBadRequest,
    ProviderError,
    ProviderNotConfigured,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
)


class OpenAIProvider(LLMProvider):
    name: ClassVar[str] = "openai"

    def __init__(self, settings: Settings) -> None:
        if settings.openai_api_key is None:
            raise ProviderNotConfigured(
                "OPENAI_API_KEY is not set", provider=self.name
            )

        self._model = settings.openai_model
        self._max_tokens = settings.llm_max_tokens
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=settings.openai_base_url,
            timeout=settings.llm_timeout_seconds,
            # The SDK retries 429/5xx with exponential backoff on its own.
            # Left at the default rather than adding our own retry loop, which
            # would multiply out to max_retries * our_retries attempts.
            max_retries=2,
        )

    @property
    def default_model(self) -> str:
        return self._model

    async def check_credentials(self) -> None:
        try:
            await self._client.models.retrieve(self._model)
        except openai.APIError as exc:
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
        try:
            stream = await self._client.chat.completions.create(
                model=model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                max_tokens=max_tokens or self._max_tokens,
                stream=True,
                # Without this, a streaming response carries NO token counts at
                # all -- `usage` is null on every chunk. Phase 6 needs those
                # numbers to compute cost, so opting in is mandatory, not
                # optional. It adds one final chunk with an empty `choices`
                # list, which is why the loop below tolerates that shape.
                stream_options={"include_usage": True},
            )

            usage = Usage()
            finish_reason: str | None = None
            resolved_model = model

            async for chunk in stream:
                resolved_model = chunk.model or resolved_model

                if chunk.choices:
                    choice = chunk.choices[0]
                    if choice.delta and choice.delta.content:
                        yield TextDelta(choice.delta.content)
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason

                # The usage-bearing chunk arrives last and has choices == [].
                if chunk.usage:
                    usage = Usage(
                        input_tokens=chunk.usage.prompt_tokens or 0,
                        output_tokens=chunk.usage.completion_tokens or 0,
                    )

            yield StreamDone(
                model=resolved_model, usage=usage, finish_reason=finish_reason
            )

        except openai.APIError as exc:
            raise self._translate(exc) from exc

    def _translate(self, exc: openai.APIError) -> ProviderError:
        """Map SDK exceptions onto our hierarchy.

        Ordered most-specific first: several of these are subclasses of
        APIStatusError, so a broad clause first would swallow the rest.
        """
        if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
            return ProviderAuthError(
                f"OpenAI rejected our credentials: {exc}", provider=self.name
            )
        if isinstance(exc, openai.RateLimitError):
            return ProviderRateLimited(f"OpenAI rate limit: {exc}", provider=self.name)
        if isinstance(exc, openai.APITimeoutError):
            return ProviderTimeout(f"OpenAI timed out: {exc}", provider=self.name)
        if isinstance(exc, openai.APIConnectionError):
            return ProviderUnavailable(
                f"Cannot reach OpenAI: {exc}", provider=self.name
            )
        if isinstance(exc, openai.NotFoundError | openai.BadRequestError):
            return ProviderBadRequest(f"OpenAI rejected the request: {exc}", provider=self.name)
        return ProviderError(f"OpenAI error: {exc}", provider=self.name)

    async def aclose(self) -> None:
        await self._client.close()
