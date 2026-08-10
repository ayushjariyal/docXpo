"""Google Gemini provider, via the official `google-genai` SDK.

Gemini diverges from the other three providers in more ways than any of them,
which is a good stress test of whether the abstraction actually holds:

1. **The assistant role is called `model`**, not `assistant`. Sending
   "assistant" is rejected, so roles are translated on the way in.
2. **System prompts go in `config.system_instruction`**, not the message list
   (same shape of problem as Anthropic, different spelling).
3. **`max_tokens` is `max_output_tokens`**, nested inside a config object.
4. **Usage arrives on every chunk, cumulatively** -- unlike Ollama (final line
   only) or OpenAI (one extra opt-in chunk at the end). We keep the most recent
   non-null value rather than summing, or token counts would be multiplied by
   the number of chunks.
5. `generate_content_stream` is a *coroutine returning an async iterator*, so
   it needs `await` before `async for` -- unlike the Anthropic and OpenAI
   helpers.

All of that is contained here. Above this file, Gemini looks like every other
provider.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import ClassVar

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

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

# docXpo's vocabulary -> Gemini's. Only the assistant role actually differs.
_ROLE_MAP = {"user": "user", "assistant": "model"}


class GeminiProvider(LLMProvider):
    name: ClassVar[str] = "gemini"

    def __init__(self, settings: Settings) -> None:
        if settings.gemini_api_key is None:
            raise ProviderNotConfigured(
                "GEMINI_API_KEY is not set", provider=self.name
            )

        self._model = settings.gemini_model
        self._max_tokens = settings.llm_max_tokens
        self._client = genai.Client(
            api_key=settings.gemini_api_key.get_secret_value(),
            http_options=genai_types.HttpOptions(
                # SDK wants milliseconds here, while our setting (like every
                # other provider's) is in seconds.
                timeout=int(settings.llm_timeout_seconds * 1000),
            ),
        )

    @property
    def default_model(self) -> str:
        return self._model

    async def check_credentials(self) -> None:
        """Cheap auth + model check: GET /v1beta/models/{id}.

        Free and non-generative, and it validates the model name too -- a typo
        like "gemini-2.0-flsh" fails at boot instead of on first use.

        Caveat worth knowing: like Anthropic's equivalent, this confirms the key
        is *accepted*. It cannot confirm you have remaining quota, so a
        exhausted free tier still surfaces on the first real request as a 429.
        """
        try:
            await self._client.aio.models.get(model=self._model)
        except genai_errors.APIError as exc:
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

        contents = [
            genai_types.Content(
                role=_ROLE_MAP[m.role],
                parts=[genai_types.Part(text=m.content)],
            )
            for m in conversation
        ]

        config = genai_types.GenerateContentConfig(
            max_output_tokens=max_tokens or self._max_tokens,
            # Omit rather than pass None: the SDK treats an explicit None
            # differently from an absent field on some versions.
            **({"system_instruction": system} if system else {}),
        )

        try:
            # Note the `await`: unlike the Anthropic/OpenAI helpers this is a
            # coroutine that *returns* the async iterator.
            stream = await self._client.aio.models.generate_content_stream(
                model=model, contents=contents, config=config
            )

            usage = Usage()
            finish_reason: str | None = None
            resolved_model = model

            async for chunk in stream:
                if chunk.model_version:
                    resolved_model = chunk.model_version

                # `.text` is a convenience property that concatenates the text
                # parts of the first candidate. It is None on chunks that carry
                # only metadata (usage, safety ratings), hence the guard.
                text = chunk.text
                if text:
                    yield TextDelta(text)

                # Cumulative, not incremental -- overwrite, never accumulate.
                if chunk.usage_metadata is not None:
                    um = chunk.usage_metadata
                    # Gemini 3.x reasons by default and reports those tokens
                    # separately in `thoughts_token_count`. They are billed as
                    # output and they consume max_output_tokens, so they MUST be
                    # included -- counting only `candidates_token_count` under-
                    # reports cost by ~10x on a typical short answer (measured:
                    # 462 thinking vs 51 visible).
                    usage = Usage(
                        input_tokens=um.prompt_token_count or 0,
                        output_tokens=(um.candidates_token_count or 0)
                        + (um.thoughts_token_count or 0),
                    )

                if chunk.candidates and chunk.candidates[0].finish_reason:
                    raw = chunk.candidates[0].finish_reason
                    # An enum on this SDK; normalize to the plain string the
                    # rest of the app (and the SSE payload) expects.
                    finish_reason = getattr(raw, "name", str(raw)).lower()

            yield StreamDone(
                model=resolved_model, usage=usage, finish_reason=finish_reason
            )

        except genai_errors.APIError as exc:
            raise self._translate(exc) from exc

    def _translate(self, exc: genai_errors.APIError) -> ProviderError:
        """Map SDK errors onto our hierarchy using the HTTP status code.

        `APIError.code` is the HTTP status int; `.status` is Google's string
        code (e.g. "INVALID_ARGUMENT") and `.message` the human-readable text.
        """
        code = getattr(exc, "code", None)
        detail = getattr(exc, "message", None) or str(exc)

        # Google returns *400 INVALID_ARGUMENT* for a malformed/invalid API key,
        # not 401. Left to the generic 400 branch it would surface as
        # ProviderBadRequest -> HTTP 400, blaming the caller for what is
        # actually our own misconfiguration. Detect it by message and treat it
        # as an auth failure (-> 502) like every other provider.
        if code in (401, 403) or (code == 400 and "api key" in detail.lower()):
            return ProviderAuthError(
                f"Gemini rejected our credentials: {detail}", provider=self.name
            )
        if code == 429:
            # The most likely error on a free-tier key: quota exhausted.
            return ProviderRateLimited(
                f"Gemini quota exceeded: {detail}", provider=self.name
            )
        if code in (400, 404):
            return ProviderBadRequest(
                f"Gemini rejected the request: {detail}", provider=self.name
            )
        if code == 504:
            return ProviderTimeout(f"Gemini timed out: {detail}", provider=self.name)
        if code is not None and code >= 500:
            return ProviderUnavailable(
                f"Gemini server error ({code}): {detail}", provider=self.name
            )
        return ProviderError(f"Gemini error: {detail}", provider=self.name)

    async def aclose(self) -> None:
        # The google-genai client manages its own transport and exposes no
        # close/aclose hook, so there is nothing to release here. Defined
        # explicitly because LLMProvider requires it.
        return None
