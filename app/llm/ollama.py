"""Ollama provider — the local, free default.

Wire format is newline-delimited JSON (not SSE), one complete object per line:

    {"model":"llama3.2","message":{"role":"assistant","content":"Hel"},"done":false}
    {"model":"llama3.2","message":{"role":"assistant","content":"lo"},"done":false}
    {"model":"llama3.2","message":{...},"done":true,
     "prompt_eval_count":26,"eval_count":298,"done_reason":"stop"}

Token counts only exist on the final line, which is why usage is accumulated
there rather than per-chunk.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import ClassVar

import httpx

from app.core.config import Settings
from app.core.logging import get_logger
from app.llm.base import LLMProvider, Message, StreamDone, StreamEvent, TextDelta, Usage
from app.llm.errors import (
    ProviderBadRequest,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
)

log = get_logger(__name__)


class OllamaProvider(LLMProvider):
    name: ClassVar[str] = "ollama"

    def __init__(self, settings: Settings) -> None:
        self._model = settings.ollama_model
        self._client = httpx.AsyncClient(
            base_url=settings.ollama_base_url,
            # Granular timeouts: connecting should be fast (Ollama is local),
            # but *reading* can be slow because a CPU-only model may take tens
            # of seconds to produce its first token. A single flat timeout
            # would force us to choose between "slow to detect a dead server"
            # and "kills legitimate slow generations".
            timeout=httpx.Timeout(
                settings.llm_timeout_seconds, connect=5.0, write=10.0
            ),
        )

    @property
    def default_model(self) -> str:
        return self._model

    async def check_credentials(self) -> None:
        """Ollama has no auth; this just confirms the server is reachable."""
        try:
            resp = await self._client.get("/api/tags", timeout=5.0)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(
                f"Cannot reach Ollama at {self._client.base_url}: {exc}",
                provider=self.name,
            ) from exc

    def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        # Plain `def` (not `async def`) so callers get the async generator
        # directly and can `async for` over it without an extra await.
        return self._stream(messages, model or self._model, max_tokens)

    async def _stream(
        self,
        messages: Sequence[Message],
        model: str,
        max_tokens: int | None,
    ) -> AsyncIterator[StreamEvent]:
        payload: dict[str, object] = {
            "model": model,
            # Ollama accepts system as a normal message role, so no split needed.
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
        }
        if max_tokens is not None:
            # Ollama spells max_tokens "num_predict", nested under options.
            payload["options"] = {"num_predict": max_tokens}

        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as resp:
                if resp.status_code >= 400:
                    # The body hasn't been read yet on a streaming response;
                    # aread() is required before we can look at it.
                    body = (await resp.aread()).decode(errors="replace")
                    raise self._translate_status(resp.status_code, body)

                usage = Usage()
                finish_reason: str | None = None
                resolved_model = model

                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        # A partial line means a truncated stream. Skipping is
                        # safer than crashing the whole response.
                        log.warning("ollama_bad_json_line", line=line[:200])
                        continue

                    resolved_model = chunk.get("model", resolved_model)

                    content = (chunk.get("message") or {}).get("content") or ""
                    if content:
                        yield TextDelta(content)

                    if chunk.get("done"):
                        usage = Usage(
                            input_tokens=chunk.get("prompt_eval_count", 0) or 0,
                            output_tokens=chunk.get("eval_count", 0) or 0,
                        )
                        finish_reason = chunk.get("done_reason")

                yield StreamDone(
                    model=resolved_model, usage=usage, finish_reason=finish_reason
                )

        except httpx.TimeoutException as exc:
            raise ProviderTimeout(f"Ollama timed out: {exc}", provider=self.name) from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailable(
                f"Cannot reach Ollama at {self._client.base_url}. "
                "Is it running? Start it with `ollama serve` and pull a model "
                f"with `ollama pull {model}`.",
                provider=self.name,
            ) from exc

    def _translate_status(self, status: int, body: str) -> ProviderError:
        # Ollama returns 404 with a "model not found" body when the model
        # hasn't been pulled — by far the most common first-run failure, so it
        # gets an actionable message instead of a bare 404.
        if status == 404:
            return ProviderBadRequest(
                f"Ollama model not found. Pull it first: `ollama pull {self._model}`. "
                f"({body[:200]})",
                provider=self.name,
            )
        if status < 500:
            return ProviderBadRequest(
                f"Ollama rejected the request ({status}): {body[:200]}",
                provider=self.name,
            )
        return ProviderUnavailable(
            f"Ollama server error ({status}): {body[:200]}", provider=self.name
        )

    async def aclose(self) -> None:
        await self._client.aclose()
