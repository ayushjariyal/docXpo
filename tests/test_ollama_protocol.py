"""Protocol-level tests for the Ollama provider.

These drive the real `_stream` code path -- including httpx's streaming
response handling and our line parsing -- against byte-for-byte copies of what
Ollama actually puts on the wire. An `httpx.MockTransport` replaces only the
socket, so everything above it is the production code path.

This is the tier of test that catches "we parsed the happy path but the final
`done` line has a different shape", which a mocked-out provider never would.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest

from app.core.config import Settings
from app.llm.base import Message, StreamDone, TextDelta
from app.llm.errors import ProviderBadRequest, ProviderUnavailable
from app.llm.ollama import OllamaProvider

# Verbatim shape of an Ollama /api/chat stream (fields trimmed to what we read).
OLLAMA_LINES = [
    {"model": "llama3.2", "message": {"role": "assistant", "content": "Hel"}, "done": False},
    {"model": "llama3.2", "message": {"role": "assistant", "content": "lo"}, "done": False},
    {
        "model": "llama3.2",
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 26,
        "eval_count": 298,
    },
]


def _provider_with(handler) -> OllamaProvider:
    provider = OllamaProvider(Settings(_env_file=None))
    # Swap the transport, keep the client configuration.
    provider._client = httpx.AsyncClient(
        base_url="http://ollama.test", transport=httpx.MockTransport(handler)
    )
    return provider


def _ndjson_handler(lines: list[dict], status: int = 200):
    async def body() -> AsyncIterator[bytes]:
        for line in lines:
            yield (json.dumps(line) + "\n").encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=body())

    return handler


async def test_parses_ndjson_stream_into_normalized_events() -> None:
    provider = _provider_with(_ndjson_handler(OLLAMA_LINES))

    events = [e async for e in provider.stream_chat([Message("user", "hi")])]
    await provider.aclose()

    assert [type(e).__name__ for e in events] == ["TextDelta", "TextDelta", "StreamDone"]
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Hello"

    done = events[-1]
    assert isinstance(done, StreamDone)
    # Ollama's prompt_eval_count / eval_count become our input/output tokens.
    assert done.usage.input_tokens == 26
    assert done.usage.output_tokens == 298
    assert done.finish_reason == "stop"
    assert done.model == "llama3.2"


async def test_request_payload_matches_the_ollama_api() -> None:
    captured: dict = {}

    async def body() -> AsyncIterator[bytes]:
        for line in OLLAMA_LINES:
            yield (json.dumps(line) + "\n").encode()

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        captured["_path"] = request.url.path
        return httpx.Response(200, content=body())

    provider = _provider_with(handler)
    async for _ in provider.stream_chat(
        [Message("system", "Be brief."), Message("user", "hi")], max_tokens=64
    ):
        pass
    await provider.aclose()

    assert captured["_path"] == "/api/chat"
    assert captured["stream"] is True
    # Ollama takes the system prompt as a normal message role (unlike Anthropic).
    assert captured["messages"][0] == {"role": "system", "content": "Be brief."}
    # max_tokens is spelled num_predict, nested under options.
    assert captured["options"] == {"num_predict": 64}


async def test_malformed_line_is_skipped_not_fatal() -> None:
    """A truncated stream should still deliver the tokens that did arrive."""

    async def body() -> AsyncIterator[bytes]:
        yield (json.dumps(OLLAMA_LINES[0]) + "\n").encode()
        yield b'{"message": {"content": "trunc\n'  # broken JSON
        yield (json.dumps(OLLAMA_LINES[2]) + "\n").encode()

    provider = _provider_with(lambda r: httpx.Response(200, content=body()))

    events = [e async for e in provider.stream_chat([Message("user", "hi")])]
    await provider.aclose()

    assert any(isinstance(e, TextDelta) for e in events)
    assert isinstance(events[-1], StreamDone)


async def test_unpulled_model_gives_an_actionable_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": 'model "llama3.2" not found'})

    provider = _provider_with(handler)

    with pytest.raises(ProviderBadRequest) as exc_info:
        async for _ in provider.stream_chat([Message("user", "hi")]):
            pass
    await provider.aclose()

    assert "ollama pull" in str(exc_info.value)


async def test_connection_refused_is_translated() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    provider = _provider_with(handler)

    with pytest.raises(ProviderUnavailable) as exc_info:
        async for _ in provider.stream_chat([Message("user", "hi")]):
            pass
    await provider.aclose()

    assert exc_info.value.status_code == 503
    assert exc_info.value.retryable is True
    assert "ollama serve" in str(exc_info.value)
