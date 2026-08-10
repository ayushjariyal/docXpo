"""Phase 2 tests: the provider abstraction and the SSE framing.

No real provider is involved. A FakeProvider implements the same interface,
which is exactly the point of having the interface -- if these tests needed
Ollama running, the abstraction would not be doing its job.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_chat_service
from app.llm.base import LLMProvider, Message, StreamDone, StreamEvent, TextDelta, Usage
from app.llm.errors import ProviderRateLimited, ProviderUnavailable
from app.services.chat_service import ChatService


class FakeProvider(LLMProvider):
    name = "fake"

    def __init__(
        self,
        chunks: Sequence[str] = ("Hello", ", ", "world"),
        *,
        fail_before_stream: Exception | None = None,
        fail_after: int | None = None,
    ) -> None:
        self._chunks = list(chunks)
        self._fail_before_stream = fail_before_stream
        self._fail_after = fail_after
        self.closed = False

    @property
    def default_model(self) -> str:
        return "fake-model-1"

    def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[StreamEvent]:
        if self._fail_before_stream is not None:
            raise self._fail_before_stream
        for i, chunk in enumerate(self._chunks):
            if self._fail_after is not None and i == self._fail_after:
                raise ProviderUnavailable("upstream died", provider=self.name)
            yield TextDelta(chunk)
        yield StreamDone(
            model=self.default_model,
            usage=Usage(input_tokens=7, output_tokens=3),
            finish_reason="stop",
        )

    async def aclose(self) -> None:
        self.closed = True


class FakeRegistry:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    async def get(self, name: str | None = None) -> LLMProvider:
        return self.provider


def _install(app: FastAPI, provider: LLMProvider) -> None:
    service = ChatService(FakeRegistry(provider))  # type: ignore[arg-type]
    app.dependency_overrides[get_chat_service] = lambda: service


def parse_sse(body: str) -> list[tuple[str, str]]:
    """Split a raw SSE body into (event, data) pairs."""
    frames = []
    for block in body.strip().split("\n\n"):
        if not block.strip():
            continue
        event = data = ""
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = line.removeprefix("data: ")
        frames.append((event, data))
    return frames


# --- the abstraction itself ------------------------------------------------


async def test_complete_is_derived_from_the_stream() -> None:
    """complete() must accumulate exactly what stream_chat produced."""
    result = await FakeProvider(["a", "b", "c"]).complete([Message("user", "hi")])

    assert result.text == "abc"
    assert result.usage.total_tokens == 10
    assert result.finish_reason == "stop"


async def test_split_system_extracts_system_messages() -> None:
    from app.llm.base import split_system

    system, rest = split_system(
        [
            Message("system", "Be brief."),
            Message("user", "hi"),
            Message("system", "Be polite."),
        ]
    )

    # Both system messages survive, joined -- none is silently dropped.
    assert system == "Be brief.\n\nBe polite."
    assert [m.role for m in rest] == ["user"]


# --- SSE framing -----------------------------------------------------------


def test_newlines_in_a_token_do_not_break_the_frame() -> None:
    """The bug that JSON-encoding the data field exists to prevent.

    A raw `data: {text}` write would split this token across two frames and
    desynchronise every subsequent event.
    """
    from app.api.v1.routes.chat import sse

    frame = sse("token", {"text": "line1\nline2"})

    assert frame.count("\n\n") == 1  # exactly one frame terminator
    assert frame.endswith("\n\n")
    body_lines = [ln for ln in frame.split("\n") if ln]
    assert len(body_lines) == 2  # "event: token" and one single-line "data: ..."


async def test_stream_emits_tokens_then_done(app: FastAPI) -> None:
    _install(app, FakeProvider(["Hel", "lo"]))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        resp = await ac.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["x-accel-buffering"] == "no"

    frames = parse_sse(resp.text)
    assert [e for e, _ in frames] == ["token", "token", "done"]

    import json

    assert json.loads(frames[0][1])["text"] == "Hel"
    done = json.loads(frames[-1][1])
    assert done["usage"] == {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
    assert done["model"] == "fake-model-1"
    assert done["latency_ms"] >= 0


# --- error handling --------------------------------------------------------


async def test_failure_before_first_token_is_a_real_http_status(app: FastAPI) -> None:
    """Priming the stream is what makes this a 429 instead of `200 + error`."""
    _install(app, FakeProvider(fail_before_stream=ProviderRateLimited("slow down")))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        resp = await ac.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]})

    assert resp.status_code == 429
    assert resp.json()["detail"]["retryable"] is True


async def test_failure_mid_stream_becomes_an_error_event(app: FastAPI) -> None:
    """After headers are sent the status can't change, so errors go in-band."""
    _install(app, FakeProvider(["one", "two", "three"], fail_after=2))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        resp = await ac.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]})

    assert resp.status_code == 200  # already committed before the failure
    frames = parse_sse(resp.text)
    assert [e for e, _ in frames] == ["token", "token", "error"]

    import json

    assert json.loads(frames[-1][1])["type"] == "ProviderUnavailable"


# --- request validation ----------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"messages": []},
        {"messages": [{"role": "system", "content": "only a system prompt"}]},
        {"messages": [{"role": "user", "content": ""}]},
        {"messages": [{"role": "user", "content": "hi"}], "provider": "nope"},
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 0},
    ],
)
async def test_invalid_requests_are_rejected(app: FastAPI, payload: dict) -> None:
    _install(app, FakeProvider())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        resp = await ac.post("/v1/chat", json=payload)

    assert resp.status_code == 422
