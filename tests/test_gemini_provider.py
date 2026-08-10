"""Gemini provider tests.

Focused on the four places Gemini's API disagrees with the others, since those
are where the abstraction actually earns its keep. The SDK call is patched at
the boundary; everything above it is production code.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

from app.core.config import Settings
from app.llm.base import Message, StreamDone, TextDelta
from app.llm.errors import (
    ProviderAuthError,
    ProviderBadRequest,
    ProviderNotConfigured,
    ProviderRateLimited,
)
from app.llm.gemini_provider import GeminiProvider


def _settings(**overrides) -> Settings:
    base = {
        "_env_file": None,
        "default_provider": "gemini",
        "gemini_api_key": "test-key",
        "log_level": "CRITICAL",
    }
    return Settings(**{**base, **overrides})


def _chunk(
    text=None, prompt=None, candidates=None, finish=None, model=None, thoughts=None
):
    """Build an object shaped like a GenerateContentResponse chunk."""
    usage = None
    if prompt is not None or candidates is not None:
        usage = SimpleNamespace(
            prompt_token_count=prompt,
            candidates_token_count=candidates,
            thoughts_token_count=thoughts,
        )
    cands = None
    if finish is not None:
        cands = [SimpleNamespace(finish_reason=SimpleNamespace(name=finish))]
    return SimpleNamespace(
        text=text, usage_metadata=usage, candidates=cands, model_version=model
    )


def _patch_stream(monkeypatch, chunks, capture: dict | None = None):
    """Replace the SDK's streaming call with a canned sequence of chunks."""

    # Patched onto the class, so it receives the bound instance as `self`.
    async def fake_stream(self, *, model, contents, config):
        if capture is not None:
            capture["model"] = model
            capture["contents"] = contents
            capture["config"] = config

        async def gen():
            for c in chunks:
                yield c

        return gen()

    monkeypatch.setattr(
        "google.genai.models.AsyncModels.generate_content_stream", fake_stream
    )


async def test_streams_text_and_reports_usage(monkeypatch) -> None:
    _patch_stream(
        monkeypatch,
        [
            _chunk(text="Hel", prompt=10, candidates=1, model="gemini-2.0-flash"),
            _chunk(text="lo", prompt=10, candidates=2),
            _chunk(prompt=10, candidates=2, finish="STOP"),
        ],
    )
    provider = GeminiProvider(_settings())

    events = [e async for e in provider.stream_chat([Message("user", "hi")])]

    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Hello"
    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.model == "gemini-2.0-flash"
    assert done.finish_reason == "stop"


async def test_cumulative_usage_is_not_summed(monkeypatch) -> None:
    """Gemini repeats cumulative totals on every chunk.

    Summing them would report 30 input tokens for a 10-token prompt and inflate
    Phase 6's cost numbers threefold.
    """
    _patch_stream(
        monkeypatch,
        [
            _chunk(text="a", prompt=10, candidates=1),
            _chunk(text="b", prompt=10, candidates=2),
            _chunk(prompt=10, candidates=3, finish="STOP"),
        ],
    )
    provider = GeminiProvider(_settings())

    events = [e async for e in provider.stream_chat([Message("user", "hi")])]
    done = events[-1]

    assert done.usage.input_tokens == 10  # not 30
    assert done.usage.output_tokens == 3  # last value, not 1+2+3


async def test_assistant_role_is_translated_to_model(monkeypatch) -> None:
    """Gemini rejects the literal role "assistant"."""
    captured: dict = {}
    _patch_stream(monkeypatch, [_chunk(text="ok", finish="STOP")], captured)
    provider = GeminiProvider(_settings())

    async for _ in provider.stream_chat(
        [
            Message("user", "hi"),
            Message("assistant", "hello"),
            Message("user", "again"),
        ]
    ):
        pass

    assert [c.role for c in captured["contents"]] == ["user", "model", "user"]


async def test_system_prompt_goes_to_system_instruction(monkeypatch) -> None:
    """It must be lifted out of the message list, like Anthropic."""
    captured: dict = {}
    _patch_stream(monkeypatch, [_chunk(text="ok", finish="STOP")], captured)
    provider = GeminiProvider(_settings())

    async for _ in provider.stream_chat(
        [Message("system", "Be brief."), Message("user", "hi")], max_tokens=64
    ):
        pass

    assert captured["config"].system_instruction == "Be brief."
    assert captured["config"].max_output_tokens == 64
    # The system message must NOT also appear in contents.
    assert all(c.role != "system" for c in captured["contents"])


async def test_chunks_without_text_are_skipped(monkeypatch) -> None:
    """Metadata-only chunks have text=None and must not yield empty deltas."""
    _patch_stream(
        monkeypatch,
        [_chunk(text="hi"), _chunk(text=None, prompt=5, candidates=1, finish="STOP")],
    )
    provider = GeminiProvider(_settings())

    events = [e async for e in provider.stream_chat([Message("user", "x")])]
    deltas = [e for e in events if isinstance(e, TextDelta)]

    assert len(deltas) == 1


async def test_thinking_tokens_are_billed_as_output(monkeypatch) -> None:
    """Gemini 3.x reports reasoning separately, but it is billed as output.

    Counting only `candidates_token_count` under-reports cost by roughly 10x on
    a short answer, which would make Phase 6's cost metrics silently wrong.
    """
    _patch_stream(
        monkeypatch,
        [_chunk(text="hi", prompt=12, candidates=51, thoughts=462, finish="STOP")],
    )
    provider = GeminiProvider(_settings())

    events = [e async for e in provider.stream_chat([Message("user", "x")])]
    done = events[-1]

    assert done.usage.input_tokens == 12
    assert done.usage.output_tokens == 513  # 51 visible + 462 thinking
    assert done.usage.total_tokens == 525


async def test_absent_thinking_count_is_treated_as_zero(monkeypatch) -> None:
    """Non-reasoning models omit the field entirely; it must not crash."""
    _patch_stream(
        monkeypatch,
        [_chunk(text="hi", prompt=10, candidates=5, thoughts=None, finish="STOP")],
    )
    provider = GeminiProvider(_settings())

    events = [e async for e in provider.stream_chat([Message("user", "x")])]

    assert events[-1].usage.output_tokens == 5


async def test_missing_key_is_a_clear_error() -> None:
    with pytest.raises(ProviderNotConfigured, match="GEMINI_API_KEY"):
        GeminiProvider(_settings(gemini_api_key=None))


@pytest.mark.parametrize(
    ("code", "message", "expected"),
    [
        (401, "boom", ProviderAuthError),
        (403, "boom", ProviderAuthError),
        (429, "boom", ProviderRateLimited),  # the free-tier failure mode
        (400, "boom", ProviderBadRequest),
        (404, "boom", ProviderBadRequest),
        # Google returns 400 -- not 401 -- for a bad key. It must still be
        # classified as auth (502), not as the caller's bad request (400).
        (400, "API key not valid. Please pass a valid API key.", ProviderAuthError),
    ],
)
async def test_http_errors_map_to_our_hierarchy(
    monkeypatch, code: int, message: str, expected: type
) -> None:
    async def raiser(*args, **kwargs):
        raise genai_errors.APIError(code, {"error": {"message": message}})

    monkeypatch.setattr(
        "google.genai.models.AsyncModels.generate_content_stream", raiser
    )
    provider = GeminiProvider(_settings())

    with pytest.raises(expected) as exc_info:
        async for _ in provider.stream_chat([Message("user", "hi")]):
            pass

    assert exc_info.value.provider == "gemini"
