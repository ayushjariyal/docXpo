"""Ingestion waits out per-minute embedding limits instead of failing.

Gemini's free tier allows 100 embeddings a minute and counts every chunk, so a
large document *will* hit a 429 mid-ingest. These tests pin the policy: wait
when the window will reopen soon, give up at once when it won't, and never
retry an error that waiting cannot fix.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.llm.embeddings import parse_retry_after
from app.llm.errors import ProviderAuthError, ProviderRateLimited
from app.services import document_service as ds
from app.services.document_service import DocumentService

GOOGLE_429 = (
    "You exceeded your current quota, please check your plan and billing details. "
    "* Quota exceeded for metric: generativelanguage.googleapis.com/"
    "embed_content_free_tier_requests, limit: 100, model: gemini-embedding-1.0\n"
    "Please retry in 27.563311189s."
)


def test_retry_after_is_read_from_googles_message() -> None:
    assert parse_retry_after(GOOGLE_429) == pytest.approx(27.563311189)


def test_retry_after_is_none_when_not_stated() -> None:
    assert parse_retry_after("quota exceeded") is None


class _FlakyEmbedder:
    """Raises the queued errors first, then succeeds."""

    def __init__(self, *errors: Exception) -> None:
        self.errors = list(errors)
        self.calls = 0

    async def embed(self, texts, *, task="document"):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return [[1.0, 0.0] for _ in texts]


@pytest.fixture
def waits(monkeypatch) -> list[float]:
    """Record sleeps instead of actually sleeping."""
    seen: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        seen.append(seconds)

    monkeypatch.setattr(ds.asyncio, "sleep", fake_sleep)
    return seen


def _service(embedder) -> DocumentService:
    # _embed_batch never touches the session, so None is safe here.
    return DocumentService(None, embedder, Settings(_env_file=None))  # type: ignore[arg-type]


async def test_per_minute_limit_is_waited_out(waits) -> None:
    emb = _FlakyEmbedder(ProviderRateLimited("429", retry_after=27.5))

    vectors = await _service(emb)._embed_batch(["a", "b"], filename="big.txt")

    assert vectors == [[1.0, 0.0], [1.0, 0.0]]
    assert emb.calls == 2
    assert waits == [28.5]  # the requested wait plus a one-second margin


async def test_default_wait_when_provider_does_not_say(waits) -> None:
    emb = _FlakyEmbedder(ProviderRateLimited("429"))

    await _service(emb)._embed_batch(["a"], filename="x.txt")

    assert waits == [ds.EMBED_DEFAULT_WAIT_S + 1]


async def test_daily_cap_fails_at_once_instead_of_hanging(waits) -> None:
    emb = _FlakyEmbedder(ProviderRateLimited("429", retry_after=3600))

    with pytest.raises(ProviderRateLimited):
        await _service(emb)._embed_batch(["a"], filename="x.txt")

    assert waits == []
    assert emb.calls == 1


async def test_gives_up_after_max_consecutive_limits(waits) -> None:
    errors = [ProviderRateLimited("429", retry_after=1) for _ in range(ds.EMBED_MAX_RETRIES + 1)]
    emb = _FlakyEmbedder(*errors)

    with pytest.raises(ProviderRateLimited):
        await _service(emb)._embed_batch(["a"], filename="x.txt")

    assert len(waits) == ds.EMBED_MAX_RETRIES
    assert emb.calls == ds.EMBED_MAX_RETRIES + 1


async def test_non_quota_errors_are_never_retried(waits) -> None:
    emb = _FlakyEmbedder(ProviderAuthError("bad key"))

    with pytest.raises(ProviderAuthError):
        await _service(emb)._embed_batch(["a"], filename="x.txt")

    assert waits == []
