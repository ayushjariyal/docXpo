"""Ingestion against real Postgres: text the database itself would reject.

A fake embedder can't tell you this, and neither can a fake repository --
"Postgres refuses NUL in a TEXT column" is a fact about the real database.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.llm.errors import ProviderRateLimited
from app.services.document_service import DocumentService
from tests.integration.conftest import needs_postgres

pytestmark = [pytest.mark.integration, needs_postgres]


class _UnitEmbedder:
    """Every text gets the same unit vector; this test is about storage."""

    dimensions = 768

    async def embed(self, texts, *, task="document"):
        return [[1.0] + [0.0] * 767 for _ in texts]


async def test_text_containing_nul_is_stored_not_rejected(session) -> None:
    """Regression: a PDF whose text layer contained U+0000 made the upload
    500 with 'invalid byte sequence for encoding UTF8: 0x00'."""
    service = DocumentService(session, _UnitEmbedder(), Settings(_env_file=None))

    doc = await service.ingest(
        filename="proposal.pdf",
        content_type="application/pdf",
        content="[page 1]\nImplementing\x00 Dijkstra\x00 driving-distance",
    )

    assert "\x00" not in doc.content
    assert "Implementing Dijkstra driving-distance" in doc.content
    assert doc.chunk_count == 1


class _QuotaExhaustedEmbedder:
    """A daily cap: asks for an hour, which ingestion must not wait out."""

    dimensions = 768

    async def embed(self, texts, *, task="document"):
        raise ProviderRateLimited("daily quota", retry_after=3600)


async def test_failed_embedding_leaves_no_partial_document(session) -> None:
    """Embedding now runs before any INSERT, so giving up writes nothing."""
    service = DocumentService(session, _QuotaExhaustedEmbedder(), Settings(_env_file=None))
    before = await service.count()

    with pytest.raises(ProviderRateLimited):
        await service.ingest(filename="big.txt", content_type="text/plain", content="x " * 5000)

    assert await service.count() == before
