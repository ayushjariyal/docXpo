"""Ingestion: text in, embedded chunks in Postgres.

The pipeline is deliberately small and linear:

    clean -> chunk -> embed (batched, rate-limit aware) -> insert -> commit

Everything expensive is the embedding step, which is why it is batched rather
than looped one chunk at a time -- and why it runs *before* anything touches the
database. On a free embedding tier a large document can spend minutes waiting
out rate limits; holding a transaction (and a pooled connection) open for that
long would be wasteful on any database and fatal on a serverless one.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.logging import get_logger
from app.db.models.document import Document
from app.llm.embeddings import EmbeddingProvider
from app.llm.errors import ProviderRateLimited
from app.repositories.document_repository import DocumentRepository
from app.services.chunking import chunk_text
from app.services.extraction import clean_text
from app.services.semantic_cache import SemanticCache

log = get_logger(__name__)

# One embed_content call per slice. Keeps a single request well inside provider
# payload limits while still amortising the round trip over many chunks -- a
# 300-chunk document is 3 calls instead of 300.
EMBED_BATCH_SIZE = 100

# Waiting out embedding rate limits. Gemini's free tier allows 100 embeddings a
# minute and counts every chunk in a batch, so a 1,000-chunk document needs ten
# windows. A per-minute limit says "retry in ~30s"; anything far longer than a
# minute is a daily cap, and waiting it out inside an HTTP request is not an
# option -- so that one fails immediately instead.
EMBED_MAX_RETRIES = 5          # consecutive 429s on one batch before giving up
EMBED_DEFAULT_WAIT_S = 60.0    # when the provider doesn't say how long
EMBED_GIVE_UP_AFTER_S = 120.0  # a longer requested wait means a daily cap


class EmptyDocumentError(ValueError):
    """Upload contained no extractable text."""


class DocumentService:
    def __init__(
        self,
        session: AsyncSession,
        embedder: EmbeddingProvider,
        settings: Settings,
        cache: SemanticCache | None = None,
    ) -> None:
        self._session = session
        self._repo = DocumentRepository(session)
        self._embedder = embedder
        self._settings = settings
        self._cache = cache

    async def ingest(
        self, *, filename: str, content_type: str, content: str
    ) -> Document:
        # Here rather than in the upload route: this is the one place every
        # ingestion path (file upload, PDF, pasted text) passes through.
        content = clean_text(content)
        chunks = chunk_text(
            content,
            size=self._settings.chunk_size,
            overlap=self._settings.chunk_overlap,
        )
        if not chunks:
            raise EmptyDocumentError("document contains no text to index")

        # Embed everything first, with no transaction open (see module docstring).
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[start : start + EMBED_BATCH_SIZE]
            vectors.extend(
                await self._embed_batch([c.text for c in batch], filename=filename)
            )

        document = await self._repo.create_document(
            filename=filename, content_type=content_type, content=content
        )
        await self._repo.add_chunks(
            document,
            [
                (c.index, c.text, c.char_start, c.char_end, v)
                for c, v in zip(chunks, vectors, strict=True)
            ],
        )
        # The service owns the transaction boundary: the document and all of its
        # chunks commit together, so a failure part-way through embedding never
        # leaves a document with a partial index.
        await self._session.commit()

        # Every cached RAG answer was derived from the *previous* corpus, so it
        # is now potentially stale. Bumping the version changes the cache
        # namespace, which orphans them all in O(1) -- no scan-and-delete.
        await self._invalidate_cache()

        log.info(
            "document_ingested",
            document_id=str(document.id),
            filename=filename,
            chars=len(content),
            chunks=len(chunks),
        )
        return document

    async def _embed_batch(
        self, texts: Sequence[str], *, filename: str
    ) -> list[list[float]]:
        """Embed one batch, waiting out per-minute rate limits.

        Retries only `ProviderRateLimited`: a per-minute window reopens on its
        own, so waiting is the correct response. Every other provider error
        (bad key, bad request, outage) is raised at once -- waiting would not
        fix it.
        """
        for attempt in range(EMBED_MAX_RETRIES + 1):
            try:
                # task="document" matters: Gemini encodes passages and questions
                # asymmetrically, and using the query encoding here would quietly
                # cost recall. See app/llm/embeddings.py.
                return await self._embedder.embed(texts, task="document")
            except ProviderRateLimited as exc:
                requested = exc.retry_after
                if attempt == EMBED_MAX_RETRIES or (
                    requested is not None and requested > EMBED_GIVE_UP_AFTER_S
                ):
                    raise
                # +1s so we land just after the window reopens, not just before.
                wait = (requested if requested is not None else EMBED_DEFAULT_WAIT_S) + 1
                log.warning(
                    "embedding_rate_limited_waiting",
                    filename=filename,
                    attempt=attempt + 1,
                    wait_seconds=round(wait, 1),
                )
                await asyncio.sleep(wait)
        raise AssertionError("unreachable")  # the loop always returns or raises

    async def _invalidate_cache(self) -> None:
        if self._cache is not None:
            await self._cache.bump_corpus_version()

    async def list_documents(self, *, limit: int = 50, offset: int = 0):
        return await self._repo.list_documents(limit=limit, offset=offset)

    async def count(self) -> int:
        return await self._repo.count_documents()

    async def get(self, document_id: uuid.UUID) -> Document | None:
        return await self._repo.get(document_id)

    async def delete(self, document_id: uuid.UUID) -> bool:
        deleted = await self._repo.delete_document(document_id)
        await self._session.commit()
        if deleted:
            # Same reasoning as ingest: answers cached from a corpus that
            # included this document must not be served any more.
            await self._invalidate_cache()
            log.info("document_deleted", document_id=str(document_id))
        return deleted
