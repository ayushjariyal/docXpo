"""Ingestion: text in, embedded chunks in Postgres.

The pipeline is deliberately small and linear:

    decode -> chunk -> embed (batched) -> insert -> commit

Everything expensive is the embedding step, which is why it is batched rather
than looped one chunk at a time.
"""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.logging import get_logger
from app.db.models.document import Document
from app.llm.embeddings import EmbeddingProvider
from app.repositories.document_repository import DocumentRepository
from app.services.chunking import chunk_text
from app.services.semantic_cache import SemanticCache

log = get_logger(__name__)

# One embed_content call per slice. Keeps a single request well inside provider
# payload limits while still amortising the round trip over many chunks -- a
# 300-chunk document is 3 calls instead of 300.
EMBED_BATCH_SIZE = 100


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
        chunks = chunk_text(
            content,
            size=self._settings.chunk_size,
            overlap=self._settings.chunk_overlap,
        )
        if not chunks:
            raise EmptyDocumentError("document contains no text to index")

        document = await self._repo.create_document(
            filename=filename, content_type=content_type, content=content
        )

        vectors: list[list[float]] = []
        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[start : start + EMBED_BATCH_SIZE]
            # task="document" matters: Gemini encodes passages and questions
            # asymmetrically, and using the query encoding here would quietly
            # cost recall. See app/llm/embeddings.py.
            vectors.extend(
                await self._embedder.embed([c.text for c in batch], task="document")
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
