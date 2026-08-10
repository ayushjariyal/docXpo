"""Data access for documents and chunks.

This is the only module that knows SQL. Services call it; it returns ORM
objects and plain dataclasses, never `Result` or `Row`. That boundary is what
lets the vector search be swapped (for a different index, or a hybrid
keyword+vector query) without touching the RAG service.

Note it does not commit -- see `get_db_session` in app/db/session.py for why
transaction control lives with the service.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document import Chunk, Document


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A retrieved chunk plus how well it matched."""

    id: uuid.UUID
    document_id: uuid.UUID
    document_filename: str
    chunk_index: int
    content: str
    similarity: float


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ---- writes ------------------------------------------------------------

    async def create_document(
        self, *, filename: str, content_type: str, content: str
    ) -> Document:
        doc = Document(
            filename=filename,
            content_type=content_type,
            content=content,
            char_count=len(content),
            chunk_count=0,
        )
        self._session.add(doc)
        await self._session.flush()  # assigns the PK without committing
        return doc

    async def add_chunks(
        self,
        document: Document,
        rows: Sequence[tuple[int, str, int, int, list[float]]],
    ) -> None:
        """Bulk-insert chunks. Each row is (index, text, start, end, embedding)."""
        self._session.add_all(
            [
                Chunk(
                    document_id=document.id,
                    chunk_index=i,
                    content=text,
                    char_start=start,
                    char_end=end,
                    embedding=vector,
                )
                for (i, text, start, end, vector) in rows
            ]
        )
        document.chunk_count = len(rows)
        await self._session.flush()

    async def delete_document(self, document_id: uuid.UUID) -> bool:
        # Chunks go via the FK's ON DELETE CASCADE -- one statement, no N+1.
        result = await self._session.execute(
            delete(Document).where(Document.id == document_id)
        )
        return result.rowcount > 0

    # ---- reads -------------------------------------------------------------

    async def get(self, document_id: uuid.UUID) -> Document | None:
        return await self._session.get(Document, document_id)

    async def list_documents(self, *, limit: int = 50, offset: int = 0) -> list[Document]:
        result = await self._session.execute(
            select(Document)
            .order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars())

    async def count_documents(self) -> int:
        return int(await self._session.scalar(select(func.count(Document.id))) or 0)

    async def search(
        self,
        embedding: list[float],
        *,
        top_k: int = 5,
        min_similarity: float = 0.0,
        document_id: uuid.UUID | None = None,
    ) -> list[ScoredChunk]:
        """Top-k nearest chunks by cosine similarity.

        `<=>` is pgvector's **cosine distance** operator, in [0, 2]. Similarity
        is `1 - distance`, so ordering by distance ascending is the same as
        ordering by similarity descending.

        Two things make this actually use the HNSW index:

        * ordering by the raw `embedding <=> :q` expression -- wrapping it in
          arithmetic (e.g. `ORDER BY 1 - (embedding <=> :q) DESC`) makes the
          planner fall back to a sequential scan;
        * a LIMIT -- an unbounded ANN query has no meaning to the index.

        So the similarity floor is applied *after* the ordered fetch rather than
        as a WHERE clause, which would also defeat the index.
        """
        distance = Chunk.embedding.cosine_distance(embedding)

        stmt = (
            select(
                Chunk.id,
                Chunk.document_id,
                Document.filename,
                Chunk.chunk_index,
                Chunk.content,
                distance.label("distance"),
            )
            .join(Document, Document.id == Chunk.document_id)
            .order_by(distance)
            .limit(top_k)
        )
        if document_id is not None:
            stmt = stmt.where(Chunk.document_id == document_id)

        rows = (await self._session.execute(stmt)).all()

        scored = [
            ScoredChunk(
                id=r.id,
                document_id=r.document_id,
                document_filename=r.filename,
                chunk_index=r.chunk_index,
                content=r.content,
                similarity=1.0 - float(r.distance),
            )
            for r in rows
        ]
        return [c for c in scored if c.similarity >= min_similarity]
