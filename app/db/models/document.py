"""Document and Chunk tables.

A Document is what the user uploaded; a Chunk is a retrievable slice of it with
its embedding. Retrieval only ever searches Chunks -- the Document row exists to
group them, carry the filename/metadata, and make deletion a single statement.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    pass

# Must equal Settings.embedding_dimensions. It is a literal here because a
# Postgres column width is fixed at migration time -- it cannot be read from
# config at runtime. Changing it means a migration AND re-embedding every row,
# which is exactly why the value is called out in DECISIONS.md.
EMBEDDING_DIM = 768


class Document(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "documents"

    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str] = mapped_column(String(128), default="text/plain")
    # The full original text. Kept so documents can be re-chunked with different
    # parameters without asking the user to upload again -- chunking strategy is
    # exactly the sort of thing you tune after seeing real retrieval quality.
    content: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document",
        # Postgres does the cascade via the FK; passive_deletes stops SQLAlchemy
        # from first SELECTing every chunk in order to delete them one by one.
        cascade="all, delete",
        passive_deletes=True,
    )


class Chunk(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "chunks"

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Position within the document, so retrieved chunks can be shown in order
    # and neighbours can be fetched later ("expand context around this hit").
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_doc_index"),
        # Plain B-tree for "give me this document's chunks in order".
        Index("ix_chunks_document_id_chunk_index", "document_id", "chunk_index"),
    )
