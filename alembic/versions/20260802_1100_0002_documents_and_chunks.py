"""documents and chunks with pgvector HNSW index

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIM = 768


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("content_type", sa.String(length=128), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_documents"),
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_chunks"),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_chunks_document_id_documents",
            # Deleting a document must take its chunks with it. Enforced by the
            # database, not the ORM, so it holds for a manual `DELETE` in psql.
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_chunks_doc_index"),
    )
    op.create_index(
        "ix_chunks_document_id_chunk_index", "chunks", ["document_id", "chunk_index"]
    )

    # ---- The vector index --------------------------------------------------
    # HNSW rather than IVFFlat:
    #
    #   * IVFFlat must be built on a table that already contains a
    #     representative sample of rows -- it clusters existing data to build
    #     its lists. Creating it in a migration on an empty table gives awful
    #     recall until you rebuild it, which is a footgun in a project where
    #     documents arrive after deploy.
    #   * HNSW builds a graph incrementally, so it is correct from row one and
    #     stays correct as rows are inserted. Better recall/speed too.
    #
    # The cost is a slower build and more memory. For a corpus of this size that
    # is irrelevant; at tens of millions of rows IVFFlat becomes worth revisiting.
    #
    # vector_cosine_ops matches the `<=>` operator used by the repository. The
    # operator class and the query operator MUST agree -- using `<->` (L2)
    # against a cosine index means Postgres silently ignores the index and
    # sequentially scans the table.
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw")
    op.drop_index("ix_chunks_document_id_chunk_index", table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("documents")
