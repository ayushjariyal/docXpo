"""enable pgvector extension

Revision ID: 0001
Revises:
Create Date: 2026-08-02

The extension is created by a migration rather than by a docker-entrypoint SQL
script so that it is version-controlled and applies identically to a managed
Postgres (RDS/Neon/Supabase) where you cannot drop files into the image.

Requires a role with CREATE privilege on the database; the compose superuser
has it, and managed providers allow `vector` on their extension allowlist.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    # IF EXISTS (not CASCADE): if any table still has a vector column, this
    # should fail loudly rather than silently drop that column.
    op.execute("DROP EXTENSION IF EXISTS vector")
