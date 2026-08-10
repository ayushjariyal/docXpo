"""Declarative base and shared column conventions.

Alembic's autogenerate compares ``Base.metadata`` against the live database, so
every model must inherit from this ``Base`` *and* be imported before Alembic
runs -- see the import in ``app/db/models/__init__.py``.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, MetaData, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Explicit naming convention for constraints and indexes.
#
# Without this, Postgres invents names for unnamed constraints. Alembic then
# generates migrations that say "drop the constraint" without knowing what it is
# called, and `alembic downgrade` breaks. Setting this on day one costs nothing;
# retrofitting it onto a live database is painful.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class UUIDPrimaryKeyMixin:
    """UUID primary keys instead of bigserial.

    Trade-off: UUIDs are wider (16 bytes vs 8) and random ones hurt B-tree
    insert locality. In exchange, ids are non-guessable (a document id in a URL
    doesn't leak how many documents exist) and can be generated client-side
    without a database round trip. For a document/chat service where ids get
    handed to API consumers, that is the right trade.
    """

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )


class TimestampMixin:
    """created_at / updated_at maintained by the database, not Python.

    ``server_default=func.now()`` means the value comes from Postgres' clock.
    That keeps timestamps consistent even if app servers have drifting clocks or
    a row is inserted by a migration or by hand in psql.

    ``DateTime(timezone=True)`` is deliberate: a bare ``Mapped[datetime]`` maps
    to TIMESTAMP WITHOUT TIME ZONE, which silently drops the offset and makes
    every latency calculation in Phase 6 wrong the moment a container runs in a
    different TZ. timestamptz stores a real instant.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
