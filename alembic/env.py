"""Alembic environment, wired for an async engine.

Alembic's migration runner is synchronous, but our driver (asyncpg) is not.
The bridge is ``connection.run_sync(...)``: we open a real async connection,
then hand it to Alembic's sync ``do_run_migrations`` through SQLAlchemy's
greenlet adapter. This is why the whole thing is wrapped in ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from app.core.config import get_settings

# Importing the model registry is what populates Base.metadata. Without it,
# autogenerate sees zero tables and cheerfully writes a migration that drops
# everything.
from app.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Inject the URL from Settings rather than alembic.ini. Escape '%' because
# ConfigParser treats it as interpolation syntax and passwords may contain it.
config.set_main_option("sqlalchemy.url", get_settings().database_url.replace("%", "%%"))

target_metadata = Base.metadata


def _include_object(obj, name, type_, reflected, compare_to) -> bool:
    """Filter for autogenerate.

    pgvector creates no tables of its own, but extensions and any tables created
    outside our metadata (e.g. by a future extension) should never be proposed
    for deletion. Extend this if you add unmanaged tables.
    """
    return True


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of executing it (``alembic upgrade --sql``).

    Useful when a DBA must review or apply migrations by hand in production.
    """
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # Detect column type changes (e.g. VARCHAR(50) -> VARCHAR(255)).
        # Off by default in Alembic, which surprises people.
        compare_type=True,
        compare_server_default=True,
        include_object=_include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        # NullPool: a migration process is short-lived and single-use. Pooling
        # would just leave idle connections open after the migration finishes.
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
