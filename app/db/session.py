"""Async engine, session factory, and the FastAPI session dependency."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings, get_settings


def build_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(
        settings.database_url,
        echo=settings.db_echo,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        # Recycle connections before Postgres/a proxy silently kills idle ones.
        pool_recycle=1800,
        # Cheap liveness check on checkout. Costs a round trip but turns
        # "connection was closed by the server" 500s into a transparent reconnect.
        pool_pre_ping=True,
    )


# Module-level singletons: one engine (and therefore one connection pool) per
# process. Creating an engine per request would open a fresh pool every time --
# a classic way to exhaust Postgres' connection limit.
engine: AsyncEngine = build_engine(get_settings())

SessionFactory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    # Without this, accessing any attribute of an ORM object after commit()
    # triggers a lazy refresh -- which in async code raises MissingGreenlet.
    # Turning it off lets us return ORM objects from a service after committing.
    expire_on_commit=False,
    autoflush=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a session scoped to one request.

    Transaction policy: this dependency owns *rollback*, not commit. Services
    decide when a unit of work is complete and call ``commit()`` themselves,
    because only the service knows whether two writes belong in one transaction.
    The rollback here is the safety net for an exception escaping the endpoint.
    """
    async with SessionFactory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def dispose_engine() -> None:
    """Close every pooled connection. Called on application shutdown."""
    await engine.dispose()
