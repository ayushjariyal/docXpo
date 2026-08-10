"""Integration tier: tests that run against the real Postgres and Redis.

The unit tests everywhere else use fakes — they are fast, hermetic, and verify
*our* logic. They cannot verify the parts we handed to the database:

* the pgvector `<=>` operator and whether the HNSW index is actually used,
* the rate limiter's Lua script, which runs inside Redis and was previously
  only ever exercised by a Python re-implementation of itself,
* the `percentile_cont` aggregation behind `/metrics`.

Those are exactly the places a subtle bug hides, because a re-implementation in
the test can be wrong in the same way the code is. This tier closes that.

Every test runs inside a transaction that is **rolled back**, so the tier leaves
no residue in the developer's database and can be run repeatedly.

If the services are not up, the whole tier skips with a clear reason rather than
failing — `pytest` on a laptop with no Docker should stay green.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.session import build_engine

REASON = (
    "integration services unavailable — start them with "
    "`docker compose up -d postgres redis` and apply migrations"
)


def _settings() -> Settings:
    """Host-side connection details, overridable from the environment.

    Defaults to localhost because the tier runs outside the compose network,
    but reads POSTGRES_HOST / REDIS_HOST so CI can point it at services
    somewhere else -- and so the skip path can be exercised.
    """
    return Settings(
        _env_file=None,
        postgres_host=os.getenv("POSTGRES_HOST", "localhost"),
        redis_host=os.getenv("REDIS_HOST", "localhost"),
        log_level="CRITICAL",
    )


async def _postgres_ready(settings: Settings) -> bool:
    engine = build_engine(settings)
    try:
        async with engine.connect() as conn:
            # Also confirms the migrations ran: without request_logs the
            # aggregation tests would fail confusingly rather than skip.
            await conn.execute(text("SELECT 1 FROM request_logs LIMIT 1"))
            await conn.execute(text("SELECT 1 FROM chunks LIMIT 1"))
        return True
    except Exception:
        return False
    finally:
        await engine.dispose()


async def _redis_ready(settings: Settings) -> bool:
    client = Redis.from_url(settings.redis_url, socket_connect_timeout=2)
    try:
        await client.ping()
        return True
    except Exception:
        return False
    finally:
        await client.aclose()


def _probe(coro_fn) -> bool:
    """Run an async probe from sync collection code."""
    try:
        return asyncio.run(coro_fn(_settings()))
    except Exception:
        return False


HAVE_POSTGRES = _probe(_postgres_ready)
HAVE_REDIS = _probe(_redis_ready)

needs_postgres = pytest.mark.skipif(not HAVE_POSTGRES, reason=REASON)
needs_redis = pytest.mark.skipif(not HAVE_REDIS, reason=REASON)


@pytest.fixture(scope="session")
def settings() -> Settings:
    return _settings()


# loop_scope="function": the project sets a *session*-scoped fixture loop
# (Phase 1, so a shared engine can span tests), but these fixtures hold a
# live asyncpg/redis connection that must live on the same loop as the test
# using it. Mismatched loops surface as "attached to a different loop".
@pytest_asyncio.fixture(loop_scope="function")
async def session(settings: Settings) -> AsyncIterator[AsyncSession]:
    """A session bound to a transaction that is always rolled back.

    Binding the session to an *outer* transaction rather than letting it manage
    its own means even code under test that calls `commit()` (the services all
    do) is contained — that commit lands in the outer transaction, which we then
    discard. Without this, running the tier twice would accumulate rows.
    """
    engine = build_engine(settings)
    conn = await engine.connect()
    trans = await conn.begin()
    factory = async_sessionmaker(bind=conn, expire_on_commit=False)
    db = factory()
    try:
        yield db
    finally:
        await db.close()
        await trans.rollback()
        await conn.close()
        await engine.dispose()


@pytest_asyncio.fixture(loop_scope="function")
async def redis(settings: Settings) -> AsyncIterator[Redis]:
    """Real Redis, with every key this test created removed afterwards."""
    client = Redis.from_url(settings.redis_url, decode_responses=False)
    prefix = "itest:"
    try:
        yield client
    finally:
        keys = [k async for k in client.scan_iter(match=f"*{prefix}*")]
        if keys:
            await client.delete(*keys)
        await client.aclose()
