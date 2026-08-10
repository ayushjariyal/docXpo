"""Async test fixtures.

Phase 1 tests deliberately run with **no Docker services up**: they exercise the
wiring (app factory, middleware, routing, dependency injection), not Postgres or
Redis themselves. Dependencies are replaced via ``app.dependency_overrides``,
which is FastAPI's supported seam for exactly this.

Phase 7 adds a second tier of integration tests that do talk to real containers.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_redis
from app.core.config import Settings
from app.core.pricing import Pricebook
from app.db.session import get_db_session
from app.main import create_app
from app.services.metrics_service import MetricsService


@pytest.fixture
def settings() -> Settings:
    # Constructed explicitly rather than read from .env so tests are
    # reproducible on any machine and in CI.
    #
    # Rate limiting is off here so these tests exercise what they are actually
    # about. The limiter has its own suite (tests/test_rate_limit.py) which
    # turns it on deliberately.
    return Settings(
        environment="local",
        log_level="WARNING",
        log_json=False,
        rate_limit_enabled=False,
    )


class FakeRedis:
    """Minimal stand-in. Grows as later phases use more of the Redis API."""

    def __init__(self, *, healthy: bool = True) -> None:
        self.healthy = healthy

    async def ping(self) -> bool:
        if not self.healthy:
            raise ConnectionError("redis unavailable")
        return True


class FakeSession:
    def __init__(self, *, healthy: bool = True) -> None:
        self.healthy = healthy

    async def execute(self, *_args, **_kwargs):
        if not self.healthy:
            raise ConnectionError("postgres unavailable")
        return None


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    application = create_app(settings)
    # The lifespan hook normally populates this; ASGITransport does not run
    # lifespan, so we seed it here.
    application.state.redis = FakeRedis()
    # The lifespan hook normally builds this; routes record to it on every
    # request, and its writes are swallowed on failure so no DB is needed.
    application.state.metrics = MetricsService(Pricebook())
    application.dependency_overrides[get_redis] = lambda: FakeRedis()
    application.dependency_overrides[get_db_session] = lambda: FakeSession()
    return application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """Calls the app in-process over ASGI -- no port bound, no network."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
