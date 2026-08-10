"""Smoke tests for the Phase 1 skeleton."""

from __future__ import annotations

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app import __version__
from app.api.deps import get_redis
from app.core.middleware import REQUEST_ID_HEADER
from app.db.session import get_db_session
from tests.conftest import FakeRedis, FakeSession


async def test_liveness_returns_ok(client: AsyncClient) -> None:
    resp = await client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__


async def test_liveness_does_not_touch_dependencies(app: FastAPI) -> None:
    """Liveness must stay green even when Postgres and Redis are both down.

    This is the property that stops a dependency outage from triggering a
    container restart storm.
    """
    app.dependency_overrides[get_redis] = lambda: FakeRedis(healthy=False)
    app.dependency_overrides[get_db_session] = lambda: FakeSession(healthy=False)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/health")

    assert resp.status_code == 200


async def test_readiness_ok_when_dependencies_healthy(client: AsyncClient) -> None:
    resp = await client.get("/health/ready")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ready", "checks": {"postgres": "ok", "redis": "ok"}}


async def test_readiness_503_when_redis_down(app: FastAPI) -> None:
    app.dependency_overrides[get_redis] = lambda: FakeRedis(healthy=False)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/health/ready")

    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    # Postgres is still reported, proving both checks ran rather than
    # short-circuiting on the first failure.
    assert body["checks"]["postgres"] == "ok"
    assert body["checks"]["redis"].startswith("error:")


async def test_request_id_is_returned(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.headers.get(REQUEST_ID_HEADER)


async def test_upstream_request_id_is_preserved(client: AsyncClient) -> None:
    resp = await client.get("/health", headers={REQUEST_ID_HEADER: "trace-abc-123"})
    assert resp.headers[REQUEST_ID_HEADER] == "trace-abc-123"
