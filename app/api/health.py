"""Liveness and readiness endpoints.

These are two different questions and conflating them causes real outages:

* **Liveness** (``/health``) -- "is this process alive?" Touches nothing
  external. If it fails, the only fix is to restart the container. Wiring a
  database check into liveness means a brief Postgres blip makes Kubernetes
  kill every healthy app pod, turning a recoverable dependency outage into a
  full restart storm.

* **Readiness** (``/health/ready``) -- "can this process serve traffic right
  now?" Checks the dependencies it cannot work without. Failing readiness pulls
  the pod out of the load-balancer pool but leaves it running, so it rejoins
  automatically once Postgres/Redis recover.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from sqlalchemy import text

from app import __version__
from app.api.deps import AppSettings, DbSession, RedisClient
from app.core.logging import get_logger

log = get_logger(__name__)
router = APIRouter(tags=["health"])

# A dependency check must never outlive the load balancer's own probe timeout,
# or a stuck check leaves the endpoint hanging instead of reporting unhealthy.
CHECK_TIMEOUT_SECONDS = 2.0


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    environment: str


class ReadinessResponse(BaseModel):
    status: str
    checks: dict[str, str]


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health(settings: AppSettings) -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=__version__,
        environment=settings.environment,
    )


async def _check_postgres(session: DbSession) -> str:
    # SELECT 1 verifies the full path: pool checkout, driver, network, and that
    # Postgres is actually accepting queries (not just that the port is open).
    await asyncio.wait_for(session.execute(text("SELECT 1")), CHECK_TIMEOUT_SECONDS)
    return "ok"


async def _check_redis(redis: RedisClient) -> str:
    await asyncio.wait_for(redis.ping(), CHECK_TIMEOUT_SECONDS)
    return "ok"


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    responses={503: {"description": "One or more dependencies are unavailable"}},
)
async def readiness(
    response: Response,
    session: DbSession,
    redis: RedisClient,
) -> ReadinessResponse:
    checks: dict[str, str] = {}

    # Run both checks even if the first fails, so the response reports the full
    # picture ("postgres ok, redis down") instead of only the first problem.
    for name, coro in (
        ("postgres", _check_postgres(session)),
        ("redis", _check_redis(redis)),
    ):
        try:
            checks[name] = await coro
        except Exception as exc:
            # Broad by design: a readiness probe must report "not ready" for any
            # failure mode (timeout, auth error, DNS, driver bug), never 500.
            # asyncio.TimeoutError is an alias of builtin TimeoutError on 3.11+,
            # so it is covered here too.
            checks[name] = f"error: {type(exc).__name__}"
            log.warning("readiness_check_failed", dependency=name, error=str(exc))

    healthy = all(v == "ok" for v in checks.values())
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(status="ready" if healthy else "degraded", checks=checks)
