"""Aggregated metrics.

JSON rather than Prometheus exposition format, deliberately: this is meant to
be read by a human debugging cost and latency, and the numbers that matter here
(cost in USD, cache-hit rate) are aggregates over a window rather than the
monotonic counters Prometheus scrapes. Adding a `/metrics/prometheus` alongside
this would be straightforward if a scraper ever needed one.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.deps import DbSession
from app.services.metrics_service import MetricsRepository

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("", summary="Aggregate usage, cost and latency")
async def metrics(
    session: DbSession,
    # 0 means "all time". Bounded above so a caller cannot ask for an
    # unindexed full-table percentile by accident.
    window_hours: int = Query(24, ge=0, le=24 * 90),
) -> dict:
    repo = MetricsRepository(session)
    window = window_hours or None

    return {
        "summary": await repo.summary(window_hours=window),
        "by_model": await repo.by_provider(window_hours=window),
    }


@router.get("/recent", summary="Most recent requests")
async def recent(session: DbSession, limit: int = Query(20, ge=1, le=200)) -> dict:
    return {"requests": await MetricsRepository(session).recent(limit=limit)}
