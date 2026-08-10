"""Recording and aggregating per-request metrics.

## Why the recorder owns its own DB session

Recording happens *after* the response body has finished streaming, at which
point FastAPI has already torn down the request's dependencies — including the
`AsyncSession` the route was given. Reusing it would raise, intermittently and
confusingly. So `record()` opens a short-lived session from the factory instead
of accepting one.

## Why a failed write is swallowed

Observability must never break the thing it observes. If Postgres is down, the
user's answer has already been delivered; refusing to return it because we
could not file a metrics row would turn a monitoring outage into a service
outage. The failure is logged loudly and dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import Numeric, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.core.pricing import Pricebook
from app.db.models.request_log import RequestLog
from app.db.session import SessionFactory

log = get_logger(__name__)


@dataclass(slots=True)
class RequestRecord:
    endpoint: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    cached: bool = False
    status_code: int = 200
    finish_reason: str | None = None
    error_type: str | None = None
    request_id: str | None = None
    principal: str | None = None
    # On a cache hit, the token counts the stored answer originally cost. Used
    # to price what the hit *avoided* spending.
    saved_input_tokens: int = 0
    saved_output_tokens: int = 0


class MetricsService:
    def __init__(self, pricebook: Pricebook) -> None:
        self._pricebook = pricebook

    async def record(self, rec: RequestRecord) -> None:
        try:
            if rec.cached:
                # A hit spends nothing; what matters is what it saved.
                cost, priced = Decimal("0"), True
                saved, _ = self._pricebook.cost(
                    rec.model, rec.saved_input_tokens, rec.saved_output_tokens
                )
            else:
                cost, priced = self._pricebook.cost(
                    rec.model, rec.input_tokens, rec.output_tokens
                )
                saved = Decimal("0")

            async with SessionFactory() as session:
                session.add(
                    RequestLog(
                        endpoint=rec.endpoint,
                        request_id=rec.request_id,
                        principal=rec.principal,
                        provider=rec.provider,
                        model=rec.model,
                        input_tokens=rec.input_tokens,
                        output_tokens=rec.output_tokens,
                        latency_ms=Decimal(str(round(rec.latency_ms, 2))),
                        ttft_ms=(
                            Decimal(str(round(rec.ttft_ms, 2)))
                            if rec.ttft_ms is not None
                            else None
                        ),
                        cached=rec.cached,
                        cost_usd=cost,
                        cost_saved_usd=saved,
                        priced=priced,
                        status_code=rec.status_code,
                        finish_reason=rec.finish_reason,
                        error_type=rec.error_type,
                    )
                )
                await session.commit()
        except Exception as exc:  # noqa: BLE001
            # See module docstring: never let telemetry break the request.
            log.error(
                "metrics_record_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                endpoint=rec.endpoint,
            )


class MetricsRepository:
    """Aggregation queries. Kept separate from recording: different lifetimes,
    different session (this one comes from the request), different concerns."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def summary(self, *, window_hours: int | None = 24) -> dict:
        since = (
            datetime.now(UTC) - timedelta(hours=window_hours)
            if window_hours is not None
            else None
        )

        def scoped(stmt):
            return stmt.where(RequestLog.created_at >= since) if since else stmt

        # percentile_cont is an *exact* percentile: Postgres sorts the matching
        # rows and interpolates. That is correct but O(n log n) over the window,
        # which is fine at this scale. At high volume the usual move is a
        # streaming approximation (t-digest / HdrHistogram) or pre-aggregated
        # buckets -- accepting ~1% error for O(1) memory.
        def pct(p: float):
            return func.percentile_cont(p).within_group(
                cast(RequestLog.latency_ms, Numeric).asc()
            )

        row = (
            await self._session.execute(
                scoped(
                    select(
                        func.count().label("requests"),
                        func.coalesce(func.sum(RequestLog.input_tokens), 0),
                        func.coalesce(func.sum(RequestLog.output_tokens), 0),
                        func.coalesce(func.sum(RequestLog.cost_usd), 0),
                        func.coalesce(func.sum(RequestLog.cost_saved_usd), 0),
                        func.count().filter(RequestLog.cached.is_(True)),
                        func.count().filter(RequestLog.error_type.isnot(None)),
                        func.count().filter(RequestLog.priced.is_(False)),
                        pct(0.50),
                        pct(0.95),
                        pct(0.99),
                        func.min(RequestLog.created_at),
                        func.max(RequestLog.created_at),
                    )
                )
            )
        ).one()

        (
            requests, tok_in, tok_out, cost, saved,
            hits, errors, unpriced, p50, p95, p99, first_seen, last_seen,
        ) = row

        # Cache-hit rate is computed here rather than stored, because it is a
        # ratio over a window -- it has no meaning on a single row.
        hit_rate = (hits / requests) if requests else 0.0

        return {
            "window_hours": window_hours,
            "requests": requests,
            "cache_hits": hits,
            "cache_hit_rate": round(hit_rate, 4),
            "errors": errors,
            "error_rate": round(errors / requests, 4) if requests else 0.0,
            "tokens": {
                "input": int(tok_in),
                "output": int(tok_out),
                "total": int(tok_in) + int(tok_out),
            },
            "cost_usd": {
                "spent": str(Decimal(cost).quantize(Decimal("0.000001"))),
                "saved_by_cache": str(Decimal(saved).quantize(Decimal("0.000001"))),
                # Honest reporting: how many rows we could not price at all.
                # Without this, an unpriced model looks free.
                "unpriced_requests": unpriced,
            },
            "latency_ms": {
                "p50": float(p50) if p50 is not None else None,
                "p95": float(p95) if p95 is not None else None,
                "p99": float(p99) if p99 is not None else None,
            },
            "first_seen": first_seen.isoformat() if first_seen else None,
            "last_seen": last_seen.isoformat() if last_seen else None,
        }

    async def by_provider(self, *, window_hours: int | None = 24) -> list[dict]:
        since = (
            datetime.now(UTC) - timedelta(hours=window_hours)
            if window_hours is not None
            else None
        )
        stmt = select(
            RequestLog.provider,
            RequestLog.model,
            func.count().label("requests"),
            func.coalesce(func.sum(RequestLog.cost_usd), 0),
            func.count().filter(RequestLog.cached.is_(True)),
            func.percentile_cont(0.95).within_group(
                cast(RequestLog.latency_ms, Numeric).asc()
            ),
        ).group_by(RequestLog.provider, RequestLog.model)

        if since:
            stmt = stmt.where(RequestLog.created_at >= since)

        rows = (await self._session.execute(stmt.order_by(func.count().desc()))).all()
        return [
            {
                "provider": r[0],
                "model": r[1],
                "requests": r[2],
                "cost_usd": str(Decimal(r[3]).quantize(Decimal("0.000001"))),
                "cache_hits": r[4],
                "p95_latency_ms": float(r[5]) if r[5] is not None else None,
            }
            for r in rows
        ]

    async def recent(self, *, limit: int = 20) -> list[dict]:
        rows = (
            await self._session.execute(
                select(RequestLog).order_by(RequestLog.created_at.desc()).limit(limit)
            )
        ).scalars()
        return [
            {
                "created_at": r.created_at.isoformat(),
                "endpoint": r.endpoint,
                "provider": r.provider,
                "model": r.model,
                "cached": r.cached,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "latency_ms": float(r.latency_ms),
                "cost_usd": str(r.cost_usd),
                "status_code": r.status_code,
                "request_id": r.request_id,
            }
            for r in rows
        ]
