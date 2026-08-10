"""The rate limiter's Lua script and the metrics aggregation SQL, for real.

Both were previously covered only by a *re-implementation* of themselves — a
Python token bucket in the test, and no coverage at all for `percentile_cont`.
A re-implementation can be wrong in exactly the same way the real thing is, so
those tests could never have caught a bug in the Lua or the SQL. These run the
actual artefacts.
"""

from __future__ import annotations

import asyncio
import uuid
from decimal import Decimal

import pytest
from sqlalchemy import delete

from app.core.rate_limit import RateLimiter
from app.services.metrics_service import MetricsRepository
from tests.integration.conftest import needs_postgres, needs_redis

pytestmark = pytest.mark.integration


# ---- the Lua script -------------------------------------------------------


@needs_redis
async def test_lua_enforces_the_burst(redis) -> None:
    key = f"itest:{uuid.uuid4().hex}"
    limiter = RateLimiter(redis, requests_per_minute=60, burst=3)

    verdicts = [(await limiter.check(key)).allowed for _ in range(5)]

    assert verdicts[:3] == [True, True, True]
    assert verdicts[3] is False


@needs_redis
async def test_lua_refills_continuously(redis) -> None:
    """A token bucket refills smoothly, not in fixed ticks."""
    key = f"itest:{uuid.uuid4().hex}"
    # 600 rpm = 10 tokens/sec, i.e. one token per 100ms. Deliberately slower
    # than it needs to be: at 100 tokens/sec the three setup round trips to
    # Redis (~10ms over Docker) refill a whole token by themselves, and the
    # test races its own network latency.
    limiter = RateLimiter(redis, requests_per_minute=600, burst=2)

    await limiter.check(key)
    await limiter.check(key)
    assert not (await limiter.check(key)).allowed

    await asyncio.sleep(0.3)

    assert (await limiter.check(key)).allowed


@needs_redis
async def test_lua_is_atomic_under_concurrency(redis) -> None:
    """The reason the arithmetic lives in Lua at all.

    Fire 30 simultaneous requests at a bucket of 5. If refill/check/decrement
    were three separate round trips, several would interleave between the read
    and the write and more than 5 would be admitted. Redis runs the script
    atomically on a single thread, so exactly 5 may pass.
    """
    key = f"itest:{uuid.uuid4().hex}"
    limiter = RateLimiter(redis, requests_per_minute=60, burst=5)

    results = await asyncio.gather(*(limiter.check(key) for _ in range(30)))

    assert sum(r.allowed for r in results) == 5


@needs_redis
async def test_lua_reports_a_usable_retry_after(redis) -> None:
    key = f"itest:{uuid.uuid4().hex}"
    limiter = RateLimiter(redis, requests_per_minute=60, burst=1)
    await limiter.check(key)

    denied = await limiter.check(key)

    assert not denied.allowed
    # Never zero: a Retry-After of 0 invites an immediate, certain-to-fail retry.
    assert denied.retry_after_seconds >= 1
    assert denied.remaining == 0


@needs_redis
async def test_lua_isolates_buckets_per_key(redis) -> None:
    a, b = f"itest:{uuid.uuid4().hex}", f"itest:{uuid.uuid4().hex}"
    limiter = RateLimiter(redis, requests_per_minute=60, burst=2)

    await limiter.check(a)
    await limiter.check(a)

    assert not (await limiter.check(a)).allowed
    assert (await limiter.check(b)).allowed


@needs_redis
async def test_lua_sets_a_ttl_so_idle_buckets_evaporate(redis) -> None:
    """Without PEXPIRE, Redis accumulates a key per API key forever."""
    key = f"itest:{uuid.uuid4().hex}"
    limiter = RateLimiter(redis, requests_per_minute=60, burst=5)

    await limiter.check(key)

    assert await redis.pttl(f"rl:{key}") > 0


# ---- the aggregation SQL --------------------------------------------------


async def _isolate(session) -> None:
    """Clear request_logs *inside the test transaction*.

    The tier rolls back, so this never touches committed data -- but without it
    every count assertion would also see whatever real traffic the developer's
    database already holds, and the test would fail on a used machine while
    passing on a fresh one.
    """
    from app.db.models.request_log import RequestLog

    await session.execute(delete(RequestLog))


async def _insert(session, **kw):
    from app.db.models.request_log import RequestLog

    session.add(
        RequestLog(
            endpoint=kw.get("endpoint", "chat"),
            provider=kw.get("provider", "gemini"),
            model=kw.get("model", "gemini-flash-latest"),
            input_tokens=kw.get("input_tokens", 0),
            output_tokens=kw.get("output_tokens", 0),
            latency_ms=Decimal(str(kw.get("latency_ms", 100))),
            cached=kw.get("cached", False),
            cost_usd=Decimal(str(kw.get("cost_usd", "0"))),
            cost_saved_usd=Decimal(str(kw.get("cost_saved_usd", "0"))),
            priced=kw.get("priced", True),
            error_type=kw.get("error_type"),
        )
    )
    await session.flush()


@needs_postgres
async def test_percentile_cont_computes_a_real_p95(session) -> None:
    """100 rows at 1..100ms: the exact p95 interpolates to ~95.05."""
    await _isolate(session)
    for i in range(1, 101):
        await _insert(session, latency_ms=i)

    summary = await MetricsRepository(session).summary(window_hours=1)

    assert summary["requests"] == 100
    assert summary["latency_ms"]["p50"] == pytest.approx(50.5, abs=0.6)
    assert summary["latency_ms"]["p95"] == pytest.approx(95.05, abs=0.6)
    assert summary["latency_ms"]["p99"] == pytest.approx(99.01, abs=0.6)


@needs_postgres
async def test_cache_hit_rate_and_costs_aggregate(session) -> None:
    await _isolate(session)
    await _insert(session, cached=False, cost_usd="0.01", input_tokens=100, output_tokens=50)
    await _insert(session, cached=False, cost_usd="0.02", input_tokens=200, output_tokens=60)
    await _insert(session, cached=True, cost_saved_usd="0.03")
    await _insert(session, cached=True, cost_saved_usd="0.04")

    s = await MetricsRepository(session).summary(window_hours=1)

    assert s["requests"] == 4
    assert s["cache_hits"] == 2
    assert s["cache_hit_rate"] == 0.5
    assert Decimal(s["cost_usd"]["spent"]) == Decimal("0.030000")
    assert Decimal(s["cost_usd"]["saved_by_cache"]) == Decimal("0.070000")
    assert s["tokens"]["total"] == 410


@needs_postgres
async def test_unpriced_requests_are_counted_separately(session) -> None:
    """An unpriced model must not silently look free."""
    await _isolate(session)
    await _insert(session, priced=True, cost_usd="0.01")
    await _insert(session, priced=False, cost_usd="0")

    s = await MetricsRepository(session).summary(window_hours=1)

    assert s["cost_usd"]["unpriced_requests"] == 1


@needs_postgres
async def test_errors_are_counted(session) -> None:
    await _isolate(session)
    await _insert(session)
    await _insert(session, error_type="ProviderTimeout")

    s = await MetricsRepository(session).summary(window_hours=1)

    assert s["errors"] == 1
    assert s["error_rate"] == 0.5


@needs_postgres
async def test_empty_window_does_not_divide_by_zero(session) -> None:
    """A fresh install hits this on the first page load."""
    await _isolate(session)

    s = await MetricsRepository(session).summary(window_hours=1)

    assert s["requests"] == 0
    assert s["cache_hit_rate"] == 0.0
    assert s["error_rate"] == 0.0
    assert s["latency_ms"]["p95"] is None


@needs_postgres
async def test_by_model_groups_and_ranks(session) -> None:
    await _isolate(session)
    await _insert(session, provider="gemini", model="a", cost_usd="0.01")
    await _insert(session, provider="gemini", model="a", cost_usd="0.01")
    await _insert(session, provider="anthropic", model="b", cost_usd="0.05")

    rows = await MetricsRepository(session).by_provider(window_hours=1)
    by_model = {r["model"]: r for r in rows}

    assert by_model["a"]["requests"] == 2
    assert by_model["b"]["requests"] == 1
    assert Decimal(by_model["a"]["cost_usd"]) == Decimal("0.020000")
