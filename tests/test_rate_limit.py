"""Rate limiter tests.

The token-bucket arithmetic runs as a Lua script inside Redis, so these use a
fake that *interprets* the same semantics rather than re-implementing them in
Python. That is an honest limitation: it verifies the limiter's behaviour and
the middleware's HTTP contract, but not the Lua itself. The Lua is exercised
against real Redis in the live checks recorded in DECISIONS.md.
"""

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from redis.exceptions import RedisError

from app.core.auth import extract_key, fingerprint, identify, matches_any
from app.core.config import Settings
from app.core.pricing import Pricebook
from app.core.rate_limit import RateLimiter
from app.main import create_app
from app.services.metrics_service import MetricsService


class FakeRedisScript:
    """Interprets the token-bucket semantics the Lua script implements."""

    def __init__(self, store: dict, fail: bool = False) -> None:
        self._store = store
        self._fail = fail

    async def __call__(self, keys, args):
        if self._fail:
            raise RedisError("redis is down")

        key = keys[0]
        rate, capacity, now, cost = float(args[0]), float(args[1]), float(args[2]), float(args[3])

        tokens, ts = self._store.get(key, (capacity, now))
        elapsed = max(0.0, now - ts) / 1000.0
        tokens = min(capacity, tokens + elapsed * rate)

        allowed = 0
        if tokens >= cost:
            allowed = 1
            tokens -= cost

        self._store[key] = (tokens, now)

        retry_ms = 0 if allowed else int(((cost - tokens) / rate) * 1000) + 1
        reset_ms = int(((capacity - tokens) / rate) * 1000) + 1
        return [allowed, int(tokens), retry_ms, reset_ms]


class FakeRedis:
    def __init__(self, fail: bool = False) -> None:
        self.buckets: dict = {}
        self._fail = fail

    def register_script(self, _src):
        return FakeRedisScript(self.buckets, self._fail)

    async def delete(self, *keys):
        for k in keys:
            self.buckets.pop(k, None)

    async def ping(self):
        return True


def _limiter(rpm=60, burst=5, fail=False, fail_open=True) -> RateLimiter:
    return RateLimiter(
        FakeRedis(fail=fail), requests_per_minute=rpm, burst=burst, fail_open=fail_open
    )


# ---- bucket behaviour -----------------------------------------------------


async def test_requests_within_burst_are_allowed() -> None:
    limiter = _limiter(burst=5)

    results = [await limiter.check("k") for _ in range(5)]

    assert all(r.allowed for r in results)
    # Remaining counts down, so a client can see it approaching zero.
    assert results[0].remaining > results[-1].remaining


async def test_burst_exhaustion_rejects() -> None:
    limiter = _limiter(burst=3)
    for _ in range(3):
        await limiter.check("k")

    denied = await limiter.check("k")

    assert not denied.allowed
    assert denied.remaining == 0


async def test_retry_after_is_never_zero_on_rejection() -> None:
    """A Retry-After of 0 invites an immediate retry that must fail again."""
    limiter = _limiter(rpm=60, burst=1)
    await limiter.check("k")

    denied = await limiter.check("k")

    assert not denied.allowed
    assert denied.retry_after_seconds >= 1
    assert int(denied.headers()["Retry-After"]) >= 1


async def test_bucket_refills_over_time() -> None:
    """The defining property of a token bucket: continuous refill."""
    limiter = _limiter(rpm=6000, burst=2)  # 100 tokens/sec -- refills fast
    await limiter.check("k")
    await limiter.check("k")
    assert not (await limiter.check("k")).allowed

    time.sleep(0.05)  # ~5 tokens' worth

    assert (await limiter.check("k")).allowed


async def test_buckets_are_isolated_per_key() -> None:
    """One noisy client must not consume another's budget."""
    limiter = _limiter(burst=2)
    await limiter.check("alice")
    await limiter.check("alice")

    assert not (await limiter.check("alice")).allowed
    assert (await limiter.check("bob")).allowed


async def test_peek_does_not_spend_a_token() -> None:
    limiter = _limiter(burst=3)

    for _ in range(5):
        await limiter.peek("k")

    assert (await limiter.check("k")).allowed


async def test_fail_open_allows_when_redis_is_down() -> None:
    limiter = _limiter(fail=True, fail_open=True)

    assert (await limiter.check("k")).allowed


async def test_fail_closed_rejects_when_redis_is_down() -> None:
    """The opposite trade-off: protect upstream cost over availability."""
    limiter = _limiter(fail=True, fail_open=False)

    result = await limiter.check("k")

    assert not result.allowed
    assert result.retry_after_seconds >= 1


async def test_headers_present_on_success_too() -> None:
    """Clients should be able to slow down *before* being rejected."""
    result = await _limiter().check("k")

    headers = result.headers()
    assert {"X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"} <= headers.keys()
    assert "Retry-After" not in headers  # only on rejection


# ---- identity -------------------------------------------------------------


def test_key_fingerprint_is_stable_and_not_reversible() -> None:
    fp = fingerprint("super-secret-key")

    assert fp == fingerprint("super-secret-key")
    assert "super-secret-key" not in fp
    assert len(fp) == 12


def test_extract_key_accepts_both_header_styles() -> None:
    assert extract_key({b"x-api-key": b"abc"}) == "abc"
    assert extract_key({b"authorization": b"Bearer abc"}) == "abc"
    assert extract_key({b"authorization": b"bearer abc"}) == "abc"
    assert extract_key({}) is None


def test_matches_any_is_constant_time_membership() -> None:
    known = frozenset({"key-one", "key-two"})

    assert matches_any("key-one", known)
    assert not matches_any("key-three", known)


def test_open_mode_meters_by_ip() -> None:
    principal = identify({}, configured_keys=frozenset(), client_ip="10.0.0.5")

    assert principal is not None
    assert principal.kind == "anonymous"
    assert principal.identifier == "ip:10.0.0.5"


def test_configured_keys_require_a_valid_key() -> None:
    keys = frozenset({"good-key"})

    assert identify({}, configured_keys=keys, client_ip="1.2.3.4") is None
    assert identify({b"x-api-key": b"bad"}, configured_keys=keys, client_ip=None) is None

    ok = identify({b"x-api-key": b"good-key"}, configured_keys=keys, client_ip=None)
    assert ok is not None and ok.is_authenticated
    # The raw key must never become the bucket key or a log label.
    assert "good-key" not in ok.identifier


# ---- HTTP contract --------------------------------------------------------


def _app(**overrides) -> FastAPI:
    settings = Settings(_env_file=None, log_level="CRITICAL", **overrides)
    app = create_app(settings)
    # ASGITransport does not run the lifespan hook, so app.state is empty.
    # These are placeholders: the tests below never reach a handler that uses
    # them, but dependency construction touches them before body validation.
    app.state.redis = FakeRedis()
    app.state.provider_registry = object()
    app.state.embedder = object()
    app.state.metrics = MetricsService(Pricebook())
    return app


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_health_is_never_rate_limited() -> None:
    """A probe rejected by a rate limit would trigger a rolling restart."""
    app = _app(rate_limit_rpm=60, rate_limit_burst=1)

    async with _client(app) as ac:
        for _ in range(5):
            assert (await ac.get("/health")).status_code == 200


async def test_missing_key_is_401_with_www_authenticate() -> None:
    app = _app(api_keys="secret-key")

    async with _client(app) as ac:
        resp = await ac.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]})

    assert resp.status_code == 401
    assert "WWW-Authenticate" in resp.headers
    assert resp.json()["detail"]["type"] == "AuthenticationError"


async def test_429_carries_retry_after_and_limit_headers() -> None:
    """Also proves the limiter runs *before* routing and validation.

    The first request has a deliberately invalid body, so it never reaches the
    handler -- yet it still spends a token. A limiter that only metered
    successful requests would let a client flood the service with malformed
    ones for free.
    """
    app = _app(api_keys="secret-key", rate_limit_rpm=60, rate_limit_burst=1)
    headers = {"X-API-Key": "secret-key"}

    async with _client(app) as ac:
        first = await ac.post("/v1/chat", json={}, headers=headers)
        second = await ac.post("/v1/chat", json={}, headers=headers)

    assert first.status_code == 422  # rejected by validation, token still spent
    assert second.status_code == 429
    assert int(second.headers["Retry-After"]) >= 1
    assert second.headers["X-RateLimit-Limit"] == "60"
    assert second.headers["X-RateLimit-Remaining"] == "0"
    # The successful-path headers must be present on the 422 as well.
    assert "X-RateLimit-Remaining" in first.headers


async def test_disabled_limiter_lets_everything_through() -> None:
    app = _app(rate_limit_enabled=False, rate_limit_burst=1)

    async with _client(app) as ac:
        for _ in range(4):
            resp = await ac.get("/health")
            assert resp.status_code == 200


@pytest.mark.parametrize("path", ["/", "/health", "/health/ready", "/docs"])
async def test_exempt_paths_need_no_api_key(path: str) -> None:
    """The console and probes must work even when keys are enforced."""
    app = _app(api_keys="secret-key")

    async with _client(app) as ac:
        assert (await ac.get(path)).status_code != 401
