"""Redis connection pool lifecycle.

``redis.asyncio.Redis`` is itself a pool wrapper, so one client per process is
correct -- it multiplexes commands over pooled connections. We keep it on
``app.state`` rather than as a module global so tests can install a fake, and
so shutdown is deterministic.
"""

from __future__ import annotations

from redis.asyncio import Redis

from app.core.config import Settings


def build_redis(settings: Settings) -> Redis:
    return Redis.from_url(
        settings.redis_url,
        # We store JSON and (from Phase 4) raw float32 vectors. Decoding
        # everything to str would corrupt the binary payloads, so decoding is
        # left to each call site.
        decode_responses=False,
        # Don't let a hung Redis turn into a hung request. Redis is a cache and
        # a rate limiter here -- neither is worth stalling a request for.
        socket_connect_timeout=2,
        socket_timeout=2,
        health_check_interval=30,
    )
