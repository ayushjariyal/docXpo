"""Per-key token-bucket rate limiting, backed by Redis.

## Why a token bucket

| Algorithm | Memory | Burst behaviour |
|---|---|---|
| Fixed window | 1 counter | Allows **2x** at a boundary: 60 at 11:59:59 and 60 at 12:00:00 |
| Sliding window log | O(n) timestamps | Exact, but stores every request |
| **Token bucket** | **2 numbers** | Smooth refill, bounded burst, O(1) |

A bucket holds `capacity` tokens and refills at `rate` tokens/second. Each
request spends one. A client that has been idle can burst up to `capacity`,
then settles into the sustained rate — which matches how real clients behave
(bursty, then quiet) far better than a hard per-minute cap.

## Why the arithmetic is a Lua script

The refill-check-decrement sequence must be **atomic**. Done as separate
GET/SET calls, two concurrent requests both read `tokens=1`, both conclude they
may proceed, and both write `tokens=0` — the limit is silently exceeded under
exactly the load it exists to control.

Redis executes a Lua script atomically against a single-threaded core, so no
other command interleaves. This is the whole reason the logic lives in Redis
rather than in Python.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from redis.asyncio import Redis

from app.core.logging import get_logger

log = get_logger(__name__)

# KEYS[1] = bucket key
# ARGV    = rate (tokens/sec), capacity, now_ms, cost
# Returns : {allowed, remaining_tokens, retry_after_ms, reset_ms}
TOKEN_BUCKET_LUA = """
local key      = KEYS[1]
local rate     = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local now      = tonumber(ARGV[3])
local cost     = tonumber(ARGV[4])

local bucket = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(bucket[1])
local ts     = tonumber(bucket[2])

-- First sight of this key: start full, so a new client is not punished.
if tokens == nil or ts == nil then
  tokens = capacity
  ts     = now
end

-- Refill for the time elapsed since the last request, capped at capacity.
-- Continuous rather than per-tick: no edge where a client gets a free window.
local elapsed = math.max(0, now - ts) / 1000.0
tokens = math.min(capacity, tokens + (elapsed * rate))

local allowed = 0
if tokens >= cost then
  allowed = 1
  tokens  = tokens - cost
end

redis.call('HSET', key, 'tokens', tokens, 'ts', now)

-- Expire once the bucket would have refilled completely: an idle key is
-- indistinguishable from a fresh one, so keeping it wastes memory. Without a
-- TTL, Redis would accumulate a key per API key forever.
local full_refill_ms = math.ceil((capacity / rate) * 1000)
redis.call('PEXPIRE', key, full_refill_ms + 1000)

local retry_after = 0
if allowed == 0 then
  -- Time until enough tokens exist for this request.
  retry_after = math.ceil(((cost - tokens) / rate) * 1000)
end

-- Time until the bucket is full again, for X-RateLimit-Reset.
local reset = math.ceil(((capacity - tokens) / rate) * 1000)

return {allowed, math.floor(tokens), retry_after, reset}
"""


@dataclass(frozen=True, slots=True)
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    retry_after_seconds: int
    reset_seconds: int

    def headers(self) -> dict[str, str]:
        """Standard rate-limit headers.

        Returned on **every** response, not only on 429s, so a well-behaved
        client can slow down before it gets rejected rather than discovering
        the limit by hitting it.
        """
        h = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(self.reset_seconds),
        }
        if not self.allowed:
            # Retry-After is the one the HTTP spec actually defines for 429,
            # and what well-written clients and proxies honour.
            h["Retry-After"] = str(max(1, self.retry_after_seconds))
        return h


class RateLimiter:
    def __init__(
        self,
        redis: Redis,
        *,
        requests_per_minute: int,
        burst: int,
        fail_open: bool = True,
    ) -> None:
        self._redis = redis
        self._rate = requests_per_minute / 60.0  # tokens per second
        self._capacity = burst
        self._limit = requests_per_minute
        self._fail_open = fail_open
        self._script = None  # registered lazily; see _ensure_script

    def _ensure_script(self):
        # register_script does not touch the network -- it only computes the
        # SHA. redis-py transparently falls back to EVAL if the script is not
        # yet cached server-side, so this survives a Redis restart.
        if self._script is None:
            self._script = self._redis.register_script(TOKEN_BUCKET_LUA)
        return self._script

    async def check(self, key: str, *, cost: int = 1) -> RateLimitResult:
        bucket_key = f"rl:{key}"
        now_ms = int(time.time() * 1000)

        try:
            allowed, remaining, retry_ms, reset_ms = await self._ensure_script()(
                keys=[bucket_key],
                args=[self._rate, self._capacity, now_ms, cost],
            )
        except Exception as exc:  # noqa: BLE001
            # Deliberately broad, not just RedisError. Fail-open only means
            # anything if it covers *every* way this can break -- a connection
            # error, a Lua bug, a client that does not behave as expected. A
            # rate limiter that can crash the request path is worse than no
            # rate limiter at all.
            #
            # The trade-off is explicit: during a Redis outage the service is
            # unprotected. Fail *closed* is right when an unmetered flood costs
            # more than an outage -- that is what fail_open=False is for.
            log.error(
                "rate_limit_backend_error",
                error=str(exc),
                error_type=type(exc).__name__,
                fail_open=self._fail_open,
            )
            if self._fail_open:
                return RateLimitResult(True, self._limit, self._capacity, 0, 0)
            return RateLimitResult(False, self._limit, 0, 5, 5)

        return RateLimitResult(
            allowed=bool(allowed),
            limit=self._limit,
            remaining=int(remaining),
            # Round *up* to whole seconds: a Retry-After of 0 invites an
            # immediate retry that is guaranteed to fail again.
            retry_after_seconds=(int(retry_ms) + 999) // 1000,
            reset_seconds=(int(reset_ms) + 999) // 1000,
        )

    async def peek(self, key: str) -> RateLimitResult:
        """Report the bucket state without spending a token."""
        return await self.check(key, cost=0)

    async def reset(self, key: str) -> None:
        await self._redis.delete(f"rl:{key}")
