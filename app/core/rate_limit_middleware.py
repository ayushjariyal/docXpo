"""Rate-limit + API-key middleware.

Raw ASGI, for the same reason as `RequestContextMiddleware` in Phase 1: it must
inject headers into responses that are **streamed**. `BaseHTTPMiddleware`
buffers the body through a memory stream, which is exactly wrong for the SSE
endpoints this most needs to protect.

Middleware rather than a FastAPI dependency, for two reasons:

1. A dependency cannot reliably attach headers to a `StreamingResponse` the
   route constructs and returns itself -- and `X-RateLimit-*` on *successful*
   streaming responses is most of the value.
2. The limit should apply uniformly, including to paths that never declared it.
   Opt-out (an explicit exempt list) is a safer default than opt-in, where
   forgetting the dependency silently leaves a route unmetered.
"""

from __future__ import annotations

import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.auth import identify
from app.core.logging import get_logger
from app.core.rate_limit import RateLimiter

log = get_logger(__name__)

# Never metered or authenticated:
#   /health*  -- an orchestrator's probe must not be rejected because a noisy
#                client exhausted a shared bucket; that would turn a rate-limit
#                event into a rolling restart.
#   /         -- the browser console itself, plus the OpenAPI docs.
EXEMPT_PREFIXES = ("/health", "/docs", "/openapi.json", "/redoc")
EXEMPT_EXACT = ("/",)


def _is_exempt(path: str) -> bool:
    return path in EXEMPT_EXACT or path.startswith(EXEMPT_PREFIXES)


class RateLimitMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        configured_keys: frozenset[str],
        requests_per_minute: int,
        burst: int,
        fail_open: bool = True,
        enabled: bool = True,
    ) -> None:
        self.app = app
        self._keys = configured_keys
        self._rpm = requests_per_minute
        self._burst = burst
        self._fail_open = fail_open
        self._enabled = enabled
        self._limiter: RateLimiter | None = None

    def _get_limiter(self, scope: Scope) -> RateLimiter:
        """Build the limiter on first request, from the app's shared client.

        Middleware is constructed while the app is being assembled, before the
        lifespan hook has created `app.state.redis`. Building the limiter here
        instead of in __init__ lets it reuse that single connection pool rather
        than opening a second one just for rate limiting.
        """
        if self._limiter is None:
            self._limiter = RateLimiter(
                scope["app"].state.redis,
                requests_per_minute=self._rpm,
                burst=self._burst,
                fail_open=self._fail_open,
            )
        return self._limiter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self._enabled or _is_exempt(scope["path"]):
            await self.app(scope, receive, send)
            return

        headers = dict(scope["headers"])
        client = scope.get("client")
        principal = identify(
            headers, configured_keys=self._keys, client_ip=client[0] if client else None
        )

        if principal is None:
            await _json_response(
                send,
                401,
                {
                    "error": "missing or invalid API key",
                    "type": "AuthenticationError",
                    "hint": "send it as 'X-API-Key: <key>' or 'Authorization: Bearer <key>'",
                },
                # WWW-Authenticate is what makes a 401 spec-compliant and tells
                # a client *how* to authenticate rather than just that it failed.
                extra={"WWW-Authenticate": 'Bearer realm="docXpo"'},
            )
            return

        # Stash for the metrics recorder: the principal is resolved here, and
        # re-deriving it in each route would duplicate the header parsing.
        scope["docxpo_principal"] = principal.label

        result = await self._get_limiter(scope).check(principal.identifier)

        if not result.allowed:
            log.warning(
                "rate_limited",
                principal=principal.label,
                path=scope["path"],
                retry_after=result.retry_after_seconds,
            )
            await _json_response(
                send,
                429,
                {
                    "error": "rate limit exceeded",
                    "type": "RateLimitExceeded",
                    "limit_per_minute": result.limit,
                    "retry_after_seconds": result.retry_after_seconds,
                },
                extra=result.headers(),
            )
            return

        # Allowed: pass through, and stamp the budget onto the real response.
        rate_headers = [
            (k.lower().encode(), v.encode()) for k, v in result.headers().items()
        ]

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                message["headers"] = [*message["headers"], *rate_headers]
            await send(message)

        await self.app(scope, receive, send_wrapper)


async def _json_response(
    send: Send, status: int, body: dict, *, extra: dict[str, str] | None = None
) -> None:
    """Emit a complete JSON response without touching the app.

    Written by hand rather than via Starlette's JSONResponse because at this
    layer we hold the raw ASGI `send` callable, not a request object.
    """
    payload = json.dumps({"detail": body}).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(payload)).encode()),
    ]
    for key, value in (extra or {}).items():
        headers.append((key.lower().encode(), value.encode()))

    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": payload})
