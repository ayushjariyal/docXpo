"""Request context + access logging middleware.

Written as raw ASGI middleware rather than Starlette's ``BaseHTTPMiddleware``.

Why that matters here: ``BaseHTTPMiddleware`` runs the endpoint in a separate
anyio task and pipes the response body through a memory stream. For ordinary
JSON responses that is invisible, but it interferes with long-lived streaming
responses -- which is exactly what Phase 2's SSE ``/v1/chat`` endpoint is. Raw
ASGI middleware just wraps ``send``, so response chunks flow straight through to
the client as they are produced.
"""

from __future__ import annotations

import time
import uuid

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.logging import get_logger

log = get_logger(__name__)

REQUEST_ID_HEADER = "x-request-id"


def current_request_id() -> str | None:
    """Read the id bound for this request, without threading it through args.

    contextvars are per-task and each request is its own task, so this returns
    the id for the request currently being handled -- which is what lets a
    metrics row be correlated with the access log line and the response header.
    """
    return structlog.contextvars.get_contextvars().get("request_id")


class RequestContextMiddleware:
    """Assigns a request id, logs one line per request, times the handler."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Lifespan and websocket scopes have no headers/status -- pass through.
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope["headers"])  # list[tuple[bytes, bytes]] -> dict
        incoming = headers.get(REQUEST_ID_HEADER.encode())
        # Reuse an upstream id if a proxy/gateway already assigned one, so a
        # single id traces the request across services.
        request_id = incoming.decode() if incoming else uuid.uuid4().hex

        # contextvars are per-task, and each request is its own task, so this is
        # safe under concurrency: every log line in this request picks up the id.
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        status_code = 500  # if the app raises before sending, that's a 500
        start = time.perf_counter()

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                # Echo the id back so a client can quote it in a bug report.
                message["headers"] = [
                    *message["headers"],
                    (REQUEST_ID_HEADER.encode(), request_id.encode()),
                ]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            # `finally` so we still log when the handler raises or the client
            # disconnects mid-stream.
            duration_ms = (time.perf_counter() - start) * 1000
            log.info(
                "http_request",
                method=scope["method"],
                path=scope["path"],
                status_code=status_code,
                duration_ms=round(duration_ms, 2),
            )
            structlog.contextvars.clear_contextvars()
