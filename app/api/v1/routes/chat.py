"""POST /v1/chat — token-by-token streaming over Server-Sent Events.

SSE rather than WebSockets because the traffic is one-directional (server ->
client) and SSE is plain HTTP: it works through proxies and load balancers
with no upgrade handshake, and browsers reconnect automatically. A WebSocket
would buy bidirectionality we don't need and cost us that simplicity.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.deps import ChatSvc, Metrics
from app.core.logging import get_logger
from app.core.middleware import current_request_id
from app.llm.base import StreamDone, StreamEvent, TextDelta
from app.llm.errors import ProviderError
from app.schemas.chat import ChatDoneOut, ChatRequest, UsageOut
from app.services.metrics_service import RequestRecord

log = get_logger(__name__)
router = APIRouter(tags=["chat"])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # Tells nginx not to buffer the response. Without it, nginx holds chunks
    # until its buffer fills and the "streaming" endpoint delivers everything
    # in one burst at the end -- the single most common reason SSE "works
    # locally but not in production".
    "X-Accel-Buffering": "no",
}


def sse(event: str, data: dict[str, object]) -> str:
    """Format one SSE frame.

    The `data:` field is JSON-encoded rather than written raw, and that is
    load-bearing: SSE is a newline-delimited protocol, so a token containing
    a literal newline would terminate the frame early and corrupt the stream.
    json.dumps escapes newlines to \\n, which keeps every frame on one line.
    """
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


@router.post(
    "/chat",
    summary="Stream a chat completion over SSE",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"text/event-stream": {}}, "description": "SSE stream"},
        400: {"description": "Provider rejected the request"},
        429: {"description": "Upstream rate limit"},
        503: {"description": "Provider unreachable or not configured"},
    },
)
async def chat(payload: ChatRequest, request: Request, service: ChatSvc, metrics: Metrics):
    provider = await _resolve(service, payload.provider)

    started = time.perf_counter()
    stream = service.stream(
        provider,
        payload.messages,
        model=payload.model,
        max_tokens=payload.max_tokens,
    )

    # --- Prime the stream --------------------------------------------------
    # Pull the first event *before* returning the response. Once a
    # StreamingResponse starts, the 200 status and headers are already on the
    # wire and cannot be taken back -- an unreachable Ollama would otherwise
    # surface as "200 OK, then an error event", which no HTTP client treats as
    # a failure. Priming lets connection-level problems become real 4xx/5xx
    # status codes, while errors that happen *mid-generation* still have to be
    # reported in-band (see _body below).
    iterator = stream.__aiter__()
    try:
        first: StreamEvent | None = await iterator.__anext__()
    except StopAsyncIteration:
        first = None
    except ProviderError as exc:
        log.warning("chat_provider_error", provider=provider.name, error=exc.message)
        raise HTTPException(status_code=exc.status_code, detail=exc.to_dict()) from exc

    seen = {"in": 0, "out": 0, "model": provider.default_model, "finish": None}
    ttft: list[float] = []

    async def _body() -> AsyncIterator[str]:
        error_type: str | None = None
        try:
            event = first
            while event is not None:
                if isinstance(event, TextDelta) and not ttft:
                    ttft.append((time.perf_counter() - started) * 1000)
                elif isinstance(event, StreamDone):
                    seen["in"] = event.usage.input_tokens
                    seen["out"] = event.usage.output_tokens
                    seen["model"] = event.model or seen["model"]
                    seen["finish"] = event.finish_reason
                yield _frame(event, provider.name, started)
                try:
                    event = await iterator.__anext__()
                except StopAsyncIteration:
                    break

        except ProviderError as exc:
            # Mid-stream failure. The status line is long gone, so the only way
            # to tell the client is an in-band `error` event. Clients must
            # treat `error` as terminal -- documented in the README.
            error_type = type(exc).__name__
            log.warning(
                "chat_stream_error", provider=provider.name, error=exc.message
            )
            yield sse("error", exc.to_dict())

        except Exception as exc:  # noqa: BLE001
            error_type = type(exc).__name__
            log.exception("chat_stream_unexpected_error", provider=provider.name)
            yield sse("error", {"error": "internal error", "type": type(exc).__name__})

        finally:
            # Recorded after the body has been delivered, so it adds nothing to
            # what the user waits for. A partial stream is still worth a row --
            # errors are exactly what you want counted.
            await metrics.record(
                RequestRecord(
                    endpoint="chat",
                    provider=provider.name,
                    model=seen["model"],
                    input_tokens=seen["in"],
                    output_tokens=seen["out"],
                    latency_ms=(time.perf_counter() - started) * 1000,
                    ttft_ms=ttft[0] if ttft else None,
                    finish_reason=seen["finish"],
                    error_type=error_type,
                    status_code=200 if error_type is None else 500,
                    request_id=current_request_id(),
                    principal=request.scope.get("docxpo_principal"),
                )
            )
            # Runs on client disconnect too: Starlette cancels this generator,
            # and the resulting GeneratorExit unwinds through here. Closing the
            # iterator releases the upstream HTTP connection instead of leaving
            # it open, generating tokens nobody will read.
            await iterator.aclose()

    return StreamingResponse(
        _body(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


def _frame(
    event: StreamEvent, provider_name: str, started: float, *, cached: bool = False
) -> str:
    if isinstance(event, TextDelta):
        return sse("token", {"text": event.text})

    assert isinstance(event, StreamDone)
    done = ChatDoneOut(
        provider=provider_name,
        model=event.model,
        usage=UsageOut(
            input_tokens=event.usage.input_tokens,
            output_tokens=event.usage.output_tokens,
            total_tokens=event.usage.total_tokens,
        ),
        finish_reason=event.finish_reason,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        cached=cached,
    )
    return sse("done", done.model_dump())


async def _resolve(service: ChatSvc, name: str | None):
    try:
        return await service.resolve_provider(name)
    except ProviderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_dict()) from exc
