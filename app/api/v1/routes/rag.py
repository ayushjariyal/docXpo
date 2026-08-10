"""RAG query endpoints.

`/rag/retrieve` returns only what retrieval found -- no generation, no tokens
spent. It exists because the two halves of RAG fail differently: a bad answer
is either bad retrieval or bad generation, and being able to inspect retrieval
in isolation is what makes chunk size and top_k tunable rather than guesswork.

`/rag/query` does the full pipeline and streams, reusing the SSE machinery from
the chat endpoint plus one extra event: `sources` is emitted *before* the first
token so a client can render citations while the answer is still arriving.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.deps import Cache, ChatSvc, Metrics, RagSvc
from app.api.v1.routes.chat import SSE_HEADERS, _frame, sse
from app.core.logging import get_logger
from app.core.middleware import current_request_id
from app.llm.base import StreamDone, TextDelta
from app.llm.errors import ProviderError
from app.repositories.document_repository import ScoredChunk
from app.schemas.documents import QueryIn, RetrieveOut, SourceOut
from app.services.metrics_service import RequestRecord
from app.services.semantic_cache import CachedAnswer

log = get_logger(__name__)
router = APIRouter(prefix="/rag", tags=["rag"])

# Enough to show why a chunk matched without shipping the whole passage twice.
EXCERPT_CHARS = 320


def _cached_response(hit: CachedAnswer, *, started: float) -> StreamingResponse:
    """Serve a cache hit in the same SSE shape as a generated answer.

    The client is not told to behave differently -- it receives `sources`,
    `token`, `done` exactly as usual, with `cached: true` on the done event.
    The answer arrives as a single token frame because it is already complete;
    re-chunking it to fake incremental typing would add latency for cosmetics.

    `sources` is empty: the answer was cached, not re-retrieved, and inventing
    citations we did not actually look up would be dishonest.
    """

    async def gen() -> AsyncIterator[str]:
        yield sse("sources", {"sources": []})
        yield sse("token", {"text": hit.answer})
        yield sse(
            "done",
            {
                "provider": hit.provider,
                "model": hit.model,
                "usage": {
                    # Zero: this request spent no generation tokens. The stored
                    # counts are reported separately as what was *saved*.
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
                "finish_reason": "cache_hit",
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "cached": True,
                "cache_similarity": round(hit.similarity, 4),
                "cache_age_seconds": round(hit.age_seconds, 1),
                "cached_question": hit.question,
                "tokens_saved": hit.input_tokens + hit.output_tokens,
            },
        )

    log.info(
        "cache_hit",
        similarity=round(hit.similarity, 4),
        age_seconds=round(hit.age_seconds, 1),
        tokens_saved=hit.input_tokens + hit.output_tokens,
    )
    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)


def _sources(chunks: Sequence[ScoredChunk]) -> list[SourceOut]:
    return [
        SourceOut(
            n=i,
            document_id=c.document_id,
            filename=c.document_filename,
            chunk_index=c.chunk_index,
            similarity=round(c.similarity, 4),
            excerpt=c.content[:EXCERPT_CHARS] + ("…" if len(c.content) > EXCERPT_CHARS else ""),
        )
        for i, c in enumerate(chunks, start=1)
    ]


@router.get("/cache", summary="Semantic cache configuration and size")
async def cache_stats(cache: Cache, chat: ChatSvc) -> dict:
    provider = await chat.resolve_provider(None)
    return await cache.stats(
        provider=provider.name,
        model=provider.default_model,
        corpus_version=await cache.corpus_version(),
    )


@router.delete("/cache", summary="Clear the cache for the default provider/model")
async def clear_cache(cache: Cache, chat: ChatSvc) -> dict:
    provider = await chat.resolve_provider(None)
    removed = await cache.clear(
        provider=provider.name,
        model=provider.default_model,
        corpus_version=await cache.corpus_version(),
    )
    return {"cleared": removed}


@router.post(
    "/retrieve",
    response_model=RetrieveOut,
    summary="Retrieve matching chunks without generating an answer",
)
async def retrieve(payload: QueryIn, rag: RagSvc) -> RetrieveOut:
    try:
        chunks = await rag.retrieve(
            payload.question,
            top_k=payload.top_k,
            min_similarity=payload.min_similarity,
            document_id=payload.document_id,
        )
    except ProviderError as exc:
        raise HTTPException(exc.status_code, exc.to_dict()) from exc

    return RetrieveOut(question=payload.question, sources=_sources(chunks))


@router.post(
    "/query",
    summary="Retrieve context and stream a grounded answer over SSE",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def query(
    payload: QueryIn,
    request: Request,
    rag: RagSvc,
    chat: ChatSvc,
    cache: Cache,
    metrics: Metrics,
):
    # Retrieval happens before the response is committed, so an embedding
    # failure is a real HTTP status rather than an in-band error event.
    try:
        provider = await chat.resolve_provider(payload.provider)
        # Embed once. The cache lookup and the vector search use the same
        # vector, so the cache adds no embedding cost -- see rag.embed_query.
        embedding = await rag.embed_query(payload.question)
        corpus_version = await cache.corpus_version()

        model = payload.model or provider.default_model
        hit = await cache.lookup(
            embedding,
            provider=provider.name,
            model=model,
            corpus_version=corpus_version,
        )
        if hit is not None:
            started_hit = time.perf_counter()
            await metrics.record(
                RequestRecord(
                    endpoint="rag_query",
                    provider=hit.provider,
                    model=hit.model,
                    latency_ms=(time.perf_counter() - started_hit) * 1000,
                    cached=True,
                    finish_reason="cache_hit",
                    # Priced as what the hit *avoided* spending, which is what
                    # makes the cache's value show up in /metrics as a number.
                    saved_input_tokens=hit.input_tokens,
                    saved_output_tokens=hit.output_tokens,
                    request_id=current_request_id(),
                    principal=request.scope.get("docxpo_principal"),
                )
            )
            return _cached_response(hit, started=started_hit)

        chunks = await rag.retrieve(
            payload.question,
            top_k=payload.top_k,
            min_similarity=payload.min_similarity,
            document_id=payload.document_id,
            embedding=embedding,
        )
    except ProviderError as exc:
        raise HTTPException(exc.status_code, exc.to_dict()) from exc

    sources = _sources(chunks)

    # Nothing cleared the similarity floor. Answering anyway would mean the
    # model inventing something from an empty context, so say so and spend no
    # tokens. Still a 200 with a well-formed stream: "no answer" is a valid
    # outcome of a valid request, not an error.
    if not chunks:
        async def empty() -> AsyncIterator[str]:
            yield sse("sources", {"sources": []})
            yield sse(
                "token",
                {"text": "I don't have anything in the indexed documents that answers that."},
            )
            yield sse(
                "done",
                {
                    "provider": provider.name,
                    "model": provider.default_model,
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "finish_reason": "no_context",
                    "latency_ms": 0.0,
                },
            )

        return StreamingResponse(empty(), media_type="text/event-stream", headers=SSE_HEADERS)

    started = time.perf_counter()
    stream = rag.stream_answer(
        provider,
        payload.question,
        chunks,
        model=payload.model,
        max_tokens=payload.max_tokens,
    )

    # Same priming trick as /v1/chat: pull the first event before returning, so
    # a dead provider is a status code rather than a 200 followed by an error.
    iterator = stream.__aiter__()
    try:
        first = await iterator.__anext__()
    except StopAsyncIteration:
        first = None
    except ProviderError as exc:
        raise HTTPException(exc.status_code, exc.to_dict()) from exc

    answer_parts: list[str] = []
    usage_seen: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
    # The *resolved* model, which can differ from the requested one when the
    # request names a rolling alias ("gemini-flash-latest" -> "gemini-3.6-flash").
    # Stored for display so a cache hit reports what actually generated the
    # answer; the namespace still keys on the requested name, since that is what
    # the caller asked for.
    resolved_model = {"name": model}
    ttft: list[float] = []

    async def body() -> AsyncIterator[str]:
        error_type: str | None = None
        try:
            # Sources first: the client can render citations immediately and
            # then fill in the prose as it streams.
            yield sse("sources", {"sources": [s.model_dump(mode="json") for s in sources]})

            event = first
            while event is not None:
                if isinstance(event, TextDelta):
                    if not ttft:
                        ttft.append((time.perf_counter() - started) * 1000)
                    answer_parts.append(event.text)
                elif isinstance(event, StreamDone):
                    usage_seen["input_tokens"] = event.usage.input_tokens
                    usage_seen["output_tokens"] = event.usage.output_tokens
                    resolved_model["name"] = event.model or model
                yield _frame(event, provider.name, started, cached=False)
                try:
                    event = await iterator.__anext__()
                except StopAsyncIteration:
                    break

            # Only cache a complete answer. Storing a partial one (client
            # disconnect, mid-stream error) would serve a truncated response to
            # every future match -- so this runs on the success path only.
            answer = "".join(answer_parts)
            if answer.strip():
                await cache.store(
                    question=payload.question,
                    embedding=embedding,
                    answer=answer,
                    provider=provider.name,
                    # Namespace by the requested name, display the resolved one.
                    model=model,
                    display_model=resolved_model["name"],
                    corpus_version=corpus_version,
                    input_tokens=usage_seen["input_tokens"],
                    output_tokens=usage_seen["output_tokens"],
                )
        except ProviderError as exc:
            error_type = type(exc).__name__
            log.warning("rag_stream_error", provider=provider.name, error=exc.message)
            yield sse("error", exc.to_dict())
        except Exception as exc:  # noqa: BLE001
            error_type = type(exc).__name__
            log.exception("rag_stream_unexpected_error", provider=provider.name)
            yield sse("error", {"error": "internal error", "type": type(exc).__name__})
        finally:
            await metrics.record(
                RequestRecord(
                    endpoint="rag_query",
                    provider=provider.name,
                    model=resolved_model["name"],
                    input_tokens=usage_seen["input_tokens"],
                    output_tokens=usage_seen["output_tokens"],
                    latency_ms=(time.perf_counter() - started) * 1000,
                    ttft_ms=ttft[0] if ttft else None,
                    error_type=error_type,
                    status_code=200 if error_type is None else 500,
                    request_id=current_request_id(),
                    principal=request.scope.get("docxpo_principal"),
                )
            )
            await iterator.aclose()

    return StreamingResponse(body(), media_type="text/event-stream", headers=SSE_HEADERS)
