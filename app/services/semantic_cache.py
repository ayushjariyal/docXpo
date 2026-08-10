"""Semantic response cache backed by Redis.

An exact-match cache is nearly useless for natural language: "what is the retry
limit?" and "what's the retry limit" are different strings and would miss. A
semantic cache compares the *meaning* of the incoming question against previous
ones by cosine similarity, and serves the stored answer when they are close
enough.

## Why brute force, not RediSearch

Plain `redis:7-alpine` has no vector index; getting one means the much larger
`redis-stack` image. Instead we fetch the namespace's vectors and score them
with a single numpy matmul.

That is O(n), so the cache is deliberately **bounded** (`cache_max_entries`).
The numbers make this fine at our scale: 500 entries x 768 dims is one
(500, 768) @ (768,) product -- roughly 0.3ms, against an LLM call costing
seconds. Above ~10k entries the linear scan stops being free and the right move
is RediSearch's HNSW, or reusing the pgvector index we already have.

## Why the dot product *is* the cosine similarity here

Because Phase 3 guarantees every embedding is unit-normalized. For unit
vectors, cos(a, b) = a . b, so no division by norms is needed. If that
invariant were ever broken, every score here would be silently wrong -- which
is why normalization lives in the embedding provider rather than being done
ad hoc at call sites.

## Layout

    sc:{ns}:idx      ZSET   id -> created_at   (bounded; oldest trimmed)
    sc:{ns}:v:{id}   BYTES  float32 vector     (TTL)
    sc:{ns}:e:{id}   JSON   the cached answer  (TTL)

Vectors are stored as raw float32 rather than JSON: 3KB instead of ~10KB, and
they load straight into numpy with no parsing. Only the *winning* entry's JSON
is fetched, so a lookup transfers vectors plus at most one answer.

The namespace embeds provider, model and corpus version, so a cached Gemini
answer can never be served to an Anthropic request, and re-indexing documents
retires the old namespace automatically.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass

import numpy as np
from redis.asyncio import Redis

from app.core.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)

CORPUS_VERSION_KEY = "docxpo:corpus_version"


@dataclass(frozen=True, slots=True)
class CachedAnswer:
    question: str
    answer: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    similarity: float
    age_seconds: float


class SemanticCache:
    def __init__(self, redis: Redis, settings: Settings) -> None:
        self._redis = redis
        self._enabled = settings.cache_enabled
        self._threshold = settings.cache_similarity_threshold
        self._ttl = settings.cache_ttl_seconds
        self._max_entries = settings.cache_max_entries

    # ---- corpus versioning -------------------------------------------------

    async def corpus_version(self) -> int:
        """Monotonic counter bumped whenever the document set changes.

        It is part of the cache namespace, so adding or deleting a document
        instantly orphans every previously cached RAG answer instead of serving
        answers derived from documents that no longer exist. The orphaned keys
        are not deleted -- they simply expire, which keeps ingestion O(1)
        rather than "scan and delete the whole cache".
        """
        raw = await self._redis.get(CORPUS_VERSION_KEY)
        return int(raw) if raw else 0

    async def bump_corpus_version(self) -> int:
        version = int(await self._redis.incr(CORPUS_VERSION_KEY))
        log.info("corpus_version_bumped", version=version)
        return version

    # ---- lookup / store ----------------------------------------------------

    def _ns(self, provider: str, model: str, corpus_version: int) -> str:
        return f"sc:{provider}:{model}:v{corpus_version}"

    async def lookup(
        self, embedding: list[float], *, provider: str, model: str, corpus_version: int
    ) -> CachedAnswer | None:
        if not self._enabled:
            return None

        ns = self._ns(provider, model, corpus_version)
        # Newest first: if the cache is over-full we would rather score recent
        # entries than ancient ones.
        ids = await self._redis.zrevrange(f"{ns}:idx", 0, self._max_entries - 1)
        if not ids:
            return None

        raw_vectors = await self._redis.mget([f"{ns}:v:{i.decode()}" for i in ids])

        live_ids: list[str] = []
        matrix: list[np.ndarray] = []
        expired: list[bytes] = []
        for entry_id, raw in zip(ids, raw_vectors, strict=True):
            if raw is None:
                # TTL expired the vector but the index still lists it.
                expired.append(entry_id)
                continue
            live_ids.append(entry_id.decode())
            matrix.append(np.frombuffer(raw, dtype=np.float32))

        if expired:
            # Lazy cleanup: cheaper than a background sweeper and self-healing.
            await self._redis.zrem(f"{ns}:idx", *expired)
        if not live_ids:
            return None

        query = np.asarray(embedding, dtype=np.float32)
        # Unit vectors => dot product is the cosine similarity.
        scores = np.stack(matrix) @ query

        best = int(np.argmax(scores))
        best_score = float(scores[best])
        if best_score < self._threshold:
            log.info("cache_miss", best_similarity=round(best_score, 4), candidates=len(live_ids))
            return None

        payload = await self._redis.get(f"{ns}:e:{live_ids[best]}")
        if payload is None:
            return None  # raced with expiry between the two reads

        data = json.loads(payload)
        return CachedAnswer(
            question=data["question"],
            answer=data["answer"],
            provider=data["provider"],
            model=data["model"],
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            similarity=best_score,
            age_seconds=max(0.0, time.time() - data.get("created_at", time.time())),
        )

    async def store(
        self,
        *,
        question: str,
        embedding: list[float],
        answer: str,
        provider: str,
        model: str,
        corpus_version: int,
        display_model: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        if not self._enabled or not answer.strip():
            return

        ns = self._ns(provider, model, corpus_version)
        entry_id = uuid.uuid4().hex
        now = time.time()

        vector = np.asarray(embedding, dtype=np.float32).tobytes()
        payload = json.dumps(
            {
                "question": question,
                "answer": answer,
                "provider": provider,
                # What actually produced the answer, which differs from the
                # namespace key when the request named a rolling alias.
                "model": display_model or model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "created_at": now,
            }
        )

        # Pipelined so one round trip does all four writes.
        pipe = self._redis.pipeline()
        pipe.setex(f"{ns}:v:{entry_id}", self._ttl, vector)
        pipe.setex(f"{ns}:e:{entry_id}", self._ttl, payload)
        pipe.zadd(f"{ns}:idx", {entry_id: now})
        # Trim oldest beyond the cap. Keeps the linear scan bounded, which is
        # the whole reason brute force is viable.
        pipe.zremrangebyrank(f"{ns}:idx", 0, -(self._max_entries + 1))
        pipe.expire(f"{ns}:idx", self._ttl)
        await pipe.execute()

        log.info("cache_stored", provider=provider, model=model, answer_chars=len(answer))

    async def stats(self, *, provider: str, model: str, corpus_version: int) -> dict:
        ns = self._ns(provider, model, corpus_version)
        return {
            "enabled": self._enabled,
            "threshold": self._threshold,
            "ttl_seconds": self._ttl,
            "max_entries": self._max_entries,
            "entries": int(await self._redis.zcard(f"{ns}:idx")),
            "namespace": ns,
        }

    async def clear(self, *, provider: str, model: str, corpus_version: int) -> int:
        """Drop every entry in one namespace. Returns how many were indexed."""
        ns = self._ns(provider, model, corpus_version)
        ids = await self._redis.zrange(f"{ns}:idx", 0, -1)
        if ids:
            keys = [f"{ns}:v:{i.decode()}" for i in ids]
            keys += [f"{ns}:e:{i.decode()}" for i in ids]
            await self._redis.delete(*keys)
        await self._redis.delete(f"{ns}:idx")
        return len(ids)
