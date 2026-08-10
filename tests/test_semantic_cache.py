"""Semantic cache tests.

Run against a fake Redis that implements only the commands the cache uses, so
they exercise the real similarity maths and the real key layout without needing
a running server.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.core.config import Settings
from app.services.semantic_cache import SemanticCache


class FakeRedis:
    """Minimal in-memory stand-in for the Redis commands the cache uses."""

    def __init__(self) -> None:
        self.strings: dict[str, bytes] = {}
        self.zsets: dict[str, dict[str, float]] = {}
        self.counters: dict[str, int] = {}

    async def get(self, key):
        return self.strings.get(key)

    async def mget(self, keys):
        return [self.strings.get(k) for k in keys]

    async def setex(self, key, _ttl, value):
        self.strings[key] = value if isinstance(value, bytes) else str(value).encode()

    async def incr(self, key):
        self.counters[key] = self.counters.get(key, 0) + 1
        self.strings[key] = str(self.counters[key]).encode()
        return self.counters[key]

    async def zadd(self, key, mapping):
        self.zsets.setdefault(key, {}).update(mapping)

    async def zrevrange(self, key, start, end):
        items = sorted(self.zsets.get(key, {}).items(), key=lambda kv: -kv[1])
        sliced = items[start : (end + 1 if end >= 0 else None)]
        return [k.encode() for k, _ in sliced]

    async def zrange(self, key, start, end):
        items = sorted(self.zsets.get(key, {}).items(), key=lambda kv: kv[1])
        sliced = items[start : (end + 1 if end >= 0 else None)]
        return [k.encode() for k, _ in sliced]

    async def zrem(self, key, *members):
        z = self.zsets.get(key, {})
        for m in members:
            z.pop(m.decode() if isinstance(m, bytes) else m, None)

    async def zcard(self, key):
        return len(self.zsets.get(key, {}))

    async def zremrangebyrank(self, key, start, end):
        items = sorted(self.zsets.get(key, {}).items(), key=lambda kv: kv[1])
        if end < 0:
            end = len(items) + end
        for k, _ in items[start : end + 1]:
            self.zsets[key].pop(k, None)

    async def delete(self, *keys):
        for k in keys:
            self.strings.pop(k, None)
            self.zsets.pop(k, None)

    async def expire(self, key, ttl):
        return True

    def pipeline(self):
        return _FakePipeline(self)


class _FakePipeline:
    """Queues commands and replays them on execute(), like redis-py does."""

    def __init__(self, redis: FakeRedis) -> None:
        self._redis = redis
        self._ops: list[tuple[str, tuple]] = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self._ops.append((name, args))
            return self

        return record

    async def execute(self):
        for name, args in self._ops:
            await getattr(self._redis, name)(*args)
        self._ops.clear()


def unit(*values: float) -> list[float]:
    """Build a unit-length 768-dim vector -- the cache assumes normalization."""
    v = np.zeros(768, dtype=np.float32)
    for i, x in enumerate(values):
        v[i] = x
    n = math.sqrt(float(v @ v))
    return (v / n).tolist()


def _cache(redis: FakeRedis, **overrides) -> SemanticCache:
    settings = Settings(_env_file=None, log_level="CRITICAL", **overrides)
    return SemanticCache(redis, settings)


NS = {"provider": "gemini", "model": "m1", "corpus_version": 0}


async def test_miss_on_empty_cache() -> None:
    cache = _cache(FakeRedis())

    assert await cache.lookup(unit(1, 0), **NS) is None


async def test_identical_question_hits() -> None:
    cache = _cache(FakeRedis())
    vec = unit(1, 0, 0)

    await cache.store(question="q", embedding=vec, answer="42", **NS)
    hit = await cache.lookup(vec, **NS)

    assert hit is not None
    assert hit.answer == "42"
    assert hit.similarity == pytest.approx(1.0, abs=1e-5)


async def test_near_duplicate_above_threshold_hits() -> None:
    cache = _cache(FakeRedis(), cache_similarity_threshold=0.98)
    await cache.store(question="q", embedding=unit(1, 0), answer="42", **NS)

    # cos ~= 0.995 -- a paraphrase-level match.
    hit = await cache.lookup(unit(1, 0.1), **NS)

    assert hit is not None and hit.similarity > 0.98


async def test_different_question_below_threshold_misses() -> None:
    """The failure this threshold exists to prevent: answering the wrong question."""
    cache = _cache(FakeRedis(), cache_similarity_threshold=0.98)
    await cache.store(question="q", embedding=unit(1, 0), answer="42", **NS)

    # cos ~= 0.707 -- clearly a different question.
    assert await cache.lookup(unit(1, 1), **NS) is None


async def test_threshold_is_configurable_and_changes_the_verdict() -> None:
    """Same vectors, opposite outcome -- the trade-off in one test."""
    vec_a, vec_b = unit(1, 0), unit(1, 0.3)  # cos ~= 0.958

    strict = _cache(FakeRedis(), cache_similarity_threshold=0.98)
    await strict.store(question="q", embedding=vec_a, answer="42", **NS)
    assert await strict.lookup(vec_b, **NS) is None

    loose = _cache(FakeRedis(), cache_similarity_threshold=0.90)
    await loose.store(question="q", embedding=vec_a, answer="42", **NS)
    assert await loose.lookup(vec_b, **NS) is not None


async def test_best_match_wins_not_first_match() -> None:
    cache = _cache(FakeRedis(), cache_similarity_threshold=0.5)
    await cache.store(question="far", embedding=unit(1, 0.9), answer="WRONG", **NS)
    await cache.store(question="near", embedding=unit(1, 0.02), answer="RIGHT", **NS)

    hit = await cache.lookup(unit(1, 0), **NS)

    assert hit is not None and hit.answer == "RIGHT"


async def test_namespaced_by_provider_and_model() -> None:
    """A Gemini answer must never be served to an Anthropic request."""
    cache = _cache(FakeRedis())
    vec = unit(1, 0)
    await cache.store(question="q", embedding=vec, answer="from-gemini", **NS)

    other = {"provider": "anthropic", "model": "m1", "corpus_version": 0}
    assert await cache.lookup(vec, **other) is None
    assert await cache.lookup(vec, **NS) is not None


async def test_corpus_version_bump_invalidates_everything() -> None:
    """Indexing a document must retire answers derived from the old corpus."""
    redis = FakeRedis()
    cache = _cache(redis)
    vec = unit(1, 0)
    await cache.store(question="q", embedding=vec, answer="old", **NS)

    new_version = await cache.bump_corpus_version()

    fresh = {"provider": "gemini", "model": "m1", "corpus_version": new_version}
    assert await cache.lookup(vec, **fresh) is None
    # The old namespace still holds it; it just expires rather than being swept.
    assert await cache.lookup(vec, **NS) is not None


async def test_disabled_cache_never_hits() -> None:
    cache = _cache(FakeRedis(), cache_enabled=False)
    vec = unit(1, 0)

    await cache.store(question="q", embedding=vec, answer="42", **NS)

    assert await cache.lookup(vec, **NS) is None


async def test_empty_answers_are_not_stored() -> None:
    """A blank answer would otherwise poison every future match."""
    redis = FakeRedis()
    cache = _cache(redis)

    await cache.store(question="q", embedding=unit(1, 0), answer="   ", **NS)

    assert await cache.lookup(unit(1, 0), **NS) is None


async def test_cache_is_bounded() -> None:
    """The linear scan is only viable because the entry count is capped."""
    redis = FakeRedis()
    cache = _cache(redis, cache_max_entries=5)

    for i in range(12):
        await cache.store(question=f"q{i}", embedding=unit(1, i * 0.01), answer="a", **NS)

    stats = await cache.stats(**NS)
    assert stats["entries"] <= 5


async def test_expired_vectors_are_pruned_from_the_index() -> None:
    """TTL removes the payload; the index entry must be cleaned up lazily."""
    redis = FakeRedis()
    cache = _cache(redis)
    await cache.store(question="q", embedding=unit(1, 0), answer="42", **NS)

    # Simulate TTL expiry of the vector while the ZSET still lists the id.
    for key in [k for k in redis.strings if ":v:" in k]:
        del redis.strings[key]

    assert await cache.lookup(unit(1, 0), **NS) is None
    assert (await cache.stats(**NS))["entries"] == 0


async def test_clear_empties_the_namespace() -> None:
    cache = _cache(FakeRedis())
    await cache.store(question="q", embedding=unit(1, 0), answer="42", **NS)

    removed = await cache.clear(**NS)

    assert removed == 1
    assert await cache.lookup(unit(1, 0), **NS) is None


def test_default_threshold_excludes_the_measured_negation() -> None:
    """Regression guard on the number that matters.

    Measured on real embeddings: a negated question ("does NOT use") scored
    0.9752 against its opposite -- higher than a legitimate looser paraphrase.
    The default must sit above it, or the cache will confidently answer the
    inverse of what was asked.
    """
    threshold = Settings(_env_file=None).cache_similarity_threshold

    assert threshold > 0.9752, "default would admit a negated question"
