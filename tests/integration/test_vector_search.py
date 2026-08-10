"""RAG retrieval against real pgvector.

This is the piece the unit tests could not reach: the `<=>` cosine-distance
operator, the ORDER BY that lets the HNSW index do the work, and the fact that
similarity is `1 - distance`. A fake repository would have happily agreed with
whatever ranking we implemented, correct or not.
"""

from __future__ import annotations

import math
import uuid

import pytest

from app.repositories.document_repository import DocumentRepository
from app.services.rag_service import build_context
from tests.integration.conftest import needs_postgres

pytestmark = [pytest.mark.integration, needs_postgres]

DIM = 768


def unit(*values: float) -> list[float]:
    """A unit-length vector. The whole pipeline assumes normalization."""
    v = [0.0] * DIM
    for i, x in enumerate(values):
        v[i] = float(x)
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


async def _search_in(repo, doc, query, **kw):
    """Search scoped to the document this test created.

    Unscoped, these assertions would also see whatever the developer's database
    already holds -- the tier rolls back its own writes, but it cannot hide rows
    committed by real use. A test that passes on a fresh machine and fails on a
    used one is worse than no test.
    """
    kw.setdefault("top_k", 10)
    kw.setdefault("min_similarity", 0.0)
    return await repo.search(query, document_id=doc.id, **kw)


async def _seed(session, vectors: list[list[float]], filename="itest.txt"):
    repo = DocumentRepository(session)
    doc = await repo.create_document(
        filename=filename, content_type="text/plain", content="x" * 50
    )
    await repo.add_chunks(
        doc,
        [(i, f"chunk {i} of {filename}", i * 10, i * 10 + 9, v) for i, v in enumerate(vectors)],
    )
    return repo, doc


async def test_results_are_ordered_by_cosine_similarity(session) -> None:
    query = unit(1, 0, 0)
    # Deliberately inserted worst-first, so a repository that returned insertion
    # order rather than ranked order would fail.
    repo, doc = await _seed(session, [unit(0, 1, 0), unit(1, 0.6, 0), unit(1, 0.05, 0)])

    hits = await _search_in(repo, doc, query, top_k=3)

    assert [h.chunk_index for h in hits] == [2, 1, 0]
    sims = [h.similarity for h in hits]
    assert sims == sorted(sims, reverse=True)


async def test_similarity_is_one_minus_cosine_distance(session) -> None:
    """An identical vector must score 1.0, not 0.0 -- the sign of `<=>` is easy
    to get backwards, and doing so inverts the entire ranking."""
    query = unit(1, 0, 0)
    repo, doc = await _seed(session, [query])

    hits = await _search_in(repo, doc, query, top_k=1)

    assert hits[0].similarity == pytest.approx(1.0, abs=1e-5)


async def test_orthogonal_vector_scores_zero(session) -> None:
    repo, doc = await _seed(session, [unit(0, 1, 0)])

    hits = await _search_in(repo, doc, unit(1, 0, 0), top_k=1)

    assert hits[0].similarity == pytest.approx(0.0, abs=1e-5)


async def test_top_k_limits_results(session) -> None:
    repo, doc = await _seed(session, [unit(1, i * 0.01, 0) for i in range(10)])

    assert len(await _search_in(repo, doc, unit(1, 0, 0), top_k=3)) == 3


async def test_min_similarity_filters_weak_matches(session) -> None:
    """The floor is applied after the ordered fetch, so it must still filter."""
    repo, doc = await _seed(session, [unit(1, 0, 0), unit(0, 1, 0)])

    hits = await _search_in(repo, doc, unit(1, 0, 0), min_similarity=0.5)

    assert len(hits) == 1
    assert hits[0].similarity > 0.5


async def test_document_filter_scopes_the_search(session) -> None:
    repo, doc_a = await _seed(session, [unit(1, 0, 0)], filename="a.txt")
    _, doc_b = await _seed(session, [unit(1, 0, 0)], filename="b.txt")

    hits = await repo.search(unit(1, 0, 0), top_k=10, min_similarity=0.0, document_id=doc_b.id)

    assert len(hits) == 1
    assert hits[0].document_id == doc_b.id
    assert hits[0].document_filename == "b.txt"
    assert all(h.document_id != doc_a.id for h in hits)


async def test_search_joins_the_filename_for_citations(session) -> None:
    """Sources are cited by filename, so the join must actually populate it."""
    repo, doc = await _seed(session, [unit(1, 0, 0)], filename="runbook.md")

    hits = await _search_in(repo, doc, unit(1, 0, 0), top_k=1)

    assert hits[0].document_filename == "runbook.md"


async def test_empty_corpus_returns_no_hits(session) -> None:
    repo = DocumentRepository(session)

    # A vector no seeded chunk is near; the tier's rollback keeps other tests'
    # rows out of this one, but the similarity floor makes it robust anyway.
    assert await repo.search(unit(0, 0, 1), top_k=5, min_similarity=0.999) == []


async def test_deleting_a_document_removes_its_chunks(session) -> None:
    """The FK cascade is a database guarantee, not an ORM one."""
    repo, doc = await _seed(session, [unit(1, 0, 0), unit(1, 0.1, 0)])
    assert len(await _search_in(repo, doc, unit(1, 0, 0))) == 2

    assert await repo.delete_document(doc.id) is True

    remaining = await repo.search(unit(1, 0, 0), top_k=10, min_similarity=0.0)
    assert all(h.document_id != doc.id for h in remaining)


async def test_deleting_a_missing_document_reports_false(session) -> None:
    repo = DocumentRepository(session)

    assert await repo.delete_document(uuid.uuid4()) is False


async def test_context_block_is_numbered_from_one(session) -> None:
    """Citations in the answer are [1]-based and must line up with `sources`."""
    repo, doc = await _seed(session, [unit(1, 0, 0), unit(1, 0.1, 0)], filename="notes.txt")
    hits = await _search_in(repo, doc, unit(1, 0, 0), top_k=2)

    context = build_context(hits)

    assert context.startswith("[1] (from notes.txt)")
    assert "[2] (from notes.txt)" in context
    assert "[0]" not in context
