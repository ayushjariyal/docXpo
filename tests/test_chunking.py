"""Chunking tests.

The properties that matter for retrieval quality, not the exact split points --
those depend on the separator list and would make the tests brittle.
"""

from __future__ import annotations

import pytest

from app.services.chunking import chunk_text, normalize_text


def test_short_text_is_one_chunk() -> None:
    chunks = chunk_text("Just a sentence.", size=1200, overlap=200)

    assert len(chunks) == 1
    assert chunks[0].text == "Just a sentence."
    assert chunks[0].index == 0


def test_empty_input_produces_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_every_chunk_respects_the_size_budget() -> None:
    """size + overlap is the real ceiling, since overlap is prepended."""
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(30))

    chunks = chunk_text(text, size=600, overlap=100)

    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 700, f"chunk {c.index} is {len(c.text)} chars"


def test_prefers_paragraph_boundaries() -> None:
    """With paragraphs that fit, splits should land on the blank lines."""
    paras = [f"Paragraph number {i} has some content in it." for i in range(8)]
    chunks = chunk_text("\n\n".join(paras), size=120, overlap=0)

    # No chunk should start mid-word.
    for c in chunks:
        assert c.text[0].isupper() or c.text[0].isdigit()


def test_overlap_preserves_boundary_spanning_facts() -> None:
    """The core reason overlap exists.

    A fact split across a boundary must survive intact in at least one chunk.
    """
    filler = "Padding sentence here. " * 20
    fact = "The retry limit is set to five attempts."
    text = filler + fact + " " + filler

    chunks = chunk_text(text, size=300, overlap=120)

    assert any(fact in c.text for c in chunks), "fact was destroyed by chunking"


def test_zero_overlap_is_allowed() -> None:
    chunks = chunk_text("word " * 500, size=200, overlap=0)

    assert len(chunks) > 1
    # Without overlap, chunk starts must be strictly increasing and non-nested.
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert b.char_start >= a.char_start


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError, match="must be smaller"):
        chunk_text("hello", size=100, overlap=100)


def test_text_with_no_separators_is_hard_cut() -> None:
    """Pathological input (e.g. a base64 blob) must still terminate."""
    chunks = chunk_text("x" * 1000, size=100, overlap=0)

    assert len(chunks) >= 10
    assert all(len(c.text) <= 100 for c in chunks)


def test_chunk_offsets_point_into_the_normalized_text() -> None:
    text = normalize_text("\n\n".join(f"Para {i} content here." for i in range(12)))
    chunks = chunk_text(text, size=100, overlap=0)

    for c in chunks:
        assert 0 <= c.char_start < len(text)
        assert c.char_end <= len(text) + 1


def test_normalize_collapses_line_endings_and_blank_runs() -> None:
    assert normalize_text("a\r\n\r\nb") == "a\n\nb"
    assert normalize_text("a\n\n\n\n\nb") == "a\n\nb"
    assert normalize_text("  padded  ") == "padded"


def test_indices_are_sequential() -> None:
    chunks = chunk_text("word " * 800, size=250, overlap=50)

    assert [c.index for c in chunks] == list(range(len(chunks)))
