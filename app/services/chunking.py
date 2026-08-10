"""Recursive character text splitting.

## Why chunk at all

Two independent reasons, and it is worth being clear that they are different:

1. **Embedding models have an input limit**, so a long document physically
   cannot be embedded in one call.
2. **More importantly: one vector per document is a bad retrieval unit.** An
   embedding is an average of everything in its input, so embedding a
   40-page manual produces a vector that means "this is a manual" and matches
   almost nothing specifically. Smaller chunks give sharper vectors.

## The strategy: recursive splitting on semantic boundaries

Naively slicing every N characters cuts mid-sentence and mid-word, which
produces chunks that begin and end on fragments and embed poorly.

Instead we try a list of separators in descending order of semantic strength:

    paragraph break  ->  line break  ->  sentence end  ->  space  ->  hard cut

The text is split on the strongest separator available. Any piece still too
large is split again by the next separator down, recursively. The hard cut is
the last resort and only fires on pathological input (a 5000-character string
with no whitespace, e.g. a base64 blob).

The effect is that chunks break at paragraph boundaries when it can, sentence
boundaries when it must, and mid-word essentially never.

## Chunk size: ~1200 characters

Roughly 300 tokens. The trade-off in both directions:

* **Too small** (say 200 chars) -- a chunk loses the context that makes it
  meaningful. "It supports up to 512 connections." is useless without knowing
  what "it" is, and retrieval returns many near-duplicate fragments.
* **Too large** (say 8000 chars) -- the embedding averages over too many
  distinct ideas and stops being specific, so retrieval degrades. It also
  wastes the generator's context window: you inject 8000 characters to deliver
  one relevant sentence.

~1200 sits near the size of a well-formed paragraph or two, which is usually
the unit at which a document actually makes a single point.

## Overlap: 200 characters (~17%)

Chunks share their edges. Without overlap, a fact that straddles a boundary is
destroyed: if the split lands between "The retry limit is" and "set to 5", then
*neither* chunk answers "what is the retry limit?" -- the first has the question
without the answer, the second the answer without the question.

Overlap means boundary-spanning content appears intact in at least one chunk.

The cost is real and worth stating: ~17% more rows, ~17% more embedding calls,
~17% more storage, and near-duplicate chunks can occupy two of your top-k slots
with the same information. 10-20% is the usual sweet spot; below that boundary
loss shows up, above it the duplication starts crowding out results.
"""

from __future__ import annotations

from dataclasses import dataclass

# Ordered strongest-to-weakest. "" is the sentinel meaning "hard cut".
SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", "")


@dataclass(frozen=True, slots=True)
class Chunk:
    index: int
    text: str
    char_start: int
    char_end: int


def normalize_text(text: str) -> str:
    """Collapse line-ending styles and runs of blank lines.

    Done before splitting so that "\r\n\r\n" is recognised as a paragraph break
    and doesn't defeat the strongest separator.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def _split_recursive(text: str, size: int, seps: tuple[str, ...]) -> list[str]:
    """Split `text` into pieces of at most `size`, preferring early separators."""
    if len(text) <= size:
        return [text] if text else []

    if not seps:
        # Ran out of separators entirely: hard-cut. Only reachable on input
        # with no whitespace at all.
        return [text[i : i + size] for i in range(0, len(text), size)]

    sep, rest = seps[0], seps[1:]
    if sep == "":
        return [text[i : i + size] for i in range(0, len(text), size)]

    parts = text.split(sep)
    out: list[str] = []
    buf = ""

    for part in parts:
        # Re-attach the separator we split on, so the text round-trips and the
        # sentence terminator stays with its sentence.
        piece = part + sep

        if len(piece) > size:
            # This single piece is oversized on its own -- flush what we have
            # and recurse into it with the next-weaker separator.
            if buf:
                out.append(buf)
                buf = ""
            out.extend(_split_recursive(piece, size, rest))
            continue

        if len(buf) + len(piece) <= size:
            buf += piece
        else:
            out.append(buf)
            buf = piece

    if buf:
        out.append(buf)

    return [p for p in (s.strip() for s in out) if p]


def chunk_text(text: str, *, size: int = 1200, overlap: int = 200) -> list[Chunk]:
    """Split text into overlapping chunks aligned to semantic boundaries.

    `char_start`/`char_end` are offsets into the *normalized* text, so a chunk
    can be traced back to its position in the source document -- useful for
    citations and for debugging bad retrievals.
    """
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")

    text = normalize_text(text)
    if not text:
        return []

    pieces = _split_recursive(text, size, SEPARATORS)

    chunks: list[Chunk] = []
    cursor = 0  # search position, so repeated text maps to the right offset

    for i, piece in enumerate(pieces):
        found = text.find(piece, cursor)
        start = found if found != -1 else cursor

        body = piece
        if i > 0 and overlap:
            # Prepend the tail of the previous piece. Taken from the source text
            # rather than the previous chunk so the overlap is contiguous even
            # where stripping removed whitespace between pieces.
            tail_start = max(0, start - overlap)
            body = text[tail_start:start].lstrip() + piece
            start = tail_start

        end = start + len(body)
        chunks.append(Chunk(index=i, text=body, char_start=start, char_end=end))
        cursor = found + len(piece) if found != -1 else cursor + len(piece)

    return chunks
