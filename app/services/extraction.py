"""Turning an uploaded file into indexable text.

## The bug this replaces

The upload route used to do `raw.decode("utf-8", errors="replace")` on every
file. For a PDF that "succeeds": you get the container's structure —
`%PDF-1.4`, `1 0 obj<</Type/Catalog…`, and the compressed content streams as
replacement characters — and that garbage is chunked, embedded and stored as
searchable content. The upload returns **201**, so nothing looks wrong until
retrieval starts returning noise.

Silently indexing garbage is worse than refusing the file, because there is no
signal that anything went wrong.

## Detect by magic bytes, not by Content-Type

`Content-Type` is supplied by the client. Browsers guess it from the file
extension, `curl -F` lets you assert anything, and some clients send
`application/octet-stream` for everything. The first few bytes of the file are
what the file actually *is*, so that is what we branch on. The declared type is
only used as a hint for the error message.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.logging import get_logger

log = get_logger(__name__)


# Tuning for the "is this text?" heuristic. See _looks_like_text.
ABSOLUTE_BAD_CHAR_ALLOWANCE = 4
MAX_BAD_CHAR_RATIO = 0.05


class UnsupportedFileType(ValueError):
    """The bytes are not something we can turn into text."""


class NoExtractableText(ValueError):
    """Recognised format, but it contains no text layer."""


@dataclass(frozen=True, slots=True)
class Extracted:
    text: str
    kind: str  # "pdf" | "text"
    pages: int | None = None


# Signatures of binary formats we can recognise well enough to reject with a
# useful message, rather than emitting a generic "not text".
_BINARY_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"PK\x03\x04", "a ZIP-based file (.docx/.xlsx/.pptx or a zip archive)"),
    (b"\x89PNG", "a PNG image"),
    (b"\xff\xd8\xff", "a JPEG image"),
    (b"GIF8", "a GIF image"),
    (b"\x1f\x8b", "a gzip archive"),
    (b"\x00\x00\x00\x18ftyp", "a video file"),
    (b"\xd0\xcf\x11\xe0", "a legacy Office file (.doc/.xls)"),
    (b"%!PS", "a PostScript file"),
    (b"\x7fELF", "a binary executable"),
)


def _looks_like_text(raw: bytes) -> bool:
    """Heuristic: does this decode as UTF-8 without heavy corruption?

    A NUL byte effectively never appears in real text and is the single
    strongest signal of a binary file, so it is checked first. Beyond that we
    decode and count replacement characters: a handful means an encoding quirk
    worth tolerating, a flood means these are not characters at all.
    """
    if b"\x00" in raw[:8192]:
        return False

    sample = raw[:8192]
    decoded = sample.decode("utf-8", errors="replace")
    if not decoded:
        return False

    bad = decoded.count("�")

    # An absolute allowance *before* the ratio, because a ratio is meaningless
    # on a short file: a 40-character note containing one mis-encoded quote is
    # 5% bad and would otherwise be rejected as binary. A real binary trips the
    # NUL check or produces far more than a handful.
    if bad <= ABSOLUTE_BAD_CHAR_ALLOWANCE:
        return True

    return (bad / len(decoded)) < MAX_BAD_CHAR_RATIO


def extract_pdf(raw: bytes) -> Extracted:
    """Pull the text layer out of a PDF.

    pypdf is pure Python with no system dependencies — unlike pdfplumber or
    anything built on poppler, which would need binaries in the Docker image.
    It reads the text layer only: it does no OCR, which is why an image-only
    scan raises rather than returning an empty document.
    """
    try:
        reader = PdfReader(io.BytesIO(raw))
    except (PdfReadError, Exception) as exc:  # noqa: B014
        raise UnsupportedFileType(f"could not parse the PDF: {exc}") from exc

    if reader.is_encrypted:
        # An empty-password PDF is common and decrypts silently; a real one
        # cannot be read and must say so rather than yielding zero pages.
        try:
            reader.decrypt("")
        except Exception as exc:
            raise UnsupportedFileType(
                "the PDF is password-protected and cannot be read"
            ) from exc

    parts: list[str] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            # One malformed page should not lose the other 200.
            log.warning("pdf_page_extract_failed", page=number, error=str(exc))
            continue
        if text.strip():
            # A page marker gives the chunker a strong separator to split on and
            # keeps a citation traceable to roughly where it came from.
            parts.append(f"[page {number}]\n{text.strip()}")

    if not parts:
        raise NoExtractableText(
            "no text found in this PDF. It is probably a scanned image — "
            "extracting that needs OCR, which docXpo does not do."
        )

    return Extracted(text="\n\n".join(parts), kind="pdf", pages=len(reader.pages))


def extract(raw: bytes, *, filename: str, content_type: str) -> Extracted:
    """Bytes to text, or a clear refusal.

    Never returns garbage: anything we cannot read raises, so the caller turns
    it into a 4xx instead of indexing noise.
    """
    if not raw.strip():
        raise UnsupportedFileType("the file is empty")

    # PDFs are detected by signature, so a .pdf sent as text/plain (or a PDF
    # renamed to .txt) is still handled correctly.
    if raw[:5] == b"%PDF-":
        extracted = extract_pdf(raw)
        log.info(
            "pdf_extracted",
            filename=filename,
            pages=extracted.pages,
            chars=len(extracted.text),
        )
        return extracted

    for signature, description in _BINARY_SIGNATURES:
        if raw.startswith(signature):
            raise UnsupportedFileType(
                f"this looks like {description}, which docXpo cannot index. "
                "Supported: plain text (.txt, .md, .csv, .json, source code) and PDF."
            )

    if not _looks_like_text(raw):
        raise UnsupportedFileType(
            f"'{filename}' does not appear to be text (declared as {content_type}). "
            "Supported: plain text (.txt, .md, .csv, .json, source code) and PDF."
        )

    # errors="replace" is still right *here*: we have established this is text,
    # so a stray bad byte should cost one character rather than the upload.
    return Extracted(text=raw.decode("utf-8", errors="replace"), kind="text")
