"""File extraction tests.

The regression these exist for: uploading a PDF used to return 201 and index
the container's raw bytes (`%PDF-1.4`, `1 0 obj<</Type/Catalog…`) as searchable
content. Silently storing garbage is worse than refusing the file, because
nothing signals that it happened.
"""

from __future__ import annotations

import zlib

import pytest

from app.services.extraction import (
    NoExtractableText,
    UnsupportedFileType,
    extract,
    extract_pdf,
)


def build_pdf(pages: list[str]) -> bytes:
    """A minimal but genuinely valid PDF with a Flate-compressed text layer.

    Compressed on purpose: real PDFs from Word or Chrome compress their content
    streams, so the text is *not* visible in the raw bytes. An uncompressed stub
    would let a broken extractor pass by accidentally finding the words in the
    container.
    """
    objects: dict[int, bytes] = {}

    def esc(s: str) -> str:
        return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")

    n_pages = len(pages)
    # 1 catalog, 2 page tree, 3 font, then (content, page) per page.
    page_ids = [4 + 2 * i + 1 for i in range(n_pages)]

    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode()
    objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    for i, text in enumerate(pages):
        stream = f"BT /F1 12 Tf 72 720 Td ({esc(text)}) Tj ET".encode()
        packed = zlib.compress(stream)
        cid = 4 + 2 * i
        objects[cid] = (
            f"<< /Length {len(packed)} /Filter /FlateDecode >>\nstream\n".encode()
            + packed
            + b"\nendstream"
        )
        objects[page_ids[i]] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {cid} 0 R /Resources << /Font << /F1 3 0 R >> >> >>".encode()
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for num in sorted(objects):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + objects[num] + b"\nendobj\n"

    xref_at = len(out)
    high = max(objects) + 1
    out += f"xref\n0 {high}\n".encode()
    out += b"0000000000 65535 f \n"
    for num in range(1, high):
        out += f"{offsets.get(num, 0):010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {high} /Root 1 0 R >>\nstartxref\n{xref_at}\n".encode()
        + b"%%EOF\n"
    )
    return bytes(out)


# ---- PDFs ------------------------------------------------------------------


def test_the_fixture_really_hides_its_text() -> None:
    """Guards the guard: if the text were visible raw, the tests below prove nothing."""
    raw = build_pdf(["The retry limit is five attempts per key."])

    assert b"%PDF-" == raw[:5]
    assert b"retry limit" not in raw, "content stream is not actually compressed"


def test_pdf_text_is_extracted_not_the_container() -> None:
    raw = build_pdf(["The retry limit is five attempts per key."])

    result = extract(raw, filename="runbook.pdf", content_type="application/pdf")

    assert result.kind == "pdf"
    assert "retry limit is five attempts" in result.text
    # The regression: none of the PDF's structure may reach the index.
    assert "%PDF" not in result.text
    assert "/Type /Catalog" not in result.text
    assert "endobj" not in result.text


def test_multipage_pdf_keeps_page_markers() -> None:
    """Page markers give the chunker a strong separator and aid citations."""
    raw = build_pdf(["Alpha section content.", "Beta section content."])

    result = extract_pdf(raw)

    assert result.pages == 2
    assert "[page 1]" in result.text and "[page 2]" in result.text
    assert "Alpha section" in result.text and "Beta section" in result.text


def test_pdf_detected_by_signature_not_by_declared_type() -> None:
    """Content-Type is client-supplied; the bytes are what the file actually is."""
    raw = build_pdf(["Detected by magic bytes."])

    result = extract(raw, filename="mislabelled.txt", content_type="text/plain")

    assert result.kind == "pdf"
    assert "Detected by magic bytes" in result.text


def test_pdf_with_no_text_layer_is_rejected_clearly() -> None:
    """A scanned PDF has pages but no text; it must not index as empty."""
    raw = build_pdf([" "])

    with pytest.raises(NoExtractableText, match="scanned"):
        extract_pdf(raw)


def test_corrupt_pdf_is_rejected_not_decoded() -> None:
    raw = b"%PDF-1.4\nthis is not really a pdf at all"

    with pytest.raises((UnsupportedFileType, NoExtractableText)):
        extract(raw, filename="broken.pdf", content_type="application/pdf")


# ---- other binaries --------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "hint"),
    [
        (b"PK\x03\x04" + b"\x00" * 40, "ZIP"),
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 40, "PNG"),
        (b"\xff\xd8\xff\xe0" + b"\x00" * 40, "JPEG"),
        (b"\xd0\xcf\x11\xe0" + b"\x00" * 40, "Office"),
        (b"\x7fELF" + b"\x00" * 40, "executable"),
    ],
)
def test_known_binaries_are_named_in_the_error(raw: bytes, hint: str) -> None:
    """A useful refusal says what it thinks the file was."""
    with pytest.raises(UnsupportedFileType, match=hint):
        extract(raw, filename="x.bin", content_type="application/octet-stream")


def test_unknown_binary_is_rejected_not_indexed() -> None:
    """The core fix: unrecognised binary must never be decoded into the index."""
    raw = bytes(range(256)) * 8

    with pytest.raises(UnsupportedFileType):
        extract(raw, filename="mystery.dat", content_type="application/octet-stream")


def test_empty_file_is_rejected() -> None:
    with pytest.raises(UnsupportedFileType, match="empty"):
        extract(b"   \n  ", filename="blank.txt", content_type="text/plain")


# ---- text ------------------------------------------------------------------


def test_plain_text_still_works() -> None:
    result = extract(b"Just some notes.\n\nSecond paragraph.", filename="n.md",
                     content_type="text/markdown")

    assert result.kind == "text"
    assert result.text.startswith("Just some notes.")


def test_utf8_is_preserved() -> None:
    result = extract("café · naïve · 日本語".encode(), filename="u.txt",
                     content_type="text/plain")

    assert "café" in result.text and "日本語" in result.text


def test_a_few_bad_bytes_do_not_lose_the_document() -> None:
    """One corrupt byte should cost one character, not the whole upload."""
    raw = b"Mostly fine text, " + b"\xff\xfe" + b" and more after it."

    result = extract(raw, filename="messy.txt", content_type="text/plain")

    assert "Mostly fine text" in result.text
    assert "and more after it" in result.text


def test_a_flood_of_bad_bytes_is_treated_as_binary() -> None:
    """The line between 'text with a glitch' and 'not text at all'."""
    raw = b"\xff\xfe\xff\xfe" * 500

    with pytest.raises(UnsupportedFileType):
        extract(raw, filename="junk.txt", content_type="text/plain")


def test_nul_bytes_mean_binary() -> None:
    """A NUL effectively never appears in real text -- the strongest signal."""
    raw = b"looks like text at first\x00\x00\x00but is not"

    with pytest.raises(UnsupportedFileType):
        extract(raw, filename="x.txt", content_type="text/plain")
