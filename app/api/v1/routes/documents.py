"""Document ingestion endpoints."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from app.api.deps import DocSvc
from app.core.logging import get_logger
from app.llm.errors import ProviderError
from app.schemas.documents import DocumentListOut, DocumentOut, TextUploadIn
from app.services.document_service import EmptyDocumentError
from app.services.extraction import NoExtractableText, UnsupportedFileType, extract

log = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

# Guards against a multi-hundred-MB upload being read into memory and then
# embedded, which would be slow and expensive rather than merely large.
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def _extract(raw: bytes, *, filename: str, content_type: str) -> str:
    """Bytes to indexable text, or a 4xx.

    Anything unreadable raises rather than being decoded into garbage -- see
    app/services/extraction.py for why silently indexing a PDF's container
    bytes was worse than refusing it.
    """
    try:
        return extract(raw, filename=filename, content_type=content_type).text
    except NoExtractableText as exc:
        # 422: the request was well-formed and the format was understood, there
        # is simply nothing in it to index.
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except UnsupportedFileType as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc


async def _ingest(service: DocSvc, *, filename: str, content_type: str, content: str):
    try:
        return await service.ingest(
            filename=filename, content_type=content_type, content=content
        )
    except EmptyDocumentError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except ProviderError as exc:
        # Embedding failed. Surfaced with the provider's own status so a quota
        # error is a 429 the caller can back off on, not an opaque 500.
        raise HTTPException(exc.status_code, exc.to_dict()) from exc


@router.post(
    "",
    response_model=DocumentOut,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document file",
)
async def upload_document(
    service: DocSvc, file: Annotated[UploadFile, File()]
) -> DocumentOut:
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit",
        )
    filename = file.filename or "upload.txt"
    content_type = file.content_type or "application/octet-stream"

    doc = await _ingest(
        service,
        filename=filename,
        content_type=content_type,
        content=_extract(raw, filename=filename, content_type=content_type),
    )
    return DocumentOut.model_validate(doc)


@router.post(
    "/text",
    response_model=DocumentOut,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest pasted text",
)
async def upload_text(payload: TextUploadIn, service: DocSvc) -> DocumentOut:
    doc = await _ingest(
        service,
        filename=payload.filename,
        content_type="text/plain",
        content=payload.content,
    )
    return DocumentOut.model_validate(doc)


@router.get("", response_model=DocumentListOut, summary="List documents")
async def list_documents(
    service: DocSvc,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> DocumentListOut:
    docs = await service.list_documents(limit=limit, offset=offset)
    return DocumentListOut(
        total=await service.count(),
        documents=[DocumentOut.model_validate(d) for d in docs],
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its chunks",
)
async def delete_document(document_id: uuid.UUID, service: DocSvc) -> None:
    if not await service.delete(document_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
