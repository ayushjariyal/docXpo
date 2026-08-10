"""Wire contract for document ingestion and retrieval."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.core.config import ProviderName


class DocumentOut(BaseModel):
    id: uuid.UUID
    filename: str
    content_type: str
    char_count: int
    chunk_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentListOut(BaseModel):
    total: int
    documents: list[DocumentOut]


class TextUploadIn(BaseModel):
    """Paste text directly, as an alternative to multipart file upload."""

    filename: str = Field(default="pasted.txt", max_length=512)
    content: str = Field(min_length=1, max_length=2_000_000)


class SourceOut(BaseModel):
    """One retrieved chunk, as returned alongside an answer."""

    n: int  # 1-based; matches the [n] citations in the generated answer
    document_id: uuid.UUID
    filename: str
    chunk_index: int
    similarity: float
    excerpt: str


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=8_000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    min_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    # Restrict retrieval to a single document.
    document_id: uuid.UUID | None = None
    provider: ProviderName | None = None
    model: str | None = Field(default=None, max_length=200)
    max_tokens: int | None = Field(default=None, ge=1, le=32_000)


class RetrieveOut(BaseModel):
    """Retrieval without generation -- useful for tuning chunking and top_k."""

    question: str
    sources: list[SourceOut]
