"""Shared FastAPI dependencies.

Annotated aliases (``DbSession``, ``ChatSvc``, …) keep route signatures short
and mean the wiring can change in one place. Later phases add ``ApiKey`` here.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.session import get_db_session
from app.llm.embeddings import EmbeddingProvider
from app.llm.registry import ProviderRegistry
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService
from app.services.metrics_service import MetricsService
from app.services.rag_service import RagService
from app.services.semantic_cache import SemanticCache


def get_redis(request: Request) -> Redis:
    """Pull the shared client off app.state (populated by the lifespan hook)."""
    return request.app.state.redis


def get_registry(request: Request) -> ProviderRegistry:
    return request.app.state.provider_registry


def get_embedder(request: Request) -> EmbeddingProvider:
    """Shared embedding client, built once in the lifespan hook."""
    return request.app.state.embedder


def get_chat_service(request: Request) -> ChatService:
    # Built per request, but it is a stateless facade over the registry -- the
    # expensive object (HTTP pools) lives in the registry on app.state.
    return ChatService(get_registry(request))


def get_metrics(request: Request) -> MetricsService:
    return request.app.state.metrics


def get_cache(request: Request) -> SemanticCache:
    return SemanticCache(request.app.state.redis, request.app.state.settings)


# Declared before the factories below so they can be used as annotations.
DbSession = Annotated[AsyncSession, Depends(get_db_session)]


def get_document_service(request: Request, session: DbSession) -> DocumentService:
    return DocumentService(
        session,
        get_embedder(request),
        request.app.state.settings,
        cache=get_cache(request),
    )


def get_rag_service(request: Request, session: DbSession) -> RagService:
    return RagService(session, get_embedder(request), request.app.state.settings)


RedisClient = Annotated[Redis, Depends(get_redis)]
AppSettings = Annotated[Settings, Depends(get_settings)]
Registry = Annotated[ProviderRegistry, Depends(get_registry)]
ChatSvc = Annotated[ChatService, Depends(get_chat_service)]
Embedder = Annotated[EmbeddingProvider, Depends(get_embedder)]
DocSvc = Annotated[DocumentService, Depends(get_document_service)]
RagSvc = Annotated[RagService, Depends(get_rag_service)]
Cache = Annotated[SemanticCache, Depends(get_cache)]
Metrics = Annotated[MetricsService, Depends(get_metrics)]
