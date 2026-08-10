"""Embedding providers.

Deliberately a *separate* interface from ``LLMProvider`` rather than more
methods on it, for one concrete reason: **Anthropic has no embedding model at
all.** Folding `embed()` into LLMProvider would force AnthropicProvider to
implement a method it can never satisfy. Keeping them separate means you can
run chat on Anthropic and embeddings on Gemini, which is a real deployment.

Everything here returns **unit-normalized** vectors -- see `_normalize` for why
that is not optional.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar, Literal

import openai
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from app.core.config import Settings
from app.core.logging import get_logger
from app.llm.errors import (
    ProviderAuthError,
    ProviderError,
    ProviderNotConfigured,
    ProviderRateLimited,
)

log = get_logger(__name__)

# Gemini produces *asymmetric* embeddings: a passage and a question about that
# passage are encoded differently on purpose, which measurably improves
# retrieval over encoding both the same way. Using the wrong one silently
# degrades recall, so the two call sites are distinguished by this type.
TaskType = Literal["document", "query"]


def _normalize(vector: list[float]) -> list[float]:
    """Scale a vector to unit length.

    Required, not cosmetic. `gemini-embedding-001` is a Matryoshka model: its
    native 3072-dim output is unit-normalized, but asking for a reduced
    dimensionality truncates the vector, which destroys that property --
    measured L2 norm at 768 dims is ~0.587, not 1.0.

    Two things break if we store un-normalized vectors:

    1. Cosine distance still ranks correctly (it divides by the norms), but any
       switch to the faster inner-product operator would silently return
       nonsense.
    2. Absolute similarity *scores* stop being comparable between vectors,
       which makes a fixed similarity threshold meaningless -- and Phase 4's
       semantic cache is built entirely on such a threshold.
    """
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector  # degenerate; nothing sensible to divide by
    return [x / norm for x in vector]


class EmbeddingProvider(ABC):
    name: ClassVar[str]

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Vector width. Must match the DB column; see the Chunk model."""

    @abstractmethod
    async def embed(
        self, texts: Sequence[str], *, task: TaskType = "document"
    ) -> list[list[float]]:
        """Embed a batch of texts. Returns one unit-normalized vector each."""

    async def embed_one(self, text: str, *, task: TaskType = "query") -> list[float]:
        return (await self.embed([text], task=task))[0]

    @abstractmethod
    async def check_credentials(self) -> None: ...


class GeminiEmbedder(EmbeddingProvider):
    name: ClassVar[str] = "gemini"

    _TASK = {
        "document": "RETRIEVAL_DOCUMENT",
        "query": "RETRIEVAL_QUERY",
    }

    def __init__(self, settings: Settings) -> None:
        if settings.gemini_api_key is None:
            raise ProviderNotConfigured(
                "GEMINI_API_KEY is required for embeddings", provider=self.name
            )
        self._model = settings.gemini_embedding_model
        self._dims = settings.embedding_dimensions
        self._client = genai.Client(
            api_key=settings.gemini_api_key.get_secret_value(),
            http_options=genai_types.HttpOptions(
                timeout=int(settings.llm_timeout_seconds * 1000)
            ),
        )

    @property
    def dimensions(self) -> int:
        return self._dims

    async def embed(
        self, texts: Sequence[str], *, task: TaskType = "document"
    ) -> list[list[float]]:
        if not texts:
            return []
        try:
            resp = await self._client.aio.models.embed_content(
                model=self._model,
                contents=list(texts),
                config=genai_types.EmbedContentConfig(
                    task_type=self._TASK[task],
                    # 768, not the native 3072: pgvector's HNSW and IVFFlat
                    # indexes both refuse columns wider than 2000 dimensions, so
                    # a 3072-dim column could only ever be scanned sequentially.
                    output_dimensionality=self._dims,
                ),
            )
            return [_normalize(list(e.values)) for e in resp.embeddings]
        except genai_errors.APIError as exc:
            raise _translate_gemini(exc, self.name) from exc

    async def check_credentials(self) -> None:
        try:
            await self._client.aio.models.get(model=self._model)
        except genai_errors.APIError as exc:
            raise _translate_gemini(exc, self.name) from exc


class OpenAIEmbedder(EmbeddingProvider):
    name: ClassVar[str] = "openai"

    def __init__(self, settings: Settings) -> None:
        if settings.openai_api_key is None:
            raise ProviderNotConfigured(
                "OPENAI_API_KEY is required for embeddings", provider=self.name
            )
        self._model = settings.openai_embedding_model
        self._dims = settings.embedding_dimensions
        self._client = openai.AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=settings.openai_base_url,
            timeout=settings.llm_timeout_seconds,
        )

    @property
    def dimensions(self) -> int:
        return self._dims

    async def embed(
        self, texts: Sequence[str], *, task: TaskType = "document"
    ) -> list[list[float]]:
        # OpenAI embeddings are symmetric -- there is no document/query
        # distinction -- so `task` is accepted and ignored. Keeping it in the
        # signature is what lets callers stay provider-agnostic.
        if not texts:
            return []
        try:
            resp = await self._client.embeddings.create(
                model=self._model, input=list(texts), dimensions=self._dims
            )
            # Already unit-normalized by OpenAI, but normalizing again is cheap
            # and guarantees the invariant holds no matter the provider.
            return [_normalize(d.embedding) for d in resp.data]
        except openai.APIError as exc:
            raise ProviderError(
                f"OpenAI embedding failed: {exc}", provider=self.name
            ) from exc

    async def check_credentials(self) -> None:
        try:
            await self._client.models.retrieve(self._model)
        except openai.APIError as exc:
            raise ProviderError(
                f"OpenAI embedding check failed: {exc}", provider=self.name
            ) from exc


def _translate_gemini(exc: genai_errors.APIError, provider: str) -> ProviderError:
    code = getattr(exc, "code", None)
    detail = getattr(exc, "message", None) or str(exc)
    if code in (401, 403) or (code == 400 and "api key" in detail.lower()):
        return ProviderAuthError(f"Gemini rejected our credentials: {detail}", provider=provider)
    if code == 429:
        return ProviderRateLimited(f"Gemini quota exceeded: {detail}", provider=provider)
    return ProviderError(f"Gemini embedding error: {detail}", provider=provider)


_EMBEDDERS: dict[str, type[EmbeddingProvider]] = {
    GeminiEmbedder.name: GeminiEmbedder,
    OpenAIEmbedder.name: OpenAIEmbedder,
}


def build_embedder(settings: Settings) -> EmbeddingProvider:
    name = settings.embedding_provider
    if name not in _EMBEDDERS:
        raise ProviderNotConfigured(
            f"Unknown embedding provider {name!r}. "
            f"Available: {', '.join(sorted(_EMBEDDERS))}"
        )
    return _EMBEDDERS[name](settings)
