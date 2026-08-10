"""Retrieval-augmented generation: retrieve, then generate grounded in what was
retrieved.

The whole value is in the prompt contract below. Retrieval is the easy half;
the hard half is making the model *use* the context and admit when the context
does not contain the answer, rather than falling back on its parametric memory
and sounding equally confident either way.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.logging import get_logger
from app.llm.base import LLMProvider, Message, StreamEvent
from app.llm.embeddings import EmbeddingProvider
from app.repositories.document_repository import DocumentRepository, ScoredChunk

log = get_logger(__name__)

# Rules chosen for specific failure modes:
#   1+2  the point of RAG -- ground the answer, and make "not in the docs" a
#        first-class allowed answer instead of a reason to improvise.
#   3    citations make a wrong answer traceable to the chunk that caused it,
#        which is the difference between a debuggable and an opaque system.
#   4    without this the model narrates the mechanism ("Based on the provided
#        context excerpts...") in every reply, which reads terribly.
SYSTEM_PROMPT = """You answer questions using only the context provided below.

Rules:
1. Base your answer solely on the context. Do not use prior knowledge.
2. If the context does not contain the answer, say so plainly and stop. Do not \
guess or fill gaps from memory.
3. Cite the sources you used by their bracketed number, like [1] or [2].
4. Do not mention "the context" or "the provided documents" in your answer. \
Just answer the question.

Context:
{context}"""


def build_context(chunks: Sequence[ScoredChunk]) -> str:
    """Render retrieved chunks into the numbered block the prompt refers to.

    Numbering is 1-based and matches the `sources` array sent to the client, so
    a [2] in the answer is resolvable by the caller.
    """
    return "\n\n".join(
        f"[{i}] (from {c.document_filename})\n{c.content}"
        for i, c in enumerate(chunks, start=1)
    )


class RagService:
    def __init__(
        self,
        session: AsyncSession,
        embedder: EmbeddingProvider,
        settings: Settings,
    ) -> None:
        self._repo = DocumentRepository(session)
        self._embedder = embedder
        self._settings = settings

    async def embed_query(self, question: str) -> list[float]:
        """Embed a question once, for both cache lookup and retrieval.

        Exposed separately because the semantic cache needs the same vector
        retrieval does. Sharing it means the cache costs **zero** extra
        embedding calls -- a cache that had to embed on its own would spend an
        API round trip before it could even tell you it was a miss.
        """
        # task="query" -- the asymmetric counterpart of the "document" encoding
        # used at ingest time.
        return await self._embedder.embed_one(question, task="query")

    async def retrieve(
        self,
        question: str,
        *,
        top_k: int | None = None,
        min_similarity: float | None = None,
        document_id: uuid.UUID | None = None,
        embedding: list[float] | None = None,
    ) -> list[ScoredChunk]:
        vector = embedding if embedding is not None else await self.embed_query(question)

        chunks = await self._repo.search(
            vector,
            top_k=top_k or self._settings.rag_top_k,
            min_similarity=(
                self._settings.rag_min_similarity
                if min_similarity is None
                else min_similarity
            ),
            document_id=document_id,
        )
        log.info(
            "rag_retrieved",
            question_chars=len(question),
            hits=len(chunks),
            top_similarity=round(chunks[0].similarity, 4) if chunks else None,
        )
        return chunks

    def build_messages(
        self, question: str, chunks: Sequence[ScoredChunk]
    ) -> list[Message]:
        """Assemble the prompt.

        The context goes in a **system** message, not the user turn. Two
        reasons: it keeps the user's question as the last thing the model reads
        (which measurably improves instruction-following on long prompts), and
        the provider abstraction already routes system messages correctly for
        each vendor -- Anthropic's top-level `system`, Gemini's
        `system_instruction`, a role for the others.
        """
        return [
            Message(role="system", content=SYSTEM_PROMPT.format(context=build_context(chunks))),
            Message(role="user", content=question),
        ]

    def stream_answer(
        self,
        provider: LLMProvider,
        question: str,
        chunks: Sequence[ScoredChunk],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        return provider.stream_chat(
            self.build_messages(question, chunks),
            model=model,
            max_tokens=max_tokens,
        )
