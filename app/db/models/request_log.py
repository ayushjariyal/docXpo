"""Per-request observability record.

One row per LLM-backed request. This is where the previous phases converge:
the request id from Phase 1's middleware, the provider/model and token counts
from Phase 2, the cache verdict from Phase 4, and the API-key fingerprint from
Phase 5 — all on one row, so a single query can answer "what did this key cost
us last week, and how often did the cache save us a call?".
"""

from __future__ import annotations

from decimal import Decimal

from sqlalchemy import Boolean, Index, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class RequestLog(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "request_logs"

    # Which endpoint: "chat", "rag_query", "rag_retrieve".
    endpoint: Mapped[str] = mapped_column(String(64), nullable=False)
    # Correlates with the x-request-id header and every log line for the request.
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Fingerprint from Phase 5 -- never the raw API key.
    principal: Mapped[str | None] = mapped_column(String(128), nullable=True)

    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)

    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Wall clock for the whole request.
    latency_ms: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False, default=0)
    # Time to first token. Null on cache hits and non-streaming calls. Kept
    # separate from latency_ms because for a streaming endpoint TTFT is the
    # number a user actually feels, while total latency is what capacity
    # planning needs.
    ttft_ms: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)

    cached: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Numeric, not float: money. Computed once at write time with the rate then
    # in effect (see app/core/pricing.py) so historical spend never changes.
    cost_usd: Mapped[Decimal] = mapped_column(
        Numeric(14, 8), nullable=False, default=Decimal("0")
    )
    # What a cache hit avoided spending. Zero on a miss. This is what turns the
    # cache from "feels fast" into a number.
    cost_saved_usd: Mapped[Decimal] = mapped_column(
        Numeric(14, 8), nullable=False, default=Decimal("0")
    )
    # False when the model had no entry in the pricebook, so /metrics can report
    # how much of the traffic it could actually price rather than implying $0.
    priced: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    status_code: Mapped[int] = mapped_column(Integer, nullable=False, default=200)
    finish_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_type: Mapped[str | None] = mapped_column(String(64), nullable=True)

    __table_args__ = (
        # Every /metrics query filters on a time window and then aggregates, so
        # this is the index that matters. DESC because queries ask for "recent".
        Index("ix_request_logs_created_at", "created_at"),
        # Per-provider and per-key breakdowns.
        Index("ix_request_logs_provider_created", "provider", "created_at"),
        Index("ix_request_logs_principal_created", "principal", "created_at"),
    )
