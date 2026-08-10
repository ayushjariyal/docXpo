"""request_logs for per-request observability

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-07
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "request_logs",
        sa.Column("id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("endpoint", sa.String(length=64), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column("principal", sa.String(length=128), nullable=True),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.Numeric(12, 2), nullable=False, server_default="0"),
        sa.Column("ttft_ms", sa.Numeric(12, 2), nullable=True),
        sa.Column("cached", sa.Boolean(), nullable=False, server_default=sa.false()),
        # Numeric, not double precision: money must not accumulate binary
        # floating-point error across thousands of tiny per-request costs.
        sa.Column("cost_usd", sa.Numeric(14, 8), nullable=False, server_default="0"),
        sa.Column("cost_saved_usd", sa.Numeric(14, 8), nullable=False, server_default="0"),
        sa.Column("priced", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("status_code", sa.Integer(), nullable=False, server_default="200"),
        sa.Column("finish_reason", sa.String(length=64), nullable=True),
        sa.Column("error_type", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_request_logs"),
    )
    # Every /metrics query is "filter by window, then aggregate", so this is the
    # index that carries the endpoint.
    op.create_index("ix_request_logs_created_at", "request_logs", ["created_at"])
    op.create_index(
        "ix_request_logs_provider_created", "request_logs", ["provider", "created_at"]
    )
    op.create_index(
        "ix_request_logs_principal_created", "request_logs", ["principal", "created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_request_logs_principal_created", table_name="request_logs")
    op.drop_index("ix_request_logs_provider_created", table_name="request_logs")
    op.drop_index("ix_request_logs_created_at", table_name="request_logs")
    op.drop_table("request_logs")
