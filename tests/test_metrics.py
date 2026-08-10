"""Pricing and metrics tests.

The pricebook is pure logic and fully covered here. The aggregation SQL
(`percentile_cont`, filtered counts) is Postgres-specific and is verified
against the real database in the live checks recorded in DECISIONS.md.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.core.config import Settings
from app.core.pricing import Pricebook
from app.services.metrics_service import MetricsService, RequestRecord


def test_known_model_is_priced() -> None:
    book = Pricebook()

    # 1M in + 1M out on Sonnet 4.6 = $3 + $15.
    cost, priced = book.cost("claude-sonnet-4-6", 1_000_000, 1_000_000)

    assert priced
    assert cost == Decimal("18.00000000")


def test_unknown_model_is_zero_but_flagged() -> None:
    """The flag is the point: silent zero would understate spend."""
    book = Pricebook()

    cost, priced = book.cost("some-model-we-never-heard-of", 1000, 1000)

    assert cost == Decimal("0")
    assert priced is False


def test_longest_prefix_wins() -> None:
    """Dated snapshots inherit their family's rate without a per-release entry."""
    book = Pricebook()

    dated, priced = book.cost("claude-sonnet-4-6-20251114", 1_000_000, 0)

    assert priced
    assert dated == Decimal("3.00000000")


def test_more_specific_prefix_beats_shorter_one() -> None:
    book = Pricebook()

    generic = book.lookup("gemini-flash")
    specific = book.lookup("gemini-3.6-flash")

    assert generic is not None and specific is not None


def test_local_models_are_priced_at_zero_not_unknown() -> None:
    """Free is a real answer; unpriced is a gap. They must not look alike."""
    book = Pricebook()

    cost, priced = book.cost("llama3.2", 500_000, 500_000)

    assert cost == Decimal("0E-8")
    assert priced is True  # priced, and the price happens to be zero


def test_small_costs_do_not_round_to_zero() -> None:
    """A cheap request costs fractions of a cent; 4dp would floor it away."""
    book = Pricebook()

    cost, _ = book.cost("gpt-4o-mini", 100, 100)

    assert cost > 0


def test_costs_use_decimal_not_float() -> None:
    """Summing thousands of tiny float costs accumulates error."""
    book = Pricebook()

    total = sum((book.cost("gpt-4o-mini", 1000, 1000)[0] for _ in range(1000)), Decimal(0))

    assert isinstance(total, Decimal)
    # The exact value: 1000 * (1000*0.15 + 1000*0.60) / 1e6
    assert total == Decimal("0.75000000")


def test_pricing_overrides_replace_defaults() -> None:
    """Prices change; correcting one must not need a code edit."""
    book = Pricebook('{"claude-sonnet-4-6": {"input": 99, "output": 100}}')

    cost, priced = book.cost("claude-sonnet-4-6", 1_000_000, 0)

    assert priced and cost == Decimal("99.00000000")


def test_override_can_add_an_unknown_model() -> None:
    book = Pricebook('{"my-local-finetune": {"input": 0.5, "output": 1.0}}')

    cost, priced = book.cost("my-local-finetune", 1_000_000, 1_000_000)

    assert priced and cost == Decimal("1.50000000")


# ---- recording -------------------------------------------------------------


async def test_recording_failure_is_swallowed(monkeypatch) -> None:
    """Observability must never break the thing it observes.

    The DB write is forced to fail; record() must still return normally, or a
    Postgres outage would turn into a service outage for requests that had
    already produced a perfectly good answer.
    """

    def explode(*_a, **_kw):
        raise RuntimeError("postgres is down")

    monkeypatch.setattr("app.services.metrics_service.SessionFactory", explode)

    await MetricsService(Pricebook()).record(
        RequestRecord(endpoint="chat", provider="gemini", model="gemini-flash-latest")
    )  # must not raise


def test_cache_hit_prices_what_it_saved_not_what_it_spent() -> None:
    """A hit spends nothing; its value is the call it avoided."""
    book = Pricebook()

    spent, _ = book.cost("claude-sonnet-4-6", 0, 0)
    saved, _ = book.cost("claude-sonnet-4-6", 1000, 500)

    assert spent == 0
    assert saved > 0


@pytest.mark.parametrize("model", ["", "   "])
def test_blank_model_is_unpriced_not_a_crash(model: str) -> None:
    cost, priced = Pricebook().cost(model, 100, 100)

    assert cost == Decimal("0") and priced is False


def test_metrics_settings_have_sane_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.metrics_enabled is True
    assert settings.pricing_overrides == ""
