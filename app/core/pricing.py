"""Token pricing, used to estimate per-request cost.

## Why cost is stored, not computed on read

The price of a model changes. If `/metrics` multiplied historical token counts
by *today's* rate, last month's spend would silently change every time a vendor
adjusted pricing — and a cost report that rewrites its own history is worse than
no cost report. So cost is calculated once, at record time, with the rate then
in effect, and stored on the row.

## Why unknown models cost zero *and say so*

A model with no entry here records `cost_usd = 0` and sets `priced = false`.
Guessing a rate would produce a number that looks authoritative and is wrong;
reporting zero silently would understate spend. The `priced` flag lets
`/metrics` report exactly how much of the traffic it could actually price, so
an unpriced model shows up as a gap rather than as free.

Rates are USD per 1,000,000 tokens. Verify against the vendor's pricing page
before trusting any figure here — these are point-in-time and go stale.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class Price:
    """USD per 1M tokens."""

    input_per_mtok: Decimal
    output_per_mtok: Decimal


# Decimal, not float: money. 0.1 + 0.2 != 0.3 in binary floating point, and
# summing thousands of tiny per-request costs is exactly where that compounds.
def _p(inp: str, out: str) -> Price:
    return Price(Decimal(inp), Decimal(out))


# Keys are matched by longest prefix, so "claude-opus-5" covers any dated
# snapshot of it without needing an entry per release.
DEFAULT_PRICING: dict[str, Price] = {
    # Anthropic (per 1M tokens)
    "claude-opus-5": _p("5.00", "25.00"),
    "claude-opus-4-8": _p("5.00", "25.00"),
    "claude-opus-4-7": _p("5.00", "25.00"),
    "claude-sonnet-5": _p("3.00", "15.00"),
    "claude-sonnet-4-6": _p("3.00", "15.00"),
    "claude-haiku-4-5": _p("1.00", "5.00"),
    # OpenAI
    "gpt-4o-mini": _p("0.15", "0.60"),
    "gpt-4o": _p("2.50", "10.00"),
    "text-embedding-3-small": _p("0.02", "0.00"),
    # Google. NOTE: the free tier bills nothing, so these are the *paid* rates
    # and will overstate cost while you are inside the free quota. Treated as
    # the honest default: showing what the traffic would cost is more useful
    # than showing zero and hiding the real number.
    "gemini-2.0-flash": _p("0.10", "0.40"),
    "gemini-2.5-flash": _p("0.30", "2.50"),
    "gemini-3-flash": _p("0.30", "2.50"),
    "gemini-3.6-flash": _p("0.30", "2.50"),
    "gemini-flash": _p("0.30", "2.50"),
    "gemini-embedding-001": _p("0.15", "0.00"),
    # Local models cost nothing to run, and that is a real answer, not a gap.
    "llama": _p("0", "0"),
    "qwen": _p("0", "0"),
    "mistral": _p("0", "0"),
}


class Pricebook:
    def __init__(self, overrides_json: str = "") -> None:
        self._prices = dict(DEFAULT_PRICING)
        if overrides_json.strip():
            # Shape: {"model-name": {"input": 1.23, "output": 4.56}}
            for name, rates in json.loads(overrides_json).items():
                self._prices[name] = _p(str(rates["input"]), str(rates["output"]))

    def lookup(self, model: str) -> Price | None:
        """Longest-prefix match, so dated snapshots inherit their family's rate."""
        if not model:
            return None
        if (exact := self._prices.get(model)) is not None:
            return exact

        best: tuple[int, Price] | None = None
        for prefix, price in self._prices.items():
            if model.startswith(prefix) and (best is None or len(prefix) > best[0]):
                best = (len(prefix), price)
        return best[1] if best else None

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> tuple[Decimal, bool]:
        """Returns (usd, priced). `priced` is False when the model is unknown."""
        price = self.lookup(model)
        if price is None:
            return Decimal("0"), False

        usd = (
            Decimal(input_tokens) * price.input_per_mtok
            + Decimal(output_tokens) * price.output_per_mtok
        ) / Decimal(1_000_000)
        # 8dp: a single cheap request can cost fractions of a cent, and rounding
        # to 4dp would floor thousands of them to zero.
        return usd.quantize(Decimal("0.00000001")), True
