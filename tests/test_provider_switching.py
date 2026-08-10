"""Provider selection must be pure configuration.

Requirement: switching between gemini/anthropic/openai/ollama needs an env var
change and nothing else. These tests fail if someone hard-codes a provider
anywhere in the resolution path.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.llm.anthropic_provider import AnthropicProvider
from app.llm.errors import ProviderNotConfigured
from app.llm.gemini_provider import GeminiProvider
from app.llm.ollama import OllamaProvider
from app.llm.openai_provider import OpenAIProvider
from app.llm.registry import ProviderRegistry

ALL_KEYS = {
    "gemini_api_key": "k",
    "anthropic_api_key": "k",
    "openai_api_key": "k",
    "_env_file": None,
    "log_level": "CRITICAL",
}


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("gemini", GeminiProvider),
        ("anthropic", AnthropicProvider),
        ("openai", OpenAIProvider),
        ("ollama", OllamaProvider),
    ],
)
async def test_default_provider_env_var_selects_the_implementation(
    provider: str, expected: type
) -> None:
    registry = ProviderRegistry(Settings(default_provider=provider, **ALL_KEYS))

    resolved = await registry.get()

    assert isinstance(resolved, expected)
    assert resolved.name == provider


async def test_per_request_override_beats_the_default() -> None:
    """The `provider` field in a request body overrides DEFAULT_PROVIDER."""
    registry = ProviderRegistry(Settings(default_provider="gemini", **ALL_KEYS))

    assert (await registry.get()).name == "gemini"
    assert (await registry.get("anthropic")).name == "anthropic"


async def test_gemini_is_the_shipped_default() -> None:
    assert Settings(_env_file=None).default_provider == "gemini"
    assert Settings(_env_file=None).gemini_model == "gemini-flash-latest"


async def test_switching_default_does_not_require_other_providers_keys() -> None:
    """Only the selected provider's credentials are needed to boot.

    An Anthropic deployment must not be forced to hold a Gemini key.
    """
    registry = ProviderRegistry(
        Settings(
            _env_file=None,
            default_provider="anthropic",
            anthropic_api_key="k",
            gemini_api_key=None,
            openai_api_key=None,
            validate_provider_on_startup=False,
        )
    )

    assert (await registry.get()).name == "anthropic"
    # ...and the unconfigured one still fails clearly when explicitly asked for.
    with pytest.raises(ProviderNotConfigured, match="GEMINI_API_KEY"):
        await registry.get("gemini")


async def test_instances_are_reused_per_provider() -> None:
    """One instance per provider per process -- pools must not be rebuilt."""
    registry = ProviderRegistry(Settings(default_provider="gemini", **ALL_KEYS))

    assert await registry.get("gemini") is await registry.get("gemini")


async def test_unknown_provider_lists_the_valid_options() -> None:
    registry = ProviderRegistry(Settings(default_provider="gemini", **ALL_KEYS))

    with pytest.raises(ProviderNotConfigured) as exc_info:
        await registry.get("bard")

    message = str(exc_info.value)
    for name in ("gemini", "anthropic", "openai", "ollama"):
        assert name in message
