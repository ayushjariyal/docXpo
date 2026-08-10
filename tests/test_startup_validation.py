"""The app must refuse to start when its default provider is unusable.

This is the behaviour that guarantees docXpo never serves anything other than a
real model response: there is no fake provider to fall back to, so a missing or
rejected API key has to stop the boot rather than surface later.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.llm.errors import ProviderAuthError, ProviderNotConfigured
from app.llm.registry import ProviderRegistry
from app.main import create_app, lifespan


def _settings(**overrides) -> Settings:
    base = {
        "environment": "local",
        "log_level": "CRITICAL",
        "default_provider": "anthropic",
        "anthropic_api_key": None,
        # Embeddings are a separate provider with its own key. Supplied here so
        # these tests exercise the *chat* provider's failure path specifically.
        "gemini_api_key": "embedding-key",
        # Don't make a network call in tests; the "missing key" path is what we
        # are exercising, and it fails before any request would be made.
        "validate_provider_on_startup": False,
        "_env_file": None,  # ignore any real .env on the developer's machine
    }
    return Settings(**{**base, **overrides})


async def test_missing_api_key_is_a_clear_error() -> None:
    registry = ProviderRegistry(_settings())

    with pytest.raises(ProviderNotConfigured) as exc_info:
        await registry.validate_default()

    assert "ANTHROPIC_API_KEY" in str(exc_info.value)


async def test_startup_aborts_when_the_key_is_missing() -> None:
    """The whole point: no key => the app does not come up."""
    app = create_app(_settings())

    with pytest.raises(RuntimeError) as exc_info:
        async with lifespan(app):
            pytest.fail("startup should not have completed")

    message = str(exc_info.value)
    assert "Cannot start" in message
    assert "anthropic" in message


async def test_startup_aborts_when_the_key_is_rejected(monkeypatch) -> None:
    """A syntactically valid but rejected key must also stop the boot."""
    settings = _settings(
        anthropic_api_key="sk-ant-not-a-real-key",
        validate_provider_on_startup=True,
    )

    async def reject(self) -> None:
        raise ProviderAuthError("invalid x-api-key", provider="anthropic")

    monkeypatch.setattr(
        "app.llm.anthropic_provider.AnthropicProvider.check_credentials", reject
    )

    app = create_app(settings)
    with pytest.raises(RuntimeError, match="Cannot start"):
        async with lifespan(app):
            pytest.fail("startup should not have completed")


async def test_startup_succeeds_when_credentials_check_passes(monkeypatch) -> None:
    settings = _settings(
        anthropic_api_key="sk-ant-looks-fine",
        validate_provider_on_startup=True,
    )

    async def accept(self) -> None:
        return None

    monkeypatch.setattr(
        "app.llm.anthropic_provider.AnthropicProvider.check_credentials", accept
    )
    monkeypatch.setattr("app.llm.embeddings.GeminiEmbedder.check_credentials", accept)

    app = create_app(settings)
    async with lifespan(app):
        registry = app.state.provider_registry
        provider = await registry.get()
        assert provider.name == "anthropic"
        assert provider.default_model == "claude-sonnet-4-6"


async def test_validation_can_be_skipped_but_key_is_still_required() -> None:
    """VALIDATE_PROVIDER_ON_STARTUP=false skips the network call only.

    It must NOT become a way to boot with no credentials at all -- that would
    reintroduce exactly the silent-failure mode we are trying to remove.
    """
    registry = ProviderRegistry(
        _settings(validate_provider_on_startup=False, anthropic_api_key=None)
    )

    with pytest.raises(ProviderNotConfigured):
        await registry.validate_default()


async def test_startup_aborts_when_the_embedding_key_is_missing(monkeypatch) -> None:
    """A working chat provider is not enough -- RAG needs an embedder too.

    This must produce the same clear message as a missing chat key, not a raw
    ProviderNotConfigured traceback.
    """

    async def accept(self) -> None:
        return None

    monkeypatch.setattr(
        "app.llm.anthropic_provider.AnthropicProvider.check_credentials", accept
    )
    monkeypatch.setattr("app.llm.embeddings.GeminiEmbedder.check_credentials", accept)

    settings = _settings(
        anthropic_api_key="sk-ant-fine",
        gemini_api_key=None,  # no embedding provider configured
        validate_provider_on_startup=True,
    )

    app = create_app(settings)
    with pytest.raises(RuntimeError, match="Cannot start"):
        async with lifespan(app):
            pytest.fail("startup should not have completed")


def test_anthropic_key_is_not_leaked_by_repr() -> None:
    """SecretStr means an accidental log of settings can't expose the key."""
    settings = _settings(anthropic_api_key="sk-ant-super-secret-value")

    assert "super-secret-value" not in repr(settings)
    assert "super-secret-value" not in str(settings.anthropic_api_key)
    # ...but the real value is still reachable where it is actually needed.
    assert settings.anthropic_api_key.get_secret_value() == "sk-ant-super-secret-value"
