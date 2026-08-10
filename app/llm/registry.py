"""Provider registry — owns provider instances and their connection pools.

Providers are built **lazily**: constructing the OpenAI provider raises if
``OPENAI_API_KEY`` is unset, so eagerly building all three at startup would
make the app refuse to boot unless you held keys for every provider. Lazy
construction means Ollama-only development works with an empty .env, and a
missing key only surfaces when someone actually asks for that provider.

One instance per provider per process: each holds an HTTP connection pool, so
rebuilding per request would defeat connection reuse and leak sockets.
"""

from __future__ import annotations

import asyncio
from typing import ClassVar

from app.core.config import ProviderName, Settings
from app.core.logging import get_logger
from app.llm.anthropic_provider import AnthropicProvider
from app.llm.base import LLMProvider
from app.llm.errors import ProviderNotConfigured
from app.llm.gemini_provider import GeminiProvider
from app.llm.ollama import OllamaProvider
from app.llm.openai_provider import OpenAIProvider

log = get_logger(__name__)


class ProviderRegistry:
    # Adding a provider is a one-line change here plus the new module. This
    # table is the *only* thing that maps the DEFAULT_PROVIDER env var onto an
    # implementation -- switching providers never requires a code change.
    _BUILDERS: ClassVar[dict[str, type[LLMProvider]]] = {
        GeminiProvider.name: GeminiProvider,
        AnthropicProvider.name: AnthropicProvider,
        OpenAIProvider.name: OpenAIProvider,
        OllamaProvider.name: OllamaProvider,
    }

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._instances: dict[str, LLMProvider] = {}
        # Two concurrent first-requests for the same provider would otherwise
        # both see an empty dict and each build a client (and a pool). The lock
        # makes construction happen exactly once.
        self._lock = asyncio.Lock()

    @property
    def default_name(self) -> ProviderName:
        return self._settings.default_provider

    async def validate_default(self) -> LLMProvider:
        """Build and verify the default provider. Raises if it is unusable.

        Called from the lifespan hook so a missing or rejected API key stops
        the app from starting, rather than surfacing on a user's first request.
        Only the *default* provider is checked -- the others stay lazy, so you
        don't need an OpenAI key just to boot an Anthropic-backed deployment.
        """
        provider = await self.get()  # raises ProviderNotConfigured if no key

        if self._settings.validate_provider_on_startup:
            await provider.check_credentials()
            log.info(
                "provider_credentials_verified",
                provider=provider.name,
                model=provider.default_model,
            )
        else:
            log.warning(
                "provider_credentials_unverified",
                provider=provider.name,
                reason="VALIDATE_PROVIDER_ON_STARTUP=false",
            )
        return provider

    async def get(self, name: str | None = None) -> LLMProvider:
        resolved = name or self._settings.default_provider

        if resolved not in self._BUILDERS:
            raise ProviderNotConfigured(
                f"Unknown provider {resolved!r}. "
                f"Available: {', '.join(sorted(self._BUILDERS))}"
            )

        if (existing := self._instances.get(resolved)) is not None:
            return existing

        async with self._lock:
            # Re-check inside the lock: another task may have built it while
            # we waited. This is the standard double-checked locking pattern.
            if (existing := self._instances.get(resolved)) is not None:
                return existing

            provider = self._BUILDERS[resolved](self._settings)
            self._instances[resolved] = provider
            log.info(
                "provider_initialised",
                provider=resolved,
                model=provider.default_model,
            )
            return provider

    async def aclose(self) -> None:
        """Close every provider that was actually built."""
        for name, provider in self._instances.items():
            try:
                await provider.aclose()
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                log.warning("provider_close_failed", provider=name, error=str(exc))
        self._instances.clear()
