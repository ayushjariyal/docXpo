"""LLM provider abstraction.

The public surface is deliberately small: callers import the types and the
registry, never a concrete provider class.
"""

from app.llm.base import (
    ChatResult,
    LLMProvider,
    Message,
    StreamDone,
    StreamEvent,
    TextDelta,
    Usage,
)
from app.llm.errors import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
)
from app.llm.registry import ProviderRegistry

__all__ = [
    "ChatResult",
    "LLMProvider",
    "Message",
    "ProviderAuthError",
    "ProviderError",
    "ProviderRateLimited",
    "ProviderRegistry",
    "ProviderTimeout",
    "ProviderUnavailable",
    "StreamDone",
    "StreamEvent",
    "TextDelta",
    "Usage",
]
