"""Provider-neutral error hierarchy.

Each provider translates its own SDK exceptions into these, so the service and
web layers never import `openai` or `anthropic` just to catch an error. That
is what keeps the abstraction from leaking: adding a fourth provider must not
require touching the router's except-clauses.

``status_code`` is the HTTP status the *gateway* should return, which is not
always the status the provider returned — a 401 from OpenAI means docXpo is
misconfigured, so it surfaces as 502, not as 401 to our caller.
"""

from __future__ import annotations


class ProviderError(Exception):
    """Base class for every provider failure."""

    status_code: int = 502
    retryable: bool = False

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider

    def to_dict(self) -> dict[str, object]:
        return {
            "error": self.message,
            "type": type(self).__name__,
            "provider": self.provider,
            "retryable": self.retryable,
        }


class ProviderUnavailable(ProviderError):
    """Could not reach the provider at all (DNS, refused, socket closed)."""

    status_code = 503
    retryable = True


class ProviderTimeout(ProviderError):
    status_code = 504
    retryable = True


class ProviderRateLimited(ProviderError):
    """Upstream 429. Propagated as 429 so callers can honour their own backoff."""

    status_code = 429
    retryable = True


class ProviderAuthError(ProviderError):
    """Bad or missing credentials.

    Deliberately 502, not 401: the caller's credentials are fine, *ours* are
    wrong. Returning 401 would tell an API consumer to re-authenticate, which
    would not fix anything.
    """

    status_code = 502
    retryable = False


class ProviderBadRequest(ProviderError):
    """The provider rejected the request (unknown model, malformed messages)."""

    status_code = 400
    retryable = False


class ProviderNotConfigured(ProviderError):
    """Provider selected but its API key / base URL is absent from settings."""

    status_code = 503
    retryable = False
