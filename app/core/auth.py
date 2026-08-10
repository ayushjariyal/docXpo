"""API key extraction and identification.

Deliberately small. This is *identity for metering*, not a full auth system:
keys come from configuration, there is no user model, no scopes, no rotation.
What it must get right is (a) constant-time comparison, and (b) never letting a
raw key reach a log line or a Redis key.

When no keys are configured the service runs **open** and meters by client IP
instead. That keeps local development and the browser console working with an
empty .env, while a deployment that sets API_KEYS gets enforcement.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Literal

API_KEY_HEADER = "x-api-key"
AUTH_HEADER = "authorization"


@dataclass(frozen=True, slots=True)
class Principal:
    """Who is being metered."""

    kind: Literal["api_key", "anonymous"]
    # Stable, non-reversible id. Used as the Redis bucket key and in logs, so a
    # leaked log or a `redis-cli KEYS *` never exposes a live credential.
    identifier: str
    label: str  # short, safe-to-display form

    @property
    def is_authenticated(self) -> bool:
        return self.kind == "api_key"


def fingerprint(secret: str) -> str:
    """Short, stable, non-reversible id for a key.

    12 hex chars = 48 bits. Collision risk is negligible at any realistic
    number of API keys, and a short id keeps Redis keys and log lines readable.
    """
    return hashlib.sha256(secret.encode()).hexdigest()[:12]


def matches_any(candidate: str, known: frozenset[str]) -> bool:
    """Constant-time membership test.

    A plain `candidate in known` short-circuits on the first differing byte,
    which leaks key material through timing. `compare_digest` does not, so we
    walk every configured key rather than using set membership.
    """
    return any(hmac.compare_digest(candidate, k) for k in known)


def extract_key(headers: dict[bytes, bytes]) -> str | None:
    """Read the API key from either accepted header.

    `X-API-Key` is the common convention for machine keys; `Authorization:
    Bearer` is accepted too because most HTTP clients and SDKs reach for it by
    default.
    """
    raw = headers.get(API_KEY_HEADER.encode())
    if raw:
        return raw.decode().strip() or None

    auth = headers.get(AUTH_HEADER.encode())
    if auth:
        value = auth.decode().strip()
        if value.lower().startswith("bearer "):
            return value[7:].strip() or None
    return None


def identify(
    headers: dict[bytes, bytes],
    *,
    configured_keys: frozenset[str],
    client_ip: str | None,
) -> Principal | None:
    """Resolve the caller. Returns None when a required key is missing/invalid.

    Open mode (no configured keys) meters by IP. Note the honest limitation:
    behind a proxy every request carries the proxy's IP unless X-Forwarded-For
    is handled, and that header is client-controlled and trivially spoofed. IP
    metering is therefore a development convenience, not a security control --
    which is exactly why configuring API_KEYS switches it off.
    """
    if not configured_keys:
        ip = client_ip or "unknown"
        return Principal(kind="anonymous", identifier=f"ip:{ip}", label=f"ip:{ip}")

    presented = extract_key(headers)
    if presented is None or not matches_any(presented, configured_keys):
        return None

    fp = fingerprint(presented)
    return Principal(kind="api_key", identifier=f"key:{fp}", label=f"key:{fp}")
