"""Application configuration.

Every knob the app has lives here, is typed, and is validated once at startup.
The rest of the codebase never reads ``os.environ`` directly -- it asks for
``Settings``. That gives us three things:

1. A wrong/missing env var crashes at boot with a readable error, instead of
   blowing up on request #10,000 as a ``NoneType`` somewhere deep in a service.
2. Tests can construct a ``Settings(...)`` with overrides instead of monkey-
   patching the environment.
3. There is exactly one place to look to see what the app can be configured to do.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "dev", "prod"]
ProviderName = Literal["gemini", "anthropic", "openai", "ollama"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Ignore unknown keys in .env rather than erroring. Later phases add
        # provider keys; a stale .env shouldn't block the app from booting.
        extra="ignore",
        # POSTGRES_HOST env var -> postgres_host field.
        case_sensitive=False,
    )

    # ---- App ---------------------------------------------------------------
    app_name: str = "docXpo"
    environment: Environment = "local"
    debug: bool = False

    # ---- Logging -----------------------------------------------------------
    log_level: str = "INFO"
    # JSON logs are what a log aggregator (Loki/CloudWatch/Datadog) wants.
    # Locally we default to human-readable coloured output instead.
    log_json: bool = False

    # ---- Postgres ----------------------------------------------------------
    # Stored as separate parts rather than one URL string so that docker-compose
    # can override just the host, and so the URL can never be half-formed.
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "docxpo"
    postgres_password: str = "docxpo"
    postgres_db: str = "docxpo"

    # Connection pool sizing. Each async worker holds up to
    # (pool_size + max_overflow) connections; Postgres' default max_connections
    # is 100, so keep this modest or you will exhaust the server under load.
    db_pool_size: int = 10
    db_max_overflow: int = 5
    db_echo: bool = False

    # ---- Redis -------------------------------------------------------------
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None

    # ---- LLM providers -----------------------------------------------------
    # Gemini is the default: its free tier is the most generous, so the project
    # runs without burning paid credits. Switching is purely an env var --
    # DEFAULT_PROVIDER=anthropic requires no code change.
    #
    # There is no fake/mock fallback: if the selected provider's credentials are
    # missing or rejected, the app refuses to start rather than serving anything
    # that isn't a real model response.
    default_provider: ProviderName = "gemini"

    # Verify the default provider's credentials during startup by making one
    # cheap, non-generative API call (GET /v1/models/{id} -- no tokens billed).
    #
    # Trade-off: this catches a typo'd key, a revoked key, or a bad model ID at
    # boot instead of on the first user request, which is what "fail loudly at
    # startup" requires. The cost is that startup now depends on the provider
    # being reachable, so a network blip prevents a deploy. Set to false for
    # offline development or if you would rather boot and find out later.
    validate_provider_on_startup: bool = True

    # Anthropic *requires* max_tokens, the others treat it as optional. The
    # abstraction always sends it so behaviour is identical across providers.
    #
    # 4096, not 1024: on reasoning-by-default models (Gemini 3.x, Claude Opus 5)
    # this budget covers thinking tokens *plus* the visible answer. At 1024 a
    # measured request spent 462 tokens thinking and got cut off mid-sentence
    # with finish_reason="max_tokens" after only 36 visible tokens.
    llm_max_tokens: int = 4096
    # Generous: a local model on CPU can take a while to produce its first
    # token. This bounds the whole request, not the gap between tokens.
    llm_timeout_seconds: float = 120.0

    gemini_api_key: SecretStr | None = None
    # NOT "gemini-2.0-flash": on newly-created API keys that model returns
    # 429 with `limit: 0` (no free-tier quota granted), and gemini-2.5-flash
    # returns 404 "no longer available to new users". The rolling alias
    # resolves to whatever current Flash model the free tier actually serves,
    # which is the only one that works on a fresh key.
    gemini_model: str = "gemini-flash-latest"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # SecretStr keeps keys out of logs and tracebacks: repr() renders as
    # '**********', so an accidental log of the settings object can't leak them.
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None  # for OpenAI-compatible gateways
    openai_model: str = "gpt-4o-mini"

    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-sonnet-4-6"

    # ---- Embeddings / RAG --------------------------------------------------
    # Separate from DEFAULT_PROVIDER on purpose: Anthropic has no embedding
    # model, so a deployment can legitimately chat on Anthropic and embed on
    # Gemini. Ollama is not offered here -- it would mean a second local model.
    embedding_provider: Literal["gemini", "openai"] = "gemini"
    gemini_embedding_model: str = "gemini-embedding-001"
    openai_embedding_model: str = "text-embedding-3-small"

    # 768, not the model's native 3072. pgvector's HNSW/IVFFlat indexes reject
    # columns wider than 2000 dims, so 3072 could only ever be sequentially
    # scanned. Changing this requires a migration AND re-embedding every chunk,
    # because a Postgres vector column has a fixed width.
    embedding_dimensions: int = 768

    # See app/services/chunking.py for the full reasoning on these two.
    chunk_size: int = 1200
    chunk_overlap: int = 200

    rag_top_k: int = 5

    # Chunks below this cosine similarity are dropped rather than injected.
    #
    # 0.50 is calibrated from measurement, not intuition. With normalized
    # gemini-embedding-001 vectors the useful range is compressed far more than
    # the textbook 0-1 suggests:
    #
    #     on-topic and answerable ............ 0.67
    #     on-topic but not in the corpus ..... 0.63
    #     completely unrelated question ...... 0.54
    #
    # So an absolute threshold is a coarse safety net, not a precision filter --
    # anything high enough to reject the 0.54 case also rejects real hits. The
    # actual defence against answering from noise is the system prompt, which
    # does reliably produce "that isn't in the documents". This floor only
    # catches degenerate matches and saves the tokens they would have cost.
    rag_min_similarity: float = 0.50

    # ---- Semantic cache ----------------------------------------------------
    cache_enabled: bool = True

    # The single most consequential number in Phase 4. Measured query-to-query
    # cosine on real embeddings:
    #
    #     identical .................. 1.000
    #     typo ....................... 0.993
    #     paraphrase ................. 0.985
    #     NEGATED ("does NOT use") ... 0.975   <-- wrong answer if admitted
    #     looser paraphrase .......... 0.917
    #     same topic, different ask .. 0.867
    #     different topic ............ 0.539
    #
    # The trade-off runs both ways and is not symmetric in cost:
    #   * too LOW  -> false hits. The cache confidently answers a *different*
    #     question. This is the expensive failure -- silently wrong output.
    #   * too HIGH -> few hits. The cache costs a little and saves nothing.
    #
    # 0.98 admits restatements and typos while excluding the measured negation
    # at 0.975. Note how thin that margin is: embeddings encode topic far more
    # strongly than polarity, so a negated question looks almost identical to
    # its opposite. No threshold fixes that -- it only trades hit rate for the
    # probability of being wrong. Hence the conservative default.
    cache_similarity_threshold: float = 0.98

    cache_ttl_seconds: int = 3600
    # Lookup scans the namespace linearly, so this bound is what keeps it fast.
    cache_max_entries: int = 500

    # ---- Rate limiting / API keys ------------------------------------------
    rate_limit_enabled: bool = True

    # Sustained rate. The bucket refills at rpm/60 tokens per second.
    rate_limit_rpm: int = 60
    # Bucket capacity = how large a burst an idle client may spend at once.
    # Larger than rpm on purpose: real clients are bursty (a page load firing
    # several requests), and a burst allowance absorbs that without raising the
    # sustained rate. Set burst == rpm for a strict per-minute cap.
    rate_limit_burst: int = 20

    # If Redis is unreachable: True = allow requests through (availability over
    # protection), False = reject. See RateLimiter.check for the trade-off.
    rate_limit_fail_open: bool = True

    # Comma-separated. **Empty means no authentication**: the service runs open
    # and meters by client IP, which keeps local dev and the browser console
    # working with an empty .env. Setting any key switches on enforcement.
    api_keys: str = ""

    @property
    def api_key_set(self) -> frozenset[str]:
        return frozenset(k.strip() for k in self.api_keys.split(",") if k.strip())

    # ---- Observability -----------------------------------------------------
    metrics_enabled: bool = True
    # JSON overrides for the pricebook, e.g.
    #   {"gemini-3.6-flash": {"input": 0.30, "output": 2.50}}
    # Prices change; this avoids a code edit and redeploy to correct one.
    pricing_overrides: str = ""

    # ---- API ---------------------------------------------------------------
    api_v1_prefix: str = "/v1"
    # Browser test console at "/". A dev tool, so it follows the same rule as
    # the OpenAPI docs: off in prod unless explicitly enabled.
    serve_ui: bool = True
    # Comma-separated in the env var; pydantic-settings parses JSON lists too.
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> str:
        """SQLAlchemy URL using the asyncpg driver.

        The ``+asyncpg`` suffix is what tells SQLAlchemy to use the async
        dialect. Swapping it for ``+psycopg2`` would silently give you a
        blocking driver inside an async event loop -- the classic way to make a
        "fully async" service serve one request at a time.
        """
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def redis_url(self) -> str:
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def is_local(self) -> bool:
        return self.environment == "local"


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so the .env file is parsed exactly once per process.

    ``lru_cache`` also makes this usable as a FastAPI dependency: every request
    gets the same instance instead of re-reading the file from disk.
    Tests can reset it with ``get_settings.cache_clear()``.
    """
    return Settings()
