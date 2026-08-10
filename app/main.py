"""Application factory and process lifecycle.

Kept deliberately thin: it wires together config, logging, middleware, routers,
and the startup/shutdown of shared clients. No business logic belongs here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.health import router as health_router
from app.api.ui import router as ui_router
from app.api.v1.router import api_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.core.middleware import RequestContextMiddleware
from app.core.pricing import Pricebook
from app.core.rate_limit_middleware import RateLimitMiddleware
from app.core.redis import build_redis
from app.db.session import dispose_engine
from app.llm.embeddings import build_embedder
from app.llm.errors import ProviderError
from app.llm.registry import ProviderRegistry
from app.services.metrics_service import MetricsService

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown hook.

    Everything before ``yield`` runs once before the server accepts traffic;
    everything after runs on graceful shutdown. This replaces the deprecated
    ``@app.on_event("startup")`` decorators and, unlike them, guarantees the
    teardown half runs even if startup partially failed.

    Note we do NOT run migrations here. Migrations are a deploy step, not an app
    step -- if three replicas boot at once they would race to migrate the same
    database.
    """
    settings: Settings = app.state.settings

    app.state.redis = build_redis(settings)
    # Stateless apart from the pricebook, so one instance per process.
    app.state.metrics = MetricsService(Pricebook(settings.pricing_overrides))
    registry = ProviderRegistry(settings)
    app.state.provider_registry = registry

    async def release() -> None:
        """Release sockets so the process can exit promptly and Postgres/Redis
        don't sit on half-open connections."""
        await registry.aclose()
        await app.state.redis.aclose()
        await dispose_engine()

    # Fail fast and loudly. There is no fake/mock provider to fall back to, so
    # a deployment that cannot reach its LLM is broken and should not accept
    # traffic while pretending otherwise. Raising here aborts startup: uvicorn
    # logs the error and exits non-zero, which is what an orchestrator needs to
    # see in order to halt a bad rollout.
    # Two independent components, checked separately so the failure message
    # names the one that actually broke -- "chat provider 'anthropic'" and
    # "embedding provider 'gemini'" are different problems with different fixes,
    # and they can genuinely be different vendors.
    component = f"chat provider '{settings.default_provider}'"
    try:
        await registry.validate_default()

        component = f"embedding provider '{settings.embedding_provider}'"
        # Built inside the try so a missing key here gives the same clear
        # message as a missing chat key, rather than a raw traceback.
        app.state.embedder = build_embedder(settings)
        if settings.validate_provider_on_startup:
            await app.state.embedder.check_credentials()
    except ProviderError as exc:
        log.error(
            "startup_provider_unavailable",
            component=component,
            error=exc.message,
            hint="Set the matching API key in .env (see .env.example).",
        )
        # Startup aborts before the `yield`, so the teardown block below never
        # runs -- clean up here or we leak the Redis pool and DB engine.
        await release()
        raise RuntimeError(
            f"Cannot start: {component} is unusable -- {exc.message}"
        ) from exc

    log.info(
        "application_startup",
        app=settings.app_name,
        version=__version__,
        environment=settings.environment,
        default_provider=settings.default_provider,
    )

    try:
        yield
    finally:
        await release()
        log.info("application_shutdown")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI app.

    A factory (rather than a module-level ``app = FastAPI()``) lets tests build
    an isolated instance with overridden settings.
    """
    settings = settings or get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.log_json)

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="Self-hostable LLM gateway and RAG service",
        lifespan=lifespan,
        # Hide interactive docs outside local/dev -- they describe the whole
        # attack surface of the service.
        docs_url="/docs" if settings.environment != "prod" else None,
        redoc_url=None,
    )
    app.state.settings = settings

    # Middleware runs in reverse registration order for the response path, so
    # RequestContextMiddleware is added last => it is the OUTERMOST layer. That
    # is what we want: it sees the real status code of anything raised deeper,
    # including errors raised inside CORS handling.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Order matters. Middleware is added outermost-last, so this registers
    # RequestContextMiddleware *outside* the rate limiter -- meaning a 429 is
    # still assigned a request id and still shows up in the access log. If the
    # limiter were outermost, rejected requests would be invisible, which is
    # precisely the traffic you most want to see.
    app.add_middleware(
        RateLimitMiddleware,
        configured_keys=settings.api_key_set,
        requests_per_minute=settings.rate_limit_rpm,
        burst=settings.rate_limit_burst,
        fail_open=settings.rate_limit_fail_open,
        enabled=settings.rate_limit_enabled,
    )
    app.add_middleware(RequestContextMiddleware)

    app.include_router(health_router)
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    if settings.serve_ui and settings.environment != "prod":
        app.include_router(ui_router)

    return app


app = create_app()
