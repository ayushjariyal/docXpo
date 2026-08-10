# Multi-stage build.
#
# Stage 1 compiles wheels (asyncpg has C extensions and needs a toolchain).
# Stage 2 copies only the built artifacts, so gcc and the build headers never
# ship in the runtime image -- smaller image, smaller attack surface.

# ---- builder ----------------------------------------------------------------
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only the dependency manifest first. This layer is cached and only
# invalidated when dependencies change -- editing app code does not trigger a
# full reinstall on rebuild.
COPY pyproject.toml ./
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# `pip install .` needs the package dir to exist to resolve the build backend.
COPY app ./app
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

# ---- runtime ----------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Run as an unprivileged user. A container escape from a root process is a much
# worse day than one from uid 1000.
RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser alembic ./alembic
COPY --chown=appuser:appuser alembic.ini ./

USER appuser
EXPOSE 8000

# --no-access-log: our RequestContextMiddleware already emits one structured
# line per request; uvicorn's default access log would duplicate it in a
# different format.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
