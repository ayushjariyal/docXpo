"""Serves the browser test console.

A single self-contained HTML file, served by FastAPI itself: no build step, no
npm, no CDN. That keeps `docker compose up` the only thing you need to run to
get a working UI, and means the UI can never drift out of sync with the API it
is testing.

It is a *development console*, not a product surface -- which is why it is
gated off in prod by default (`SERVE_UI`).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(tags=["ui"])

# Resolved relative to this module, not the process working directory, so it
# works the same whether uvicorn is started from the repo root or from /app in
# the container.
INDEX = Path(__file__).resolve().parent.parent / "web" / "index.html"


@router.get("/", include_in_schema=False)
async def index() -> FileResponse:
    # no-cache: during development the file changes constantly, and a cached
    # copy of the console is a confusing thing to debug.
    return FileResponse(INDEX, media_type="text/html", headers={"Cache-Control": "no-cache"})
