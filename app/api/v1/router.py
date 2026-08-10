"""Aggregates every v1 route module into a single router.

``main.py`` mounts only this object under the ``/v1`` prefix, so adding an
endpoint in a later phase means touching one line here and never touching the
app factory.

Health checks deliberately live *outside* this router (``app/api/health.py``,
mounted at the root). They are infrastructure endpoints for load balancers and
orchestrators, not part of the public API contract, so they must not be
versioned or move when v2 ships.
"""

from fastapi import APIRouter

from app.api.v1.routes import chat, documents, metrics, rag

api_router = APIRouter()
api_router.include_router(chat.router)
api_router.include_router(documents.router)
api_router.include_router(rag.router)
api_router.include_router(metrics.router)
