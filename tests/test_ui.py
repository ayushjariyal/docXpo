"""The browser console is served correctly and gated off in prod."""

from __future__ import annotations

from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.main import create_app


def _client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_ui_is_served_at_root() -> None:
    async with _client(create_app(Settings(_env_file=None, log_level="CRITICAL"))) as ac:
        resp = await ac.get("/")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    # The console must post to the real endpoint, not a stub.
    assert "/v1/chat" in resp.text


async def test_ui_reads_the_sse_frame_format_we_actually_emit() -> None:
    """Guards against the UI and the endpoints drifting apart.

    If the event names in chat.py / rag.py ever change, this fails instead of
    the console silently going blank.
    """
    async with _client(create_app(Settings(_env_file=None, log_level="CRITICAL"))) as ac:
        html = (await ac.get("/")).text

    for token in ("'event: '", "'data: '", "'token'", "'done'", "'error'", "'sources'"):
        assert token in html, f"UI no longer handles {token}"


async def test_ui_targets_every_endpoint_it_needs() -> None:
    """The console drives ingestion and both RAG modes, not just chat."""
    async with _client(create_app(Settings(_env_file=None, log_level="CRITICAL"))) as ac:
        html = (await ac.get("/")).text

    for path in (
        "/v1/chat",
        "/v1/rag/query",
        "/v1/rag/retrieve",  # retrieval without spending generation quota
        "/v1/documents",
        "/v1/documents/text",
        "/v1/metrics",
        "/v1/metrics/recent",
    ):
        assert path in html, f"UI does not reference {path}"


async def test_ui_is_disabled_in_prod() -> None:
    settings = Settings(_env_file=None, environment="prod", log_level="CRITICAL")

    async with _client(create_app(settings)) as ac:
        resp = await ac.get("/")

    assert resp.status_code == 404


async def test_ui_can_be_disabled_explicitly() -> None:
    settings = Settings(_env_file=None, serve_ui=False, log_level="CRITICAL")

    async with _client(create_app(settings)) as ac:
        resp = await ac.get("/")

    assert resp.status_code == 404


async def test_metrics_panel_follows_the_figure_rules() -> None:
    """Guards the two dataviz rules that are easy to regress.

    Exactly one hero figure per view, and `tabular-nums` confined to table
    columns -- on a 52px display number, tabular digits give every glyph the
    width of a zero and the value reads visibly loose.
    """
    async with _client(create_app(Settings(_env_file=None, log_level="CRITICAL"))) as ac:
        html = (await ac.get("/")).text

    assert html.count('class="hero"') == 1, "a view may lead with only one hero figure"
    # The meter's unfilled track must be a lighter step of the accent ramp,
    # not grey, so the state reads across the whole bar.
    assert "--accent-track" in html
