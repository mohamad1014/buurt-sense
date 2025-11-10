"""Tests for serving the frontend assets."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


def test_index_route_serves_html() -> None:
    """The control panel should be served at the root path."""

    app = create_app()
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Buurt Sense MVP Control Panel" in response.text


def test_static_assets_are_available() -> None:
    """Static assets must be served alongside the frontend."""

    app = create_app()
    client = TestClient(app)

    response = client.get("/static/app.js")

    assert response.status_code == 200
    assert "application/javascript" in response.headers["content-type"]
    assert "startSession" in response.text
