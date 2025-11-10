"""Shared pytest fixtures for API tests."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import create_app
from app.storage import SessionStore
from buurtsense.storage import SessionStorage


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    """Provide a TestClient backed by a fresh app instance for each test."""

    db_url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    storage = SessionStorage(db_url=db_url)
    store = SessionStore(storage=storage)
    app = create_app(session_store=store)
    with TestClient(app) as test_client:
        yield test_client
