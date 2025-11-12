"""Shared pytest fixtures for API tests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import create_app
from app.storage import SessionStore


class _NoOpBackend:
    """Recording backend stub used for tests that manage artifacts manually."""

    async def start(self, session_id: str, *, started_at: datetime) -> None:
        return None

    async def stop(self, session_id: str, *, ended_at: datetime) -> None:
        return None


from buurtsense.storage import SessionStorage


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Provide a TestClient backed by a fresh app instance for each test."""

    capture_root = tmp_path / "captures"
    monkeypatch.setenv("BUURT_CAPTURE_ROOT", str(capture_root))
    monkeypatch.setenv("BUURT_SEGMENT_LENGTH_SEC", "0.25")
    monkeypatch.setenv("BUURT_SEGMENT_BYTES_PER_SEC", "4000")
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    storage = SessionStorage(db_url=db_url)
    store = SessionStore(storage=storage)
    app = create_app(session_store=store)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def passive_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Provide a TestClient with a no-op backend for manual artifact scenarios."""

    capture_root = tmp_path / "captures"
    monkeypatch.setenv("BUURT_CAPTURE_ROOT", str(capture_root))
    monkeypatch.setenv("BUURT_SEGMENT_LENGTH_SEC", "60")
    monkeypatch.setenv("BUURT_SEGMENT_BYTES_PER_SEC", "1024")
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    storage = SessionStorage(db_url=db_url)
    store = SessionStore(storage=storage, recording_backend=_NoOpBackend())
    app = create_app(session_store=store)
    with TestClient(app) as test_client:
        yield test_client
