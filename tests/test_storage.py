"""Tests for the session store conversion helpers."""
from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from app.storage import SessionStore


def _make_store() -> SessionStore:
    """Return a SessionStore instance with mocked dependencies."""

    return SessionStore(storage=Mock(), recording_backend=Mock())


def test_to_session_normalizes_naive_datetimes() -> None:
    """Naive datetimes from SQLite should be coerced to UTC-aware values."""

    store = _make_store()
    record = SimpleNamespace(
        id=str(uuid4()),
        started_at=datetime(2024, 1, 1, 12, 30, 0),
        ended_at=None,
    )

    session = store._to_session(record)

    assert session.started_at.tzinfo is UTC
    assert session.started_at.hour == 12


def test_to_session_parses_iso_strings() -> None:
    """ISO8601 strings from queries should be parsed and normalized."""

    store = _make_store()
    started_at = "2024-05-01T08:15:00"
    ended_at = "2024-05-01T08:45:00"
    record = SimpleNamespace(
        id=str(uuid4()),
        started_at=started_at,
        ended_at=ended_at,
    )

    session = store._to_session(record)

    assert session.started_at.tzinfo is UTC
    assert session.ended_at is not None and session.ended_at.tzinfo is UTC
    assert session.ended_at.minute == 45


def test_to_session_requires_started_at_value() -> None:
    """Missing started_at values should raise a ValueError."""

    store = _make_store()
    record = SimpleNamespace(id=str(uuid4()), ended_at=None)

    with pytest.raises(ValueError):
        store._to_session(record)
