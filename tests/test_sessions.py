"""Tests for the session management API."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator
from uuid import uuid4

from fastapi.testclient import TestClient

from app.models import Session
from app.storage import SessionStore


def validate_session_payload(payload: dict) -> Session:
    """Validate and return a Session object from a JSON payload."""
    session = Session.model_validate(payload)
    assert session.started_at.tzinfo is not None, "started_at must be timezone-aware"
    if session.ended_at is not None:
        assert session.ended_at.tzinfo is not None, "ended_at must be timezone-aware"
        assert (
            session.ended_at >= session.started_at
        ), "ended_at cannot precede started_at"
    return session


def test_session_lifecycle(client: TestClient) -> None:
    """Test complete session lifecycle: create, fetch, stop, and list."""
    response = client.post("/sessions")
    assert response.status_code == 201
    session = validate_session_payload(response.json())
    assert session.ended_at is None

    get_response = client.get(f"/sessions/{session.id}")
    assert get_response.status_code == 200
    fetched = validate_session_payload(get_response.json())
    assert fetched.ended_at is None
    assert fetched.started_at == session.started_at

    stop_response = client.post(f"/sessions/{session.id}/stop")
    assert stop_response.status_code == 200
    stopped = validate_session_payload(stop_response.json())
    assert stopped.ended_at is not None

    final_get = client.get(f"/sessions/{session.id}")
    assert final_get.status_code == 200
    final_session = validate_session_payload(final_get.json())
    assert final_session.ended_at == stopped.ended_at

    list_response = client.get("/sessions")
    assert list_response.status_code == 200
    sessions = [validate_session_payload(item) for item in list_response.json()]
    assert session.id in {item.id for item in sessions}


def test_stopping_unknown_session_returns_not_found(client: TestClient) -> None:
    """Test that stopping a non-existent session returns 404."""
    unknown_id = uuid4()
    response = client.post(f"/sessions/{unknown_id}/stop")
    assert response.status_code == 404
    assert response.json()["detail"].lower() == "session not found"


def test_double_stop_returns_conflict(client: TestClient) -> None:
    """Test that stopping an already stopped session returns 409."""
    session = validate_session_payload(client.post("/sessions").json())

    first_stop = client.post(f"/sessions/{session.id}/stop")
    assert first_stop.status_code == 200
    validate_session_payload(first_stop.json())

    second_stop = client.post(f"/sessions/{session.id}/stop")
    assert second_stop.status_code == 409
    assert second_stop.json()["detail"].lower() == "session already stopped"


def test_retrieving_unknown_session_returns_not_found(client: TestClient) -> None:
    """Test that fetching a non-existent session returns 404."""
    response = client.get(f"/sessions/{uuid4()}")
    assert response.status_code == 404
    assert response.json()["detail"].lower() == "session not found"


def test_stop_session_persists_backend_artifacts(client: TestClient) -> None:
    """Test that stopping a session persists recording artifacts."""
    session = validate_session_payload(client.post("/sessions").json())
    stop_response = client.post(f"/sessions/{session.id}/stop")
    assert stop_response.status_code == 200

    store: SessionStore = client.app.state.session_store

    async def _fetch() -> object:
        return await store.storage.get_session(str(session.id))

    record = asyncio.run(_fetch())
    segments = getattr(record, "segments", [])
    assert segments, "recording backend should persist at least one segment"
    detections = getattr(segments[0], "detections", [])
    assert detections, "recording backend should persist detections for the segment"


def _read_sse_snapshot(
    lines: Iterator[str | bytes], *, timeout: float = 5.0
) -> list[Session]:
    """Return the next session snapshot emitted by the SSE stream."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            line = next(lines)
        except StopIteration as exc:  # pragma: no cover - defensive guard
            raise AssertionError("SSE stream ended unexpectedly") from exc

        if not line:
            continue

        if isinstance(line, bytes):
            line = line.decode("utf-8")

        if not line.startswith("data:"):
            continue

        payload = json.loads(line.removeprefix("data:").strip())
        assert isinstance(payload, list)
        return [Session.model_validate(item) for item in payload]

    raise AssertionError("Timed out waiting for SSE payload")


def test_session_events_stream_provides_live_updates(client: TestClient) -> None:
    """The SSE endpoint should emit snapshots for lifecycle changes."""

    with client.stream("GET", "/sessions/events") as stream:
        assert stream.status_code == 200
        assert stream.headers.get("content-type", "").startswith("text/event-stream")
        snapshots = stream.iter_lines()

        initial = _read_sse_snapshot(snapshots, timeout=2.0)
        assert initial == []

        session = validate_session_payload(client.post("/sessions").json())
        update = _read_sse_snapshot(snapshots, timeout=3.0)
        updated_sessions = {item.id: item for item in update}
        assert session.id in updated_sessions
        assert updated_sessions[session.id].ended_at is None

        client.post(f"/sessions/{session.id}/stop")
        final = _read_sse_snapshot(snapshots, timeout=3.0)
        stopped_sessions = {item.id: item for item in final}
        assert session.id in stopped_sessions
        assert stopped_sessions[session.id].ended_at is not None


def test_session_websocket_provides_live_updates(client: TestClient) -> None:
    """Test that WebSocket provides real-time session updates."""
    with client.websocket_connect("/ws/sessions") as websocket:
        # Get initial empty payload
        initial = websocket.receive_json()
        assert isinstance(initial, list)
        assert not initial

        # Create session and verify update
        session = validate_session_payload(client.post("/sessions").json())
        update = websocket.receive_json()
        assert any(item["id"] == str(session.id) for item in update)

        # Stop session and verify final update
        client.post(f"/sessions/{session.id}/stop")
        final = websocket.receive_json()
        stopped = next(item for item in final if item["id"] == str(session.id))
        assert stopped["ended_at"] is not None

