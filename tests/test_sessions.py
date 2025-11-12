"""Tests for the session management API."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi.testclient import TestClient

from app.models import Session
from app.storage import SessionStore


def make_session_payload() -> dict[str, Any]:
    """Return a representative session creation payload."""

    now = datetime.now(UTC).isoformat()
    return {
        "started_at": now,
        "device_info": {"manufacturer": "BuurtSense", "model": "MVP"},
        "gps_origin": {
            "lat": 52.3676,
            "lon": 4.9041,
            "accuracy_m": 4.2,
            "captured_at": now,
        },
        "orientation_origin": {
            "heading_deg": 135.0,
            "captured_at": now,
        },
        "config_snapshot": {
            "segment_length_sec": 30,
            "overlap_sec": 5,
            "confidence_threshold": 0.6,
        },
    }


def expected_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the subset of payload fields persisted alongside sessions."""

    keys = ("device_info", "gps_origin", "orientation_origin", "config_snapshot")
    return {key: payload[key] for key in keys if key in payload}


def start_session(client: TestClient) -> tuple[Session, dict[str, Any]]:
    """Create a session via the API and return the parsed response and metadata."""

    payload = make_session_payload()
    response = client.post("/sessions", json=payload)
    assert response.status_code == 201
    metadata = expected_metadata(payload)
    session = validate_session_payload(response.json(), metadata)
    return session, metadata


def validate_session_payload(
    payload: dict, metadata: dict[str, Any] | None = None
) -> Session:
    """Validate and return a Session object from a JSON payload."""
    session = Session.model_validate(payload)
    assert session.started_at.tzinfo is not None, "started_at must be timezone-aware"
    if session.ended_at is not None:
        assert session.ended_at.tzinfo is not None, "ended_at must be timezone-aware"
        assert (
            session.ended_at >= session.started_at
        ), "ended_at cannot precede started_at"
    if metadata is not None:
        for key, value in metadata.items():
            assert getattr(session, key) == value, f"{key} metadata mismatch"
    return session


def test_session_lifecycle(client: TestClient) -> None:
    """Test complete session lifecycle: create, fetch, stop, and list."""
    session, metadata = start_session(client)
    assert session.ended_at is None

    get_response = client.get(f"/sessions/{session.id}")
    assert get_response.status_code == 200
    fetched = validate_session_payload(get_response.json(), metadata)
    assert fetched.ended_at is None
    assert fetched.started_at == session.started_at

    stop_response = client.post(f"/sessions/{session.id}/stop")
    assert stop_response.status_code == 200
    stopped = validate_session_payload(stop_response.json(), metadata)
    assert stopped.ended_at is not None

    final_get = client.get(f"/sessions/{session.id}")
    assert final_get.status_code == 200
    final_session = validate_session_payload(final_get.json(), metadata)
    assert final_session.ended_at == stopped.ended_at

    list_response = client.get("/sessions")
    assert list_response.status_code == 200
    session_list_payload = list_response.json()
    sessions = [validate_session_payload(item) for item in session_list_payload]
    assert session.id in {item.id for item in sessions}
    stored_payload = next(
        item for item in session_list_payload if item["id"] == str(session.id)
    )
    validate_session_payload(stored_payload, metadata)


def test_stopping_unknown_session_returns_not_found(client: TestClient) -> None:
    """Test that stopping a non-existent session returns 404."""
    unknown_id = uuid4()
    response = client.post(f"/sessions/{unknown_id}/stop")
    assert response.status_code == 404
    assert response.json()["detail"].lower() == "session not found"


def test_double_stop_returns_conflict(client: TestClient) -> None:
    """Test that stopping an already stopped session returns 409."""
    session, metadata = start_session(client)

    first_stop = client.post(f"/sessions/{session.id}/stop")
    assert first_stop.status_code == 200
    validate_session_payload(first_stop.json(), metadata)

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
    session, metadata = start_session(client)
    stop_response = client.post(f"/sessions/{session.id}/stop")
    assert stop_response.status_code == 200
    validate_session_payload(stop_response.json(), metadata)

    store: SessionStore = client.app.state.session_store

    async def _fetch() -> object:
        return await store.storage.get_session(str(session.id))

    record = asyncio.run(_fetch())
    segments = getattr(record, "segments", [])
    assert segments, "recording backend should persist at least one segment"
    detections = getattr(segments[0], "detections", [])
    assert detections, "recording backend should persist detections for the segment"


def _fetch_snapshot(
    client: TestClient,
    *,
    cursor: int | None = None,
    timeout: float = 0.5,
) -> tuple[int, list[Session]]:
    """Return the latest session snapshot from the update endpoint."""

    response = client.get(
        "/sessions/updates",
        params={"cursor": cursor, "timeout": timeout},
    )
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["revision"], int)
    sessions = [Session.model_validate(item) for item in payload["sessions"]]
    return payload["revision"], sessions


def test_session_updates_accepts_none_cursor_value(client: TestClient) -> None:
    """Explicitly passing None as cursor should be treated as no filter."""

    response = client.get(
        "/sessions/updates",
        params={"cursor": None, "timeout": 0.0},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["revision"] == 0
    assert payload["sessions"] == []


def test_session_updates_endpoint_provides_live_changes(client: TestClient) -> None:
    """The updates endpoint should provide successive lifecycle snapshots."""

    revision, initial = _fetch_snapshot(client, timeout=0.1)
    assert initial == []

    session, metadata = start_session(client)
    revision, update = _fetch_snapshot(client, cursor=revision, timeout=0.1)
    updated_sessions = {item.id: item for item in update}
    assert session.id in updated_sessions
    assert updated_sessions[session.id].ended_at is None
    assert updated_sessions[session.id].gps_origin == metadata["gps_origin"]
    assert updated_sessions[session.id].config_snapshot == metadata["config_snapshot"]

    client.post(f"/sessions/{session.id}/stop")
    revision, final = _fetch_snapshot(client, cursor=revision, timeout=0.1)
    stopped_sessions = {item.id: item for item in final}
    assert session.id in stopped_sessions
    assert stopped_sessions[session.id].ended_at is not None
    assert stopped_sessions[session.id].device_info == metadata["device_info"]


def test_session_websocket_provides_live_updates(client: TestClient) -> None:
    """Test that WebSocket provides real-time session updates."""
    with client.websocket_connect("/ws/sessions") as websocket:
        # Get initial empty payload
        initial = websocket.receive_json()
        assert isinstance(initial, list)
        assert not initial

        # Create session and verify update
        session, metadata = start_session(client)
        update = websocket.receive_json()
        created = next(item for item in update if item["id"] == str(session.id))
        assert created["gps_origin"] == metadata["gps_origin"]

        # Stop session and verify final update
        client.post(f"/sessions/{session.id}/stop")
        final = websocket.receive_json()
        stopped = next(item for item in final if item["id"] == str(session.id))
        assert stopped["ended_at"] is not None
        assert stopped["config_snapshot"] == metadata["config_snapshot"]
