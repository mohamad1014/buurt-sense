"""Tests for the session management API."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
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


def make_segment_payload() -> dict[str, Any]:
    """Return a representative segment payload."""

    start = datetime.now(UTC)
    end = start + timedelta(seconds=30)
    return {
        "index": 0,
        "start_ts": start.isoformat(),
        "end_ts": end.isoformat(),
        "file_uri": "recordings/session-1/segment-0.wav",
        "gps_trace": [
            {
                "lat": 52.3676,
                "lon": 4.9041,
                "ts": start.isoformat(),
                "accuracy_m": 2.5,
            }
        ],
        "orientation_trace": [
            {
                "heading_deg": 140.0,
                "ts": start.isoformat(),
            }
        ],
    }


def make_detection_payload() -> dict[str, Any]:
    """Return a representative detection payload."""

    timestamp = datetime.now(UTC)
    return {
        "class": "gunshot",
        "confidence": 0.92,
        "timestamp": timestamp.isoformat(),
        "gps_point": {
            "lat": 52.3677,
            "lon": 4.9042,
            "ts": timestamp.isoformat(),
            "accuracy_m": 3.1,
        },
        "orientation_heading_deg": 145.0,
    }


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


def test_segment_and_detection_uploads_persist_and_broadcast(
    client: TestClient,
) -> None:
    """Segments and detections should persist and trigger update broadcasts."""

    revision, _ = _fetch_snapshot(client, timeout=0.1)
    session, _ = start_session(client)
    revision, _ = _fetch_snapshot(client, cursor=revision, timeout=0.1)

    segment_payload = make_segment_payload()
    segment_response = client.post(
        f"/sessions/{session.id}/segments", json=segment_payload
    )
    assert segment_response.status_code == 201
    segment_data = segment_response.json()
    assert segment_data["file_path"] == segment_payload["file_uri"]

    segment_revision, _ = _fetch_snapshot(client, cursor=revision, timeout=0.1)
    assert segment_revision > revision

    detection_payload = make_detection_payload()
    detection_response = client.post(
        f"/segments/{segment_data['id']}/detections", json=detection_payload
    )
    assert detection_response.status_code == 201
    detection_data = detection_response.json()
    assert detection_data["label"] == detection_payload["class"]

    detection_revision, _ = _fetch_snapshot(
        client, cursor=segment_revision, timeout=0.1
    )
    assert detection_revision > segment_revision

    store: SessionStore = client.app.state.session_store

    async def _fetch() -> object:
        return await store.storage.get_session(str(session.id))

    record = asyncio.run(_fetch())
    assert record.segments
    stored_segment = next(
        seg for seg in record.segments if seg.id == segment_data["id"]
    )
    assert stored_segment.file_path == segment_payload["file_uri"]
    assert stored_segment.index == segment_payload["index"]
    assert stored_segment.gps_trace
    assert stored_segment.gps_trace[0]["lat"] == segment_payload["gps_trace"][0]["lat"]
    assert stored_segment.orientation_trace
    assert (
        stored_segment.orientation_trace[0]["heading_deg"]
        == segment_payload["orientation_trace"][0]["heading_deg"]
    )
    assert stored_segment.detections
    stored_detection = stored_segment.detections[0]
    assert stored_detection.label == detection_payload["class"]
    assert stored_detection.confidence == detection_payload["confidence"]


def _assert_timezone(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    assert dt.tzinfo is not None, "datetime values must include timezone information"
    return dt


def test_session_detail_endpoint_returns_expected_payload(
    client: TestClient,
) -> None:
    session, metadata = start_session(client)

    segment_payload = make_segment_payload()
    segment_response = client.post(
        f"/sessions/{session.id}/segments", json=segment_payload
    )
    assert segment_response.status_code == 201
    segment_data = segment_response.json()

    detection_payload = make_detection_payload()
    detection_response = client.post(
        f"/segments/{segment_data['id']}/detections", json=detection_payload
    )
    assert detection_response.status_code == 201

    detail_response = client.get(f"/sessions/{session.id}/detail")
    assert detail_response.status_code == 200
    detail = detail_response.json()

    assert detail["id"] == str(session.id)
    assert detail["device_info"] == metadata.get("device_info")
    assert detail["gps_origin"]["lat"] == metadata["gps_origin"]["lat"]
    _assert_timezone(detail["started_at"])
    assert detail["detections"], "session detail should include detections"

    segment_detail = detail["segments"][0]
    assert segment_detail["file_uri"] == segment_payload["file_uri"]
    _assert_timezone(segment_detail["start_ts"])
    _assert_timezone(segment_detail["end_ts"])

    detection_detail = detail["detections"][0]
    assert detection_detail["class"] == detection_payload["class"]
    _assert_timezone(detection_detail["timestamp"])


def test_session_detection_pagination_returns_expected_pages(
    client: TestClient,
) -> None:
    session, _ = start_session(client)

    segment_payload = make_segment_payload()
    segment_response = client.post(
        f"/sessions/{session.id}/segments", json=segment_payload
    )
    assert segment_response.status_code == 201
    segment_id = segment_response.json()["id"]

    first_detection = make_detection_payload()
    first_response = client.post(
        f"/segments/{segment_id}/detections", json=first_detection
    )
    assert first_response.status_code == 201

    second_detection = make_detection_payload()
    base_ts = datetime.fromisoformat(first_detection["timestamp"])
    adjusted_ts = (base_ts + timedelta(seconds=5)).isoformat()
    second_detection["class"] = "siren"
    second_detection["timestamp"] = adjusted_ts
    second_detection["gps_point"]["ts"] = adjusted_ts
    second_response = client.post(
        f"/segments/{segment_id}/detections", json=second_detection
    )
    assert second_response.status_code == 201

    first_page = client.get(
        f"/sessions/{session.id}/detections",
        params={"limit": 1, "offset": 0},
    )
    assert first_page.status_code == 200
    first_payload = first_page.json()
    assert first_payload["total"] == 2
    assert first_payload["limit"] == 1
    assert first_payload["offset"] == 0
    assert len(first_payload["items"]) == 1
    first_item = first_payload["items"][0]
    assert first_item["class"] == first_detection["class"]
    assert first_item["segment_id"] == segment_id
    _assert_timezone(first_item["timestamp"])

    second_page = client.get(
        f"/sessions/{session.id}/detections",
        params={"limit": 1, "offset": 1},
    )
    assert second_page.status_code == 200
    second_payload = second_page.json()
    assert second_payload["total"] == 2
    assert second_payload["offset"] == 1
    assert len(second_payload["items"]) == 1
    second_item = second_payload["items"][0]
    assert second_item["class"] == second_detection["class"]
    assert second_item["segment_id"] == segment_id
    _assert_timezone(second_item["timestamp"])


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
