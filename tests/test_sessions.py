"""Tests for the session management API."""

from __future__ import annotations

import asyncio
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from app.models import Session
from app.storage import SessionStore
from app.utils import derive_timezone


def make_session_payload() -> dict[str, Any]:
    """Return a representative session creation payload."""

    now = datetime.now(UTC).isoformat()
    return {
        "started_at": now,
        "device_id": str(uuid4()),
        "operator_alias": "FieldOps",
        "notes": "Night shift patrol",
        "app_version": "1.2.3",
        "model_bundle_version": "2024.01",
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
        "detection_summary": {
            "total_detections": 0,
            "by_class": {},
        },
        "redact_location": False,
    }


def make_frontend_payload() -> dict[str, Any]:
    """Mirror the payload produced by the browser control panel."""

    now = datetime.now(UTC).isoformat()
    return {
        "started_at": now,
        "operator_alias": "Browser Operator",
        "notes": "Started from local UI",
        "app_version": "web-ui",
        "model_bundle_version": "demo",
        "gps_origin": {
            "lat": 52.3676,
            "lon": 4.9041,
            "accuracy_m": 5,
            "captured_at": now,
        },
        "orientation_origin": {
            "heading_deg": 0,
            "captured_at": now,
        },
        "config_snapshot": {
            "segment_length_sec": 30,
            "overlap_sec": 5,
            "confidence_threshold": 0.6,
        },
        "detection_summary": {
            "total_detections": 0,
            "by_class": {},
        },
        "redact_location": False,
        "skip_backend_capture": True,
    }


def expected_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the subset of payload fields persisted alongside sessions."""

    metadata: dict[str, Any] = {
        "device_id": UUID(payload["device_id"]),
        "operator_alias": payload.get("operator_alias"),
        "notes": payload.get("notes"),
        "app_version": payload.get("app_version"),
        "model_bundle_version": payload.get("model_bundle_version"),
        "device_info": payload.get("device_info"),
        "gps_origin": payload.get("gps_origin"),
        "orientation_origin": payload.get("orientation_origin"),
        "config_snapshot": payload.get("config_snapshot"),
        "redact_location": payload.get("redact_location", False),
    }
    metadata["timezone"] = derive_timezone(
        payload["gps_origin"]["lat"], payload["gps_origin"]["lon"]
    )
    return metadata


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
    assert session.detection_summary is not None
    assert session.detection_summary.get("total_detections") == 0
    assert session.detection_summary.get("by_class") == {}
    return session, metadata


def validate_session_payload(
    payload: dict, metadata: dict[str, Any] | None = None
) -> Session:
    """Validate and return a Session object from a JSON payload."""
    session = Session.model_validate(payload)
    assert session.started_at.tzinfo is not None, "started_at must be timezone-aware"
    assert session.device_id is not None, "device_id should always be populated"
    assert session.timezone is not None, "timezone should be derived"
    assert session.detection_summary is not None, "detection summary should be present"
    assert session.created_at is not None, "created_at should be persisted"
    assert session.updated_at is not None, "updated_at should be persisted"
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
    summary_after_stop = stopped.detection_summary or {}
    assert summary_after_stop["total_detections"] >= 1
    ambient_total = summary_after_stop["by_class"].get("ambient_noise", 0)
    assert ambient_total >= 1
    assert summary_after_stop["total_detections"] >= ambient_total
    high_confidence = summary_after_stop.get("high_confidence")
    assert high_confidence is not None
    assert high_confidence["class"] == "ambient_noise"
    _assert_timezone(high_confidence["ts"])

    final_get = client.get(f"/sessions/{session.id}")
    assert final_get.status_code == 200
    final_session = validate_session_payload(final_get.json(), metadata)
    assert final_session.ended_at == stopped.ended_at
    assert final_session.detection_summary == summary_after_stop

    list_response = client.get("/sessions")
    assert list_response.status_code == 200
    session_list_payload = list_response.json()
    sessions = [validate_session_payload(item) for item in session_list_payload]
    assert session.id in {item.id for item in sessions}
    stored_payload = next(
        item for item in session_list_payload if item["id"] == str(session.id)
    )
    stored_session = validate_session_payload(stored_payload, metadata)
    assert stored_session.detection_summary == summary_after_stop


def test_frontend_payload_contract(client: TestClient) -> None:
    """The UI payload should satisfy the API schema without manual tweaks."""

    payload = make_frontend_payload()
    response = client.post("/sessions", json=payload)
    assert response.status_code == 201
    payload_with_device = dict(payload)
    payload_with_device["device_id"] = response.json()["device_id"]
    metadata = expected_metadata(payload_with_device)
    session = validate_session_payload(response.json(), metadata)
    assert session.notes == payload["notes"]
    summary = session.detection_summary or {}
    assert summary.get("total_detections", 0) == 0


def test_backend_stream_persists_segments_and_detections(
    client: TestClient,
) -> None:
    """The recording backend should generate files and detections while running."""

    session, metadata = start_session(client)

    time.sleep(0.35)

    stop_response = client.post(f"/sessions/{session.id}/stop")
    assert stop_response.status_code == 200
    stopped_session = validate_session_payload(stop_response.json(), metadata)

    detail_response = client.get(f"/sessions/{session.id}/detail")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    segments = detail.get("segments", [])
    detections = detail.get("detections", [])

    assert segments, "at least one segment should be recorded"
    assert detections, "detections should be emitted for recorded segments"

    summary = stopped_session.detection_summary or {}
    assert summary.get("total_detections", 0) == len(detections)
    assert {item["class"] for item in detections} == {"ambient_noise"}

    capture_root = Path(os.environ["BUURT_CAPTURE_ROOT"])
    for segment in segments:
        file_uri = segment["file_uri"]
        file_path = capture_root / file_uri
        assert file_path.exists(), f"segment file missing: {file_uri}"
        assert file_path.stat().st_size > 0
        assert segment.get("audio_duration_ms", 0) >= 1
        assert segment.get("frame_count", 0) >= 1
        assert segment.get("checksum"), "segment checksum should be populated"
        assert segment.get("size_bytes", 0) > 0

    store: SessionStore = client.app.state.session_store

    async def _fetch() -> object:
        return await store.storage.get_session(str(session.id))

    record = asyncio.run(_fetch())
    assert record.segments, "session should persist segments in storage"
    db_segment = record.segments[0]
    assert db_segment.frame_count is not None and db_segment.frame_count > 0
    assert db_segment.audio_duration_ms is not None and db_segment.audio_duration_ms > 0
    assert db_segment.checksum, "checksum should be stored in the database"
    assert db_segment.size_bytes is not None and db_segment.size_bytes > 0


def test_skip_backend_capture_flag_disables_backend(client: TestClient) -> None:
    """Sessions can opt out of the synthetic capture backend."""

    payload = make_session_payload()
    payload["skip_backend_capture"] = True
    response = client.post("/sessions", json=payload)
    assert response.status_code == 201
    session = validate_session_payload(response.json())

    time.sleep(0.35)

    stop_response = client.post(f"/sessions/{session.id}/stop")
    assert stop_response.status_code == 200

    detail_response = client.get(f"/sessions/{session.id}/detail")
    assert detail_response.status_code == 200
    detail = detail_response.json()

    assert detail.get("segments", []) == []
    assert detail.get("detections", []) == []
    summary = detail.get("detection_summary") or {}
    assert summary.get("total_detections", 0) == 0


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
    assert updated_sessions[session.id].timezone == metadata["timezone"]
    assert updated_sessions[session.id].device_id == metadata["device_id"]


def test_segment_and_detection_uploads_persist_and_broadcast(
    passive_client: TestClient,
) -> None:
    """Segments and detections should persist and trigger update broadcasts."""

    revision, _ = _fetch_snapshot(passive_client, timeout=0.1)
    session, _ = start_session(passive_client)
    revision, _ = _fetch_snapshot(passive_client, cursor=revision, timeout=0.1)

    segment_payload = make_segment_payload()
    segment_response = passive_client.post(
        f"/sessions/{session.id}/segments", json=segment_payload
    )
    assert segment_response.status_code == 201
    segment_data = segment_response.json()
    assert segment_data["file_path"] == segment_payload["file_uri"]

    segment_revision, _ = _fetch_snapshot(passive_client, cursor=revision, timeout=0.1)
    assert segment_revision > revision

    detection_payload = make_detection_payload()
    detection_response = passive_client.post(
        f"/segments/{segment_data['id']}/detections", json=detection_payload
    )
    assert detection_response.status_code == 201


def test_segment_upload_endpoint_writes_media_file(
    passive_client: TestClient,
) -> None:
    """Uploading a media blob should write to disk and persist metadata."""

    session, _ = start_session(passive_client)
    start_ts = datetime.now(UTC)
    end_ts = start_ts + timedelta(seconds=2)
    blob = b"fake-media-bytes"

    response = passive_client.post(
        f"/sessions/{session.id}/segments/upload",
        data={
            "index": "0",
            "start_ts": start_ts.isoformat(),
            "end_ts": end_ts.isoformat(),
        },
        files={"file": ("segment.webm", blob, "audio/webm")},
    )
    assert response.status_code == 201
    payload = response.json()
    assert payload["size_bytes"] == len(blob)
    assert payload["audio_duration_ms"] >= 2000

    capture_root = Path(os.environ["BUURT_CAPTURE_ROOT"])
    stored_path = capture_root / payload["file_path"]
    assert stored_path.exists()
    assert stored_path.read_bytes() == blob

    detail_response = passive_client.get(f"/sessions/{session.id}/detail")
    assert detail_response.status_code == 200
    segments = detail_response.json().get("segments", [])
    assert any(seg["id"] == payload["id"] for seg in segments)


def _assert_timezone(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    assert dt.tzinfo is not None, "datetime values must include timezone information"
    return dt


def test_session_detail_endpoint_returns_expected_payload(
    passive_client: TestClient,
) -> None:
    session, metadata = start_session(passive_client)

    segment_payload = make_segment_payload()
    segment_response = passive_client.post(
        f"/sessions/{session.id}/segments", json=segment_payload
    )
    assert segment_response.status_code == 201
    segment_data = segment_response.json()

    detection_payload = make_detection_payload()
    detection_response = passive_client.post(
        f"/segments/{segment_data['id']}/detections", json=detection_payload
    )
    assert detection_response.status_code == 201

    second_detection_payload = make_detection_payload()
    second_detection_payload["class"] = "siren"
    second_detection_payload["confidence"] = 0.41
    second_base_ts = datetime.fromisoformat(second_detection_payload["timestamp"])
    second_timestamp = (second_base_ts + timedelta(seconds=5)).isoformat()
    second_detection_payload["timestamp"] = second_timestamp
    second_detection_payload["gps_point"]["ts"] = second_timestamp
    second_response = passive_client.post(
        f"/segments/{segment_data['id']}/detections", json=second_detection_payload
    )
    assert second_response.status_code == 201

    detail_response = passive_client.get(f"/sessions/{session.id}/detail")
    assert detail_response.status_code == 200
    detail = detail_response.json()

    assert detail["id"] == str(session.id)
    assert detail["device_id"] == str(metadata["device_id"])
    assert detail["operator_alias"] == metadata["operator_alias"]
    assert detail["notes"] == metadata["notes"]
    assert detail["timezone"] == metadata["timezone"]
    assert detail["app_version"] == metadata["app_version"]
    assert detail["model_bundle_version"] == metadata["model_bundle_version"]
    assert detail["device_info"] == metadata.get("device_info")
    assert detail["gps_origin"]["lat"] == metadata["gps_origin"]["lat"]
    summary = detail["detection_summary"]
    assert summary["total_detections"] >= 2
    assert summary["by_class"].get(detection_payload["class"], 0) >= 1
    assert summary["by_class"].get(second_detection_payload["class"], 0) >= 1
    first_ts = _assert_timezone(summary["first_ts"])
    last_ts = _assert_timezone(summary["last_ts"])
    assert first_ts <= last_ts
    high_conf_summary = summary["high_confidence"]
    assert high_conf_summary["class"] == detection_payload["class"]
    assert high_conf_summary["confidence"] == detection_payload["confidence"]
    high_conf_ts = _assert_timezone(high_conf_summary["ts"])
    assert high_conf_ts == _assert_timezone(detection_payload["timestamp"])
    assert detail["redact_location"] == metadata["redact_location"]
    _assert_timezone(detail["started_at"])
    assert detail["detections"], "session detail should include detections"
    _assert_timezone(detail["created_at"])
    _assert_timezone(detail["updated_at"])

    segment_detail = detail["segments"][0]
    assert segment_detail["file_uri"] == segment_payload["file_uri"]
    _assert_timezone(segment_detail["start_ts"])
    _assert_timezone(segment_detail["end_ts"])

    detection_classes = {item["class"] for item in detail["detections"]}
    assert detection_payload["class"] in detection_classes
    assert second_detection_payload["class"] in detection_classes
    for detection_detail in detail["detections"]:
        _assert_timezone(detection_detail["timestamp"])


def test_session_detection_pagination_returns_expected_pages(
    passive_client: TestClient,
) -> None:
    session, _ = start_session(passive_client)

    segment_payload = make_segment_payload()
    segment_response = passive_client.post(
        f"/sessions/{session.id}/segments", json=segment_payload
    )
    assert segment_response.status_code == 201
    segment_id = segment_response.json()["id"]

    first_detection = make_detection_payload()
    first_response = passive_client.post(
        f"/segments/{segment_id}/detections", json=first_detection
    )
    assert first_response.status_code == 201

    second_detection = make_detection_payload()
    base_ts = datetime.fromisoformat(first_detection["timestamp"])
    adjusted_ts = (base_ts + timedelta(seconds=5)).isoformat()
    second_detection["class"] = "siren"
    second_detection["timestamp"] = adjusted_ts
    second_detection["gps_point"]["ts"] = adjusted_ts
    second_response = passive_client.post(
        f"/segments/{segment_id}/detections", json=second_detection
    )
    assert second_response.status_code == 201

    first_page = passive_client.get(
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

    second_page = passive_client.get(
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
        time.sleep(0.35)
        client.post(f"/sessions/{session.id}/stop")
        stopped = None
        for _ in range(3):
            final = websocket.receive_json()
            candidate = next(item for item in final if item["id"] == str(session.id))
            if candidate["ended_at"] is not None:
                stopped = candidate
                break
        assert stopped is not None and stopped["ended_at"] is not None
        assert stopped["config_snapshot"] == metadata["config_snapshot"]
        summary = stopped.get("detection_summary") or {}
        assert summary.get("total_detections", 0) >= 1
