from datetime import UTC, datetime, timedelta

import pytest
from app.storage import ContinuousCaptureBackend
from buurtsense.storage import (
    DetectionCreate,
    RecordingSessionCreate,
    RecordingSessionUpdate,
    SegmentCreate,
    SessionStorage,
)


@pytest.mark.asyncio
async def test_session_lifecycle(tmp_path):
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    storage = SessionStorage(db_url=db_url)
    await storage.initialize()

    session = await storage.create_session(
        RecordingSessionCreate(started_at=datetime.now(UTC), device_info={"device": "test"})
    )

    segment = await storage.create_segment(
        SegmentCreate(
            session_id=session.id,
            index=0,
            start_ts=datetime.now(UTC),
            end_ts=datetime.now(UTC) + timedelta(seconds=30),
            file_path="segment-0.mp4",
        )
    )

    detection = await storage.create_detection(
        DetectionCreate(
            segment_id=segment.id,
            label="gunshot",
            confidence=0.95,
            timestamp=datetime.now(UTC),
        )
    )

    assert detection.segment_id == segment.id

    ended = await storage.end_session(
        session.id, RecordingSessionUpdate(ended_at=datetime.now(UTC))
    )
    assert ended.ended_at is not None

    sessions = await storage.list_sessions()
    assert sessions

    fetched = await storage.get_session(session.id)
    assert fetched.segments[0].detections[0].label == "gunshot"

    await storage.close()


def test_capture_backend_uses_environment_overrides(monkeypatch, tmp_path):
    capture_root = tmp_path / "custom-captures"
    monkeypatch.setenv("BUURT_CAPTURE_ROOT", str(capture_root))
    monkeypatch.setenv("BUURT_SEGMENT_LENGTH_SEC", "1.25")
    monkeypatch.setenv("BUURT_SEGMENT_BYTES_PER_SEC", "2048")

    backend = ContinuousCaptureBackend(store=None)

    assert backend._capture_root == capture_root.resolve()
    assert backend._segment_length == pytest.approx(1.25)
    assert backend._bytes_per_second == 2048


def test_capture_backend_prefers_explicit_parameters(tmp_path):
    capture_root = tmp_path / "explicit"
    backend = ContinuousCaptureBackend(
        store=None,
        capture_root=capture_root,
        segment_length=3.5,
        bytes_per_second=8192,
    )

    assert backend._capture_root == capture_root.resolve()
    assert backend._segment_length == pytest.approx(3.5)
    assert backend._bytes_per_second == 8192
