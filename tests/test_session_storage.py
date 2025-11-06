from datetime import UTC, datetime, timedelta

import pytest
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
