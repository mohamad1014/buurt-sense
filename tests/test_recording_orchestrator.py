from __future__ import annotations

import asyncio

import pytest

from buurtsense.runtime import RecordingOrchestrator
from buurtsense.storage import SessionStorage


@pytest.mark.anyio("asyncio")
async def test_orchestrator_writes_segments(tmp_path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'runtime.db'}"
    storage = SessionStorage(db_url=db_url)
    await storage.initialize()

    orchestrator = RecordingOrchestrator(storage, segment_interval=0.05)
    session = await orchestrator.start_session()

    await asyncio.sleep(0.12)

    stopped = await orchestrator.stop_session(session.id)
    assert stopped.ended_at is not None

    fetched = await storage.get_session(session.id)
    assert fetched.ended_at is not None
    assert fetched.segments, "segments should be recorded during the session"
    assert fetched.segments[0].detections, "detections should be tied to a segment"

    await orchestrator.shutdown()
    await storage.close()
