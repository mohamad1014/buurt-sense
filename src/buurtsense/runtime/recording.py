"""Background orchestration for recording and inference flows."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict

from buurtsense.db import RecordingSession
from buurtsense.storage import (
    DetectionCreate,
    RecordingSessionCreate,
    RecordingSessionUpdate,
    SegmentCreate,
    SessionStorage,
)

from .exceptions import SessionAlreadyStoppedError, SessionNotFoundError

logger = logging.getLogger(__name__)


class RecordingOrchestrator:
    """Manage the lifecycle of recording sessions and simulated inference."""

    def __init__(
        self,
        storage: SessionStorage,
        *,
        segment_interval: float = 0.25,
    ) -> None:
        """Initialise the orchestrator with a persistent storage backend."""

        self._storage = storage
        self._segment_interval = segment_interval
        self._tasks: Dict[str, asyncio.Task[None]] = {}
        self._stop_events: Dict[str, asyncio.Event] = {}
        self._completed_sessions: set[str] = set()
        self._lock = asyncio.Lock()

    async def start_session(
        self,
        *,
        device_info: Dict[str, object] | None = None,
        gps_origin: Dict[str, object] | None = None,
        orientation_origin: Dict[str, object] | None = None,
    ) -> RecordingSession:
        """Persist a new session and launch its recording worker."""

        async with self._lock:
            record = await self._storage.create_session(
                RecordingSessionCreate(
                    started_at=datetime.now(timezone.utc),
                    device_info=device_info,
                    gps_origin=gps_origin,
                    orientation_origin=orientation_origin,
                )
            )
            stop_event = asyncio.Event()
            task = asyncio.create_task(
                self._run_pipeline(record.id, stop_event),
                name=f"recording-session-{record.id}",
            )
            self._stop_events[record.id] = stop_event
            self._tasks[record.id] = task
            return record

    async def stop_session(
        self,
        session_id: str,
        *,
        device_info: Dict[str, object] | None = None,
    ) -> RecordingSession:
        """Signal a running session to stop and persist the final state."""

        if session_id in self._completed_sessions:
            raise SessionAlreadyStoppedError(session_id)

        async with self._lock:
            stop_event = self._stop_events.get(session_id)
            task = self._tasks.get(session_id)
            if stop_event is None or task is None:
                raise SessionNotFoundError(session_id)
            if stop_event.is_set():
                raise SessionAlreadyStoppedError(session_id)
            stop_event.set()

        try:
            await task
        finally:
            async with self._lock:
                self._tasks.pop(session_id, None)
                self._stop_events.pop(session_id, None)
                self._completed_sessions.add(session_id)

        record = await self._storage.end_session(
            session_id,
            RecordingSessionUpdate(
                ended_at=datetime.now(timezone.utc),
                device_info=device_info,
            ),
        )
        return record

    async def shutdown(self) -> None:
        """Gracefully stop all running sessions."""

        async with self._lock:
            events = list(self._stop_events.values())
            tasks = list(self._tasks.values())
            for event in events:
                event.set()
            self._stop_events.clear()
            self._tasks.clear()

        for task in tasks:
            try:
                await task
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Recording pipeline task terminated with an error")

        self._completed_sessions.clear()

    async def _run_pipeline(self, session_id: str, stop_event: asyncio.Event) -> None:
        """Simulate media segmentation and inference events for a session."""

        segment_index = 0
        segment_start = datetime.now(timezone.utc)

        try:
            while True:
                try:
                    await asyncio.wait_for(
                        stop_event.wait(), timeout=self._segment_interval
                    )
                except asyncio.TimeoutError:
                    await self._emit_segment(session_id, segment_index, segment_start)
                    segment_index += 1
                    segment_start = datetime.now(timezone.utc)
                else:
                    if segment_index == 0:
                        await self._emit_segment(
                            session_id, segment_index, segment_start
                        )
                    break
        except asyncio.CancelledError:  # pragma: no cover - cooperative cancellation
            logger.info("Recording pipeline for session %s cancelled", session_id)
            raise
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Recording pipeline for session %s failed", session_id)
            raise

    async def _emit_segment(
        self,
        session_id: str,
        index: int,
        start_ts: datetime,
    ) -> None:
        """Persist a synthetic segment and detection for demonstration purposes."""

        end_ts = datetime.now(timezone.utc)
        if end_ts <= start_ts:
            end_ts = start_ts + timedelta(milliseconds=100)

        segment = await self._storage.create_segment(
            SegmentCreate(
                session_id=session_id,
                index=index,
                start_ts=start_ts,
                end_ts=end_ts,
                file_path=f"segments/{session_id}/{index:05d}.mp4",
                gps_trace=None,
                orientation_trace=None,
            )
        )

        await self._storage.create_detection(
            DetectionCreate(
                segment_id=segment.id,
                label="placeholder",
                confidence=0.5,
                timestamp=end_ts,
                gps_point=None,
                orientation=None,
            )
        )
