"""High-level API for interacting with the durable session store."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from ..db import Detection, RecordingSession, Segment, create_engine, get_sessionmaker
from ..db.session import init_db, session_scope


@dataclass(slots=True)
class RecordingSessionCreate:
    """Payload required to start a recording session."""

    started_at: datetime
    device_info: dict[str, Any] | None = None
    gps_origin: dict[str, Any] | None = None
    orientation_origin: dict[str, Any] | None = None


@dataclass(slots=True)
class RecordingSessionUpdate:
    """Payload for updating an existing session."""

    ended_at: datetime | None = None
    device_info: dict[str, Any] | None = None


@dataclass(slots=True)
class SegmentCreate:
    """Payload required to append a media segment to a session."""

    session_id: str
    index: int
    start_ts: datetime
    end_ts: datetime
    file_path: str
    gps_trace: list[dict[str, Any]] | None = None
    orientation_trace: list[dict[str, Any]] | None = None


@dataclass(slots=True)
class DetectionCreate:
    """Payload for persisting a detection event."""

    segment_id: str
    label: str
    confidence: float
    timestamp: datetime
    gps_point: dict[str, Any] | None = None
    orientation: dict[str, Any] | None = None


class SessionStorage:
    """Facade responsible for durable session persistence."""

    def __init__(self, *, engine: AsyncEngine | None = None, db_url: str | None = None) -> None:
        if engine is None:
            self.engine = create_engine(db_url)
        else:
            self.engine = engine
        self._sessionmaker: async_sessionmaker[AsyncSession] | None = None

    @property
    def sessionmaker(self) -> async_sessionmaker[AsyncSession]:
        if self._sessionmaker is None:
            self._sessionmaker = get_sessionmaker(self.engine)
        return self._sessionmaker

    async def initialize(self) -> None:
        """Create database tables on first launch."""

        await init_db(self.engine)

    async def close(self) -> None:
        """Dispose of the underlying connection pool."""

        await self.engine.dispose()

    async def __aenter__(self) -> "SessionStorage":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        await self.close()

    async def create_session(self, payload: RecordingSessionCreate) -> RecordingSession:
        """Persist a new :class:`RecordingSession`."""

        async with session_scope(self.sessionmaker) as session:
            record = RecordingSession(
                started_at=payload.started_at,
                device_info=payload.device_info,
                gps_origin=payload.gps_origin,
                orientation_origin=payload.orientation_origin,
            )
            session.add(record)
            await session.flush()
            await session.refresh(record)
            return record

    async def end_session(self, session_id: str, payload: RecordingSessionUpdate) -> RecordingSession:
        """Mark a session as ended and optionally update device metadata."""

        async with session_scope(self.sessionmaker) as session:
            result = await session.execute(
                select(RecordingSession).where(RecordingSession.id == session_id)
            )
            record = result.scalar_one()

            if payload.ended_at is not None:
                record.ended_at = payload.ended_at
            if payload.device_info is not None:
                record.device_info = payload.device_info

            await session.flush()
            await session.refresh(record)
            return record

    async def create_segment(self, payload: SegmentCreate) -> Segment:
        """Persist a new :class:`Segment` for a session."""

        async with session_scope(self.sessionmaker) as session:
            segment = Segment(
                session_id=payload.session_id,
                index=payload.index,
                start_ts=payload.start_ts,
                end_ts=payload.end_ts,
                file_path=payload.file_path,
                gps_trace=payload.gps_trace or [],
                orientation_trace=payload.orientation_trace or [],
            )
            session.add(segment)
            await session.flush()
            await session.refresh(segment)
            return segment

    async def create_detection(self, payload: DetectionCreate) -> Detection:
        """Persist a new detection tied to a segment."""

        async with session_scope(self.sessionmaker) as session:
            detection = Detection(
                segment_id=payload.segment_id,
                label=payload.label,
                confidence=payload.confidence,
                timestamp=payload.timestamp,
                gps_point=payload.gps_point,
                orientation=payload.orientation,
            )
            session.add(detection)
            await session.flush()
            await session.refresh(detection)
            return detection

    async def get_session(self, session_id: str) -> RecordingSession:
        """Retrieve a session and its related data."""

        async with self.sessionmaker() as session:
            result = await session.execute(
                select(RecordingSession)
                .options(
                    selectinload(RecordingSession.segments).selectinload(
                        Segment.detections
                    )
                )
                .where(RecordingSession.id == session_id)
            )
            return result.scalar_one()

    async def list_sessions(self, limit: int = 20, offset: int = 0) -> list[RecordingSession]:
        """Return a paginated list of sessions."""

        async with self.sessionmaker() as session:
            result = await session.execute(
                select(RecordingSession)
                .order_by(RecordingSession.started_at.desc())
                .offset(offset)
                .limit(limit)
            )
            return list(result.scalars().all())
