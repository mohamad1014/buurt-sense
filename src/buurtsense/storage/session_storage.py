"""High-level API for interacting with the durable session store."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from ..db import Detection, RecordingSession, Segment, create_engine, get_sessionmaker
from ..db.session import init_db, session_scope


@dataclass(slots=True)
class RecordingSessionCreate:
    """Payload required to start a recording session."""

    started_at: datetime
    ended_at: datetime | None = None
    device_id: str | None = None
    operator_alias: str | None = None
    notes: str | None = None
    timezone: str = "UTC"
    app_version: str | None = None
    model_bundle_version: str | None = None
    device_info: dict[str, Any] | None = None
    gps_origin: dict[str, Any] | None = None
    orientation_origin: dict[str, Any] | None = None
    config_snapshot: dict[str, Any] | None = None
    detection_summary: dict[str, Any] | None = None
    redact_location: bool = False


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

    def __init__(
        self, *, engine: AsyncEngine | None = None, db_url: str | None = None
    ) -> None:
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
        """Apply database migrations on first launch."""

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
                ended_at=payload.ended_at,
                device_id=payload.device_id,
                operator_alias=payload.operator_alias,
                notes=payload.notes,
                timezone=payload.timezone,
                app_version=payload.app_version,
                model_bundle_version=payload.model_bundle_version,
                device_info=payload.device_info,
                gps_origin=payload.gps_origin,
                orientation_origin=payload.orientation_origin,
                config_snapshot=payload.config_snapshot,
                detection_summary=payload.detection_summary or {},
                redact_location=payload.redact_location,
            )
            session.add(record)
            await session.flush()
            await session.refresh(record)
            return record

    async def end_session(
        self, session_id: str, payload: RecordingSessionUpdate
    ) -> RecordingSession:
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
            await self._update_detection_summary(
                session, detection.segment_id, detection
            )
            await session.refresh(detection)
            return detection

    @staticmethod
    def _parse_summary_timestamp(value: Any) -> datetime | None:
        """Return a datetime parsed from stored summary metadata."""

        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None

    async def _update_detection_summary(
        self, session: AsyncSession, segment_id: str, detection: Detection
    ) -> None:
        """Update a session's detection summary after persisting a detection."""

        segment = await session.get(Segment, segment_id)
        if segment is None:
            return

        record = await session.get(RecordingSession, segment.session_id)
        if record is None:
            return

        summary: dict[str, Any] = dict(record.detection_summary or {})

        total = summary.get("total_detections", 0)
        try:
            total_int = int(total)
        except (TypeError, ValueError):
            total_int = 0
        summary["total_detections"] = total_int + 1

        by_class_raw = summary.get("by_class") or {}
        by_class: dict[str, int] = {}
        if isinstance(by_class_raw, dict):
            for label, count in by_class_raw.items():
                try:
                    by_class[str(label)] = int(count)
                except (TypeError, ValueError):
                    continue
        by_class[detection.label] = by_class.get(detection.label, 0) + 1
        summary["by_class"] = by_class

        ts = detection.timestamp
        first_ts = self._parse_summary_timestamp(summary.get("first_ts"))
        if first_ts is None or ts < first_ts:
            summary["first_ts"] = ts.isoformat()
        last_ts = self._parse_summary_timestamp(summary.get("last_ts"))
        if last_ts is None or ts > last_ts:
            summary["last_ts"] = ts.isoformat()

        high_confidence_raw = summary.get("high_confidence")
        existing_confidence: float | None = None
        if isinstance(high_confidence_raw, dict):
            try:
                existing_confidence = float(high_confidence_raw.get("confidence", 0.0))
            except (TypeError, ValueError):
                existing_confidence = None

        if existing_confidence is None or detection.confidence > existing_confidence:
            summary["high_confidence"] = {
                "class": detection.label,
                "confidence": detection.confidence,
                "ts": ts.isoformat(),
            }

        record.detection_summary = summary
        session.add(record)
        await session.flush([record])

    async def get_segment(self, segment_id: str) -> Segment:
        """Retrieve a segment by identifier."""

        async with self.sessionmaker() as session:
            result = await session.execute(
                select(Segment).where(Segment.id == segment_id)
            )
            return result.scalar_one()

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

    async def list_sessions(
        self, limit: int = 20, offset: int = 0
    ) -> list[RecordingSession]:
        """Return a paginated list of sessions."""

        async with self.sessionmaker() as session:
            result = await session.execute(
                select(RecordingSession)
                .order_by(RecordingSession.started_at.desc())
                .offset(offset)
                .limit(limit)
            )
            return list(result.scalars().all())

    async def list_session_detections(
        self, session_id: str, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[Detection], int]:
        """Return detections for a session with pagination support."""

        async with self.sessionmaker() as session:
            detections_stmt = (
                select(Detection)
                .join(Segment)
                .where(Segment.session_id == session_id)
                .order_by(Detection.timestamp.asc())
                .offset(offset)
                .limit(limit)
            )
            detection_records = await session.execute(detections_stmt)
            detections = list(detection_records.scalars().all())

            total_stmt = (
                select(func.count())
                .select_from(Detection)
                .join(Segment)
                .where(Segment.session_id == session_id)
            )
            total_result = await session.execute(total_stmt)
            total = int(total_result.scalar_one())

            return detections, total
