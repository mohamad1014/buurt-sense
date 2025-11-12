"""Durable storage management and backend hooks for recording sessions."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Dict, Mapping, Protocol
from uuid import UUID

from fastapi.encoders import jsonable_encoder

from buurtsense.storage import (
    DetectionCreate as StorageDetectionCreate,
    RecordingSessionCreate,
    RecordingSessionUpdate,
    SegmentCreate as StorageSegmentCreate,
    SessionStorage,
)
from sqlalchemy.exc import NoResultFound

from .models import Session
from .schemas import DetectionCreate as DetectionCreateSchema
from .schemas import DetectionRead
from .schemas import PaginatedDetections
from .schemas import SegmentCreate as SegmentCreateSchema
from .schemas import SegmentRead
from .schemas import SessionDetail


class SessionNotFoundError(KeyError):
    """Raised when a session identifier is unknown to the store."""


class SessionAlreadyStoppedError(RuntimeError):
    """Raised when attempting to stop a session that already has an end timestamp."""


class SegmentNotFoundError(KeyError):
    """Raised when a segment identifier is unknown to the store."""


class RecordingBackend(Protocol):
    """Protocol describing the recording/inference hooks used by the store."""

    async def start(self, session_id: str, *, started_at: datetime) -> None:
        """Begin recording and inference for ``session_id``."""

    async def stop(self, session_id: str, *, ended_at: datetime) -> None:
        """Stop recording and inference for ``session_id``."""


class SimpleRecordingBackend:
    """Minimal backend that persists a synthetic segment and detection on stop."""

    def __init__(self, storage: SessionStorage) -> None:
        self._storage = storage
        self._active: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()

    async def start(self, session_id: str, *, started_at: datetime) -> None:
        async with self._lock:
            self._active[session_id] = started_at

    async def stop(self, session_id: str, *, ended_at: datetime) -> None:
        async with self._lock:
            started_at = self._active.pop(session_id, ended_at)

        if ended_at <= started_at:
            ended_at = started_at + timedelta(seconds=1)

        segment = await self._storage.create_segment(
            StorageSegmentCreate(
                session_id=session_id,
                index=0,
                start_ts=started_at,
                end_ts=ended_at,
                file_path=f"recordings/{session_id}/segment-0.dat",
            )
        )

        await self._storage.create_detection(
            StorageDetectionCreate(
                segment_id=segment.id,
                label="ambient_noise",
                confidence=0.5,
                timestamp=ended_at,
            )
        )


@dataclass(slots=True)
class _RecordAdapter:
    """Internal adapter mapping persistence models onto the API dataclass."""

    id: str
    started_at: datetime
    ended_at: datetime | None
    device_info: dict[str, Any] | None
    gps_origin: dict[str, Any] | None
    orientation_origin: dict[str, Any] | None
    config_snapshot: dict[str, Any] | None

    def to_session(self) -> Session:
        return Session(
            id=UUID(self.id),
            started_at=self.started_at,
            ended_at=self.ended_at,
            device_info=self.device_info,
            gps_origin=self.gps_origin,
            orientation_origin=self.orientation_origin,
            config_snapshot=self.config_snapshot,
        )


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    """Immutable payload describing a point-in-time view of all sessions."""

    revision: int
    sessions: tuple[Session, ...]


class SessionStore:
    """Coordinate durable storage with recording and inference hooks."""

    def __init__(
        self,
        *,
        storage: SessionStorage | None = None,
        recording_backend: RecordingBackend | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._storage = storage or SessionStorage()
        self._backend = recording_backend or SimpleRecordingBackend(self._storage)
        self._now = now or (lambda: datetime.now(UTC))
        self._lock = asyncio.Lock()
        self._initialized = False
        self._revision = 0
        self._subscribers: set[asyncio.Queue[SessionSnapshot]] = set()
        self._subscribers_lock = asyncio.Lock()

    @property
    def storage(self) -> SessionStorage:
        """Expose the underlying :class:`SessionStorage` for inspection/tests."""

        return self._storage

    async def __aenter__(self) -> "SessionStore":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        await self.close()

    async def initialize(self) -> None:
        """Ensure the backing storage is ready for use."""

        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return
            await self._storage.initialize()
            self._initialized = True

    async def close(self) -> None:
        """Dispose of the storage engine and reset internal state."""

        if not self._initialized:
            return

        async with self._lock:
            if not self._initialized:
                return
            await self._storage.close()
            self._initialized = False

    async def create(self, metadata: dict[str, Any] | None = None) -> Session:
        """Persist a new session and trigger the recording backend."""

        await self.initialize()
        started_at = self._now()
        metadata = metadata or {}
        record = await self._storage.create_session(
            RecordingSessionCreate(
                started_at=started_at,
                device_info=self._coerce_optional_mapping(
                    metadata.get("device_info"), field="device_info"
                ),
                gps_origin=self._coerce_optional_mapping(
                    metadata.get("gps_origin"), field="gps_origin"
                ),
                orientation_origin=self._coerce_optional_mapping(
                    metadata.get("orientation_origin"), field="orientation_origin"
                ),
                config_snapshot=self._coerce_optional_mapping(
                    metadata.get("config_snapshot"),
                    required=True,
                    field="config_snapshot",
                ),
            )
        )
        await self._backend.start(record.id, started_at=started_at)
        session = self._to_session(record)
        await self._broadcast()
        return session

    async def stop(self, session_id: UUID) -> Session:
        """Stop an active session and persist the end timestamp."""

        record = await self._get_record(session_id)
        if record.ended_at is not None:
            raise SessionAlreadyStoppedError(str(session_id))

        ended_at = self._now()
        await self._backend.stop(record.id, ended_at=ended_at)
        updated = await self._storage.end_session(
            record.id,
            RecordingSessionUpdate(ended_at=ended_at),
        )
        session = self._to_session(updated)
        await self._broadcast()
        return session

    async def create_segment(
        self, session_id: UUID, payload: SegmentCreateSchema
    ) -> dict[str, Any]:
        """Persist a new segment for a session and broadcast the update."""

        record = await self._get_record(session_id)
        encoded = jsonable_encoder(payload)
        start_ts = self._normalize_datetime(
            encoded.get("start_ts"), field="start_ts", required=True
        )
        end_ts = self._normalize_datetime(
            encoded.get("end_ts"), field="end_ts", required=True
        )
        if encoded.get("file_uri") is None:
            raise ValueError("file_uri is required for segment creation")

        segment = await self._storage.create_segment(
            StorageSegmentCreate(
                session_id=record.id,
                index=int(encoded.get("index", 0)),
                start_ts=start_ts,
                end_ts=end_ts,
                file_path=str(encoded["file_uri"]),
                gps_trace=list(encoded.get("gps_trace", [])),
                orientation_trace=list(encoded.get("orientation_trace", [])),
            )
        )

        await self._broadcast()
        return {
            "id": segment.id,
            "session_id": segment.session_id,
            "index": segment.index,
            "start_ts": self._normalize_datetime(
                getattr(segment, "start_ts", start_ts),
                field="start_ts",
                required=True,
            ),
            "end_ts": self._normalize_datetime(
                getattr(segment, "end_ts", end_ts),
                field="end_ts",
                required=True,
            ),
            "file_path": segment.file_path,
            "gps_trace": list(getattr(segment, "gps_trace", [])),
            "orientation_trace": list(getattr(segment, "orientation_trace", [])),
        }

    async def create_detection(
        self, segment_id: UUID, payload: DetectionCreateSchema
    ) -> dict[str, Any]:
        """Persist a detection for an existing segment and broadcast the update."""

        segment = await self._get_segment(segment_id)
        encoded = jsonable_encoder(payload)
        timestamp = self._normalize_datetime(
            encoded.get("timestamp"), field="timestamp", required=True
        )
        label = getattr(payload, "detection_class", None) or encoded.get(
            "detection_class"
        )
        if label is None:
            raise ValueError("detection_class is required for detection creation")

        detection = await self._storage.create_detection(
            StorageDetectionCreate(
                segment_id=segment.id,
                label=str(label),
                confidence=float(encoded.get("confidence", 0.0)),
                timestamp=timestamp,
                gps_point=encoded.get("gps_point"),
                orientation=(
                    {"heading_deg": encoded["orientation_heading_deg"]}
                    if encoded.get("orientation_heading_deg") is not None
                    else None
                ),
            )
        )

        await self._broadcast()
        return {
            "id": detection.id,
            "segment_id": detection.segment_id,
            "label": detection.label,
            "confidence": detection.confidence,
            "timestamp": self._normalize_datetime(
                getattr(detection, "timestamp", timestamp),
                field="timestamp",
                required=True,
            ),
            "gps_point": detection.gps_point,
            "orientation": detection.orientation,
        }

    async def get(self, session_id: UUID) -> Session:
        """Fetch a single session by identifier."""

        record = await self._get_record(session_id)
        return self._to_session(record)

    async def get_detail(self, session_id: UUID) -> SessionDetail:
        """Return a detailed view of a session including segments and detections."""

        record = await self._get_record(session_id)
        started_at = self._normalize_datetime(
            getattr(record, "started_at", None),
            field="started_at",
            required=True,
        )
        if started_at is None:  # pragma: no cover - defensive guard
            raise ValueError("started_at cannot be None")

        ended_at = self._normalize_datetime(
            getattr(record, "ended_at", None),
            field="ended_at",
            required=False,
        )

        gps_origin = self._coerce_optional_mapping(
            getattr(record, "gps_origin", None), field="gps_origin"
        )
        if gps_origin is None:
            raise ValueError("gps_origin metadata is required")

        config_snapshot = self._coerce_optional_mapping(
            getattr(record, "config_snapshot", None),
            required=True,
            field="config_snapshot",
        )

        orientation_origin = self._coerce_optional_mapping(
            getattr(record, "orientation_origin", None),
            field="orientation_origin",
        )

        segments = [
            self._segment_to_schema(segment)
            for segment in getattr(record, "segments", [])
        ]

        detections = [
            self._detection_to_schema(detection)
            for segment in getattr(record, "segments", [])
            for detection in getattr(segment, "detections", [])
        ]

        return SessionDetail(
            id=UUID(str(record.id)),
            started_at=started_at,
            ended_at=ended_at,
            device_info=self._coerce_optional_mapping(
                getattr(record, "device_info", None), field="device_info"
            ),
            gps_origin=gps_origin,
            orientation_origin=orientation_origin,
            config_snapshot=config_snapshot,
            detection_summary=self._coerce_optional_mapping(
                getattr(record, "detection_summary", None),
                field="detection_summary",
            ),
            segments=segments,
            detections=detections,
        )

    async def list_detections(
        self, session_id: UUID, *, limit: int = 50, offset: int = 0
    ) -> PaginatedDetections:
        """Return a paginated list of detections for a session."""

        await self._get_record(session_id)
        detections, total = await self._storage.list_session_detections(
            str(session_id), limit=limit, offset=offset
        )
        items = [self._detection_to_schema(detection) for detection in detections]
        return PaginatedDetections(items=items, total=total, limit=limit, offset=offset)

    async def list(self) -> list[Session]:
        """List all sessions ordered by most recent start time."""

        return await self._snapshot()

    @property
    def revision(self) -> int:
        """Return the current monotonic revision for the session snapshot."""

        return self._revision

    async def subscribe(self) -> AsyncIterator[SessionSnapshot]:
        """Yield session snapshots whenever the store mutates."""

        queue: asyncio.Queue[SessionSnapshot] = asyncio.Queue(maxsize=1)
        async with self._subscribers_lock:
            self._subscribers.add(queue)

        try:
            await queue.put(
                SessionSnapshot(self._revision, tuple(await self._snapshot()))
            )
            while True:
                snapshot = await queue.get()
                yield snapshot
        finally:
            async with self._subscribers_lock:
                self._subscribers.discard(queue)

    async def _get_record(self, session_id: UUID) -> Any:
        await self.initialize()
        try:
            return await self._storage.get_session(str(session_id))
        except NoResultFound as exc:  # pragma: no cover - defensive guard
            raise SessionNotFoundError(str(session_id)) from exc
        except KeyError as exc:  # pragma: no cover - secondary guard
            raise SessionNotFoundError(str(session_id)) from exc

    async def _get_segment(self, segment_id: UUID) -> Any:
        await self.initialize()
        try:
            return await self._storage.get_segment(str(segment_id))
        except NoResultFound as exc:  # pragma: no cover - defensive guard
            raise SegmentNotFoundError(str(segment_id)) from exc
        except KeyError as exc:  # pragma: no cover - secondary guard
            raise SegmentNotFoundError(str(segment_id)) from exc

    @staticmethod
    def _coerce_optional_mapping(
        value: Any, *, required: bool = False, field: str = "metadata"
    ) -> dict[str, Any] | None:
        """Return a shallow copy of a mapping or ``None`` if not provided."""

        if value is None:
            if required:
                raise ValueError(f"{field} metadata is required")
            return None

        if isinstance(value, dict):
            return dict(value)

        if isinstance(value, Mapping):
            return dict(value.items())

        raise TypeError(f"Unsupported metadata value for {field}: {value!r}")

    @staticmethod
    def _normalize_datetime(
        value: Any, *, field: str, required: bool
    ) -> datetime | None:
        """
        Return a timezone-aware datetime converted to UTC.

        SQLite stores timestamps without timezone information by default, so we
        defensively attach UTC to any naive values returned by SQLAlchemy and
        coerce ISO8601 strings to datetime instances.
        """

        if value is None:
            if required:
                raise ValueError(f"{field} cannot be None")
            return None

        dt: datetime
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid datetime string for {field}: {value}"
                ) from exc
        else:
            raise TypeError(f"Unsupported {field} value: {value!r}")

        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)

    def _to_session(self, record: Any) -> Session:
        """Convert a database record to a Session, ensuring timezone-aware datetimes."""

        started_at = self._normalize_datetime(
            getattr(record, "started_at", None),
            field="started_at",
            required=True,
        )
        if started_at is None:  # pragma: no cover - defensive guard
            raise ValueError("started_at cannot be None")

        ended_at = self._normalize_datetime(
            getattr(record, "ended_at", None),
            field="ended_at",
            required=False,
        )

        adapter = _RecordAdapter(
            id=str(record.id),
            started_at=started_at,
            ended_at=ended_at,
            device_info=self._coerce_optional_mapping(
                getattr(record, "device_info", None), field="device_info"
            ),
            gps_origin=self._coerce_optional_mapping(
                getattr(record, "gps_origin", None), field="gps_origin"
            ),
            orientation_origin=self._coerce_optional_mapping(
                getattr(record, "orientation_origin", None),
                field="orientation_origin",
            ),
            config_snapshot=self._coerce_optional_mapping(
                getattr(record, "config_snapshot", None),
                field="config_snapshot",
            ),
        )
        return adapter.to_session()

    def _segment_to_schema(self, segment: Any) -> SegmentRead:
        """Convert a persisted segment into the API schema."""

        start_ts = self._normalize_datetime(
            getattr(segment, "start_ts", None),
            field="start_ts",
            required=True,
        )
        end_ts = self._normalize_datetime(
            getattr(segment, "end_ts", None),
            field="end_ts",
            required=True,
        )
        if start_ts is None or end_ts is None:
            raise ValueError("segment timestamps cannot be None")

        payload: dict[str, Any] = {
            "id": segment.id,
            "index": getattr(segment, "index", 0),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "file_uri": getattr(segment, "file_path", None),
            "frame_count": getattr(segment, "frame_count", None),
            "audio_duration_ms": getattr(segment, "audio_duration_ms", None),
            "gps_trace": list(getattr(segment, "gps_trace", []) or []),
            "orientation_trace": list(getattr(segment, "orientation_trace", []) or []),
            "checksum": getattr(segment, "checksum", None),
            "size_bytes": getattr(segment, "size_bytes", None),
        }
        if payload["file_uri"] is None:
            raise ValueError("segment file path is required")
        return SegmentRead.model_validate(payload)

    def _detection_to_schema(self, detection: Any) -> DetectionRead:
        """Convert a persisted detection into the API schema."""

        timestamp = self._normalize_datetime(
            getattr(detection, "timestamp", None),
            field="timestamp",
            required=True,
        )
        if timestamp is None:
            raise ValueError("detection timestamp cannot be None")

        orientation = getattr(detection, "orientation", None)
        heading = None
        if isinstance(orientation, Mapping):
            heading = orientation.get("heading_deg")

        payload: dict[str, Any] = {
            "id": detection.id,
            "segment_id": getattr(detection, "segment_id", None),
            "class": getattr(detection, "label", None),
            "confidence": getattr(detection, "confidence", 0.0),
            "timestamp": timestamp,
            "gps_point": getattr(detection, "gps_point", None),
            "orientation_heading_deg": heading,
            "model_id": getattr(detection, "model_id", None),
            "inference_latency_ms": getattr(detection, "inference_latency_ms", None),
        }
        if payload["segment_id"] is None or payload["class"] is None:
            raise ValueError("detection requires a segment_id and class")
        return DetectionRead.model_validate(payload)

    async def _snapshot(self) -> list[Session]:
        """Return a sorted snapshot of all persisted sessions."""

        await self.initialize()
        records = await self._storage.list_sessions()
        sessions = [self._to_session(record) for record in records]
        return sorted(sessions, key=lambda session: session.started_at, reverse=True)

    async def _broadcast(self) -> None:
        """Push the current snapshot to all registered subscribers."""

        sessions = tuple(await self._snapshot())
        async with self._subscribers_lock:
            subscribers = list(self._subscribers)

        self._revision += 1
        snapshot = SessionSnapshot(self._revision, sessions)
        for queue in subscribers:
            try:
                queue.put_nowait(snapshot)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(snapshot)
                except asyncio.QueueFull:
                    # Drop the update if the subscriber is unresponsive; the next
                    # snapshot will overwrite any stale data.
                    continue
