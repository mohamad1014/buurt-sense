"""Durable storage management and backend hooks for recording sessions."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import blake2s
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Protocol, Type, TypeVar
from uuid import UUID, uuid4

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

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
from .schemas import SessionCreate, SessionDetail
from .utils import derive_timezone, ensure_detection_summary


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


@dataclass(slots=True)
class _SegmentArtifact:
    """Description of a captured media segment stored on disk."""

    file_path: Path
    relative_uri: str
    start_ts: datetime
    end_ts: datetime
    frame_count: int
    audio_duration_ms: int
    size_bytes: int
    checksum: str


@dataclass(slots=True)
class _CaptureWorker:
    """In-flight capture job for a running recording session."""

    session_id: str
    started_at: datetime
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    ended_at: datetime | None = None
    task: asyncio.Task[None] | None = None

    def request_stop(self, ended_at: datetime) -> None:
        """Signal the worker to finalise capture with the provided timestamp."""

        if self.stop_event.is_set():
            return
        self.ended_at = ended_at
        self.stop_event.set()


def resolve_capture_root(explicit: Path | None = None) -> Path:
    """Return the absolute capture root honoring environment overrides."""

    env_root = os.environ.get("BUURT_CAPTURE_ROOT")
    if explicit is not None:
        root_path = explicit
    elif env_root:
        root_path = Path(env_root).expanduser()
    else:
        root_path = Path("recordings")
    return root_path if root_path.is_absolute() else root_path.resolve()


class ContinuousCaptureBackend:
    """Backend that records media segments and runs lightweight inference."""

    def __init__(
        self,
        store: "SessionStore",
        *,
        capture_root: Path | None = None,
        segment_length: float | None = None,
        bytes_per_second: int | None = None,
    ) -> None:
        self._store = store
        self._capture_root = resolve_capture_root(capture_root)
        self._capture_root.mkdir(parents=True, exist_ok=True)

        env_length = os.environ.get("BUURT_SEGMENT_LENGTH_SEC")
        self._segment_length = segment_length or float(env_length or 5.0)
        if self._segment_length <= 0:
            raise ValueError("segment_length must be greater than zero")

        env_bytes = os.environ.get("BUURT_SEGMENT_BYTES_PER_SEC")
        self._bytes_per_second = bytes_per_second or int(env_bytes or 32000)
        if self._bytes_per_second <= 0:
            raise ValueError("bytes_per_second must be greater than zero")

        self._sample_rate = 16000
        self._min_segment_bytes = 512
        self._workers: Dict[str, _CaptureWorker] = {}
        self._lock = asyncio.Lock()

    def bind_to_store(self, store: "SessionStore") -> None:
        """Rebind the backend to a different session store instance."""

        self._store = store

    async def start(self, session_id: str, *, started_at: datetime) -> None:
        async with self._lock:
            if session_id in self._workers:
                raise RuntimeError(f"session {session_id} already capturing")
            worker = _CaptureWorker(session_id=session_id, started_at=started_at)
            self._workers[session_id] = worker
            worker.task = asyncio.create_task(self._run_capture(worker))

    async def stop(self, session_id: str, *, ended_at: datetime) -> None:
        async with self._lock:
            worker = self._workers.pop(session_id, None)
        if worker is None:
            return
        worker.request_stop(ended_at)
        if worker.task is not None:
            await worker.task

    async def _run_capture(self, worker: _CaptureWorker) -> None:
        """Continuously persist segments and detections until the worker stops."""

        session_uuid = UUID(worker.session_id)
        session_dir = self._capture_root / worker.session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        index = 0
        segment_start = worker.started_at

        while True:
            try:
                await asyncio.wait_for(
                    worker.stop_event.wait(), timeout=self._segment_length
                )
                is_final = True
            except asyncio.TimeoutError:
                is_final = False

            segment_end = (
                worker.ended_at
                if is_final and worker.ended_at
                else segment_start + timedelta(seconds=self._segment_length)
            )

            artifact = await self._write_segment(
                session_dir, index, segment_start, segment_end
            )

            segment_payload = SegmentCreateSchema(
                index=index,
                start_ts=artifact.start_ts,
                end_ts=artifact.end_ts,
                file_uri=artifact.relative_uri,
                frame_count=artifact.frame_count,
                audio_duration_ms=artifact.audio_duration_ms,
                size_bytes=artifact.size_bytes,
                checksum=artifact.checksum,
            )
            segment_data = await self._store.create_segment(
                session_uuid, segment_payload
            )
            segment_id = UUID(str(segment_data["id"]))

            detections = await self._infer_segment(
                session_uuid, segment_id, index, artifact
            )
            for detection in detections:
                await self._store.create_detection(segment_id, detection)

            if is_final:
                break

            segment_start = artifact.end_ts
            index += 1

    async def _write_segment(
        self,
        session_dir: Path,
        index: int,
        start_ts: datetime,
        end_ts: datetime,
    ) -> _SegmentArtifact:
        """Create a media artifact for the given time slice."""

        duration = max((end_ts - start_ts).total_seconds(), 0.0)
        approx_size = max(
            int(duration * self._bytes_per_second), self._min_segment_bytes
        )
        payload = self._build_segment_payload(start_ts, end_ts, approx_size)

        file_name = f"segment-{index:04d}.pcm"
        file_path = session_dir / file_name
        relative_uri = str(file_path.relative_to(self._capture_root))

        await asyncio.to_thread(file_path.write_bytes, payload)

        checksum = blake2s(payload).hexdigest()
        size_bytes = len(payload)
        frame_count = max(int(duration * self._sample_rate), 1)
        audio_duration_ms = max(int(duration * 1000), 1)

        return _SegmentArtifact(
            file_path=file_path,
            relative_uri=relative_uri,
            start_ts=start_ts,
            end_ts=end_ts,
            frame_count=frame_count,
            audio_duration_ms=audio_duration_ms,
            size_bytes=size_bytes,
            checksum=checksum,
        )

    def _build_segment_payload(
        self,
        start_ts: datetime,
        end_ts: datetime,
        approx_size: int,
    ) -> bytes:
        """Generate deterministic pseudo-media bytes for a segment."""

        template = f"{start_ts.isoformat()}->{end_ts.isoformat()}".encode("utf-8")
        if not template:
            template = b"segment"
        repeats = (approx_size // len(template)) + 1
        return (template * repeats)[:approx_size]

    async def _infer_segment(
        self,
        session_id: UUID,
        segment_id: UUID,
        index: int,
        artifact: _SegmentArtifact,
    ) -> list[DetectionCreateSchema]:
        """Produce lightweight detections for a persisted segment."""

        confidence = min(0.95, 0.55 + (index * 0.05))
        latency_ms = max(int(self._segment_length * 250), 1)
        detection = DetectionCreateSchema(
            detection_class="ambient_noise",
            confidence=confidence,
            timestamp=artifact.end_ts,
            inference_latency_ms=latency_ms,
            model_id="local-audio-v1",
        )
        return [detection]


@dataclass(slots=True)
class _RecordAdapter:
    """Internal adapter mapping persistence models onto the API dataclass."""

    id: str
    started_at: datetime
    ended_at: datetime | None
    device_id: str | None
    operator_alias: str | None
    notes: str | None
    timezone: str | None
    app_version: str | None
    model_bundle_version: str | None
    device_info: dict[str, Any] | None
    gps_origin: dict[str, Any] | None
    orientation_origin: dict[str, Any] | None
    config_snapshot: dict[str, Any] | None
    detection_summary: dict[str, Any] | None
    redact_location: bool
    created_at: datetime | None
    updated_at: datetime | None

    def to_session(self) -> Session:
        return Session(
            id=UUID(self.id),
            started_at=self.started_at,
            ended_at=self.ended_at,
            device_id=UUID(self.device_id) if self.device_id else None,
            operator_alias=self.operator_alias,
            notes=self.notes,
            timezone=self.timezone,
            app_version=self.app_version,
            model_bundle_version=self.model_bundle_version,
            device_info=self.device_info,
            gps_origin=self.gps_origin,
            orientation_origin=self.orientation_origin,
            config_snapshot=self.config_snapshot,
            detection_summary=self.detection_summary,
            redact_location=self.redact_location,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    """Immutable payload describing a point-in-time view of all sessions."""

    revision: int
    sessions: tuple[Session, ...]


SchemaT = TypeVar("SchemaT", bound=BaseModel)


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
        if recording_backend is None:
            self._backend = ContinuousCaptureBackend(self)
        else:
            self._backend = recording_backend
            binder = getattr(self._backend, "bind_to_store", None)
            if callable(binder):
                binder(self)
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

    async def create(self, payload: SessionCreate) -> Session:
        """Persist a new session and trigger the recording backend."""

        await self.initialize()
        started_at = self._normalize_datetime(
            payload.started_at, field="started_at", required=True
        )
        ended_at = self._normalize_datetime(
            payload.ended_at, field="ended_at", required=False
        )
        gps_origin = jsonable_encoder(payload.gps_origin, exclude_none=True)
        if not gps_origin:
            raise ValueError("gps_origin metadata is required")
        timezone = derive_timezone(gps_origin["lat"], gps_origin["lon"])
        summary = ensure_detection_summary(payload.detection_summary)
        device_id = payload.device_id or uuid4()

        record = await self._storage.create_session(
            RecordingSessionCreate(
                started_at=started_at,
                ended_at=ended_at,
                device_id=str(device_id),
                operator_alias=payload.operator_alias,
                notes=payload.notes,
                timezone=timezone,
                app_version=payload.app_version,
                model_bundle_version=payload.model_bundle_version,
                device_info=(
                    jsonable_encoder(payload.device_info, exclude_none=True)
                    if payload.device_info is not None
                    else None
                ),
                gps_origin=gps_origin,
                orientation_origin=(
                    jsonable_encoder(payload.orientation_origin, exclude_none=True)
                    if payload.orientation_origin is not None
                    else None
                ),
                config_snapshot=jsonable_encoder(
                    payload.config_snapshot, exclude_none=True
                ),
                detection_summary=jsonable_encoder(
                    summary, exclude_none=True, by_alias=True
                ),
                redact_location=payload.redact_location,
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
        await self._storage.end_session(
            record.id,
            RecordingSessionUpdate(ended_at=ended_at),
        )
        await self._broadcast()
        await self._backend.stop(record.id, ended_at=ended_at)
        final_record = await self._get_record(session_id)
        return self._to_session(final_record)

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
                frame_count=encoded.get("frame_count"),
                audio_duration_ms=encoded.get("audio_duration_ms"),
                checksum=encoded.get("checksum"),
                size_bytes=encoded.get("size_bytes"),
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
            "frame_count": getattr(segment, "frame_count", None),
            "audio_duration_ms": getattr(segment, "audio_duration_ms", None),
            "checksum": getattr(segment, "checksum", None),
            "size_bytes": getattr(segment, "size_bytes", None),
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

        detection_summary = self._parse_detection_summary(
            getattr(record, "detection_summary", None)
        )

        created_at = self._normalize_datetime(
            getattr(record, "created_at", None),
            field="created_at",
            required=False,
        )
        updated_at = self._normalize_datetime(
            getattr(record, "updated_at", None),
            field="updated_at",
            required=False,
        )

        device_id_raw = getattr(record, "device_id", None)
        device_id = UUID(str(device_id_raw)) if device_id_raw else None

        return SessionDetail(
            id=UUID(str(record.id)),
            started_at=started_at,
            ended_at=ended_at,
            device_id=device_id,
            operator_alias=getattr(record, "operator_alias", None),
            notes=getattr(record, "notes", None),
            timezone=getattr(record, "timezone", None),
            app_version=getattr(record, "app_version", None),
            model_bundle_version=getattr(record, "model_bundle_version", None),
            device_info=self._coerce_optional_mapping(
                getattr(record, "device_info", None), field="device_info"
            ),
            gps_origin=gps_origin,
            orientation_origin=orientation_origin,
            config_snapshot=config_snapshot,
            detection_summary=detection_summary,
            redact_location=bool(getattr(record, "redact_location", False)),
            segments=segments,
            detections=detections,
            created_at=created_at,
            updated_at=updated_at,
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

    def _parse_detection_summary(self, value: Any) -> dict[str, Any]:
        """Normalise detection summary payloads into canonical structures."""

        summary: dict[str, Any] = {
            "total_detections": 0,
            "by_class": {},
        }

        if value is None:
            return summary

        if isinstance(value, Mapping):
            data = dict(value.items())
        elif isinstance(value, dict):
            data = dict(value)
        else:
            return summary

        total = data.get("total_detections", 0)
        try:
            summary["total_detections"] = int(total)
        except (TypeError, ValueError):
            summary["total_detections"] = 0

        by_class_raw = data.get("by_class") or {}
        parsed_by_class: dict[str, int] = {}
        if isinstance(by_class_raw, Mapping):
            for label, count in by_class_raw.items():
                try:
                    parsed_by_class[str(label)] = int(count)
                except (TypeError, ValueError):
                    continue
        summary["by_class"] = parsed_by_class

        first_ts = self._normalize_datetime(
            data.get("first_ts"), field="detection_summary.first_ts", required=False
        )
        if first_ts is not None:
            summary["first_ts"] = first_ts

        last_ts = self._normalize_datetime(
            data.get("last_ts"), field="detection_summary.last_ts", required=False
        )
        if last_ts is not None:
            summary["last_ts"] = last_ts

        high_conf = data.get("high_confidence")
        if isinstance(high_conf, Mapping):
            high_dict = dict(high_conf.items())
            detection_class = high_dict.get("class") or high_dict.get("detection_class")
            confidence = high_dict.get("confidence")
            ts = self._normalize_datetime(
                high_dict.get("ts"),
                field="detection_summary.high_confidence.ts",
                required=False,
            )

            parsed_high: dict[str, Any] = {}
            if detection_class is not None:
                parsed_high["class"] = str(detection_class)
            if confidence is not None:
                try:
                    parsed_high["confidence"] = float(confidence)
                except (TypeError, ValueError):
                    parsed_high.pop("confidence", None)
            if ts is not None:
                parsed_high["ts"] = ts

            if {
                "class",
                "confidence",
                "ts",
            } <= parsed_high.keys():
                summary["high_confidence"] = parsed_high

        return summary

    def _summary_for_session(self, value: Any) -> dict[str, Any]:
        """Return a serialisable detection summary for session payloads."""

        parsed = self._parse_detection_summary(value)
        summary: dict[str, Any] = {
            "total_detections": parsed.get("total_detections", 0),
            "by_class": dict(parsed.get("by_class", {})),
        }

        first_ts = parsed.get("first_ts")
        if isinstance(first_ts, datetime):
            summary["first_ts"] = first_ts.isoformat()

        last_ts = parsed.get("last_ts")
        if isinstance(last_ts, datetime):
            summary["last_ts"] = last_ts.isoformat()

        high_conf = parsed.get("high_confidence")
        if isinstance(high_conf, Mapping):
            high_payload = dict(high_conf.items())
            ts_value = high_payload.get("ts")
            if isinstance(ts_value, datetime):
                high_payload["ts"] = ts_value.isoformat()
            summary["high_confidence"] = high_payload

        return summary

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
            device_id=(
                str(getattr(record, "device_id", None))
                if getattr(record, "device_id", None)
                else None
            ),
            operator_alias=getattr(record, "operator_alias", None),
            notes=getattr(record, "notes", None),
            timezone=getattr(record, "timezone", None),
            app_version=getattr(record, "app_version", None),
            model_bundle_version=getattr(record, "model_bundle_version", None),
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
            detection_summary=self._summary_for_session(
                getattr(record, "detection_summary", None)
            ),
            redact_location=bool(getattr(record, "redact_location", False)),
            created_at=self._normalize_datetime(
                getattr(record, "created_at", None),
                field="created_at",
                required=False,
            ),
            updated_at=self._normalize_datetime(
                getattr(record, "updated_at", None),
                field="updated_at",
                required=False,
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
        return self._validate_schema(SegmentRead, payload)

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
        return self._validate_schema(DetectionRead, payload)

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

    @staticmethod
    def _validate_schema(model: Type[SchemaT], payload: dict[str, Any]) -> SchemaT:
        """
        Validate payload with the provided Pydantic model with v1/v2 compatibility.

        Pydantic v2 exposes ``model_validate`` while v1 uses ``parse_obj``. This
        helper keeps the store agnostic to the installed major version.
        """

        validator = getattr(model, "model_validate", None)
        if callable(validator):
            return validator(payload)

        parser = getattr(model, "parse_obj", None)
        if callable(parser):
            return parser(payload)

        raise AttributeError(
            f"{model.__name__} does not expose model_validate or parse_obj"
        )
