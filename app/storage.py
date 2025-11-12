"""Durable storage management and backend hooks for recording sessions."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Dict, Protocol
from uuid import UUID

from buurtsense.storage import (
    DetectionCreate,
    RecordingSessionCreate,
    RecordingSessionUpdate,
    SegmentCreate,
    SessionStorage,
)
from sqlalchemy.exc import NoResultFound

from .models import Session


class SessionNotFoundError(KeyError):
    """Raised when a session identifier is unknown to the store."""


class SessionAlreadyStoppedError(RuntimeError):
    """Raised when attempting to stop a session that already has an end timestamp."""


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
            SegmentCreate(
                session_id=session_id,
                index=0,
                start_ts=started_at,
                end_ts=ended_at,
                file_path=f"recordings/{session_id}/segment-0.dat",
            )
        )

        await self._storage.create_detection(
            DetectionCreate(
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

    def to_session(self) -> Session:
        return Session(
            id=UUID(self.id),
            started_at=self.started_at,
            ended_at=self.ended_at,
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

    async def create(self) -> Session:
        """Persist a new session and trigger the recording backend."""

        await self.initialize()
        started_at = self._now()
        record = await self._storage.create_session(
            RecordingSessionCreate(started_at=started_at)
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

    async def get(self, session_id: UUID) -> Session:
        """Fetch a single session by identifier."""

        record = await self._get_record(session_id)
        return self._to_session(record)

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
        )
        return adapter.to_session()

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
