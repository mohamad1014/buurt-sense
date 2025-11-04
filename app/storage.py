"""In-memory storage for recording sessions."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Dict
from uuid import UUID

from .models import Session


class SessionNotFoundError(KeyError):
    """Raised when a session id is unknown."""


class SessionAlreadyStoppedError(RuntimeError):
    """Raised when attempting to stop a session that already has an end timestamp."""


@dataclass(frozen=True, slots=True)
class _SessionRecord:
    """Immutable record stored in-memory."""

    session: Session


class SessionStore:
    """Thread-safe in-memory session store."""

    def __init__(self) -> None:
        self._sessions: Dict[UUID, _SessionRecord] = {}
        self._lock = Lock()

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def create(self) -> Session:
        with self._lock:
            session = Session.new()
            self._sessions[session.id] = _SessionRecord(session=session)
            return session

    def stop(self, session_id: UUID) -> Session:
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                raise SessionNotFoundError(str(session_id))

            if record.session.ended_at is not None:
                raise SessionAlreadyStoppedError(str(session_id))

            updated = record.session.model_copy(update={"ended_at": self._now()})
            self._sessions[session_id] = _SessionRecord(session=updated)
            return updated

    def get(self, session_id: UUID) -> Session:
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                raise SessionNotFoundError(str(session_id))
            return record.session

    def list(self) -> list[Session]:
        with self._lock:
            return [record.session for record in self._sessions.values()]
