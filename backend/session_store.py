"""Session storage interfaces and in-memory implementation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional


@dataclass(slots=True)
class Session:
    """Representation of a capture session managed by the backend service."""

    session_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, str] = field(default_factory=dict)


class SessionStore(ABC):
    """Abstract storage contract for managing capture sessions.

    Implementations can back the store with different persistence layers such as
    in-memory dictionaries, relational databases, or cloud storage solutions.
    """

    @abstractmethod
    def list_sessions(self) -> Iterable[Session]:
        """Return an iterable of all active sessions."""

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve a session by its identifier."""

    @abstractmethod
    def upsert_session(self, session: Session) -> Session:
        """Create or update a session entry and return the stored session."""

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Delete a session by identifier.

        Returns
        -------
        bool
            ``True`` if the session existed and was deleted, ``False`` otherwise.
        """


class InMemorySessionStore(SessionStore):
    """A minimal in-memory session store suitable for tests and local runs."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}

    def list_sessions(self) -> Iterable[Session]:
        return list(self._sessions.values())

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def upsert_session(self, session: Session) -> Session:
        existing = self._sessions.get(session.session_id)
        if existing is not None:
            session.created_at = existing.created_at
        self._sessions[session.session_id] = session
        return session

    def delete_session(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None


__all__ = [
    "Session",
    "SessionStore",
    "InMemorySessionStore",
]
