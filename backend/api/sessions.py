"""Session management routes."""
from __future__ import annotations

from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from ..session_store import Session, SessionStore

router = APIRouter(prefix="/sessions", tags=["sessions"])


class SessionPayload(BaseModel):
    """Payload for creating or updating a session."""

    metadata: Dict[str, str] = Field(default_factory=dict, description="Arbitrary metadata about the session")


class SessionResponse(BaseModel):
    session_id: str
    created_at: str
    metadata: Dict[str, str]

    @classmethod
    def from_session(cls, session: Session) -> "SessionResponse":
        return cls(
            session_id=session.session_id,
            created_at=session.created_at.isoformat(),
            metadata=dict(session.metadata),
        )


class DeleteResponse(BaseModel):
    deleted: bool


def get_session_store(request: Request) -> SessionStore:
    store = getattr(request.app.state, "session_store", None)
    if store is None:
        raise RuntimeError("Session store dependency has not been configured on the application state.")
    return store


@router.get("/", response_model=List[SessionResponse])
async def list_sessions(store: SessionStore = Depends(get_session_store)) -> List[SessionResponse]:
    """Return all currently active sessions."""

    return [SessionResponse.from_session(session) for session in store.list_sessions()]


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, store: SessionStore = Depends(get_session_store)) -> SessionResponse:
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return SessionResponse.from_session(session)


@router.put("/{session_id}", response_model=SessionResponse, status_code=status.HTTP_200_OK)
async def upsert_session(
    session_id: str,
    payload: SessionPayload,
    store: SessionStore = Depends(get_session_store),
) -> SessionResponse:
    session = Session(session_id=session_id, metadata=dict(payload.metadata))
    stored = store.upsert_session(session)
    return SessionResponse.from_session(stored)


@router.delete("/{session_id}", response_model=DeleteResponse)
async def delete_session(session_id: str, store: SessionStore = Depends(get_session_store)) -> DeleteResponse:
    deleted = store.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return DeleteResponse(deleted=True)


__all__ = [
    "router",
]
