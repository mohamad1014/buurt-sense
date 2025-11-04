"""Minimal FastAPI application for the Buurt Sense MVP."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class Session(BaseModel):
    """Represents a lightweight recording session."""

    id: UUID
    started_at: datetime
    ended_at: datetime | None = None


app = FastAPI(title="Buurt Sense MVP", version="0.1.0")
_sessions: Dict[UUID, Session] = {}


@app.get("/health")
def health() -> dict[str, str]:
    """Health probe endpoint."""

    return {"status": "ok"}


@app.post("/sessions", response_model=Session, status_code=201)
def start_session() -> Session:
    """Start a new recording session."""

    session = Session(id=uuid4(), started_at=datetime.now(timezone.utc))
    _sessions[session.id] = session
    return session


@app.post("/sessions/{session_id}/stop", response_model=Session)
def stop_session(session_id: UUID) -> Session:
    """Stop an active recording session."""

    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.ended_at is None:
        session = session.copy(update={"ended_at": datetime.now(timezone.utc)})
        _sessions[session_id] = session

    return session
