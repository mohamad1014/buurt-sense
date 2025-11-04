"""FastAPI-like application factory for the Buurt Sense service."""
from __future__ import annotations

from uuid import UUID

from fastapi import FastAPI, HTTPException, Request, status

from .storage import (
    SessionAlreadyStoppedError,
    SessionNotFoundError,
    SessionStore,
)


def create_app() -> FastAPI:
    app = FastAPI(title="Buurt Sense API")
    app.state.store = SessionStore()

    @app.post("/sessions", status_code=status.HTTP_201_CREATED)
    def start_session(request: Request) -> dict:
        store: SessionStore = request.app.state.store
        session = store.create()
        return session.to_dict()

    @app.post("/sessions/{session_id}/stop")
    def stop_session(session_id: UUID, request: Request) -> dict:
        store: SessionStore = request.app.state.store
        try:
            session = store.stop(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc
        except SessionAlreadyStoppedError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Session already stopped") from exc
        return session.to_dict()

    @app.get("/sessions/{session_id}")
    def get_session(session_id: UUID, request: Request) -> dict:
        store: SessionStore = request.app.state.store
        try:
            session = store.get(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc
        return session.to_dict()

    @app.get("/sessions")
    def list_sessions(request: Request) -> list[dict]:
        store: SessionStore = request.app.state.store
        return [session.to_dict() for session in store.list()]

    return app
