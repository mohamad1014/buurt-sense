"""FastAPI-like application factory for the Buurt Sense service."""
from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from .storage import (
    SessionAlreadyStoppedError,
    SessionNotFoundError,
    SessionStore,
)


FRONTEND_DIR = Path(__file__).parent / "frontend"


def create_app() -> FastAPI:
    app = FastAPI(title="Buurt Sense API")
    app.state.store = SessionStore()

    if FRONTEND_DIR.exists():
        static_dir = FRONTEND_DIR / "static"
        if static_dir.exists():
            app.mount(
                "/static",
                StaticFiles(directory=static_dir, html=False),
                name="static",
            )

        @app.get("/", response_class=HTMLResponse)
        def read_frontend() -> Response:
            """Serve the control panel single-page application."""

            index_path = FRONTEND_DIR / "index.html"
            if not index_path.exists():
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Frontend not built")

            return HTMLResponse(index_path.read_text(encoding="utf-8"))

    @app.post("/sessions", status_code=status.HTTP_201_CREATED)
    def start_session(request: Request) -> dict:
        """Create a new recording session."""

        store: SessionStore = request.app.state.store
        session = store.create()
        return session.to_dict()

    @app.post("/sessions/{session_id}/stop")
    def stop_session(session_id: UUID, request: Request) -> dict:
        """Mark the specified session as stopped."""

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
        """Retrieve a single session by its identifier."""

        store: SessionStore = request.app.state.store
        try:
            session = store.get(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found") from exc
        return session.to_dict()

    @app.get("/sessions")
    def list_sessions(request: Request) -> list[dict]:
        """List all sessions held in memory."""

        store: SessionStore = request.app.state.store
        return [session.to_dict() for session in store.list()]

    return app
