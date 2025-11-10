"""FastAPI application factory for the Buurt Sense MVP."""
from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .models import Session
from .storage import SessionAlreadyStoppedError, SessionNotFoundError, SessionStore

FRONTEND_DIR = Path(__file__).parent / "frontend"


def create_app(session_store: SessionStore | None = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Parameters
    ----------
    session_store:
        Optional store instance, primarily used for injecting fakes in tests.

    Returns
    -------
    FastAPI
        Configured app instance.
    """

    app = FastAPI(title="Buurt Sense API")
    store = session_store or SessionStore()
    app.state.session_store = store

    @app.get("/health")
    def health() -> dict[str, str]:
        """Return a simple health status payload."""

        return {"status": "ok"}

    @app.post("/sessions", status_code=status.HTTP_201_CREATED)
    def create_session() -> Session:
        """Start a new recording session."""

        return store.create()

    @app.get("/sessions")
    def list_sessions() -> list[Session]:
        """List sessions ordered by most recent start timestamp."""

        sessions = store.list()
        return sorted(sessions, key=lambda session: session.started_at, reverse=True)

    @app.get("/sessions/{session_id}")
    def get_session(session_id: UUID) -> Session:
        """Retrieve a single session by identifier."""

        try:
            return store.get(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            ) from exc

    @app.post("/sessions/{session_id}/stop")
    def stop_session(session_id: UUID) -> Session:
        """Stop an active session and return the updated payload."""

        try:
            return store.stop(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            ) from exc
        except SessionAlreadyStoppedError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Session already stopped",
            ) from exc

    if FRONTEND_DIR.exists():
        static_dir = FRONTEND_DIR / "static"
        if static_dir.exists():
            app.mount("/static", StaticFiles(directory=static_dir, html=False), name="static")

        @app.get("/", response_class=HTMLResponse)
        def serve_frontend() -> HTMLResponse:
            """Serve the single-page application bundled with the backend."""

            index_path = FRONTEND_DIR / "index.html"
            if not index_path.exists():
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Frontend not built",
                )
            return HTMLResponse(index_path.read_text(encoding="utf-8"))

    return app


# ASGI entrypoint for uvicorn and similar servers.
app = create_app()
