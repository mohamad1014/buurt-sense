"""FastAPI application factory for the Buurt Sense MVP."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from buurtsense.db import RecordingSession
from buurtsense.runtime import (
    RecordingOrchestrator,
    SessionAlreadyStoppedError,
    SessionNotFoundError,
)
from buurtsense.storage import SessionStorage
from sqlalchemy.exc import NoResultFound

from .models import Session

FRONTEND_DIR = Path(__file__).parent / "frontend"


def create_app(
    *,
    session_storage: SessionStorage | None = None,
    segment_interval: float = 0.25,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(title="Buurt Sense API")
    storage = session_storage or SessionStorage()
    orchestrator = RecordingOrchestrator(storage, segment_interval=segment_interval)
    app.state.session_storage = storage
    app.state.recording_orchestrator = orchestrator

    @app.on_event("startup")
    async def _startup() -> None:
        await storage.initialize()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await orchestrator.shutdown()
        await storage.close()

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Return a simple health status payload."""

        return {"status": "ok"}

    def _get_orchestrator(request: Request) -> RecordingOrchestrator:
        orchestrator_dep = getattr(request.app.state, "recording_orchestrator", None)
        if orchestrator_dep is None:
            raise RuntimeError("Recording orchestrator not configured")
        return orchestrator_dep

    def _get_storage(request: Request) -> SessionStorage:
        storage_dep = getattr(request.app.state, "session_storage", None)
        if storage_dep is None:
            raise RuntimeError("Session storage not configured")
        return storage_dep

    def _ensure_timezone(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)

    def _serialise_session(record: RecordingSession) -> dict[str, str | None]:
        session = Session(
            id=UUID(record.id),
            started_at=_ensure_timezone(record.started_at),
            ended_at=_ensure_timezone(record.ended_at),
        )
        return session.to_dict()

    @app.post("/sessions", status_code=status.HTTP_201_CREATED)
    async def create_session(
        orchestrator_dep: RecordingOrchestrator = Depends(_get_orchestrator),
    ) -> dict[str, str | None]:
        """Start a new recording session."""

        record = await orchestrator_dep.start_session()
        return _serialise_session(record)

    @app.get("/sessions")
    async def list_sessions(
        storage_dep: SessionStorage = Depends(_get_storage),
    ) -> list[dict[str, str | None]]:
        """List sessions ordered by most recent start timestamp."""

        records: list[RecordingSession] = await storage_dep.list_sessions()
        sorted_records = sorted(
            records, key=lambda record: record.started_at, reverse=True
        )
        return [_serialise_session(record) for record in sorted_records]

    @app.get("/sessions/{session_id}")
    async def get_session(
        session_id: UUID,
        storage_dep: SessionStorage = Depends(_get_storage),
    ) -> dict[str, str | None]:
        """Retrieve a single session by identifier."""

        try:
            record = await storage_dep.get_session(str(session_id))
        except NoResultFound as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            ) from exc
        return _serialise_session(record)

    @app.post("/sessions/{session_id}/stop")
    async def stop_session(
        session_id: UUID,
        orchestrator_dep: RecordingOrchestrator = Depends(_get_orchestrator),
    ) -> dict[str, str | None]:
        """Stop an active session and return the updated payload."""

        try:
            record = await orchestrator_dep.stop_session(str(session_id))
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
        return _serialise_session(record)

    if FRONTEND_DIR.exists():
        static_dir = FRONTEND_DIR / "static"
        if static_dir.exists():
            app.mount(
                "/static", StaticFiles(directory=static_dir, html=False), name="static"
            )

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
