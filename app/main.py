"""FastAPI application factory for the Buurt Sense MVP."""

from __future__ import annotations

import asyncio
from contextlib import aclosing, asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .models import Session
from .schemas import SessionCreate
from .storage import (
    SessionAlreadyStoppedError,
    SessionNotFoundError,
    SessionSnapshot,
    SessionStore,
)

FRONTEND_DIR = Path(__file__).parent / "frontend"


def _parse_cursor(raw_cursor: str | None) -> int | None:
    """Convert a raw cursor query parameter into an optional integer."""

    if raw_cursor is None:
        return None

    candidate = raw_cursor.strip()
    if not candidate or candidate.lower() in {"none", "null"}:
        return None

    try:
        return int(candidate)
    except ValueError as exc:  # pragma: no cover - surfaced via FastAPI response
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="cursor must be an integer value",
        ) from exc


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

    store = session_store or SessionStore()

    @asynccontextmanager
    async def lifespan(
        app: FastAPI,
    ):  # pragma: no cover - exercised in integration tests
        await store.initialize()
        try:
            yield
        finally:
            await store.close()

    app = FastAPI(title="Buurt Sense API", lifespan=lifespan)
    app.state.session_store = store

    @app.get("/health")
    def health() -> dict[str, str]:
        """Return a simple health status payload."""

        return {"status": "ok"}

    @app.post("/sessions", status_code=status.HTTP_201_CREATED)
    async def create_session(payload: SessionCreate) -> Session:
        """Start a new recording session."""

        metadata: dict[str, Any] = {}
        if payload.device_info is not None:
            metadata["device_info"] = jsonable_encoder(payload.device_info)
        metadata["gps_origin"] = jsonable_encoder(payload.gps_origin)
        if payload.orientation_origin is not None:
            metadata["orientation_origin"] = jsonable_encoder(
                payload.orientation_origin
            )
        metadata["config_snapshot"] = jsonable_encoder(payload.config_snapshot)

        return await store.create(metadata=metadata)

    @app.get("/sessions")
    async def list_sessions() -> list[Session]:
        """List sessions ordered by most recent start timestamp."""

        return await store.list()

    @app.get("/sessions/updates")
    async def session_updates(
        cursor: str | None = Query(default=None, alias="cursor"),
        timeout: float = 10.0,
    ) -> dict[str, object]:
        """Return the latest session snapshot, optionally waiting for changes."""

        filter_cursor = _parse_cursor(cursor)

        async def wait_for_update() -> SessionSnapshot:
            async with aclosing(store.subscribe()) as iterator:
                async for snapshot in iterator:
                    if filter_cursor is None or snapshot.revision > filter_cursor:
                        return snapshot

        try:
            snapshot = await asyncio.wait_for(wait_for_update(), timeout=timeout)
        except asyncio.TimeoutError:
            sessions = await store.list()
            snapshot = SessionSnapshot(store.revision, tuple(sessions))

        return {
            "revision": snapshot.revision,
            "sessions": [session.to_dict() for session in snapshot.sessions],
        }

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: UUID) -> Session:
        """Retrieve a single session by identifier."""

        try:
            return await store.get(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            ) from exc

    @app.post("/sessions/{session_id}/stop")
    async def stop_session(session_id: UUID) -> Session:
        """Stop an active session and return the updated payload."""

        try:
            return await store.stop(session_id)
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

    @app.websocket("/ws/sessions")
    async def session_websocket(websocket: WebSocket) -> None:
        """Push live session snapshots to connected websocket clients."""

        await websocket.accept()
        try:
            async for snapshot in store.subscribe():
                await websocket.send_json(
                    [session.to_dict() for session in snapshot.sessions]
                )
        except WebSocketDisconnect:  # pragma: no cover - handled by FastAPI runtime
            return
        except RuntimeError:  # pragma: no cover - defensive guard for closed sockets
            return

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
