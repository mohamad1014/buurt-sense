"""FastAPI application factory for the Buurt Sense MVP."""

from __future__ import annotations

import asyncio
import json
from contextlib import aclosing, asynccontextmanager
from pathlib import Path
from uuid import UUID

from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import HTMLResponse, StreamingResponse
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
    async def create_session() -> Session:
        """Start a new recording session."""

        return await store.create()

    @app.get("/sessions")
    async def list_sessions() -> list[Session]:
        """List sessions ordered by most recent start timestamp."""

        return await store.list()

    @app.get("/sessions/events")
    async def session_events() -> StreamingResponse:
        """Stream session snapshots using the Server-Sent Events protocol."""

        async def event_generator():
            async with aclosing(store.subscribe()) as iterator:
                try:
                    async for sessions in iterator:
                        payload = json.dumps(
                            [session.to_dict() for session in sessions]
                        )
                        yield f"data: {payload}\n\n"
                except asyncio.CancelledError:  # pragma: no cover - cancellation ends stream
                    return

        headers = {"Cache-Control": "no-store", "Connection": "keep-alive"}
        return StreamingResponse(
            event_generator(), media_type="text/event-stream", headers=headers
        )

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
            async for sessions in store.subscribe():
                await websocket.send_json([session.to_dict() for session in sessions])
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
