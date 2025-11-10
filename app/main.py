from typing import List
import uuid

from fastapi import Depends, FastAPI, HTTPException, Query, status
from sqlmodel import Session

from . import crud
from .database import get_db, init_db
from .schemas import (
    DetectionCreate,
    DetectionRead,
    PaginatedDetections,
    SegmentCreate,
    SegmentRead,
    SessionCreate,
    SessionDetail,
    SessionListResponse,
    SessionRead,
)

app = FastAPI(title="Buurt Sense API", version="0.1.0")


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.post("/sessions", response_model=SessionRead, status_code=status.HTTP_201_CREATED)
def create_session(
    payload: SessionCreate,
    db: Session = Depends(get_db),
) -> SessionRead:
    return crud.create_session(db, payload)


@app.get("/sessions", response_model=SessionListResponse)
def list_sessions(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> SessionListResponse:
    items, total = crud.list_sessions(db, offset=offset, limit=limit)
    return SessionListResponse(items=items, total=total)


@app.get("/sessions/{session_id}", response_model=SessionDetail)
def get_session_detail(
    session_id: uuid.UUID,
    expand: List[str] = Query(default=[]),
    include: List[str] = Query(default=[]),
    db: Session = Depends(get_db),
) -> SessionDetail:
    include_traces = "traces" in expand
    include_detections = "full_detections" in include
    detail = crud.get_session_detail(
        db,
        session_id=session_id,
        include_traces=include_traces,
        include_detections=include_detections,
    )
    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return detail


@app.post(
    "/sessions/{session_id}/segments",
    response_model=SegmentRead,
    status_code=status.HTTP_201_CREATED,
)
def create_segment(
    session_id: uuid.UUID,
    payload: SegmentCreate,
    db: Session = Depends(get_db),
) -> SegmentRead:
    segment = crud.create_segment(db, session_id=session_id, segment_in=payload)
    if segment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return segment


@app.get(
    "/sessions/{session_id}/detections",
    response_model=PaginatedDetections,
)
def list_session_detections(
    session_id: uuid.UUID,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> PaginatedDetections:
    return crud.list_detections(db, session_id=session_id, limit=limit, offset=offset)


@app.post(
    "/segments/{segment_id}/detections",
    response_model=DetectionRead,
    status_code=status.HTTP_201_CREATED,
)
def create_detection(
    segment_id: uuid.UUID,
    payload: DetectionCreate,
    db: Session = Depends(get_db),
) -> DetectionRead:
    detection = crud.create_detection(db, segment_id=segment_id, detection_in=payload)
    if detection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Segment not found")
    return detection
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
