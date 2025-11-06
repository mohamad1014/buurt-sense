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
