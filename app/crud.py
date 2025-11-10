import uuid
from typing import List, Optional, Tuple

from sqlalchemy import func, select
from sqlmodel import Session

from .models import Detection, RecordingSession, Segment
from .schemas import (
    ConfigSnapshot,
    DetectionCreate,
    DetectionRead,
    DetectionSummary,
    DetectionSummaryHighConfidence,
    GPSOrigin,
    OrientationOrigin,
    OrientationTracePoint,
    PaginatedDetections,
    SegmentCreate,
    SegmentRead,
    SessionCreate,
    SessionDetail,
    SessionListItem,
    SessionRead,
    TracePoint,
)
from .utils import derive_timezone, ensure_detection_summary, round_gps_origin, session_status


def create_session(db: Session, session_in: SessionCreate) -> SessionRead:
    device_id = session_in.device_id or uuid.uuid4()
    timezone = derive_timezone(session_in.gps_origin.lat, session_in.gps_origin.lon)
    summary = ensure_detection_summary(session_in.detection_summary)

    record = RecordingSession(
        started_at=session_in.started_at,
        ended_at=session_in.ended_at,
        device_id=device_id,
        operator_alias=session_in.operator_alias,
        timezone=timezone,
        app_version=session_in.app_version,
        model_bundle_version=session_in.model_bundle_version,
        notes=session_in.notes,
        gps_origin=session_in.gps_origin.dict(exclude_none=True),
        orientation_origin=(
            session_in.orientation_origin.dict(exclude_none=True)
            if session_in.orientation_origin
            else None
        ),
        config_snapshot=session_in.config_snapshot.dict(exclude_none=True),
        detection_summary=summary.dict(by_alias=True, exclude_none=True),
        redact_location=session_in.redact_location,
    )
    db.add(record)
    db.flush()
    db.refresh(record)
    return SessionRead.from_orm(record)


def list_sessions(db: Session, offset: int = 0, limit: int = 50) -> Tuple[List[SessionListItem], int]:
    statement = (
        select(RecordingSession)
        .order_by(RecordingSession.started_at.desc())
        .offset(offset)
        .limit(limit)
    )
    sessions = list(db.exec(statement))
    count = db.exec(select(func.count()).select_from(RecordingSession)).one()

    items: List[SessionListItem] = []
    for record in sessions:
        item = SessionListItem(
            id=record.id,
            started_at=record.started_at,
            ended_at=record.ended_at,
            device_id=record.device_id,
            detection_summary=DetectionSummary.parse_obj(record.detection_summary or {}),
            config_snapshot=ConfigSnapshot.parse_obj(record.config_snapshot or {}),
            gps_origin=round_gps_origin(GPSOrigin.parse_obj(record.gps_origin or {})),
            status=session_status(record.ended_at),
        )
        items.append(item)
    return items, count


def get_session(db: Session, session_id: uuid.UUID) -> Optional[RecordingSession]:
    return db.get(RecordingSession, session_id)


def get_session_detail(
    db: Session,
    session_id: uuid.UUID,
    include_traces: bool = False,
    include_detections: bool = False,
) -> Optional[SessionDetail]:
    record = get_session(db, session_id)
    if record is None:
        return None

    segments_query = select(Segment).where(Segment.session_id == session_id)
    segment_records = list(db.exec(segments_query))
    segments = []
    for segment in segment_records:
        gps_trace = (
            [TracePoint.parse_obj(point) for point in (segment.gps_trace or [])]
            if include_traces
            else []
        )
        orientation_trace = (
            [
                OrientationTracePoint.parse_obj(point)
                for point in (segment.orientation_trace or [])
            ]
            if include_traces
            else []
        )
        segments.append(
            SegmentRead(
                id=segment.id,
                index=segment.index,
                start_ts=segment.start_ts,
                end_ts=segment.end_ts,
                file_uri=segment.file_uri,
                frame_count=segment.frame_count,
                audio_duration_ms=segment.audio_duration_ms,
                gps_trace=gps_trace,
                orientation_trace=orientation_trace,
                checksum=segment.checksum,
                size_bytes=segment.size_bytes,
            )
        )

    detections_payload: Optional[List[DetectionRead]] = None
    if include_detections:
        detections_query = (
            select(Detection)
            .join(Segment)
            .where(Segment.session_id == session_id)
        )
        detection_records = list(db.exec(detections_query))
        detections_payload = [
            DetectionRead(
                id=detection.id,
                segment_id=detection.segment_id,
                detection_class=detection.detection_class,
                confidence=detection.confidence,
                timestamp=detection.timestamp,
                gps_point=(
                    TracePoint.parse_obj(detection.gps_point)
                    if detection.gps_point
                    else None
                ),
                orientation_heading_deg=detection.orientation_heading_deg,
                model_id=detection.model_id,
                inference_latency_ms=detection.inference_latency_ms,
            )
            for detection in detection_records
        ]

    detail = SessionDetail(
        id=record.id,
        started_at=record.started_at,
        ended_at=record.ended_at,
        device_id=record.device_id,
        operator_alias=record.operator_alias,
        timezone=record.timezone,
        app_version=record.app_version,
        model_bundle_version=record.model_bundle_version,
        notes=record.notes,
        gps_origin=GPSOrigin.parse_obj(record.gps_origin or {}),
        orientation_origin=(
            OrientationOrigin.parse_obj(record.orientation_origin)
            if record.orientation_origin
            else None
        ),
        config_snapshot=ConfigSnapshot.parse_obj(record.config_snapshot or {}),
        detection_summary=DetectionSummary.parse_obj(record.detection_summary or {}),
        redact_location=record.redact_location,
        segments=segments,
        detections=detections_payload,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )
    return detail


def create_segment(
    db: Session,
    session_id: uuid.UUID,
    segment_in: SegmentCreate,
) -> Optional[SegmentRead]:
    session_record = get_session(db, session_id)
    if session_record is None:
        return None

    segment = Segment(
        session_id=session_id,
        index=segment_in.index,
        start_ts=segment_in.start_ts,
        end_ts=segment_in.end_ts,
        file_uri=segment_in.file_uri,
        frame_count=segment_in.frame_count,
        audio_duration_ms=segment_in.audio_duration_ms,
        gps_trace=[point.dict(exclude_none=True) for point in segment_in.gps_trace],
        orientation_trace=[
            point.dict(exclude_none=True)
            for point in segment_in.orientation_trace
        ],
        checksum=segment_in.checksum,
        size_bytes=segment_in.size_bytes,
    )
    db.add(segment)
    db.flush()
    db.refresh(segment)
    return SegmentRead.from_orm(segment)


def create_detection(
    db: Session,
    segment_id: uuid.UUID,
    detection_in: DetectionCreate,
) -> Optional[DetectionRead]:
    segment = db.get(Segment, segment_id)
    if segment is None:
        return None

    detection = Detection(
        segment_id=segment_id,
        detection_class=detection_in.detection_class,
        confidence=detection_in.confidence,
        timestamp=detection_in.timestamp,
        gps_point=(
            detection_in.gps_point.dict(exclude_none=True)
            if detection_in.gps_point
            else None
        ),
        orientation_heading_deg=detection_in.orientation_heading_deg,
        model_id=detection_in.model_id,
        inference_latency_ms=detection_in.inference_latency_ms,
    )
    db.add(detection)
    db.flush()
    db.refresh(detection)

    session_record = segment.session
    summary = DetectionSummary.parse_obj(session_record.detection_summary or {})
    summary.total_detections += 1
    summary.by_class[detection_in.detection_class] = (
        summary.by_class.get(detection_in.detection_class, 0) + 1
    )
    if summary.first_ts is None or detection_in.timestamp < summary.first_ts:
        summary.first_ts = detection_in.timestamp
    if summary.last_ts is None or detection_in.timestamp > summary.last_ts:
        summary.last_ts = detection_in.timestamp
    if (
        summary.high_confidence is None
        or detection_in.confidence > summary.high_confidence.confidence
    ):
        summary.high_confidence = DetectionSummaryHighConfidence(
            detection_class=detection_in.detection_class,
            confidence=detection_in.confidence,
            ts=detection_in.timestamp,
        )

    session_record.detection_summary = summary.dict(by_alias=True, exclude_none=True)
    db.add(session_record)

    return DetectionRead(
        id=detection.id,
        segment_id=detection.segment_id,
        detection_class=detection.detection_class,
        confidence=detection.confidence,
        timestamp=detection.timestamp,
        gps_point=(
            TracePoint.parse_obj(detection.gps_point) if detection.gps_point else None
        ),
        orientation_heading_deg=detection.orientation_heading_deg,
        model_id=detection.model_id,
        inference_latency_ms=detection.inference_latency_ms,
    )


def list_detections(
    db: Session,
    session_id: uuid.UUID,
    limit: int = 50,
    offset: int = 0,
) -> PaginatedDetections:
    statement = (
        select(Detection)
        .join(Segment)
        .where(Segment.session_id == session_id)
        .order_by(Detection.timestamp)
        .offset(offset)
        .limit(limit)
    )
    detections = list(db.exec(statement))
    total = db.exec(
        select(func.count(Detection.id)).join(Segment).where(Segment.session_id == session_id)
    ).one()

    items = [
        DetectionRead(
            id=detection.id,
            segment_id=detection.segment_id,
            detection_class=detection.detection_class,
            confidence=detection.confidence,
            timestamp=detection.timestamp,
            gps_point=(
                TracePoint.parse_obj(detection.gps_point)
                if detection.gps_point
                else None
            ),
            orientation_heading_deg=detection.orientation_heading_deg,
            model_id=detection.model_id,
            inference_latency_ms=detection.inference_latency_ms,
        )
        for detection in detections
    ]

    return PaginatedDetections(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )
