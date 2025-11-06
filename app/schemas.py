import uuid
from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, root_validator, validator


class GPSOrigin(BaseModel):
    lat: float
    lon: float
    accuracy_m: Optional[float] = Field(default=None, ge=0)
    captured_at: datetime


class OrientationOrigin(BaseModel):
    heading_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    roll_deg: Optional[float] = None
    captured_at: Optional[datetime] = None


class ConfigSnapshot(BaseModel):
    segment_length_sec: int = Field(..., ge=1)
    overlap_sec: int = Field(..., ge=0)
    confidence_threshold: float = Field(..., ge=0.0, le=1.0)
    class_cooldown_sec: Optional[int] = Field(default=None, ge=0)

    @validator("segment_length_sec")
    def validate_segment_length(cls, value: int) -> int:
        if value < 10 or value > 60:
            raise ValueError("segment_length_sec must be between 10 and 60 seconds")
        return value

    @root_validator
    def validate_overlap(cls, values: Dict[str, object]) -> Dict[str, object]:
        segment_length = values.get("segment_length_sec")
        overlap = values.get("overlap_sec")
        if segment_length is not None and overlap is not None and overlap >= segment_length:
            raise ValueError("overlap_sec must be less than segment_length_sec")
        return values


class DetectionSummaryHighConfidence(BaseModel):
    detection_class: str = Field(..., alias="class")
    confidence: float
    ts: datetime

    class Config:
        allow_population_by_field_name = True


class DetectionSummary(BaseModel):
    total_detections: int = 0
    by_class: Dict[str, int] = Field(default_factory=dict)
    first_ts: Optional[datetime] = None
    last_ts: Optional[datetime] = None
    high_confidence: Optional[DetectionSummaryHighConfidence] = None


class SessionBase(BaseModel):
    started_at: datetime
    ended_at: Optional[datetime] = None
    operator_alias: Optional[str] = None
    notes: Optional[str] = None
    app_version: Optional[str] = None
    model_bundle_version: Optional[str] = None
    gps_origin: GPSOrigin
    orientation_origin: Optional[OrientationOrigin] = None
    config_snapshot: ConfigSnapshot
    detection_summary: Optional[DetectionSummary] = None
    redact_location: bool = False

    @validator("operator_alias")
    def validate_operator_alias(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        trimmed = value.strip()
        if len(trimmed) > 64:
            raise ValueError("operator_alias must be 64 characters or fewer")
        return trimmed

    @validator("notes")
    def validate_notes(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if len(value) > 500:
            raise ValueError("notes must be 500 characters or fewer")
        return value


class SessionCreate(SessionBase):
    device_id: Optional[uuid.UUID] = None


class SessionRead(SessionBase):
    id: uuid.UUID
    device_id: uuid.UUID
    timezone: str
    detection_summary: DetectionSummary
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class SessionListItem(BaseModel):
    id: uuid.UUID
    started_at: datetime
    ended_at: Optional[datetime]
    device_id: uuid.UUID
    detection_summary: DetectionSummary
    config_snapshot: ConfigSnapshot
    gps_origin: GPSOrigin
    status: Literal["active", "completed"]

    class Config:
        orm_mode = True


class TracePoint(BaseModel):
    lat: float
    lon: float
    ts: datetime
    accuracy_m: Optional[float] = Field(default=None, ge=0)


class OrientationTracePoint(BaseModel):
    heading_deg: Optional[float] = None
    ts: Optional[datetime] = None


class SegmentBase(BaseModel):
    index: int = Field(..., ge=0)
    start_ts: datetime
    end_ts: datetime
    file_uri: str
    frame_count: Optional[int] = Field(default=None, ge=0)
    audio_duration_ms: Optional[int] = Field(default=None, ge=0)
    gps_trace: List[TracePoint] = Field(default_factory=list)
    orientation_trace: List[OrientationTracePoint] = Field(default_factory=list)
    checksum: Optional[str] = None
    size_bytes: Optional[int] = Field(default=None, ge=0)


class SegmentCreate(SegmentBase):
    pass


class SegmentRead(SegmentBase):
    id: uuid.UUID

    class Config:
        orm_mode = True


class DetectionBase(BaseModel):
    detection_class: str = Field(..., alias="class")
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime
    gps_point: Optional[TracePoint] = None
    orientation_heading_deg: Optional[float] = None
    model_id: Optional[str] = None
    inference_latency_ms: Optional[int] = Field(default=None, ge=0)

    class Config:
        allow_population_by_field_name = True


class DetectionCreate(DetectionBase):
    pass


class DetectionRead(DetectionBase):
    id: uuid.UUID
    segment_id: uuid.UUID

    class Config:
        orm_mode = True


class PaginatedDetections(BaseModel):
    items: List[DetectionRead]
    total: int
    limit: int
    offset: int


class SessionDetail(SessionRead):
    segments: List[SegmentRead] = Field(default_factory=list)
    detections: Optional[List[DetectionRead]] = None


class SessionListResponse(BaseModel):
    items: List[SessionListItem]
    total: int


class SegmentResponse(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    segment: SegmentRead


class DetectionResponse(BaseModel):
    id: uuid.UUID
    segment_id: uuid.UUID
    detection: DetectionRead
