import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.sql import func
from sqlmodel import Column, Field, Relationship, SQLModel


class GPSPoint(SQLModel):
    lat: float
    lon: float
    ts: datetime
    accuracy_m: Optional[float] = None


class OrientationPoint(SQLModel):
    heading_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    roll_deg: Optional[float] = None
    ts: Optional[datetime] = None


class DetectionSummary(SQLModel):
    total_detections: int = 0
    by_class: Dict[str, int] = Field(default_factory=dict)
    first_ts: Optional[datetime] = None
    last_ts: Optional[datetime] = None
    high_confidence: Optional[Dict[str, Any]] = None


class ConfigSnapshot(SQLModel):
    segment_length_sec: int
    overlap_sec: int
    confidence_threshold: float
    class_cooldown_sec: Optional[int] = None


class RecordingSession(SQLModel, table=True):
    __tablename__ = "recording_sessions"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        nullable=False,
    )
    started_at: datetime
    ended_at: Optional[datetime] = None
    device_id: uuid.UUID = Field(default_factory=uuid.uuid4, index=True)
    operator_alias: Optional[str] = Field(default=None, index=True)
    timezone: str
    app_version: Optional[str] = None
    model_bundle_version: Optional[str] = None
    notes: Optional[str] = None
    redact_location: bool = False

    gps_origin: Dict[str, Any] = Field(
        sa_column=Column(JSON, nullable=False)
    )
    orientation_origin: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSON, nullable=True)
    )
    config_snapshot: Dict[str, Any] = Field(
        sa_column=Column(JSON, nullable=False)
    )
    detection_summary: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON, nullable=False, server_default="{}"),
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column_kwargs={"server_default": func.now()},
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column_kwargs={"server_default": func.now(), "onupdate": func.now()},
    )

    segments: List["Segment"] = Relationship(back_populates="session")


class Segment(SQLModel, table=True):
    __tablename__ = "segments"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        nullable=False,
    )
    session_id: uuid.UUID = Field(
        foreign_key="recording_sessions.id",
        nullable=False,
        index=True,
    )
    index: int
    start_ts: datetime
    end_ts: datetime
    file_uri: str
    frame_count: Optional[int] = None
    audio_duration_ms: Optional[int] = None
    gps_trace: List[Dict[str, Any]] = Field(
        default_factory=list,
        sa_column=Column(JSON, nullable=False),
    )
    orientation_trace: List[Dict[str, Any]] = Field(
        default_factory=list,
        sa_column=Column(JSON, nullable=False),
    )
    checksum: Optional[str] = None
    size_bytes: Optional[int] = None

    session: RecordingSession = Relationship(back_populates="segments")
    detections: List["Detection"] = Relationship(back_populates="segment")


class Detection(SQLModel, table=True):
    __tablename__ = "detections"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        nullable=False,
    )
    segment_id: uuid.UUID = Field(
        foreign_key="segments.id",
        nullable=False,
        index=True,
    )
    detection_class: str = Field(alias="class")
    confidence: float
    timestamp: datetime
    gps_point: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
    )
    orientation_heading_deg: Optional[float] = None
    model_id: Optional[str] = None
    inference_latency_ms: Optional[int] = None

    segment: Segment = Relationship(back_populates="detections")
