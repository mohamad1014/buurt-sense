"""Data models used by the Buurt Sense API."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional
from uuid import UUID, uuid4

try:  # pragma: no cover - optional dependency
    from sqlalchemy.dialects.sqlite import JSON
    from sqlalchemy.sql import func
    from sqlmodel import Column, Field, Relationship, SQLModel
except ModuleNotFoundError:  # pragma: no cover - executed only without SQLAlchemy
    SQLMODEL_AVAILABLE = False
else:  # pragma: no cover - trivial branch
    SQLMODEL_AVAILABLE = True

__all__ = ["Session"]


@dataclass(frozen=True, slots=True)
class Session:
    """Lightweight representation of a recording session."""

    id: UUID
    started_at: datetime
    ended_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""

        return {
            "id": str(self.id),
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
        }

    def model_copy(self, *, update: Mapping[str, Any] | None = None) -> "Session":
        """Return a cloned session with optional field overrides."""

        data = {
            "id": self.id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }
        if update:
            data.update(update)
        return Session(**data)

    @classmethod
    def _parse_datetime(cls, value: Any) -> datetime:
        """Parse ISO8601 strings into timezone-aware datetime objects."""

        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        raise TypeError(f"Unsupported datetime value: {value!r}")

    @classmethod
    def _parse_uuid(cls, value: Any) -> UUID:
        """Parse UUID strings or instances into UUID objects."""

        if isinstance(value, UUID):
            return value
        if isinstance(value, str):
            return UUID(value)
        raise TypeError(f"Unsupported UUID value: {value!r}")

    @classmethod
    def model_validate(cls, payload: Mapping[str, Any]) -> "Session":
        """Construct a Session from a mapping payload."""

        data = dict(payload)
        return cls(
            id=cls._parse_uuid(data["id"]),
            started_at=cls._parse_datetime(data["started_at"]),
            ended_at=cls._parse_datetime(data["ended_at"]) if data.get("ended_at") else None,
        )

    @classmethod
    def new(cls) -> "Session":
        """Factory helper that returns a freshly started session."""

        from datetime import timezone

        return cls(id=uuid4(), started_at=datetime.now(timezone.utc))


if SQLMODEL_AVAILABLE:

    class GPSPoint(SQLModel):
        """ORM model describing a GPS point."""

        lat: float
        lon: float
        ts: datetime
        accuracy_m: Optional[float] = None


    class OrientationPoint(SQLModel):
        """ORM model describing an orientation data point."""

        heading_deg: Optional[float] = None
        pitch_deg: Optional[float] = None
        roll_deg: Optional[float] = None
        ts: Optional[datetime] = None


    class DetectionSummary(SQLModel):
        """ORM payload representing aggregated detection metadata."""

        total_detections: int = 0
        by_class: Dict[str, int] = Field(default_factory=dict)
        first_ts: Optional[datetime] = None
        last_ts: Optional[datetime] = None
        high_confidence: Optional[Dict[str, Any]] = None


    class ConfigSnapshot(SQLModel):
        """Persisted configuration for a capture session."""

        segment_length_sec: int
        overlap_sec: int
        confidence_threshold: float
        class_cooldown_sec: Optional[int] = None


    class RecordingSession(SQLModel, table=True):
        """ORM definition for a recording session."""

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

        gps_origin: Dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
        orientation_origin: Optional[Dict[str, Any]] = Field(
            default=None,
            sa_column=Column(JSON, nullable=True),
        )
        config_snapshot: Dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
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
        """ORM definition for captured media segments."""

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
        """ORM definition for individual detections."""

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


    __all__.extend(
        [
            "GPSPoint",
            "OrientationPoint",
            "DetectionSummary",
            "ConfigSnapshot",
            "RecordingSession",
            "Segment",
            "Detection",
        ]
    )
